# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from chex import dataclass

import lithox.defaults as d
import lithox.paths as p
from lithox.utilities.fft import centered_fft_2d, centered_ifft_2d
from lithox.utilities.spatial import pad_to_shape_2d, crop_margin_2d, pad_margin_2d
from lithox.utilities.io import load_npy


@dataclass
class SimulationOutput:
    """Container for simulator outputs.

    Attributes:
      aerial: Aerial intensity image as a JAX array with shape (height, width) or
        with leading batch-like dimensions.
      resist: Continuous resist activation (0..1) after thresholding/sigmoid.
      printed: Binary printed result (0 or 1) after applying print threshold.
    """
    aerial: jax.Array
    resist: jax.Array
    printed: jax.Array


class LithographySimulator(eqx.Module):
    """End-to-end lithography simulator module.

    This module performs three stages:
    1) Aerial image simulation from a binary/real-valued mask via frequency-domain
       convolution with precomputed Fourier-space kernels.
    2) Resist response via a sigmoid nonlinearity.
    3) Printed pattern via thresholding of the resist response.

    The module can be configured with different kernel sets (e.g., "focus", "defocus"),
    dose levels, thresholds, and numeric dtype. Kernels/scales are treated as constants
    unless `trainable=True`, in which case gradients may flow through them.

    Attributes:
      dose: Exposure dose multiplier applied to the input mask.
      resist_threshold: Midpoint parameter for the resist sigmoid.
      resist_steepness: Slope parameter for the resist sigmoid.
      print_threshold: Threshold applied to the resist to produce the binary print.
      kernels: Fourier-domain kernels with shape [K, Hk, Wk] (complex).
      kernels_ct: Conjugate/transpose-related Fourier-domain kernels used in backward pass.
      scales: Non-negative per-kernel weights with shape [K].
      margin: Optional symmetric padding (in pixels) applied around inputs/outputs.
      kernel_type: String identifier of the kernel set ("focus" or "defocus").
      dtype: JAX dtype used for internal computations (e.g., jnp.float32).
      trainable: If False (default), gradients are stopped through kernels and scales.
    """

    dose: float
    resist_threshold: float
    resist_steepness: float
    print_threshold: float

    kernels: jax.Array
    kernels_ct: jax.Array
    scales: jax.Array

    margin: int = eqx.field(static=True)
    kernel_type: str = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    trainable: bool = eqx.field(static=True, default=False)

    def __init__(
            self,
            kernel_type: Literal["focus", "defocus"] = "focus",
            *,
            dose: float = d.DOSE,
            resist_threshold: float = d.RESIST_THRESHOLD,
            resist_steepness: float = d.RESIST_STEEPNESS,
            print_threshold: float = d.PRINT_THRESHOLD,
            dtype: jnp.dtype = d.DTYPE,
            trainable: bool = False,
            margin: int = 0,
    ):
        """Initialize a LithographySimulator.

        Loads kernel stacks and per-kernel scales from package resources with a
        filesystem fallback.

        Args:
          kernel_type: Which kernel set to use ("focus" or "defocus").
          dose: Exposure dose multiplier.
          resist_threshold: Sigmoid midpoint for the resist response.
          resist_steepness: Sigmoid steepness for the resist response.
          print_threshold: Threshold applied to the resist to get a binary print.
          dtype: Numeric dtype for internal computations.
          trainable: If True, allow gradients to flow through kernels/scales.
          margin: Symmetric padding (in pixels) applied around inputs and removed
            from outputs; useful to reduce boundary artifacts.
        """
        self.kernel_type: str = kernel_type

        # Load kernels and scales; each call returns a JAX array.
        self.kernels =  load_npy(module="lithox.kernels", path=p.KERNELS_DIRECTORY, filename=f"{kernel_type}.npy")
        self.kernels_ct =  load_npy(module="lithox.kernels", path=p.KERNELS_DIRECTORY, filename=f"{kernel_type}_ct.npy")
        self.scales = load_npy(module="lithox.scales", path=p.SCALES_DIRECTORY, filename=f"{kernel_type}.npy")

        self.dose = dose
        self.margin = margin
        self.resist_threshold = resist_threshold
        self.resist_steepness = resist_steepness
        self.print_threshold = print_threshold
        self.dtype = dtype
        self.trainable = trainable

    def __call__(self, mask: jax.Array, margin: int | None = None) -> SimulationOutput:
        """Run the full simulation pipeline on a mask.

        Steps:
          1) Optional symmetric padding by `margin` (or `self.margin` if None).
          2) Aerial simulation.
          3) Resist response (sigmoid).
          4) Printed result (binary threshold).
          5) Optional cropping to remove the initial padding.

        Args:
          mask: Input mask array with last two axes (height, width); leading
            dimensions are preserved (e.g., batch).
          margin: Overrides the instance padding when provided.

        Returns:
          SimulationOutput with fields (aerial, resist, printed), each matching
          the spatial size of the original `mask`.
        """
        margin_to_use = self.margin if margin is None else margin

        if margin_to_use > 0:
            # Pad to mitigate boundary effects in frequency-domain convolution.
            mask = pad_margin_2d(mask, margin_to_use)

        # Run each stage (aerial -> resist -> printed).
        aerial = self.simulate_aerial_from_mask(mask=mask, margin=0)
        resist = self.simulate_resist_from_aerial(aerial=aerial)
        printed = self.simulate_printed_from_resist(resist=resist)

        if margin_to_use > 0:
            # Remove the extra border introduced for convolution stability.
            aerial = crop_margin_2d(aerial, margin_to_use)
            resist = crop_margin_2d(resist, margin_to_use)
            printed = crop_margin_2d(printed, margin_to_use)

        return SimulationOutput(
            aerial=aerial,
            resist=resist,
            printed=printed,
        )

    def simulate_aerial_from_mask(self, mask: jax.Array, margin: int | None = None) -> jax.Array:
        """Simulate aerial intensity from a mask.

        Applies frequency-domain convolution with a bank of kernels and combines
        per-kernel intensities via non-negative scales.

        Args:
          mask: Input mask with last two axes (height, width). Can include
            leading batch-like dimensions.
          margin: Optional symmetric padding in pixels. If None, uses `self.margin`.

        Returns:
          Aerial intensity image with the same spatial size as `mask` (after any
          optional padding/cropping).
        """
        margin_to_use = self.margin if margin is None else margin

        if margin_to_use > 0:
            mask = pad_margin_2d(mask, margin_to_use)

        kernels = self.kernels
        kernels_ct = self.kernels_ct
        scales = self.scales

        if not self.trainable:
            # Treat as constants; avoids computing/keeping grads.
            kernels = jax.lax.stop_gradient(kernels)
            kernels_ct = jax.lax.stop_gradient(kernels_ct)
            scales = jax.lax.stop_gradient(scales)

        aerial = simulate_aerial_from_mask(
            mask=mask.astype(self.dtype),
            dose=self.dose,
            kernels_fourier=kernels,  # [K,Hk,Wk] complex
            kernels_fourier_ct=kernels_ct,
            scales=scales,            # [K,] non-negative
        )

        if margin_to_use > 0:
            aerial = crop_margin_2d(aerial, margin_to_use)

        return aerial

    def simulate_resist_from_aerial(self, aerial: jax.Array) -> jax.Array:
        """Compute resist activation from the aerial intensity via a sigmoid.

        Args:
          aerial: Aerial intensity array.

        Returns:
          Resist activation in [0, 1] with the same shape as `aerial`.
        """
        return jax.nn.sigmoid(
            self.resist_steepness * (aerial - self.resist_threshold)
        )

    def simulate_printed_from_resist(self, resist: jax.Array) -> jax.Array:
        """Threshold the resist activation to obtain a binary printed result.

        Args:
          resist: Resist activation in [0, 1].

        Returns:
          Binary array (0 or 1) of the same shape as `resist`.
        """
        return (resist > self.print_threshold).astype(resist.dtype)

    @classmethod
    def nominal(cls, **overrides) -> "LithographySimulator":
        """Factory: nominal dose, focused kernels."""
        return cls(kernel_type="focus", dose=d.DOSE_NOMINAL, **overrides)

    @classmethod
    def maximum(cls, **overrides) -> "LithographySimulator":
        """Factory: maximum dose, focused kernels."""
        return cls(kernel_type="focus", dose=d.DOSE_MAX, **overrides)

    @classmethod
    def minimum(cls, **overrides) -> "LithographySimulator":
        """Factory: minimum dose, defocused kernels."""
        return cls(kernel_type="defocus", dose=d.DOSE_MIN, **overrides)


def convolve_frequency_domain(
    image_stack: jax.Array,
    kernels_fourier: jax.Array,
) -> jax.Array:
    """Apply frequency-domain convolution to a stack of fields.

    No additional padding is applied; callers should manage padding/cropping if
    they need to mitigate boundary effects.

    Args:
      image_stack: Array of shape [..., K, H, W] (complex or real) or [..., 1, H, W].
        Real inputs are cast to complex64 for convolution.
      kernels_fourier: Fourier-domain kernels of shape [K, Hk, Wk] (complex).
        They will be zero-padded to match the input spatial size.

    Returns:
      Convolved complex stack with shape [..., K, H, W].
    """
    # Ensure complex dtype for frequency-domain multiplication.
    image_stack_c = image_stack.astype(jnp.complex64)

    # Spatial dimensions of the input.
    H, W = image_stack_c.shape[-2:]

    # Pad kernels to match input spatial size.
    kernels_padded = pad_to_shape_2d(kernels_fourier, (H, W))  # [K, H, W]

    # Centered FFT of the input stack.
    stack_ft = centered_fft_2d(image_stack_c)  # [..., K, H, W]

    # Broadcast kernels across leading dimensions.
    bshape = (1,) * (stack_ft.ndim - 3) + kernels_padded.shape
    product_ft = stack_ft * kernels_padded.reshape(bshape)

    # Inverse transform back to spatial domain.
    return centered_ifft_2d(product_ft)


@jax.custom_vjp
def simulate_aerial_from_mask(
    mask: jax.Array,
    dose: float,
    kernels_fourier: jax.Array,      # [K,Hk,Wk] complex
    kernels_fourier_ct: jax.Array,   # [K,Hk,Wk] complex (used in backward)
    scales: jax.Array,               # [K] ≥ 0
) -> jax.Array:
    """Compute aerial intensity from a mask using kernel bank convolution.

    The forward model:
      I = sum_k scales[k] * | F^{-1}( F(dose * mask) * kernels_fourier[k] ) |^2

    Args:
      mask: Real-valued mask array with last two axes (height, width).
      dose: Exposure dose multiplier applied to the mask.
      kernels_fourier: Fourier-domain kernels with shape [K, Hk, Wk] (complex).
      kernels_fourier_ct: Kernels used during the backward pass (same shape).
      scales: Non-negative per-kernel weights with shape [K].

    Returns:
      Aerial intensity image with the same spatial size as `mask`.
    """
    # Treat constants as stop-gradient to avoid unnecessary backprop into them.
    kernels_fourier = jax.lax.stop_gradient(kernels_fourier)
    kernels_fourier_ct = jax.lax.stop_gradient(kernels_fourier_ct)
    scales = jax.lax.stop_gradient(scales)

    # Apply dose and ensure a stable float dtype.
    dosed_mask = (dose * mask).astype(jnp.float32)

    # Convolve mask with all kernels in one go by expanding a kernel axis.
    fields = convolve_frequency_domain(
        image_stack=jnp.expand_dims(dosed_mask, axis=-3),
        kernels_fourier=kernels_fourier,
    )  # [..., K, H, W]

    # Intensity is squared magnitude of complex field.
    intensities = jnp.square(jnp.abs(fields))

    # Weighted sum across kernels.
    return jnp.sum(scales[..., None, None] * intensities, axis=-3)


def simulate_aerial_from_mask_fwd(
    mask: jax.Array,
    dose: float,
    kernels_fourier: jax.Array,
    kernels_fourier_ct: jax.Array,
    scales: jax.Array,
):
    """Forward pass for custom VJP.

    Returns both the primal output and residuals required by the backward pass.
    Residuals include intermediate fields and constants to efficiently compute
    gradients without recomputing convolutions.

    Args:
      mask: Input mask.
      dose: Exposure dose.
      kernels_fourier: Fourier-domain kernels.
      kernels_fourier_ct: Kernels used in backward computations.
      scales: Non-negative per-kernel weights.

    Returns:
      A tuple (y, residuals) where:
        y: Aerial intensity image.
        residuals: Tuple containing (dosed_mask, fields_main, kernels_fourier,
          kernels_fourier_ct, scales, dose).
    """
    # Treat constants as non-differentiable for efficiency/stability.
    kernels_fourier = jax.lax.stop_gradient(kernels_fourier)
    kernels_fourier_ct = jax.lax.stop_gradient(kernels_fourier_ct)
    scales = jax.lax.stop_gradient(scales)

    dosed_mask = (dose * mask).astype(jnp.float32)

    # Main convolution to obtain complex fields.
    fields_main = convolve_frequency_domain(
        image_stack=jnp.expand_dims(dosed_mask, axis=-3),
        kernels_fourier=kernels_fourier,
    )  # [..., K, H, W]

    intensities = jnp.square(jnp.abs(fields_main))
    y = jnp.sum(scales[..., None, None] * intensities, axis=-3)

    # Save intermediates for the backward rule.
    residuals = (dosed_mask, fields_main, kernels_fourier, kernels_fourier_ct, scales, dose)
    return y, residuals


def simulate_aerial_from_mask_bwd(
    residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, float],
    grad_aerial: jax.Array,
):
    """Backward pass (VJP) for `simulate_aerial_from_mask`.

    Computes the gradient w.r.t. the input mask given the gradient of the aerial
    intensity. Gradients w.r.t. dose and kernels/scales are not propagated by
    design (returned as None), which matches the stop-gradient behavior in the
    forward pass unless `trainable=True` is implemented at a higher level.

    Args:
      residuals: Tuple saved by the forward pass:
        (dosed_mask, fields_main, kernels_fourier, kernels_fourier_ct, scales, dose)
      grad_aerial: Incoming gradient w.r.t. the aerial intensity.

    Returns:
      A tuple of tangents aligned with the primal inputs:
        (grad_mask, None, None, None, None)
    """
    dosed_mask, fields_main, kernels_fourier, kernels_fourier_ct, scales, dose = residuals

    # Align grad with the kernel axis.
    grad = jnp.expand_dims(grad_aerial, axis=-3)  # [..., 1, H, W]

    # Convolution with the "ct" kernels using the mask.
    fields_ct = convolve_frequency_domain(
        image_stack=jnp.expand_dims(dosed_mask, axis=-3),
        kernels_fourier=kernels_fourier_ct,
    )

    # Two conjugate-like terms that arise from differentiating |field|^2.
    term1 = convolve_frequency_domain(
        image_stack=fields_ct * grad,
        kernels_fourier=kernels_fourier,
    )
    term2 = convolve_frequency_domain(
        image_stack=fields_main * grad,
        kernels_fourier=kernels_fourier_ct,
    )

    # Sum over kernels with non-negative scales.
    summed = jnp.sum(scales[..., None, None] * (term1 + term2), axis=-3)

    # Only the real part contributes to the mask gradient in spatial domain.
    grad_mask = dose * summed.real

    # Return tangents for: (mask, dose, kernels_fourier, kernels_fourier_ct, scales)
    return (grad_mask, None, None, None, None)

# Bind custom_vjp rules.
simulate_aerial_from_mask.defvjp(
    simulate_aerial_from_mask_fwd,
    simulate_aerial_from_mask_bwd,
)
