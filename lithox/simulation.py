# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import Final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Complex, Float

import lithox.defaults as d
import lithox.paths as p
from lithox.utilities.fft import centered_fft_2d, centered_ifft_2d
from lithox.utilities.io import load_npy
from lithox.utilities.spatial import pad_to_shape_2d, crop_margin_2d, pad_margin_2d

DTYPE_COMPUTE_REAL: Final = jnp.float32
DTYPE_COMPUTE_COMPLEX: Final = jnp.complex64

# Bundled kernels are static Equinox fields; suppress the expected JAX warning here only.
warnings.filterwarnings(
    "ignore",
    message=r"A JAX array is being set as static!.*",
    category=UserWarning,
)

Image: TypeAlias = Float[Array, "*batch H W"]
Kernels: TypeAlias = Complex[Array, "K H W"]
Scales: TypeAlias = Float[Array, "K"]


def printed_from_resist(resist: Image) -> Image:
    """Hard print P = 𝟙[R > BINARIZATION_THRESHOLD] from resist activation."""
    resist_f = resist.astype(DTYPE_COMPUTE_REAL)
    tau_b = jnp.asarray(d.BINARIZATION_THRESHOLD, DTYPE_COMPUTE_REAL)
    return (resist_f > tau_b).astype(resist.dtype)


class SimulationOutput(eqx.Module):
    """Container for simulator outputs.

    Notation (MOSAIC [Gao DAC'14], Neural-ILT [Jiang ICCAD'20]):
      I — aerial intensity (`aerial`)
      R — resist activation (`resist`); many papers denote this Z
      P — printed pattern (`printed` property) with fixed binarization at 0.5

    Attributes:
      aerial: Aerial intensity I.
      resist: Resist activation R = σ(α(I − τ)) in (0, 1).

    Properties:
      printed: Binary print P = 𝟙[R > 0.5].
      printed_ste: Binary forward, gradients through R (straight-through).
    """
    aerial: Image
    resist: Image

    @property
    def printed(self) -> Image:
        """Hard print P = 𝟙[R > BINARIZATION_THRESHOLD]."""
        return printed_from_resist(self.resist)

    @property
    def printed_ste(self) -> Image:
        """Binary forward with straight-through gradients through `resist`."""
        hard = self.printed
        soft = self.resist.astype(hard.dtype)
        return jax.lax.stop_gradient(hard - soft) + soft


class LithographySimulator(eqx.Module):
    """End-to-end lithography simulator module.

    This module performs two forward stages:
    1) Aerial image simulation from a mask via frequency-domain convolution with
       precomputed Fourier-space kernels.
    2) Resist response R = σ(α(I − τ)) via a sigmoid on aerial intensity.

    Binary print P is derived on `SimulationOutput` (hard threshold or STE).

    The module can be configured with different kernel sets (e.g., "focus", "defocus"),
    dose levels, and thresholds. Kernels/scales are static (not traced).
    All arrays use float32 (complex64 in Fourier space).

    Attributes:
      dose: Exposure dose multiplier applied to the input mask.
      resist_threshold: Intensity threshold τ on I (sigmoid midpoint).
      resist_steepness: Sigmoid steepness α for the resist response.
      kernels: Fourier-domain kernels with shape [K, Hk, Wk] (complex).
      kernels_ct: Conjugate/transpose-related Fourier-domain kernels used in backward pass.
      scales: Non-negative per-kernel weights with shape [K].
      margin: Optional symmetric padding (in pixels) applied around inputs/outputs.
      kernel_type: String identifier of the kernel set ("focus" or "defocus").
    """

    dose: float = eqx.field(static=True)
    resist_threshold: float = eqx.field(static=True)
    resist_steepness: float = eqx.field(static=True)

    kernels: Kernels = eqx.field(static=True)
    kernels_ct: Kernels = eqx.field(static=True)
    scales: Scales = eqx.field(static=True)

    margin: int = eqx.field(static=True)
    kernel_type: Literal["focus", "defocus"] = eqx.field(static=True)

    @staticmethod
    def _validate_kernel_bank(
        kernel_type: str,
        kernels: jax.Array,
        kernels_ct: jax.Array,
        scales: jax.Array,
    ) -> None:
        """Fail fast on mismatched or invalid packaged kernel assets."""
        if kernels.shape != kernels_ct.shape:
            raise ValueError(
                f"{kernel_type}: kernels shape {kernels.shape} != kernels_ct {kernels_ct.shape}."
            )
        num_modes = kernels.shape[0]
        if scales.shape != (num_modes,):
            raise ValueError(
                f"{kernel_type}: scales shape {scales.shape} != ({num_modes},) kernel modes."
            )
        if bool(jnp.any(scales < 0)):
            raise ValueError(f"{kernel_type}: scales must be non-negative.")

    def __init__(
            self,
            kernel_type: Literal["focus", "defocus"] = "focus",
            *,
            dose: float = d.DOSE,
            resist_threshold: float = d.RESIST_THRESHOLD,
            resist_steepness: float = d.RESIST_STEEPNESS,
            margin: int = 0,
    ):
        """Initialize a LithographySimulator.

        Loads kernel stacks and per-kernel scales from package resources with a
        filesystem fallback.

        Args:
          kernel_type: Which kernel set to use ("focus" or "defocus").
          dose: Exposure dose multiplier.
          resist_threshold: Intensity threshold τ on I (sigmoid midpoint).
          resist_steepness: Sigmoid steepness α for the resist response.
          margin: Symmetric padding (in pixels) applied around inputs and removed
            from outputs; useful to reduce boundary artifacts.
        """
        self.margin = margin
        self.kernel_type = kernel_type

        self.dose = float(dose)
        self.resist_threshold = float(resist_threshold)
        self.resist_steepness = float(resist_steepness)

        kernels = load_npy(module="lithox.kernels", path=p.KERNELS_DIRECTORY, filename=f"{kernel_type}.npy")
        kernels_ct = load_npy(module="lithox.kernels", path=p.KERNELS_DIRECTORY, filename=f"{kernel_type}_ct.npy")
        scales = load_npy(module="lithox.scales", path=p.SCALES_DIRECTORY, filename=f"{kernel_type}.npy")

        self._validate_kernel_bank(kernel_type, kernels, kernels_ct, scales)

        self.scales = scales.astype(dtype=DTYPE_COMPUTE_REAL)
        self.kernels = kernels.astype(dtype=DTYPE_COMPUTE_COMPLEX)
        self.kernels_ct = kernels_ct.astype(dtype=DTYPE_COMPUTE_COMPLEX)

    def __call__(self, mask: ArrayLike, margin: int | None = None) -> SimulationOutput:
        """Run the full simulation pipeline on a mask.

        Steps:
          1) Optional symmetric padding by `margin` (or `self.margin` if None).
          2) Aerial simulation I.
          3) Resist response R = σ(α(I − τ)).
          4) Optional cropping to remove the initial padding.

        Use `SimulationOutput.printed` or `.printed_ste` for P (not computed here).

        Args:
          mask: Input mask (any real dtype); cast to float32 internally. Last two
            axes are (height, width), each at least `defaults.MIN_MASK_SIZE` (35).
            Leading dimensions are preserved (e.g., batch).
          margin: Overrides the instance padding when provided.

        Returns:
          SimulationOutput with float32 `aerial` and `resist`, same spatial shape
          as the input `mask`.
        """
        mask = jnp.asarray(mask, dtype=DTYPE_COMPUTE_REAL)
        self._check_mask_size(mask)

        aerial = self.simulate_aerial_from_mask(mask=mask, margin=margin)
        resist = self.simulate_resist_from_aerial(aerial=aerial)
        return SimulationOutput(aerial=aerial, resist=resist)

    def _check_mask_size(self, mask: jax.Array) -> None:
        height, width = mask.shape[-2:]
        min_size = d.MIN_MASK_SIZE
        if height < min_size or width < min_size:
            raise ValueError(
                f"Mask spatial size must be at least {min_size}×{min_size} "
                f"(kernel extent); got {height}×{width}."
            )

    def simulate_aerial_from_mask(self, mask: Image, margin: int | None = None) -> Image:
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
        if mask.ndim < 2:
            raise TypeError("mask must have at least 2 dims with trailing H, W")

        margin_to_use = self.margin if margin is None else margin
        if margin_to_use > 0:
            mask = pad_margin_2d(mask, margin_to_use)

        aerial = simulate_aerial_from_mask(
            mask=mask.astype(DTYPE_COMPUTE_REAL),
            dose=self.dose,
            kernels_fourier=self.kernels,  # [K,Hk,Wk] complex
            kernels_fourier_ct=self.kernels_ct,
            scales=self.scales,  # [K,] non-negative
        )

        if margin_to_use > 0:
            aerial = crop_margin_2d(aerial, margin_to_use)

        return aerial

    def simulate_resist_from_aerial(self, aerial: Image) -> Image:
        """Compute resist activation R = σ(α(I − τ)) from aerial intensity.

        Matches the compact resist model in MOSAIC and Neural-ILT (one sigmoid
        on intensity; no separate development nonlinearity). Many ILT papers
        write Z for this quantity.

        Args:
          aerial: Aerial intensity I.

        Returns:
          Resist activation R in (0, 1) with the same shape as `aerial`.
        """
        aerial = aerial.astype(dtype=DTYPE_COMPUTE_REAL)
        resist_steepness = jnp.asarray(self.resist_steepness, DTYPE_COMPUTE_REAL)
        resist_threshold = jnp.asarray(self.resist_threshold, DTYPE_COMPUTE_REAL)
        return jax.nn.sigmoid(resist_steepness * (aerial - resist_threshold))

    # NOTE: `printed` / `printed_ste` are derived on `SimulationOutput`.

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
    image_stack: Complex[Array, "*batch K H W"] | Complex[Array, "*batch 1 H W"] | Float[Array, "*batch K H W"] | Float[Array, "*batch 1 H W"],
    kernels_fourier: Kernels,
) -> Complex[Array, "*batch K H W"]:
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
    image_stack_complex = image_stack.astype(dtype=DTYPE_COMPUTE_COMPLEX)

    # Spatial dimensions of the input.
    height, width = image_stack_complex.shape[-2:]

    # Pad kernels to match input spatial size.
    kernels_padded = pad_to_shape_2d(kernels_fourier, target_shape=(height, width))  # [K, H, W]

    # Centered FFT of the input stack.
    stack_ft = centered_fft_2d(image_stack_complex)  # [..., K, H, W]

    # Broadcast kernels across leading dimensions.
    bshape = (1,) * (stack_ft.ndim - 3) + kernels_padded.shape
    product_ft = stack_ft * kernels_padded.reshape(bshape)

    # Inverse transform back to spatial domain.
    return centered_ifft_2d(product_ft)


@jax.custom_vjp
def simulate_aerial_from_mask(
    mask: Image,
    dose: float,
    kernels_fourier: Kernels,
    kernels_fourier_ct: Kernels,
    scales: Scales,
) -> Image:
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
    # Apply dose and ensure a stable float dtype.
    dosed_mask = jnp.asarray(dose, DTYPE_COMPUTE_REAL) * mask.astype(DTYPE_COMPUTE_REAL)

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
    mask: Image,
    dose: float,
    kernels_fourier: Kernels,
    kernels_fourier_ct: Kernels,
    scales: Scales,
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
    dosed_mask = jnp.asarray(dose, DTYPE_COMPUTE_REAL) * mask.astype(DTYPE_COMPUTE_REAL)

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
    residuals: tuple[Image, Complex[Array, "*batch K H W"], Kernels, Kernels, Scales, float],
    grad_aerial: Image,
):
    """Backward pass (VJP) for `simulate_aerial_from_mask`.

    Computes the gradient w.r.t. the input mask given the gradient of the aerial
    intensity.

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
