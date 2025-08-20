# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from chex import dataclass

import lithox.defaults as d
import lithox.paths as p
from lithox.utilities import center_pad_2d, centered_fft_2d, centered_ifft_2d, load_npy


@dataclass
class SimulationOutput:
    aerial: jax.Array
    resist: jax.Array
    printed: jax.Array


class LithographySimulator(eqx.Module):
    dose: float
    resist_threshold: float
    resist_steepness: float
    print_threshold: float

    kernels: jax.Array
    kernels_ct: jax.Array
    scales: jax.Array

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
    ):
        self.kernel_type: str = kernel_type

        self.kernels =  load_npy(module="lithox.kernels", path=p.KERNELS_DIRECTORY, filename=f"{kernel_type}.npy")
        self.kernels_ct =  load_npy(module="lithox.kernels", path=p.KERNELS_DIRECTORY, filename=f"{kernel_type}_ct.npy")
        self.scales = load_npy(module="lithox.scales", path=p.SCALES_DIRECTORY, filename=f"{kernel_type}.npy")

        self.dose = dose
        self.resist_threshold = resist_threshold
        self.resist_steepness = resist_steepness
        self.print_threshold = print_threshold
        self.dtype = dtype
        self.trainable = trainable

    def __call__(self, mask: jax.Array) -> SimulationOutput:
        aerial = self.simulate_aerial_from_mask(mask=mask)
        resist = self.simulate_resist_from_aerial(aerial=aerial)
        printed = self.simulate_printed_from_resist(resist=resist)
        return SimulationOutput(
            aerial=aerial,
            resist=resist,
            printed=printed,
        )

    def simulate_aerial_from_mask(self, mask: jax.Array) -> jax.Array:
        kernels = self.kernels
        kernels_ct = self.kernels_ct
        scales = self.scales

        if not self.trainable:
            kernels = jax.lax.stop_gradient(kernels)
            kernels_ct = jax.lax.stop_gradient(kernels_ct)
            scales = jax.lax.stop_gradient(scales)

        return simulate_aerial_from_mask(
            mask=mask.astype(self.dtype),
            dose=self.dose,
            kernels_fourier=kernels,  # [K,Hk,Wk] complex
            kernels_fourier_ct=kernels_ct,
            scales=scales,  # [K] non-negative
        )

    def simulate_resist_from_aerial(self, aerial: jax.Array) -> jax.Array:
        return jax.nn.sigmoid(
            self.resist_steepness * (aerial - self.resist_threshold)
        )

    def simulate_printed_from_resist(self, resist: jax.Array) -> jax.Array:
        return (resist > self.print_threshold).astype(resist.dtype)

    @classmethod
    def nominal(cls, **overrides) -> "LithographySimulator":
        return cls(kernel_type="focus", dose=d.DOSE_NOMINAL, **overrides)

    @classmethod
    def maximum(cls, **overrides) -> "LithographySimulator":
        return cls(kernel_type="focus", dose=d.DOSE_MAX, **overrides)

    @classmethod
    def minimum(cls, **overrides) -> "LithographySimulator":
        return cls(kernel_type="defocus", dose=d.DOSE_MIN, **overrides)


def convolve_frequency_domain(
    image_stack: jax.Array,
    kernels_fourier: jax.Array,
) -> jax.Array:
    """
    Frequency-domain convolution of a stack of complex fields by a stack of Fourier kernels,
    without additional padding.

    Args:
        image_stack: [..., K, H, W] or [..., 1, H, W], complex or real (cast to complex64).
        kernels_fourier: [K, Hk, Wk] complex Fourier-domain kernels.

    Returns:
        Convolved complex stack with spatial size [..., K, H, W].
    """
    # ensure complex dtype
    image_stack_c = image_stack.astype(jnp.complex64)
    # spatial dimensions
    H, W = image_stack_c.shape[-2:]
    # pad kernels to image size
    kernels_padded = center_pad_2d(kernels_fourier, (H, W))  # [K, H, W]
    # FFT of input fields
    stack_ft = centered_fft_2d(image_stack_c)  # [..., K, H, W]
    # broadcast kernels
    bshape = (1,) * (stack_ft.ndim - 3) + kernels_padded.shape
    product_ft = stack_ft * kernels_padded.reshape(bshape)
    # inverse FFT
    return centered_ifft_2d(product_ft)


@jax.custom_vjp
def simulate_aerial_from_mask(
    mask: jax.Array,
    dose: float,
    kernels_fourier: jax.Array,      # [K,Hk,Wk] complex
    kernels_fourier_ct: jax.Array,   # [K,Hk,Wk] complex (used in backward)
    scales: jax.Array,               # [K] ≥ 0
) -> jax.Array:
    """
    Forward aerial image:
        I = sum_k scales[k] * | F^{-1}( F(dose * mask) * kernels_fourier[k] ) |^2
    """
    # treat constants
    kernels_fourier = jax.lax.stop_gradient(kernels_fourier)
    kernels_fourier_ct = jax.lax.stop_gradient(kernels_fourier_ct)
    scales = jax.lax.stop_gradient(scales)

    dosed_mask = (dose * mask).astype(jnp.float32)
    fields = convolve_frequency_domain(
        image_stack=jnp.expand_dims(dosed_mask, axis=-3),
        kernels_fourier=kernels_fourier,
    )  # [..., K, H, W]
    intensities = jnp.square(jnp.abs(fields))
    return jnp.sum(scales[..., None, None] * intensities, axis=-3)


def simulate_aerial_from_mask_fwd(
    mask: jax.Array,
    dose: float,
    kernels_fourier: jax.Array,
    kernels_fourier_ct: jax.Array,
    scales: jax.Array,
):
    # constants
    kernels_fourier = jax.lax.stop_gradient(kernels_fourier)
    kernels_fourier_ct = jax.lax.stop_gradient(kernels_fourier_ct)
    scales = jax.lax.stop_gradient(scales)

    dosed_mask = (dose * mask).astype(jnp.float32)
    fields_main = convolve_frequency_domain(
        image_stack=jnp.expand_dims(dosed_mask, axis=-3),
        kernels_fourier=kernels_fourier,
    )  # [..., K, H, W]
    intensities = jnp.square(jnp.abs(fields_main))
    y = jnp.sum(scales[..., None, None] * intensities, axis=-3)

    residuals = (dosed_mask, fields_main, kernels_fourier, kernels_fourier_ct, scales, dose)
    return y, residuals


def simulate_aerial_from_mask_bwd(
    residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, float],
    grad_aerial: jax.Array,
):
    dosed_mask, fields_main, kernels_fourier, kernels_fourier_ct, scales, dose = residuals

    grad = jnp.expand_dims(grad_aerial, axis=-3)  # [..., 1, H, W]

    # convMask(mask, kernels_ct)
    fields_ct = convolve_frequency_domain(
        image_stack=jnp.expand_dims(dosed_mask, axis=-3),
        kernels_fourier=kernels_fourier_ct,
    )

    # term1: convMatrix(fields_ct * grad, kernels)
    term1 = convolve_frequency_domain(
        image_stack=fields_ct * grad,
        kernels_fourier=kernels_fourier,
    )
    # term2: convMatrix(fields_main * grad, kernels_ct)
    term2 = convolve_frequency_domain(
        image_stack=fields_main * grad,
        kernels_fourier=kernels_fourier_ct,
    )

    summed = jnp.sum(scales[..., None, None] * (term1 + term2), axis=-3)
    grad_mask = dose * summed.real

    # return tangents for mask, dose, kernels_fourier, kernels_fourier_ct, scales
    return (grad_mask, None, None, None, None)

# bind custom_vjp rules
simulate_aerial_from_mask.defvjp(
    simulate_aerial_from_mask_fwd,
    simulate_aerial_from_mask_bwd,
)
