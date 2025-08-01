# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from chex import dataclass

from lithox.defaults import DOSE, RESIST_STEEPNESS, RESIST_THRESHOLD, PRINT_THRESHOLD, DTYPE, DOSE_MAX, DOSE_MIN, DOSE_NOMINAL
from lithox.paths import SCALES_DIRECTORY, KERNELS_DIRECTORY
from lithox.utilities import center_pad_2d, centered_fft_2d, centered_ifft_2d, _load_npy


@dataclass
class SimulationOutput:
    aerial: jnp.ndarray
    resist: jnp.ndarray
    printed: jnp.ndarray


class LithographySimulator(eqx.Module):
    kernels: jnp.ndarray
    kernels_ct: jnp.ndarray
    scales: jnp.ndarray

    kernel_type: str
    dose: float
    resist_threshold: float
    resist_steepness: float
    print_threshold: float
    dtype: jnp.dtype

    def __init__(
            self,
            kernel_type: Literal["focus", "defocus"] = "focus",
            *,
            dose: float = DOSE,
            resist_threshold: float = RESIST_THRESHOLD,
            resist_steepness: float = RESIST_STEEPNESS,
            print_threshold: float = PRINT_THRESHOLD,
            dtype: jnp.dtype = DTYPE,
    ):
        self.kernel_type: str = kernel_type
        kernels_path: Path = KERNELS_DIRECTORY / f"{kernel_type}.npy"
        kernels_ct_path: Path = KERNELS_DIRECTORY / f"{kernel_type}_ct.npy"
        scales_path: Path = SCALES_DIRECTORY / f"{kernel_type}.npy"

        self.kernels =  _load_npy(package="lithox.kernels", filename=f"{kernel_type}.npy")
        self.kernels_ct =  _load_npy(package="lithox.kernels", filename=f"{kernel_type}_ct.npy")
        self.scales = _load_npy(package="lithox.scales", filename=f"{kernel_type}.npy")

        self.dose = dose
        self.resist_threshold = resist_threshold
        self.resist_steepness = resist_steepness
        self.print_threshold = print_threshold
        self.dtype = dtype

    def __call__(self, mask: jnp.ndarray) -> SimulationOutput:
        aerial = self.simulate_aerial_from_mask(mask=mask)
        resist = self.simulate_resist_from_aerial(aerial=aerial)
        printed = self.simulate_printed_from_resist(resist=resist)
        return SimulationOutput(
            aerial=aerial,
            resist=resist,
            printed=printed,
        )

    def simulate_aerial_from_mask(self, mask: jnp.ndarray) -> jnp.ndarray:
        return simulate_aerial_from_mask(
            mask=mask.astype(self.dtype),
            dose=self.dose,
            kernels_fourier=self.kernels,  # [K,Hk,Wk] complex
            kernels_fourier_ct=self.kernels_ct,
            scales=self.scales,  # [K] non-negative
        )

    def simulate_resist_from_aerial(self, aerial: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.sigmoid(
            self.resist_steepness * (aerial - self.resist_threshold)
        )

    def simulate_printed_from_resist(self, resist: jnp.ndarray) -> jnp.ndarray:
        return (resist > self.print_threshold).astype(resist.dtype)

    @classmethod
    def nominal(cls, **overrides) -> "LithographySimulator":
        return cls(kernel_type="focus", dose=DOSE_NOMINAL, **overrides)

    @classmethod
    def maximum(cls,**overrides) -> "LithographySimulator":
        return cls(kernel_type="focus", dose=DOSE_MAX, **overrides)

    @classmethod
    def minimum(cls, **overrides) -> "LithographySimulator":
        return cls(kernel_type="defocus", dose=DOSE_MIN, **overrides)


def convolve_frequency_domain(
    image_stack: jnp.ndarray,
    kernels_fourier: jnp.ndarray,
) -> jnp.ndarray:
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
    mask: jnp.ndarray,
    dose: float,
    kernels_fourier: jnp.ndarray,      # [K,Hk,Wk] complex
    kernels_fourier_ct: jnp.ndarray,   # [K,Hk,Wk] complex (used in backward)
    scales: jnp.ndarray,               # [K] ≥ 0
) -> jnp.ndarray:
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
    mask: jnp.ndarray,
    dose: float,
    kernels_fourier: jnp.ndarray,
    kernels_fourier_ct: jnp.ndarray,
    scales: jnp.ndarray,
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
    residuals: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float],
    grad_aerial: jnp.ndarray,
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
