# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

import jax
from jax import jit, numpy as jnp


@jit
def centered_fft_2d(array: jax.Array) -> jax.Array:
    """Compute a centered 2D Fast Fourier Transform over the last two axes.

    The zero-frequency component is shifted to the center of the spectrum
    before and after the FFT.

    Args:
      array: Input array. The FFT is computed over the last two axes (height, width).

    Returns:
      A JAX array with the same shape as `array` containing the centered 2D FFT.
    """
    array = jnp.fft.ifftshift(array, axes=(-2, -1))
    array = jnp.fft.fftn(array, axes=(-2, -1))
    array = jnp.fft.fftshift(array, axes=(-2, -1))
    return array


@jit
def centered_ifft_2d(array: jax.Array) -> jax.Array:
    """Compute a centered 2D inverse Fast Fourier Transform over the last two axes.

    The zero-frequency component is shifted to the center of the spectrum
    before and after the inverse FFT.

    Args:
      array: Input array. The inverse FFT is computed over the last two axes (height, width).

    Returns:
      A JAX array with the same shape as `array` containing the centered 2D iFFT.
    """
    array = jnp.fft.ifftshift(array, axes=(-2, -1))
    array = jnp.fft.ifftn(array, axes=(-2, -1))
    array = jnp.fft.fftshift(array, axes=(-2, -1))
    return array
