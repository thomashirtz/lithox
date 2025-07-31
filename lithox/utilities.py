# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

import jax.numpy as jnp
from jax import jit


@jit
def centered_fft_2d(
    data: jnp.ndarray,
) -> jnp.ndarray:
    """Perform a centered 2-dimensional Fast Fourier Transform.

    This function shifts the zero-frequency component to the center of
    the spectrum before and after applying the FFT.

    Args:
        data: Input array. The FFT is computed over the last two axes.

    Returns:
        A jnp.ndarray of the same shape as `data`, containing the centered 2D FFT result.
    """
    data = jnp.fft.ifftshift(data, axes=(-2, -1))
    data = jnp.fft.fftn(data, axes=(-2, -1))
    data = jnp.fft.fftshift(data, axes=(-2, -1))
    return data


@jit
def centered_ifft_2d(data: jnp.ndarray) -> jnp.ndarray:
    """Perform a centered 2-dimensional inverse Fast Fourier Transform.

    This function shifts the zero-frequency component to the center of
    the spectrum before and after applying the inverse FFT.

    Args:
        data: Input array. The inverse FFT is computed over the last two axes.

    Returns:
        A jnp.ndarray of the same shape as `data`, containing the centered 2D inverse FFT result.
    """
    data = jnp.fft.ifftshift(data, axes=(-2, -1))
    data = jnp.fft.ifftn(data, axes=(-2, -1))
    data = jnp.fft.fftshift(data, axes=(-2, -1))
    return data


def center_pad_2d(arr: jnp.ndarray, out_shape: tuple[int, int]) -> jnp.ndarray:
    """
    Zero-pad the *last two* (H, W) axes of `arr` so that they match `out_shape`,
    while keeping every leading (batch) axis unchanged.

    Works for shapes
        (H, W)                  ⟶ (H_out, W_out)
        (K, H, W)               ⟶ (K, H_out, W_out)
        (N₁, …, Nₘ, H, W)       ⟶ (N₁, …, Nₘ, H_out, W_out)
    """
    H_out, W_out = out_shape
    *prefix, H_in, W_in = arr.shape

    assert (
        H_in <= H_out and W_in <= W_out
    ), f"Kernel size ({H_in}×{W_in}) larger than target ({H_out}×{W_out})."

    pad_top = (H_out - H_in) // 2
    pad_bottom = H_out - H_in - pad_top
    pad_left = (W_out - W_in) // 2
    pad_right = W_out - W_in - pad_left

    pad_width = [(0, 0)] * len(prefix) + [
        (pad_top, pad_bottom),  # H axis
        (pad_left, pad_right),  # W axis
    ]

    return jnp.pad(arr, pad_width, mode="constant")
