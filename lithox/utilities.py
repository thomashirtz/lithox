# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

from importlib import resources
from io import BytesIO
from pathlib import Path

import jax
import jax.numpy as jnp
import requests
from PIL import Image
from jax import jit


@jit
def centered_fft_2d(
    data: jax.Array,
) -> jax.Array:
    """Perform a centered 2-dimensional Fast Fourier Transform.

    This function shifts the zero-frequency component to the center of
    the spectrum before and after applying the FFT.

    Args:
        data: Input array. The FFT is computed over the last two axes.

    Returns:
        A jax.Array of the same shape as `data`, containing the centered 2D FFT result.
    """
    data = jnp.fft.ifftshift(data, axes=(-2, -1))
    data = jnp.fft.fftn(data, axes=(-2, -1))
    data = jnp.fft.fftshift(data, axes=(-2, -1))
    return data


@jit
def centered_ifft_2d(data: jax.Array) -> jax.Array:
    """Perform a centered 2-dimensional inverse Fast Fourier Transform.

    This function shifts the zero-frequency component to the center of
    the spectrum before and after applying the inverse FFT.

    Args:
        data: Input array. The inverse FFT is computed over the last two axes.

    Returns:
        A jax.Array of the same shape as `data`, containing the centered 2D inverse FFT result.
    """
    data = jnp.fft.ifftshift(data, axes=(-2, -1))
    data = jnp.fft.ifftn(data, axes=(-2, -1))
    data = jnp.fft.fftshift(data, axes=(-2, -1))
    return data


def center_pad_2d(arr: jax.Array, out_shape: tuple[int, int]) -> jax.Array:
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


def load_image(
    path_or_url: str | Path,
    size: int,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """
    Load a lithography image from a file path or URL, convert to grayscale, resize, and normalize.

    Args:
        path_or_url: Path to the image file or an HTTP/HTTPS URL.
        size: Target width and height (pixels). Image will be resized to (size, size).
        dtype: Desired JAX array dtype (default: jnp.float32).

    Returns:
        A (size, size) JAX array with values in [0, 1].
    """
    # Check if input is a URL
    if isinstance(path_or_url, str) and path_or_url.startswith(("http://", "https://")):
        response = requests.get(path_or_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(path_or_url)

    # Convert to grayscale and resize
    img = img.convert("L").resize((size, size), Image.NEAREST)

    # Convert to JAX array and scale to [0, 1]
    return jnp.array(img, dtype=dtype) / 255.0


def load_npy(
    filename: str,
    module: str | None = None,
    path: Path | None = None
) -> jax.Array:
    """Load a .npy file via importlib.resources with an optional filesystem fallback.

    Attempts to load `filename` from the given Python package `module`
    using `importlib.resources` (works even in zipped installs). If that
    fails and `path` is provided, loads from the filesystem at `path/filename`.

    Args:
        filename: Name of the .npy file to load (e.g. "focus.npy").
        module: Dot-separated module/package name to load from. If `None`,
            importlib loading is skipped.
        path: Directory on the filesystem to load the file from. If `None`,
            filesystem loading is skipped.

    Returns:
        A JAX Array containing the data from the .npy file.
    """
    if not module and not path:
        raise ValueError("At least one of `module` or `path` must be provided.")

    # 1. Try loading from the package resource first.
    if module:
        try:
            pkg_files = resources.files(module)
            resource = pkg_files / filename
            with resources.as_file(resource) as file_path:
                return jnp.load(file_path)
        except (ModuleNotFoundError, FileNotFoundError) as e:
            if not path:
                raise FileNotFoundError(
                    f"Could not load '{filename}' from module '{module}' and no fallback path was provided."
                ) from e

    # 2. Fallback to filesystem (or primary if module was not provided).
    if path:
        try:
            fs_path = path / filename
            return jnp.load(fs_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Failed to load file from filesystem path: {fs_path}") from e
