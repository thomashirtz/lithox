# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

from importlib import resources
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import jax
import jax.numpy as jnp
import requests
from PIL import Image
from jax import jit


# ---------- internal helpers ----------

def _is_http_url(value: str | Path) -> bool:
    """Return True if the string looks like an HTTP/HTTPS URL."""
    if not isinstance(value, str):
        return False
    scheme = urlparse(value).scheme.lower()
    return scheme in ("http", "https")


def _normalize_height_width_pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    """Normalize a single int or (height, width) tuple into (height, width).

    Args:
      value: Either a single non-negative integer or a tuple of two non-negative integers.
      name: Name of the parameter for error messages.

    Returns:
      A tuple (height, width).

    Raises:
      TypeError: If value is neither int nor a length-2 tuple of ints.
      ValueError: If any entry is negative.
    """
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
        return value, value
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(v, int) for v in value)
    ):
        height, width = value
        if height < 0 or width < 0:
            raise ValueError(f"Both entries of {name} must be non-negative.")
        return height, width
    raise TypeError(f"{name} must be an int or a (height, width) tuple of ints.")


# ---------- FFT utilities ----------

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


# ---------- shape/padding utilities ----------

def pad_to_shape_2d(array: jax.Array, target_shape: tuple[int, int]) -> jax.Array:
    """Zero-pad the last two axes (height, width) to match a target (height, width).

    Leading (batch) axes are preserved unchanged.

    Examples:
      (H, W)            -> (h_out, w_out)
      (K, H, W)         -> (K, h_out, w_out)
      (N1,…,Nm,H,W)     -> (N1,…,Nm,h_out,w_out)

    Args:
      array: Input array whose last two axes are (height, width).
      target_shape: Target (height, width) to pad to.

    Returns:
      The zero-padded array with the same leading axes and last-two axes
      equal to `target_shape`.

    Raises:
      ValueError: If the input spatial size exceeds `target_shape`.
    """
    target_height, target_width = target_shape
    *leading_shape, input_height, input_width = array.shape

    if input_height > target_height or input_width > target_width:
        raise ValueError(
            f"Input size ({input_height}×{input_width}) exceeds target "
            f"({target_height}×{target_width})."
        )

    pad_top = (target_height - input_height) // 2
    pad_bottom = target_height - input_height - pad_top
    pad_left = (target_width - input_width) // 2
    pad_right = target_width - input_width - pad_left

    pad_width_spec = [(0, 0)] * len(leading_shape) + [
        (pad_top, pad_bottom),   # height axis
        (pad_left, pad_right),   # width axis
    ]
    return jnp.pad(array, pad_width_spec, mode="constant")


def crop_margin_2d(array: jax.Array, margin: int | tuple[int, int]) -> jax.Array:
    """Crop the last two dimensions (height, width) by the given margin.

    Args:
      array: Array with arbitrary leading dims followed by (height, width).
      margin: Either a single integer `p` (crop `p` pixels on all sides),
        or a tuple `(crop_height, crop_width)`.

    Returns:
      The cropped array with unchanged leading dims and
      shape `[..., height - 2*crop_height, width - 2*crop_width]`.

    Raises:
      ValueError: If the crop would remove all pixels along a dimension.
    """
    crop_height, crop_width = _normalize_height_width_pair(margin, "margin")

    if crop_height == 0 and crop_width == 0:
        return array

    input_height, input_width = array.shape[-2:]
    if 2 * crop_height >= input_height or 2 * crop_width >= input_width:
        raise ValueError(
            f"Crop ({crop_height}, {crop_width}) too large for input "
            f"size ({input_height}, {input_width})."
        )

    return array[..., crop_height: input_height - crop_height, crop_width: input_width - crop_width]


def pad_margin_2d(array: jax.Array, padding: int | tuple[int, int]) -> jax.Array:
    """Zero-pad the last two dimensions (height, width) by the given padding.

    Args:
      array: Array with arbitrary leading dims followed by (height, width).
      padding: Either a single integer `p` (pad `p` pixels on all sides),
        or a tuple `(pad_height, pad_width)`.

    Returns:
      The padded array with unchanged leading dims and
      shape `[..., height + 2*pad_height, width + 2*pad_width]`.
    """
    pad_height, pad_width = _normalize_height_width_pair(padding, "padding")
    pad_width_spec = [(0, 0)] * (array.ndim - 2) + [(pad_height, pad_height), (pad_width, pad_width)]
    return jnp.pad(array, pad_width_spec, mode="constant")


# ---------- I/O utilities ----------

def load_image(
    path_or_url: str | Path,
    size: int,
    dtype: jnp.dtype = jnp.float32,
    resample: int = Image.NEAREST,
    request_timeout: float | None = 10.0,
) -> jax.Array:
    """Load a grayscale image from file or URL, resize square, and normalize to [0, 1].

    Args:
      path_or_url: File path or HTTP/HTTPS URL to the image.
      size: Target side length in pixels. Output shape is `(size, size)`.
      dtype: Desired JAX dtype for the returned array.
      resample: PIL resampling filter used during resize (default: `Image.NEAREST`).
      request_timeout: Timeout in seconds for HTTP requests (ignored for files).

    Returns:
      A `(size, size)` JAX array with values in `[0, 1]`.

    Raises:
      requests.HTTPError: On non-2xx HTTP response.
      FileNotFoundError: If the local image file is missing.
      OSError: If the image cannot be opened/decoded.
    """
    if _is_http_url(path_or_url):
        response = requests.get(str(path_or_url), timeout=request_timeout)
        response.raise_for_status()
        pil_image = Image.open(BytesIO(response.content))
    else:
        pil_image = Image.open(Path(path_or_url))

    pil_image = pil_image.convert("L").resize((size, size), resample=resample)
    # Convert to JAX array and scale to [0, 1]
    return jnp.asarray(pil_image, dtype=dtype) / jnp.asarray(255, dtype=dtype)


def load_npy(
    filename: str,
    module: str | None = None,
    path: Path | None = None,
    allow_pickle: bool = False,
    mmap_mode: str | None = None,
) -> jax.Array:
    """Load a .npy file via importlib.resources with optional filesystem fallback.

    Attempts to load `filename` from Python package `module` using
    `importlib.resources`. If that fails (or `module` is None) and `path`
    is provided, falls back to `path / filename`.

    Args:
      filename: Name of the .npy file to load (e.g., "focus.npy").
      module: Dotted module/package name to load from. If `None`, skip package lookup.
      path: Directory on the filesystem to load from if package loading fails or is skipped.
      allow_pickle: Forwarded to `numpy.load`.
      mmap_mode: Forwarded to `numpy.load` (e.g., "r" to memory-map read-only).

    Returns:
      A JAX array containing the data from the .npy file.

    Raises:
      ValueError: If neither `module` nor `path` is provided.
      FileNotFoundError: If the file cannot be found in the package or filesystem.
      OSError: If the file cannot be read/parsed as a .npy file.
    """
    if not module and not path:
        raise ValueError("Provide at least one of `module` or `path`.")

    def _np_load(path_obj: Path) -> jax.Array:
        np_array = np.load(path_obj, allow_pickle=allow_pickle, mmap_mode=mmap_mode)
        return jnp.asarray(np_array)

    # 1) Try package resource first
    if module:
        try:
            resource = resources.files(module) / filename
            with resources.as_file(resource) as file_path:
                return _np_load(file_path)
        except (ModuleNotFoundError, FileNotFoundError) as exc:
            if not path:
                raise FileNotFoundError(
                    f"Could not load '{filename}' from module '{module}' and no fallback path was provided."
                ) from exc

    # 2) Fallback to filesystem
    if path:
        fs_path = Path(path) / filename
        try:
            return _np_load(fs_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Failed to load file from filesystem path: {fs_path}") from exc

    # Should be unreachable because of the early validation.
    raise RuntimeError("Unreachable: load_npy exhausted all options.")
