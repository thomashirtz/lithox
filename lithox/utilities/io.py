# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

from importlib import resources
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import jax
import numpy as np
import requests
from PIL import Image
from jax import numpy as jnp


def _is_http_url(value: str | Path) -> bool:
    """Return True if the string looks like an HTTP/HTTPS URL."""
    if not isinstance(value, str):
        return False
    scheme = urlparse(value).scheme.lower()
    return scheme in ("http", "https")


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
