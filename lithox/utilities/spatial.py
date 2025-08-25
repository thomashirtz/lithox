# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

import jax
from jax import numpy as jnp


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
