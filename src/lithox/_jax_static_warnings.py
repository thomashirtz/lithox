# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause
"""Suppress expected JAX warnings when bundling kernels as Equinox static fields."""

import warnings

# Precomputed Hopkins kernels/scales are stored as static Equinox fields so they are
# not traced under jit/grad; JAX still emits a UserWarning on assignment.
warnings.filterwarnings(
    "ignore",
    message=r"A JAX array is being set as static!.*",
    category=UserWarning,
)
