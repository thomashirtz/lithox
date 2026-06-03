# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

import jax.numpy as jnp


DOSE : float = 1.00
DOSE_NOMINAL: float = DOSE
DOSE_MAX: float = 1.02
DOSE_MIN: float = 0.98
RESIST_THRESHOLD: float = 0.225
RESIST_STEEPNESS: float = 50.0
BINARIZATION_THRESHOLD: float = 0.5
# Minimum mask height/width (LithoBench Fourier kernels are 35×35).
MIN_MASK_SIZE: int = 35
DTYPE: jnp.dtype = jnp.float32
