# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

from importlib.metadata import PackageNotFoundError, version

from lithox.simulation import LithographySimulator, SimulationOutput
from lithox.variation import ProcessVariationSimulator, ProcessVariationOutput, Variants
from lithox.utilities.io import load_image

try:
    __version__ = version("lithox")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "LithographySimulator",
    "SimulationOutput",
    "ProcessVariationSimulator",
    "ProcessVariationOutput",
    "Variants",
    "load_image",
]
