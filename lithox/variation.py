# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

from chex import dataclass

import equinox as eqx
import jax.numpy as jnp

from lithox.simulation import LithographySimulator
from lithox.defaults import DOSE_NOMINAL, RESIST_STEEPNESS, RESIST_THRESHOLD, PRINT_THRESHOLD, DTYPE, DOSE_MAX, DOSE_MIN

@dataclass
class Variants:
    nominal: jnp.ndarray
    max: jnp.ndarray
    min: jnp.ndarray

@dataclass
class ProcessVariationOutput:
    aerial: Variants
    resist: Variants
    printed: Variants

class ProcessVariationSimulator(eqx.Module):
    """
    Combines three LithographySimulator instances (nominal, max, min)
    to model process variations in aerial, resist, and printed outputs.
    """

    nominal_simulator: LithographySimulator
    max_simulator: LithographySimulator
    min_simulator: LithographySimulator

    def __init__(
        self,
        dose_nominal: float = DOSE_NOMINAL,
        dose_min: float = DOSE_MIN,
        dose_max: float = DOSE_MAX,
        resist_threshold: float = RESIST_THRESHOLD,
        resist_steepness: float = RESIST_STEEPNESS,
        print_threshold: float = PRINT_THRESHOLD,
        dtype: jnp.dtype = DTYPE,
    ):
        # Nominal-dose simulator (in-focus)
        self.nominal_simulator = LithographySimulator(
            kernel_type="focus",
            dose=dose_nominal,
            resist_threshold=resist_threshold,
            resist_steepness=resist_steepness,
            print_threshold=print_threshold,
            dtype=dtype,
        )

        # High-dose simulator (in-focus, max dose)
        self.max_simulator = LithographySimulator(
            kernel_type="focus",
            dose=dose_max,
            resist_threshold=resist_threshold,
            resist_steepness=resist_steepness,
            print_threshold=print_threshold,
            dtype=dtype,
        )

        # Low-dose/defocus simulator (min dose)
        self.min_simulator = LithographySimulator(
            kernel_type="defocus",
            dose=dose_min,
            resist_threshold=resist_threshold,
            resist_steepness=resist_steepness,
            print_threshold=print_threshold,
            dtype=dtype,
        )

    def __call__(self, mask: jnp.ndarray) -> ProcessVariationOutput:
        out_nom = self.nominal_simulator(mask)
        out_max = self.max_simulator(mask)
        out_min = self.min_simulator(mask)

        aerial = Variants(nominal=out_nom.aerial, max=out_max.aerial, min=out_min.aerial)
        resist = Variants(nominal=out_nom.resist, max=out_max.resist, min=out_min.resist)
        printed = Variants(nominal=out_nom.printed, max=out_max.printed, min=out_min.printed)

        return ProcessVariationOutput(aerial=aerial, resist=resist, printed=printed)

    def get_pvb(
        self,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        simulation = self(mask)
        printed_min, printed_max = simulation.printed.min, simulation.printed.max
        # Peak-to-valley: fraction of pixels that change
        pvb = (printed_max - printed_min).astype(jnp.float32)
        return pvb.mean(axis=(-2, -1))
