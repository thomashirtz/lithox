# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

import equinox as eqx
import jax
import jax.numpy as jnp
from chex import dataclass
from jaxtyping import Array, Float

import lithox.defaults as d
from lithox.simulation import LithographySimulator, SimulationOutput

Image = Float[Array, "*batch H W"]

@dataclass
class Variants:
    """Container for nominal / max / min variants of a quantity.

    Attributes:
      nominal: Value for the nominal (baseline) process setting.
      max: Value for the maximum-dose (or most aggressive) process setting.
      min: Value for the minimum-dose (or most conservative/defocused) setting.
    """
    nominal: Image
    max: Image
    min: Image


@dataclass
class ProcessVariationOutput:
    """Outputs of a process-variation sweep grouped by stage.

    After `__call__`, aerial/resist are stacked on corner axis 0 (nominal, max, min)
    inside `_variants_from_stacked`; PVB maps are derived without re-simulation.

    Attributes:
      aerial: Aerial intensity images for (nominal, max, min).
      resist: Resist activations for (nominal, max, min).
      printed: Hard prints P at (nominal, max, min) corners.
    """
    aerial: Variants
    resist: Variants
    printed: Variants

    @property
    def pvb_map(self) -> Image:
        """Metric PVB: P_max − P_min per pixel (values in {0, 1})."""
        p_max = self.printed.max.astype(jnp.float32)
        p_min = self.printed.min.astype(jnp.float32)
        return p_max - p_min

    @property
    def pvb_loss_map(self) -> Image:
        """Differentiable PVB proxy: R_max − R_min per pixel."""
        return (self.resist.max - self.resist.min).astype(jnp.float32)

    @property
    def pvb_mean(self) -> jax.Array:
        """Mean metric PVB over spatial dimensions."""
        return self.pvb_map.mean(axis=(-2, -1))

    @property
    def pvb_loss_mean(self) -> jax.Array:
        """Mean differentiable PVB proxy over spatial dimensions."""
        return self.pvb_loss_map.mean(axis=(-2, -1))


def _variants_from_stacked(stacked: SimulationOutput) -> ProcessVariationOutput:
    """Split a corner-stacked `SimulationOutput` into per-stage `Variants`."""
    tau_b = jnp.asarray(d.BINARIZATION_THRESHOLD, jnp.float32)
    printed = (stacked.resist > tau_b).astype(stacked.resist.dtype)

    def _split(field: jax.Array) -> Variants:
        return Variants(nominal=field[0], max=field[1], min=field[2])

    return ProcessVariationOutput(
        aerial=_split(stacked.aerial),
        resist=_split(stacked.resist),
        printed=_split(printed),
    )


class ProcessVariationSimulator(eqx.Module):
    """Wrap three lithography simulators to model process variations.

    A single `__call__` evaluates all corners; use `ProcessVariationOutput.pvb_map`
    (and related properties) so PVB metrics do not trigger another simulation.

    Attributes:
      nominal_simulator: In-focus simulator at nominal dose.
      max_simulator: In-focus simulator at maximum dose.
      min_simulator: Defocused (or min-dose) simulator.
    """

    nominal_simulator: LithographySimulator
    max_simulator: LithographySimulator
    min_simulator: LithographySimulator

    def __init__(
        self,
        dose_nominal: float = d.DOSE_NOMINAL,
        dose_min: float = d.DOSE_MIN,
        dose_max: float = d.DOSE_MAX,
        resist_threshold: float = d.RESIST_THRESHOLD,
        resist_steepness: float = d.RESIST_STEEPNESS,
        dtype: jnp.dtype = d.DTYPE,
        margin: int = 0,
    ):
        """Initialize the trio of simulators used for the variation sweep.

        All three simulators share the same resist/print parameters and dtype.
        They differ only by kernel set and dose.

        Args:
          dose_nominal: Dose for the nominal (baseline) simulator.
          dose_min: Dose for the minimum-dose/defocus simulator.
          dose_max: Dose for the maximum-dose simulator.
          resist_threshold: Intensity threshold τ on I (sigmoid midpoint).
          resist_steepness: Sigmoid steepness α.
          dtype: Numeric dtype for internal computations.
          margin: Symmetric padding in pixels applied inside each simulator.
        """
        self.nominal_simulator = LithographySimulator(
            kernel_type="focus",
            dose=dose_nominal,
            resist_threshold=resist_threshold,
            resist_steepness=resist_steepness,
            dtype=dtype,
            margin=margin,
        )
        self.max_simulator = LithographySimulator(
            kernel_type="focus",
            dose=dose_max,
            resist_threshold=resist_threshold,
            resist_steepness=resist_steepness,
            dtype=dtype,
            margin=margin,
        )
        self.min_simulator = LithographySimulator(
            kernel_type="defocus",
            dose=dose_min,
            resist_threshold=resist_threshold,
            resist_steepness=resist_steepness,
            dtype=dtype,
            margin=margin,
        )

    def __call__(self, mask: Image, margin: int | None = None) -> ProcessVariationOutput:
        """Run nominal, max, and min corners.

        Args:
          mask: Input mask; last two axes are (height, width), each ≥ `MIN_MASK_SIZE`.
          margin: Optional override for the simulator margin.

        Returns:
          ProcessVariationOutput; use `.pvb_map` / `.pvb_loss_map` without re-running.
        """
        corners = (
            self.nominal_simulator(mask, margin=margin),
            self.max_simulator(mask, margin=margin),
            self.min_simulator(mask, margin=margin),
        )
        stacked = SimulationOutput(
            aerial=jnp.stack([c.aerial for c in corners], axis=0),
            resist=jnp.stack([c.resist for c in corners], axis=0),
        )
        return _variants_from_stacked(stacked)
