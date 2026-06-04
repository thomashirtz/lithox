# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

import equinox as eqx
import jax
import jax.numpy as jnp
from chex import dataclass
from jaxtyping import Array, Float

import lithox.defaults as d
from lithox.simulation import (
    DTYPE_COMPUTE_REAL,
    Kernels,
    LithographySimulator,
    Scales,
    printed_from_resist,
    simulate_aerial_from_mask,
)
from lithox.utilities.spatial import crop_margin_2d, pad_margin_2d

Image = Float[Array, "*batch H W"]


def _simulate_pv_corner(
    dose: float,
    kernels_fourier: Kernels,
    kernels_fourier_ct: Kernels,
    scales: Scales,
    mask: Image,
    margin: int,
    resist_threshold: float,
    resist_steepness: float,
) -> tuple[Image, Image]:
    """One process corner: aerial I then resist R = σ(α(I − τ))."""
    mask_work = mask
    if margin > 0:
        mask_work = pad_margin_2d(mask_work, margin)

    aerial = simulate_aerial_from_mask(
        mask=mask_work.astype(DTYPE_COMPUTE_REAL),
        dose=dose,
        kernels_fourier=kernels_fourier,
        kernels_fourier_ct=kernels_fourier_ct,
        scales=scales,
    )
    if margin > 0:
        aerial = crop_margin_2d(aerial, margin)

    steepness = jnp.asarray(resist_steepness, DTYPE_COMPUTE_REAL)
    threshold = jnp.asarray(resist_threshold, DTYPE_COMPUTE_REAL)
    resist = jax.nn.sigmoid(steepness * (aerial.astype(DTYPE_COMPUTE_REAL) - threshold))
    return aerial, resist


# Batched over corner index 0 (nominal, max, min); mask and resist params are shared.
_simulate_pv_corners = jax.vmap(
    _simulate_pv_corner,
    in_axes=(0, 0, 0, 0, None, None, None, None),
)


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

    After `__call__`, corners are split into `Variants`; PVB maps are derived
    without re-simulation. Printed images use each corner's `SimulationOutput.printed`.

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


class ProcessVariationSimulator(eqx.Module):
    """Wrap three lithography simulators to model process variations.

    A single `__call__` evaluates all three corners in one `jax.vmap` batch over
    the corner axis (nominal, max, min), sharing the same mask. Use
    `ProcessVariationOutput.pvb_map` (and related properties) so PVB metrics do
    not trigger another simulation.

    The attributes `nominal_simulator`, `max_simulator`, and `min_simulator` are
    still available for single-corner runs or custom workflows.

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
        margin: int = 0,
    ):
        """Initialize the trio of simulators used for the variation sweep.

        All three simulators share the same resist/print parameters.
        They differ only by kernel set and dose.

        Args:
          dose_nominal: Dose for the nominal (baseline) simulator.
          dose_min: Dose for the minimum-dose/defocus simulator.
          dose_max: Dose for the maximum-dose simulator.
          resist_threshold: Intensity threshold τ on I (sigmoid midpoint).
          resist_steepness: Sigmoid steepness α.
          margin: Symmetric padding in pixels applied inside each simulator.
        """
        self.nominal_simulator = LithographySimulator(
            kernel_type="focus",
            dose=dose_nominal,
            resist_threshold=resist_threshold,
            resist_steepness=resist_steepness,
            margin=margin,
        )
        self.max_simulator = LithographySimulator(
            kernel_type="focus",
            dose=dose_max,
            resist_threshold=resist_threshold,
            resist_steepness=resist_steepness,
            margin=margin,
        )
        self.min_simulator = LithographySimulator(
            kernel_type="defocus",
            dose=dose_min,
            resist_threshold=resist_threshold,
            resist_steepness=resist_steepness,
            margin=margin,
        )

    def __call__(self, mask: Image, margin: int | None = None) -> ProcessVariationOutput:
        """Run nominal, max, and min corners in one vmapped batch.

        Corners are fused with `jax.vmap` over the process axis (length 3), not
        three separate Python calls. Leading dimensions on `mask` (e.g. batch) are
        preserved on each variant field.

        Args:
          mask: Input mask; last two axes are (height, width), each ≥ `MIN_MASK_SIZE`.
          margin: Optional override for the simulator margin (all corners).

        Returns:
          ProcessVariationOutput; use `.pvb_map` / `.pvb_loss_map` without re-running.
        """
        mask = jnp.asarray(mask, dtype=DTYPE_COMPUTE_REAL)
        self.nominal_simulator._check_mask_size(mask)

        margin_to_use = self.nominal_simulator.margin if margin is None else margin
        sims = (self.nominal_simulator, self.max_simulator, self.min_simulator)

        doses = jnp.asarray([sim.dose for sim in sims], dtype=DTYPE_COMPUTE_REAL)
        kernels = jnp.stack([sim.kernels for sim in sims])
        kernels_ct = jnp.stack([sim.kernels_ct for sim in sims])
        scales = jnp.stack([sim.scales for sim in sims])

        ref = self.nominal_simulator
        aerial_stack, resist_stack = _simulate_pv_corners(
            doses,
            kernels,
            kernels_ct,
            scales,
            mask,
            margin_to_use,
            ref.resist_threshold,
            ref.resist_steepness,
        )
        printed_stack = jax.vmap(printed_from_resist)(resist_stack)

        return ProcessVariationOutput(
            aerial=Variants(
                nominal=aerial_stack[0],
                max=aerial_stack[1],
                min=aerial_stack[2],
            ),
            resist=Variants(
                nominal=resist_stack[0],
                max=resist_stack[1],
                min=resist_stack[2],
            ),
            printed=Variants(
                nominal=printed_stack[0],
                max=printed_stack[1],
                min=printed_stack[2],
            ),
        )
