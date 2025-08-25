# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

import equinox as eqx
import jax
import jax.numpy as jnp
from chex import dataclass

import lithox.defaults as d
from lithox.simulation import LithographySimulator


@dataclass
class Variants:
    """Container for nominal / max / min variants of a quantity.

    Attributes:
      nominal: Value for the nominal (baseline) process setting.
      max: Value for the maximum-dose (or most aggressive) process setting.
      min: Value for the minimum-dose (or most conservative/defocused) setting.
    """
    nominal: jax.Array
    max: jax.Array
    min: jax.Array


@dataclass
class ProcessVariationOutput:
    """Outputs of a process-variation sweep grouped by stage.

    Attributes:
      aerial: Aerial intensity images for (nominal, max, min).
      resist: Resist activations for (nominal, max, min).
      printed: Binary printed results for (nominal, max, min).
    """
    aerial: Variants
    resist: Variants
    printed: Variants


class ProcessVariationSimulator(eqx.Module):
    """Wrap three lithography simulators to model process variations.

    Combines three `LithographySimulator` instances (nominal, max, min) that
    differ in dose and kernel focus/defocus to evaluate sensitivity across the
    pipeline (aerial → resist → printed).

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
        print_threshold: float = d.PRINT_THRESHOLD,
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
          resist_threshold: Midpoint of the resist sigmoid.
          resist_steepness: Slope of the resist sigmoid.
          print_threshold: Threshold to binarize the resist activation.
          dtype: Numeric dtype for internal computations.
          margin: Symmetric padding in pixels applied inside each simulator.
        """
        # Nominal-dose simulator (in-focus).
        self.nominal_simulator = LithographySimulator(
            kernel_type="focus",
            dose=dose_nominal,
            resist_threshold=resist_threshold,
            resist_steepness=resist_steepness,
            print_threshold=print_threshold,
            dtype=dtype,
            margin=margin,
        )

        # High-dose simulator (in-focus, max dose).
        self.max_simulator = LithographySimulator(
            kernel_type="focus",
            dose=dose_max,
            resist_threshold=resist_threshold,
            resist_steepness=resist_steepness,
            print_threshold=print_threshold,
            dtype=dtype,
            margin=margin,
        )

        # Low-dose / defocus simulator (min dose, defocused kernels).
        self.min_simulator = LithographySimulator(
            kernel_type="defocus",
            dose=dose_min,
            resist_threshold=resist_threshold,
            resist_steepness=resist_steepness,
            print_threshold=print_threshold,
            dtype=dtype,
            margin=margin,
        )

    def __call__(self, mask: jax.Array, margin: int | None = None) -> ProcessVariationOutput:
        """Run all three simulators on a given mask.

        This evaluates the pipeline at nominal, maximum, and minimum doses and
        groups the results by stage (aerial, resist, printed).

        Args:
          mask: Input mask array; last two axes are (height, width). Leading
            dimensions (e.g., batch) are preserved.
          margin: Optional override for the simulator margin; if None, each
            simulator uses its configured margin.

        Returns:
          ProcessVariationOutput containing per-stage `Variants` (nominal/max/min).
        """
        out_nom = self.nominal_simulator(mask=mask, margin=margin)
        out_max = self.max_simulator(mask=mask, margin=margin)
        out_min = self.min_simulator(mask=mask, margin=margin)

        # Group results by stage for convenience.
        aerial = Variants(nominal=out_nom.aerial, max=out_max.aerial, min=out_min.aerial)
        resist = Variants(nominal=out_nom.resist, max=out_max.resist, min=out_min.resist)
        printed = Variants(nominal=out_nom.printed, max=out_max.printed, min=out_min.printed)

        return ProcessVariationOutput(aerial=aerial, resist=resist, printed=printed)

    def get_pvb_map(self, mask: jax.Array, margin: int | None = None) -> jax.Array:
        """Compute the Process Variation Band (PVB) map on the printed layer.

        The PVB is the per-pixel difference between the max-dose and min-dose
        printed images: `printed_max - printed_min`. Since the printed outputs
        are binary, the PVB map is 0 where the pixel is robust (no change) and
        1 where it is sensitive (changes across the process window).

        Args:
          mask: Input mask array.
          margin: Optional override for the simulator margin.

        Returns:
          Float32 array with the same spatial size as `mask`, values in {0, 1}.
        """
        simulation = self(mask=mask, margin=margin)
        printed_min, printed_max = simulation.printed.min, simulation.printed.max
        return (printed_max - printed_min).astype(jnp.float32)

    def get_pvb_mean(self, mask: jax.Array, margin: int | None = None) -> jax.Array:
        """Return the mean Process Variation Band (PVB) over spatial dimensions.

        This summarizes sensitivity as a single scalar (or one per leading
        dimension if `mask` is batched).

        Args:
          mask: Input mask array.
          margin: Optional override for the simulator margin.

        Returns:
          The mean of the PVB map over the last two axes (height, width).
        """
        return self.get_pvb_map(mask=mask, margin=margin).mean(axis=(-2, -1))
