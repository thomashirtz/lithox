import jax
import jax.numpy as jnp
import pytest

from lithox.variation import ProcessVariationSimulator, ProcessVariationOutput

from tests.constants import MASK_SIZE


@pytest.fixture
def pvs():
    return ProcessVariationSimulator()


def test_process_variation_smoke(pvs, small_mask):
    out = pvs(small_mask)
    assert isinstance(out, ProcessVariationOutput)
    assert out.aerial.nominal.shape == (MASK_SIZE, MASK_SIZE)
    assert out.resist.max.shape == (MASK_SIZE, MASK_SIZE)
    assert out.printed.min.shape == (MASK_SIZE, MASK_SIZE)


def test_pvb_map_binary(pvs, small_mask):
    pvb = jax.device_get(pvs(small_mask).pvb_map)
    assert pvb.shape == (MASK_SIZE, MASK_SIZE)
    assert set(jnp.unique(pvb).tolist()).issubset({0.0, 1.0})


def test_pvb_loss_finite(pvs, small_mask):
    loss_map = jax.device_get(pvs(small_mask).pvb_loss_map)
    assert loss_map.shape == (MASK_SIZE, MASK_SIZE)
    assert jnp.isfinite(loss_map).all()
    assert loss_map.min() >= 0.0
    assert loss_map.max() <= 1.0


def test_gradient_through_pvb_loss_mean(pvs, random_mask):
    def loss(mask):
        return pvs(mask).pvb_loss_mean

    grad = jax.grad(loss)(random_mask)
    grad = jax.device_get(grad)
    assert grad.shape == random_mask.shape
    assert jnp.isfinite(grad).all()


def test_printed_matches_per_corner_simulator(pvs, random_mask):
    """PV printed variants use each corner's SimulationOutput.printed."""
    pv = pvs(random_mask)
    for corner_name, sim in (
        ("nominal", pvs.nominal_simulator),
        ("max", pvs.max_simulator),
        ("min", pvs.min_simulator),
    ):
        expected = jax.device_get(sim(random_mask).printed)
        actual = jax.device_get(getattr(pv.printed, corner_name))
        assert jnp.allclose(actual, expected)


def test_batched_mask_corners(pvs, batched_mask):
    """Vmapped PV preserves leading batch dimensions on each variant."""
    out = pvs(batched_mask)
    assert out.aerial.nominal.shape == batched_mask.shape
    assert out.resist.max.shape == batched_mask.shape
    assert out.printed.min.shape == batched_mask.shape


def test_pv_margin_preserves_shape(random_mask):
    margin = 8
    pvs = ProcessVariationSimulator(margin=margin)
    out = pvs(random_mask)
    assert out.aerial.nominal.shape == random_mask.shape
    assert out.resist.max.shape == random_mask.shape
    assert out.printed.min.shape == random_mask.shape


def test_pv_margin_matches_per_corner_simulator(random_mask):
    margin = 8
    pvs = ProcessVariationSimulator(margin=margin)
    pv = pvs(random_mask)
    for corner_name, sim in (
        ("nominal", pvs.nominal_simulator),
        ("max", pvs.max_simulator),
        ("min", pvs.min_simulator),
    ):
        expected = jax.device_get(sim(random_mask).aerial)
        actual = jax.device_get(getattr(pv.aerial, corner_name))
        assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_batched_pv_matches_per_corner_simulator(pvs, batched_mask):
    """Batched vmapped PV matches single-corner simulators on each batch element."""
    pv = pvs(batched_mask)
    for corner_name, sim in (
        ("nominal", pvs.nominal_simulator),
        ("max", pvs.max_simulator),
        ("min", pvs.min_simulator),
    ):
        expected = jax.device_get(sim(batched_mask).aerial)
        actual = jax.device_get(getattr(pv.aerial, corner_name))
        assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-6)
