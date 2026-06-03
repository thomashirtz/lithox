import jax
import jax.numpy as jnp
import pytest

from lithox.variation import ProcessVariationSimulator, ProcessVariationOutput

from conftest import MASK_SIZE


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
