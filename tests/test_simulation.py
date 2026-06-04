import jax
import jax.numpy as jnp
import pytest

import lithox.defaults as d
from lithox.simulation import LithographySimulator, SimulationOutput, _validate_kernel_bank

from tests.constants import MASK_SIZE


@pytest.fixture
def simulator():
    return LithographySimulator()


def test_simulator_smoke(simulator, small_mask):
    out = simulator(small_mask)
    assert isinstance(out, SimulationOutput)


def test_output_shapes(simulator, small_mask, batched_mask):
    out = simulator(small_mask)
    assert out.aerial.shape == (MASK_SIZE, MASK_SIZE)
    assert out.resist.shape == (MASK_SIZE, MASK_SIZE)
    assert out.printed.shape == (MASK_SIZE, MASK_SIZE)

    out_b = simulator(batched_mask)
    assert out_b.aerial.shape == (2, MASK_SIZE, MASK_SIZE)
    assert out_b.resist.shape == (2, MASK_SIZE, MASK_SIZE)
    assert out_b.printed.shape == (2, MASK_SIZE, MASK_SIZE)


def test_resist_in_unit_interval(simulator, small_mask):
    out = simulator(small_mask)
    r = jax.device_get(out.resist)
    assert jnp.all(r >= 0.0)
    assert jnp.all(r <= 1.0)


def test_printed_is_binary(simulator, small_mask):
    out = simulator(small_mask)
    p = jax.device_get(out.printed)
    assert set(jnp.unique(p).tolist()).issubset({0.0, 1.0})


def test_printed_ste_forward_matches_printed(simulator, small_mask):
    out = simulator(small_mask)
    ste = jax.device_get(out.printed_ste)
    hard = jax.device_get(out.printed)
    assert jnp.allclose(ste, hard)


def test_factory_variants_run(small_mask):
    for factory in (LithographySimulator.nominal, LithographySimulator.maximum, LithographySimulator.minimum):
        out = factory()(small_mask)
        assert out.aerial.shape == small_mask.shape


def test_gradient_through_resist(simulator, small_mask):
    def loss(mask):
        return simulator(mask).resist.sum()

    grad = jax.grad(loss)(small_mask)
    grad = jax.device_get(grad)
    assert grad.shape == small_mask.shape
    assert jnp.isfinite(grad).all()


def test_gradient_through_printed_ste(simulator, small_mask):
    def loss(mask):
        return simulator(mask).printed_ste.sum()

    grad = jax.grad(loss)(small_mask)
    grad = jax.device_get(grad)
    assert grad.shape == small_mask.shape
    assert jnp.isfinite(grad).all()


def test_binarization_threshold_constant():
    assert d.BINARIZATION_THRESHOLD == 0.5


def test_validate_kernel_bank_rejects_bad_scales():
    kernels = jnp.ones((2, 3, 3), dtype=jnp.complex64)
    scales = jnp.array([1.0, -0.1], dtype=jnp.float32)
    with pytest.raises(ValueError, match="non-negative"):
        _validate_kernel_bank("focus", kernels, kernels, scales)


def test_mask_below_minimum_size_raises(simulator):
    mask = jnp.ones((d.MIN_MASK_SIZE - 1, d.MIN_MASK_SIZE - 1), dtype=jnp.float32)
    with pytest.raises(ValueError, match=f"at least {d.MIN_MASK_SIZE}"):
        simulator(mask)


def test_gradient_through_aerial(simulator, random_mask):
    def loss(mask):
        return simulator(mask).aerial.sum()

    grad = jax.grad(loss)(random_mask)
    grad = jax.device_get(grad)
    assert grad.shape == random_mask.shape
    assert jnp.isfinite(grad).all()
    assert not jnp.allclose(grad, 0.0)


def test_random_mask_outputs_finite(simulator, random_mask):
    out = simulator(random_mask)
    for field in (out.aerial, out.resist, out.printed):
        data = jax.device_get(field)
        assert jnp.isfinite(data).all()


def test_margin_preserves_spatial_shape(simulator, random_mask):
    margin = 8
    sim_padded = LithographySimulator(margin=margin)
    out = sim_padded(random_mask)
    assert out.aerial.shape == random_mask.shape
    assert out.resist.shape == random_mask.shape
