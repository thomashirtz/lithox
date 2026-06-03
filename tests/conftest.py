import jax.numpy as jnp
import pytest

MASK_SIZE = 64


@pytest.fixture
def small_mask():
    """Single 64×64 mask in [0, 1]."""
    return jnp.ones((MASK_SIZE, MASK_SIZE), dtype=jnp.float32) * 0.5


@pytest.fixture
def batched_mask():
    """Batch of two 64×64 masks."""
    return jnp.ones((2, MASK_SIZE, MASK_SIZE), dtype=jnp.float32) * 0.5
