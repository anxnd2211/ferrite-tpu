import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ferrite.kernels.matmul import matmul, matmul_reference


def _rand(key, shape, dtype):
    return jax.random.normal(key, shape, jnp.float32).astype(dtype)


@pytest.mark.parametrize("dtype,atol", [(jnp.float32, 1e-3), (jnp.bfloat16, 0.5)])
def test_matches_reference_aligned(key, dtype, atol):
    k1, k2 = jax.random.split(key)
    x, y = _rand(k1, (256, 384), dtype), _rand(k2, (384, 512), dtype)
    out = jax.jit(lambda a, b: matmul(a, b, block_m=128, block_n=256, block_k=128))(x, y)
    ref = matmul_reference(x, y)
    assert out.shape == (256, 512) and out.dtype == dtype
    np.testing.assert_allclose(out.astype(jnp.float32), ref.astype(jnp.float32), atol=atol, rtol=1e-2)


def test_ragged_shapes_are_padded_exactly(key):
    k1, k2 = jax.random.split(key)
    x, y = _rand(k1, (77, 130), jnp.float32), _rand(k2, (130, 203), jnp.float32)
    out = matmul(x, y, block_m=64, block_n=128, block_k=128)
    np.testing.assert_allclose(out, matmul_reference(x, y), atol=1e-3, rtol=1e-3)


def test_f32_output_from_bf16_inputs(key):
    k1, k2 = jax.random.split(key)
    x, y = _rand(k1, (128, 256), jnp.bfloat16), _rand(k2, (256, 128), jnp.bfloat16)
    out = matmul(x, y, out_dtype=jnp.float32)
    assert out.dtype == jnp.float32
    np.testing.assert_allclose(out, matmul_reference(x, y, out_dtype=jnp.float32), atol=1e-2, rtol=1e-2)


def test_rejects_bad_blocks(key):
    x = jnp.zeros((128, 128), jnp.float32)
    with pytest.raises(ValueError):
        matmul(x, x, block_n=100)
