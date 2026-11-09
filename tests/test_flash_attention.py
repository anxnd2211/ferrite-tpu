import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from ferrite import platform
from ferrite.kernels.flash_attention import BlockSizes, flash_attention, flash_attention_reference


def _qkv(key, b, h, s, d, dtype):
    ks = jax.random.split(key, 3)
    return tuple(jax.random.normal(k, (b, h, s, d), jnp.float32).astype(dtype) for k in ks)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("s,d,bs", [(256, 128, BlockSizes(128, 128)), (512, 64, BlockSizes(128, 256)), (256, 256, BlockSizes(64, 128))])
def test_forward_f32(key, causal, s, d, bs):
    q, k, v = _qkv(key, 1, 2, s, d, jnp.float32)
    out = jax.jit(lambda q, k, v: flash_attention(q, k, v, causal=causal, block_sizes=bs))(q, k, v)
    ref = flash_attention_reference(q, k, v, causal=causal)
    np.testing.assert_allclose(out, ref, atol=2e-5, rtol=2e-5)


def test_forward_bf16(key):
    q, k, v = _qkv(key, 2, 2, 256, 128, jnp.bfloat16)
    out = flash_attention(q, k, v, causal=True)
    ref = flash_attention_reference(q, k, v, causal=True)
    assert out.dtype == jnp.bfloat16
    np.testing.assert_allclose(out.astype(jnp.float32), ref.astype(jnp.float32), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("s,d,bs", [(256, 128, BlockSizes()), (384, 64, BlockSizes(128, 128, 64, 128))])
def test_backward_matches_reference(key, causal, s, d, bs):
    q, k, v = _qkv(key, 1, 2, s, d, jnp.float32)
    cot = jax.random.normal(jax.random.key(1), q.shape, jnp.float32)

    def loss(fn):
        return lambda q, k, v: jnp.sum(fn(q, k, v, causal=causal) * cot)

    g = jax.jit(jax.grad(loss(lambda q, k, v, causal: flash_attention(q, k, v, causal=causal, block_sizes=bs)), argnums=(0, 1, 2)))(q, k, v)
    g_ref = jax.grad(loss(flash_attention_reference), argnums=(0, 1, 2))(q, k, v)
    for name, a, b in zip("qkv", g, g_ref):
        np.testing.assert_allclose(a, b, atol=5e-4, rtol=5e-4, err_msg=f"d{name}")


def test_backward_bf16_is_close(key):
    q, k, v = _qkv(key, 1, 2, 256, 128, jnp.bfloat16)
    f = lambda fn: (lambda q, k, v: jnp.sum(fn(q, k, v, causal=True).astype(jnp.float32) ** 2))
    g = jax.grad(f(flash_attention), argnums=(0, 1, 2))(q, k, v)
    g_ref = jax.grad(f(flash_attention_reference), argnums=(0, 1, 2))(q, k, v)
    for a, b in zip(g, g_ref):
        a, b = a.astype(jnp.float32), b.astype(jnp.float32)
        rel = float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
        assert rel < 5e-2, rel


def test_causal_output_is_prefix_invariant(key):
    """Causal attention on the first 256 tokens must equal the first 256 rows of a 512-token run."""
    q, k, v = _qkv(key, 1, 1, 512, 128, jnp.float32)
    full = flash_attention(q, k, v, causal=True)
    short = flash_attention(q[:, :, :256], k[:, :, :256], v[:, :, :256], causal=True)
    np.testing.assert_allclose(full[:, :, :256], short, atol=2e-5, rtol=2e-5)


def test_validation_errors(key):
    q, k, v = _qkv(key, 1, 1, 256, 128, jnp.float32)
    with pytest.raises(ValueError):
        flash_attention(q, k, v, block_sizes=BlockSizes(block_k=100))
    with pytest.raises(ValueError):
        flash_attention(q, k[:, :, :128], v[:, :, :128], causal=True)
    with pytest.raises(ValueError):
        flash_attention(q, k, v.astype(jnp.bfloat16))


@pytest.mark.skipif(platform.is_tpu(), reason="race detector is an interpret-mode feature")
def test_kernels_are_race_free(key):
    """Run forward+backward under the interpreter's data-race detector."""
    q, k, v = _qkv(key, 1, 1, 256, 128, jnp.float32)
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        g = jax.grad(lambda q, k, v: jnp.sum(flash_attention(q, k, v, causal=True)), argnums=(0, 1, 2))(q, k, v)
        jax.block_until_ready(g)
