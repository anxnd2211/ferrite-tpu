import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ferrite.kernels.rmsnorm import rmsnorm, rmsnorm_reference


@pytest.mark.parametrize("shape", [(4, 64, 256), (300, 384), (2, 8, 128)])
def test_forward_matches_reference(key, shape):
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, shape, jnp.float32)
    w = 1.0 + 0.1 * jax.random.normal(k2, shape[-1:], jnp.float32)
    np.testing.assert_allclose(rmsnorm(x, w), rmsnorm_reference(x, w), atol=1e-5, rtol=1e-5)


def test_fused_residual_forward(key):
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (2, 256, 256), jnp.bfloat16)
    r = jax.random.normal(k2, (2, 256, 256), jnp.bfloat16)
    w = jnp.ones((256,), jnp.bfloat16)
    h, y = rmsnorm(x, w, residual=r)
    h_ref, y_ref = rmsnorm_reference(x, w, residual=r)
    np.testing.assert_allclose(h.astype(jnp.float32), h_ref.astype(jnp.float32), atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(y.astype(jnp.float32), y_ref.astype(jnp.float32), atol=2e-2, rtol=2e-2)


def test_gradients_match_reference(key):
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (64, 256), jnp.float32)
    r = jax.random.normal(k2, (64, 256), jnp.float32)
    w = 1.0 + 0.1 * jax.random.normal(k3, (256,), jnp.float32)
    cot = jax.random.normal(jax.random.key(9), (64, 256), jnp.float32)

    def loss(fn):
        def f(x, r, w):
            h, y = fn(x, r, w, residual=None) if False else fn(x, w, residual=r)
            return jnp.sum(y * cot) + jnp.sum(h * cot[::-1])
        return f

    g = jax.grad(loss(rmsnorm), argnums=(0, 1, 2))(x, r, w)
    g_ref = jax.grad(loss(rmsnorm_reference), argnums=(0, 1, 2))(x, r, w)
    for a, b in zip(g, g_ref):
        np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)


def test_plain_gradient(key):
    x = jax.random.normal(key, (32, 128), jnp.float32)
    w = jnp.ones((128,), jnp.float32)
    f = lambda fn: (lambda x, w: jnp.sum(jnp.tanh(fn(x, w))))
    gx, gw = jax.grad(f(rmsnorm), argnums=(0, 1))(x, w)
    gx_ref, gw_ref = jax.grad(f(rmsnorm_reference), argnums=(0, 1))(x, w)
    np.testing.assert_allclose(gx, gx_ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(gw, gw_ref, atol=1e-5, rtol=1e-5)
