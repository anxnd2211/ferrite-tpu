import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ferrite.model import data as data_lib
from ferrite.model.gpt import GPTConfig, forward, init_params, loss_fn, param_count
from ferrite.model.train import TrainConfig, train

TINY = GPTConfig(vocab_size=256, seq_len=128, d_model=256, n_head=2, n_layer=2, d_ff=512, dtype=jnp.float32)


def test_pallas_and_xla_paths_agree(key):
    params = init_params(key, TINY)
    tokens = jax.random.randint(jax.random.key(1), (2, 128), 0, 256)
    out_pallas = forward(params, tokens, TINY)
    out_xla = forward(params, tokens, TINY.replace(attention="xla", norm="xla"))
    assert out_pallas.shape == (2, 128, 256)
    np.testing.assert_allclose(out_pallas, out_xla, atol=2e-4, rtol=2e-4)


def test_pallas_linear_path_agrees(key):
    params = init_params(key, TINY)
    tokens = jax.random.randint(jax.random.key(1), (2, 128), 0, 256)
    a = forward(params, tokens, TINY.replace(linear="pallas"))
    b = forward(params, tokens, TINY.replace(attention="xla", norm="xla", linear="xla"))
    np.testing.assert_allclose(a, b, atol=5e-4, rtol=5e-4)


def test_gradients_agree_between_paths(key):
    params = init_params(key, TINY)
    x = jax.random.randint(jax.random.key(1), (2, 128), 0, 256)
    y = jnp.roll(x, -1, axis=1)
    g1 = jax.grad(loss_fn)(params, x, y, TINY)
    g2 = jax.grad(loss_fn)(params, x, y, TINY.replace(attention="xla", norm="xla"))
    for a, b in zip(jax.tree_util.tree_leaves(g1), jax.tree_util.tree_leaves(g2)):
        np.testing.assert_allclose(a, b, atol=2e-4, rtol=2e-3)


def test_param_count_matches_formula():
    params = init_params(jax.random.key(0), TINY)
    d, f, L, V, S = TINY.d_model, TINY.d_ff, TINY.n_layer, TINY.vocab_size, TINY.seq_len
    expected = V * d + S * d + d + L * (2 * d + 3 * d * d + d * d + d * f + f * d)
    assert param_count(params) == expected


def test_synthetic_data_batches():
    corpus = data_lib.synthetic(20_000, seed=1)
    x, y = data_lib.get_batch(np.random.default_rng(0), corpus, 4, 16)
    assert x.shape == (4, 16) and y.shape == (4, 16)
    np.testing.assert_array_equal(x[:, 1:], y[:, :-1])


def test_short_training_run_reduces_loss(tmp_path):
    cfg = TINY.replace(seq_len=128, d_model=128, n_head=1, n_layer=1, d_ff=256)
    tcfg = TrainConfig(steps=12, batch=8, lr=3e-3, warmup=2, log_every=100, data="synthetic", out=str(tmp_path / "r.json"), save_params=str(tmp_path / "p.npz"))
    res = train(cfg, tcfg, verbose=False)
    assert res["loss_last"] < res["loss_first"]
    assert (tmp_path / "r.json").exists() and (tmp_path / "p.npz").exists()
    assert res["tokens_per_second"] > 0
