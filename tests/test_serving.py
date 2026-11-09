import jax
import jax.numpy as jnp
import numpy as np

from ferrite.model.gpt import GPTConfig, init_params
from ferrite.serving.engine import Engine, ServeConfig, bucket_for, reference_greedy

CFG = GPTConfig(vocab_size=256, seq_len=256, d_model=128, n_head=1, n_layer=2, d_ff=256, dtype=jnp.float32)


def test_engine_matches_reference_greedy(key):
    params = init_params(key, CFG)
    eng = Engine(params, CFG, ServeConfig(batch=2, max_len=256, buckets=(128,)))
    assert eng.n_compiles == 2  # one prefill bucket + decode
    prompts = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [42, 7, 99]]
    gen, stats = eng.generate(prompts, max_new_tokens=6)
    assert gen.shape == (2, 6) and stats["ttft_ms"] > 0
    for i, p in enumerate(prompts):
        ref = reference_greedy(params, CFG.replace(attention="xla", norm="xla"), p, 6)
        np.testing.assert_array_equal(gen[i], ref)


def test_export_round_trip(key):
    params = init_params(key, CFG)
    eng = Engine(params, CFG, ServeConfig(batch=1, max_len=256, buckets=(128,)))
    token, pos, kc, vc = eng.prefill([[5, 6, 7]])
    blob = eng.export_decode()
    assert len(blob) > 1000
    exp = Engine.load_exported(blob)
    logits_a, _, _ = exp.call(params, token, pos, kc, vc)
    logits_b, _, _ = eng._decode_jit(params, token, pos, kc, vc)
    np.testing.assert_allclose(logits_a, logits_b, atol=1e-5, rtol=1e-5)


def test_bucket_selection():
    assert bucket_for(1, (64, 128)) == 64 and bucket_for(64, (64, 128)) == 64 and bucket_for(65, (64, 128)) == 128
