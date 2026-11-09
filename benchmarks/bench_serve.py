"""Serving benchmark: time-to-first-token, inter-token latency and throughput."""

from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import results_dir, write_json  # noqa: E402

from ferrite import platform  # noqa: E402
from ferrite.model.gpt import GPTConfig, init_params, param_count  # noqa: E402
from ferrite.model.train import load_params  # noqa: E402
from ferrite.serving.engine import Engine, ServeConfig  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None, help=".npz from training; random init if omitted")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--prompt-len", type=int, default=100)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    quick = a.quick or not platform.is_tpu()
    cfg = GPTConfig(seq_len=a.seq, d_model=a.d_model, n_head=a.heads, n_layer=a.layers, d_ff=a.d_ff, dtype=jnp.bfloat16)
    if quick:
        cfg = cfg.replace(seq_len=256, d_model=256, n_head=2, n_layer=2, d_ff=512)
    batches = [1, 2] if quick else [int(b) for b in a.batches.split(",")]
    prompt_len = 20 if quick else a.prompt_len
    new_tokens = 8 if quick else a.new_tokens
    params = load_params(a.params, cfg) if a.params else init_params(jax.random.key(0), cfg)
    rng = np.random.default_rng(0)
    rows = []
    with platform.interpret_mode():
        for batch in batches:
            for kernels in ("pallas", "xla"):
                eng = Engine(params, cfg, ServeConfig(batch=batch, max_len=cfg.seq_len, buckets=(64, 128, 256) if cfg.seq_len >= 256 else (64, 128), prefill_kernels=kernels))
                prompts = [list(rng.integers(1, 256, size=prompt_len)) for _ in range(batch)]
                eng.generate(prompts, 4)  # warm the executables / transfers
                _, stats = eng.generate(prompts, new_tokens)
                blob = eng.export_decode()
                rows.append({"batch": batch, "prefill_kernels": kernels, "prompt_len": prompt_len, "new_tokens": new_tokens, **stats, "compile_seconds": eng.compile_seconds, "exported_decode_bytes": len(blob)})
                print(rows[-1])
    out = a.out or os.path.join(results_dir(), "serve.json")
    write_json(out, {"platform": platform.describe(), "quick": quick, "model": {"params": param_count(params), "seq_len": cfg.seq_len, "d_model": cfg.d_model, "n_layer": cfg.n_layer}, "runs": rows})


if __name__ == "__main__":
    main()
