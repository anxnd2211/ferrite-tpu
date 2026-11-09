"""Capture a device trace of the flash-attention kernel (fwd+bwd) next to the XLA baseline.

Writes a TensorBoard/Perfetto-compatible trace to ``traces/flash_attention_vs_xla``.
Runs as its own process on purpose: a TPU chip is owned by one process at a time.
"""

from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
from _common import device_slug  # noqa: E402

from ferrite import platform  # noqa: E402
from ferrite.kernels.flash_attention import BlockSizes, flash_attention  # noqa: E402
from ferrite.profiling import trace  # noqa: E402


def _xla_attention(q, k, v):
    t = lambda a: jnp.swapaxes(a, 1, 2)
    return t(jax.nn.dot_product_attention(t(q), t(k), t(v), is_causal=True, implementation="xla"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="4,8,2048,128", help="batch,heads,seq,head_dim")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    quick = a.quick or not platform.is_tpu()
    b, h, s, d = (1, 2, 256, 128) if quick else tuple(int(x) for x in a.shape.split(","))
    out = a.out or os.path.join("traces", device_slug(), "flash_attention_vs_xla")
    with platform.interpret_mode():
        ks = jax.random.split(jax.random.key(0), 3)
        q, k, v = (jax.random.normal(kk, (b, h, s, d), jnp.float32).astype(jnp.bfloat16) for kk in ks)
        bs = BlockSizes.for_device() if s % platform.default_tile() == 0 else BlockSizes()
        f_pallas = jax.jit(jax.grad(lambda q, k, v: jnp.sum(flash_attention(q, k, v, causal=True, block_sizes=bs).astype(jnp.float32)), argnums=(0, 1, 2)))
        f_xla = jax.jit(jax.grad(lambda q, k, v: jnp.sum(_xla_attention(q, k, v).astype(jnp.float32)), argnums=(0, 1, 2)))
        for f in (f_pallas, f_xla):
            jax.block_until_ready(f(q, k, v))  # compile outside the trace
        with trace(out):
            for _ in range(a.steps):
                jax.block_until_ready(f_pallas(q, k, v))
                jax.block_until_ready(f_xla(q, k, v))
    files = [os.path.join(dp, fn) for dp, _, fns in os.walk(out) for fn in fns]
    print(f"trace written to {out}: {len(files)} file(s)")
    for fpath in files[:5]:
        print("  ", fpath, os.path.getsize(fpath), "bytes")


if __name__ == "__main__":
    main()
