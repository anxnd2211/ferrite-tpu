"""Time each Pallas kernel against its XLA baseline and place it on the roofline.

Run on a TPU for meaningful numbers; ``--quick`` uses tiny shapes so the script
also completes on CPU through interpret mode as a smoke test.
"""

from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
from _common import results_dir, write_json  # noqa: E402

from ferrite import platform  # noqa: E402
from ferrite.kernels.flash_attention import BlockSizes, attention_flops, flash_attention, flash_attention_reference  # noqa: E402
from ferrite.kernels.matmul import matmul, matmul_flops, matmul_reference  # noqa: E402
from ferrite.kernels.rmsnorm import rmsnorm, rmsnorm_reference  # noqa: E402
from ferrite.profiling import analyze, compile_report, format_table, timeit  # noqa: E402


def _try_time(fn, *args, iters):
    """Time fn, or return the error so one failing shape does not end the benchmark."""
    try:
        return timeit(fn, *args, iters=iters), None
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
        print(f"  !! {type(e).__name__}: {msg}")
        return None, f"{type(e).__name__}: {msg}"


def bench_matmul(shapes, iters):
    rows, points = [], []
    for m, k, n in shapes:
        x = jax.random.normal(jax.random.key(0), (m, k), jnp.float32).astype(jnp.bfloat16)
        y = jax.random.normal(jax.random.key(1), (k, n), jnp.float32).astype(jnp.bfloat16)
        flops = matmul_flops(m, n, k)
        bytes_ = (m * k + k * n + m * n) * 2
        for name, fn in [("pallas", jax.jit(matmul)), ("xla", jax.jit(matmul_reference))]:
            t, err = _try_time(fn, x, y, iters=iters)
            if err:
                rows.append({"shape": [m, k, n], "impl": name, "error": err})
                continue
            p = analyze(f"matmul {m}x{k}x{n} {name}", flops, bytes_, t.median_s)
            points.append(p)
            rows.append({"shape": [m, k, n], "impl": name, **p.to_dict(), "ms": t.median_ms})
    return rows, points


def bench_rmsnorm(shapes, iters):
    rows, points = [], []
    for r, d in shapes:
        x = jax.random.normal(jax.random.key(0), (r, d), jnp.float32).astype(jnp.bfloat16)
        res = jax.random.normal(jax.random.key(1), (r, d), jnp.float32).astype(jnp.bfloat16)
        w = jnp.ones((d,), jnp.float32)
        flops = 5 * r * d
        bytes_ = 4 * r * d * 2  # read x, residual; write h, y
        for name, fn in [("pallas", jax.jit(lambda x, w, res: rmsnorm(x, w, residual=res))), ("xla", jax.jit(lambda x, w, res: rmsnorm_reference(x, w, residual=res)))]:
            t, err = _try_time(fn, x, w, res, iters=iters)
            if err:
                rows.append({"shape": [r, d], "impl": name, "error": err})
                continue
            p = analyze(f"add+rmsnorm {r}x{d} {name}", flops, bytes_, t.median_s)
            points.append(p)
            rows.append({"shape": [r, d], "impl": name, **p.to_dict(), "ms": t.median_ms})
    return rows, points


def _xla_attention(q, k, v):
    t = lambda a: jnp.swapaxes(a, 1, 2)
    return t(jax.nn.dot_product_attention(t(q), t(k), t(v), is_causal=True, implementation="xla"))


def _official_attention():
    try:
        from jax.experimental.pallas.ops.tpu import flash_attention as fa
    except Exception:
        return None
    bs = fa.BlockSizes(block_q=128, block_k_major=128, block_k=128, block_b=1, block_q_major_dkv=128, block_k_major_dkv=128, block_k_dkv=128, block_q_dkv=128, block_k_major_dq=128, block_k_dq=128, block_q_dq=128)

    def fn(q, k, v):
        return fa.flash_attention(q, k, v, causal=True, sm_scale=float(1.0 / (q.shape[-1] ** 0.5)), block_sizes=bs)

    return fn


def bench_attention(shapes, iters, include_official):
    rows, points = [], []
    official = _official_attention() if include_official else None
    for b, h, s, d in shapes:
        ks = jax.random.split(jax.random.key(0), 3)
        q, k, v = (jax.random.normal(kk, (b, h, s, d), jnp.float32).astype(jnp.bfloat16) for kk in ks)
        bs = BlockSizes.for_device() if s % platform.default_tile() == 0 else BlockSizes()
        impls = [("pallas", jax.jit(lambda q, k, v: flash_attention(q, k, v, causal=True, block_sizes=bs))), ("xla", jax.jit(_xla_attention))]
        if official is not None:
            impls.append(("jax-official-pallas", jax.jit(official)))
        fwd_flops = attention_flops(b, h, s, s, d, causal=True)
        bytes_ = 4 * b * h * s * d * 2  # q, k, v read + o written (bf16)
        for name, fn in impls:
            t, err = _try_time(fn, q, k, v, iters=iters)
            if err:
                rows.append({"shape": [b, h, s, d], "impl": name, "pass": "fwd", "error": err})
                continue
            p = analyze(f"attn fwd b{b} h{h} s{s} d{d} {name}", fwd_flops, bytes_, t.median_s)
            points.append(p)
            rows.append({"shape": [b, h, s, d], "impl": name, "pass": "fwd", **p.to_dict(), "ms": t.median_ms})
        # forward + backward
        cot = jnp.ones((b, h, s, d), jnp.bfloat16)
        for name, fn in impls[:2]:
            g = jax.jit(jax.grad(lambda q, k, v, _f=fn: jnp.sum(_f(q, k, v).astype(jnp.float32) * cot), argnums=(0, 1, 2)))
            t, err = _try_time(g, q, k, v, iters=max(2, iters // 2))
            if err:
                rows.append({"shape": [b, h, s, d], "impl": name, "pass": "fwd+bwd", "error": err})
                continue
            p = analyze(f"attn fwd+bwd b{b} h{h} s{s} d{d} {name}", attention_flops(b, h, s, s, d, causal=True, backward=True) + fwd_flops, bytes_ * 3, t.median_s)
            points.append(p)
            rows.append({"shape": [b, h, s, d], "impl": name, "pass": "fwd+bwd", **p.to_dict(), "ms": t.median_ms})
    return rows, points


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny shapes; smoke test on CPU")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    quick = a.quick or not platform.is_tpu()
    mm_shapes = [(512, 512, 512)] if quick else [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)]
    norm_shapes = [(1024, 256)] if quick else [(8192, 1024), (32768, 2048), (65536, 4096)]
    attn_shapes = [(1, 2, 256, 128)] if quick else [(4, 8, 1024, 128), (4, 8, 2048, 128), (2, 8, 4096, 128), (1, 8, 8192, 128)]
    iters = 3 if quick else a.iters
    with platform.interpret_mode():
        print(platform.describe())
        mm_rows, mm_pts = bench_matmul(mm_shapes, iters)
        print(format_table(mm_pts))
        n_rows, n_pts = bench_rmsnorm(norm_shapes, iters)
        print(format_table(n_pts))
        a_rows, a_pts = bench_attention(attn_shapes, iters, include_official=platform.is_tpu())
        print(format_table(a_pts))
        q, k, v = (jnp.zeros(attn_shapes[0], jnp.bfloat16) for _ in range(3))
        hlo = compile_report(lambda q, k, v: flash_attention(q, k, v, causal=True), q, k, v, name="flash_attention_fwd")
    out = a.out or os.path.join(results_dir(), "kernels.json")
    write_json(out, {"platform": platform.describe(), "quick": quick, "matmul": mm_rows, "rmsnorm": n_rows, "attention": a_rows, "hlo_flash_attention_fwd": hlo})


if __name__ == "__main__":
    main()
