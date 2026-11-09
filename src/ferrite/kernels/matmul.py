"""Blocked bf16 matmul with f32 accumulation on the MXU.

Grid is (M / bm, N / bn, K / bk). The two leading axes are ``parallel`` so Mosaic
may split them across TensorCores (megacore on v4/v5p); the K axis is
``arbitrary`` because it carries the f32 accumulator in VMEM scratch between
grid steps. Pallas double-buffers the input blocks, so while the MXU works on
block (i, j, kk) the DMA engine is already fetching (i, j, kk + 1).

XLA's own dot is the baseline to beat (or match). This kernel exists to show
the tiling/pipelining model explicitly and to give the roofline tooling a
kernel whose FLOPs and bytes are known exactly.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ferrite import platform
from ferrite.kernels._compat import compiler_params, dot_precision, round_up


def _matmul_kernel(x_ref, y_ref, o_ref, acc_ref, *, nk: int):
    kk = pl.program_id(2)

    @pl.when(kk == 0)
    def _init():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32, precision=dot_precision(x_ref.dtype))

    @pl.when(kk == nk - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _pick_blocks(m: int, n: int, k: int, bm, bn, bk):
    tile = platform.default_tile()
    big = 4 * tile  # 512 on 128-wide MXUs, 1024 on 256-wide ones
    bm = bm or (big if m >= big else round_up(m, platform.SUBLANES))
    bn = bn or (big if n >= big else round_up(n, platform.LANES))
    bk = bk or (big if k >= big else round_up(k, platform.LANES))
    if bm % platform.SUBLANES or bn % platform.LANES or bk % platform.LANES:
        raise ValueError(f"blocks must be multiples of ({platform.SUBLANES}, {platform.LANES}); got {bm=} {bn=} {bk=}")
    return bm, bn, bk


def _matmul_impl(x, y, block_m=None, block_n=None, block_k=None, out_dtype=None):
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[0]:
        raise ValueError(f"expected [M,K] @ [K,N], got {x.shape} @ {y.shape}")
    m, k = x.shape
    _, n = y.shape
    out_dtype = out_dtype or x.dtype
    bm, bn, bk = _pick_blocks(m, n, k, block_m, block_n, block_k)
    mp, np_, kp = round_up(m, bm), round_up(n, bn), round_up(k, bk)
    if (mp, kp) != (m, k):
        x = jnp.pad(x, ((0, mp - m), (0, kp - k)))
    if (kp, np_) != (k, n):
        y = jnp.pad(y, ((0, kp - k), (0, np_ - n)))
    grid = (mp // bm, np_ // bn, kp // bk)
    itemsize = jnp.dtype(x.dtype).itemsize
    out = pl.pallas_call(
        functools.partial(_matmul_kernel, nk=grid[2]),
        grid=grid,
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
        out_shape=jax.ShapeDtypeStruct((mp, np_), out_dtype),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        compiler_params=compiler_params(("parallel", "parallel", "arbitrary")),
        cost_estimate=pl.CostEstimate(
            flops=2 * mp * np_ * kp,
            transcendentals=0,
            bytes_accessed=(mp * kp + kp * np_) * itemsize + mp * np_ * jnp.dtype(out_dtype).itemsize,
        ),
        name="ferrite_matmul",
    )(x, y)
    if (mp, np_) != (m, n):
        out = out[:m, :n]
    return out


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _matmul_vjp(x, y, cfg):
    bm, bn, bk, out_dtype = cfg
    return _matmul_impl(x, y, bm, bn, bk, jnp.dtype(out_dtype) if out_dtype else None)


def _matmul_vjp_fwd(x, y, cfg):
    return _matmul_vjp(x, y, cfg), (x, y)


def _matmul_vjp_bwd(cfg, res, g):
    x, y = res
    # dX = G @ Y^T, dY = X^T @ G; both go through the same kernel with auto block sizes.
    dx = _matmul_impl(g.astype(x.dtype), y.T, out_dtype=x.dtype)
    dy = _matmul_impl(x.T, g.astype(y.dtype), out_dtype=y.dtype)
    return dx, dy


_matmul_vjp.defvjp(_matmul_vjp_fwd, _matmul_vjp_bwd)


def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    out_dtype=None,
) -> jax.Array:
    """``x @ y`` for 2-D ``x: [M, K]`` and ``y: [K, N]`` as a Pallas TPU kernel.

    Any shape is accepted: operands are zero-padded up to the block grid and the
    result is sliced back, which is exact because padded K contributes zeros.
    Differentiable: the VJP runs the transposed products through the same kernel.
    """
    cfg = (block_m, block_n, block_k, jnp.dtype(out_dtype).name if out_dtype is not None else None)
    return _matmul_vjp(x, y, cfg)


def matmul_reference(x: jax.Array, y: jax.Array, *, out_dtype=None) -> jax.Array:
    """XLA baseline with the same numerics contract (f32 accumulation)."""
    return jnp.dot(x, y, preferred_element_type=jnp.float32).astype(out_dtype or x.dtype)


def matmul_flops(m: int, n: int, k: int) -> int:
    return 2 * m * n * k
