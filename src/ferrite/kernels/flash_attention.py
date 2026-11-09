"""Causal flash attention, forward and backward, as Pallas TPU kernels.

Layout: ``q, k, v: [batch, heads, seq, head_dim]``. The forward grid is
``(batch, heads, seq/bq, seq/bk)``; each grid step multiplies one ``bq x d`` query
block against one ``bk x d`` key block on the MXU, updates the online-softmax row
statistics (kept in f32, lane-replicated as ``(bq, 128)`` VMEM scratch, which is
the TPU-friendly layout for per-row scalars) and accumulates ``P @ V`` in an f32
``(bq, d)`` scratch. The accumulator is normalised once, at the last key block,
rather than at every step.

Causal masking is applied with an iota compare, but the real saving is in
skipping: the body runs under ``pl.when(block is on or below the diagonal)`` so
fully-masked blocks cost no MXU time, and the K/V ``index_map`` redirects skipped
blocks to the block that will be needed next, so the pipeline issues no DMA for
them either. For a sequence of n blocks that halves both compute and traffic.

Backward uses the FlashAttention-2 recipe: ``lse`` from the forward pass and
``delta = rowsum(dO * O)`` let two kernels recompute ``P`` blockwise. ``dK, dV``
are accumulated over query blocks (grid: kv outer, q inner) and ``dQ`` over key
blocks (grid: q outer, kv inner), so neither kernel needs atomics.

API shape and skipping idioms follow ``jax.experimental.pallas.ops.tpu.flash_attention``;
the code here is written from the paper and the Pallas docs.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ferrite import platform
from ferrite.kernels._compat import compiler_params, dot_precision

NUM_LANES = platform.LANES
# Large negative finite value instead of -inf so masked rows never produce inf - inf.
MASK_VALUE = -0.7 * float(np.finfo(np.float32).max)
_NT = (((1,), (1,)), ((), ()))  # a @ b.T
_NN = (((1,), (0,)), ((), ()))  # a @ b


@dataclasses.dataclass(frozen=True)
class BlockSizes:
    """Tile sizes. ``block_q`` is a multiple of 8, ``block_k`` a multiple of 128."""

    block_q: int = 128
    block_k: int = 128
    block_q_bwd: int = 128
    block_k_bwd: int = 128

    @classmethod
    def for_device(cls) -> "BlockSizes":
        """Larger tiles on 256-wide MXUs (v6e, v7x), 128 elsewhere."""
        t = platform.default_tile()
        return cls(t, t, t, t)


def _on_or_below_diag(qi, bq: int, ki, bk: int):
    """True when block (qi, ki) has at least one unmasked (col <= row) entry."""
    return (qi + 1) * bq - 1 >= ki * bk


def _lane_tile(x: jax.Array, n: int) -> jax.Array:
    """(rows, 128) lane-replicated statistic -> (rows, n)."""
    reps, rem = divmod(n, NUM_LANES)
    if rem:
        if reps:
            raise ValueError(f"dim {n} must be <= 128 or a multiple of 128")
        return x[:, :n]
    return x if reps == 1 else jnp.tile(x, (1, reps))


def _causal_mask(s, qi, ki, bq: int, bk: int):
    rows = qi * bq + lax.broadcasted_iota(jnp.int32, (bq, bk), 0)
    cols = ki * bk + lax.broadcasted_iota(jnp.int32, (bq, bk), 1)
    return jnp.where(cols <= rows, s, MASK_VALUE)


# --------------------------------------------------------------------------
# forward
# --------------------------------------------------------------------------


def _fwd_kernel(q_ref, k_ref, v_ref, o_ref, lse_ref, m_sc, l_sc, acc_sc, *, scale, causal, bq, bk, nk):
    qi, ki = pl.program_id(2), pl.program_id(3)

    @pl.when(ki == 0)
    def _init():
        m_sc[...] = jnp.full(m_sc.shape, -jnp.inf, jnp.float32)
        l_sc[...] = jnp.zeros(l_sc.shape, jnp.float32)
        acc_sc[...] = jnp.zeros(acc_sc.shape, jnp.float32)

    def _step():
        q, k, v = q_ref[...], k_ref[...], v_ref[...]
        prec = dot_precision(q.dtype)
        s = lax.dot_general(q, k, _NT, precision=prec, preferred_element_type=jnp.float32)
        if scale != 1.0:
            s = s * scale
        if causal:
            s = _causal_mask(s, qi, ki, bq, bk)
        m_prev, l_prev = m_sc[...], l_sc[...]
        m_next = jnp.maximum(m_prev, jnp.max(s, axis=1)[:, None])  # (bq, 128)
        p = jnp.exp(s - _lane_tile(m_next, bk))  # (bq, bk)
        alpha = jnp.exp(m_prev - m_next)  # (bq, 128)
        l_sc[...] = alpha * l_prev + jnp.sum(p, axis=1)[:, None]
        m_sc[...] = m_next
        pv = jnp.dot(p.astype(v.dtype), v, precision=prec, preferred_element_type=jnp.float32)  # (bq, d)
        acc_sc[...] = acc_sc[...] * _lane_tile(alpha, acc_sc.shape[1]) + pv

    if causal:
        pl.when(_on_or_below_diag(qi, bq, ki, bk))(_step)
    else:
        _step()

    @pl.when(ki == nk - 1)
    def _finish():
        l = l_sc[...]
        inv = 1.0 / l
        o_ref[...] = (acc_sc[...] * _lane_tile(inv, acc_sc.shape[1])).astype(o_ref.dtype)
        lse_ref[...] = m_sc[...] + jnp.log(l)


def _flash_forward(q, k, v, *, causal: bool, scale: float, bs: BlockSizes):
    b, h, sq, d = q.shape
    sk = k.shape[2]
    bq, bk = bs.block_q, bs.block_k
    nq, nk = sq // bq, sk // bk

    def q_map(bi, hi, qi, ki):
        return (bi, hi, qi, 0)

    def kv_map(bi, hi, qi, ki):
        if causal:  # skipped blocks point at block 0, the first block of the next q row
            ki = lax.select(_on_or_below_diag(qi, bq, ki, bk), ki, 0)
        return (bi, hi, ki, 0)

    itemsize = jnp.dtype(q.dtype).itemsize
    frac = 0.5 if causal else 1.0
    out, lse = pl.pallas_call(
        functools.partial(_fwd_kernel, scale=scale, causal=causal, bq=bq, bk=bk, nk=nk),
        grid=(b, h, nq, nk),
        in_specs=[
            pl.BlockSpec((None, None, bq, d), q_map),
            pl.BlockSpec((None, None, bk, d), kv_map),
            pl.BlockSpec((None, None, bk, d), kv_map),
        ],
        out_specs=[
            pl.BlockSpec((None, None, bq, d), q_map),
            pl.BlockSpec((None, None, bq, NUM_LANES), q_map),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((b, h, sq, d), q.dtype),
            jax.ShapeDtypeStruct((b, h, sq, NUM_LANES), jnp.float32),
        ],
        scratch_shapes=[
            pltpu.VMEM((bq, NUM_LANES), jnp.float32),
            pltpu.VMEM((bq, NUM_LANES), jnp.float32),
            pltpu.VMEM((bq, d), jnp.float32),
        ],
        compiler_params=compiler_params(("parallel", "parallel", "parallel", "arbitrary")),
        cost_estimate=pl.CostEstimate(
            flops=int(4 * b * h * sq * sk * d * frac),
            transcendentals=int(b * h * sq * sk * frac),
            bytes_accessed=int((2 * q.size + (k.size + v.size) * frac) * itemsize),  # q read, o written, k/v read once (half if causal)
        ),
        name="ferrite_flash_attention_fwd",
    )(q, k, v)
    return out, lse



# --------------------------------------------------------------------------
# backward
# --------------------------------------------------------------------------


def _recompute_p_ds(q, k, v, do, lse, delta, *, scale, causal, qi, ki, bq, bk):
    prec = dot_precision(q.dtype)
    s = lax.dot_general(q, k, _NT, precision=prec, preferred_element_type=jnp.float32)
    if scale != 1.0:
        s = s * scale
    if causal:
        s = _causal_mask(s, qi, ki, bq, bk)
    p = jnp.exp(s - _lane_tile(lse, bk))  # (bq, bk), already normalised
    dp = lax.dot_general(do, v, _NT, precision=prec, preferred_element_type=jnp.float32)  # (bq, bk)
    ds = p * (dp - _lane_tile(delta, bk))  # (bq, bk), gradient wrt scaled scores
    return p, ds


def _bwd_dkv_kernel(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, dk_ref, dv_ref, dk_sc, dv_sc, *, scale, causal, bq, bk, nq):
    ki, qi = pl.program_id(2), pl.program_id(3)  # kv block outer, q block inner

    @pl.when(qi == 0)
    def _init():
        dk_sc[...] = jnp.zeros(dk_sc.shape, jnp.float32)
        dv_sc[...] = jnp.zeros(dv_sc.shape, jnp.float32)

    def _step():
        q, k, v, do = q_ref[...], k_ref[...], v_ref[...], do_ref[...]
        p, ds = _recompute_p_ds(q, k, v, do, lse_ref[...], delta_ref[...], scale=scale, causal=causal, qi=qi, ki=ki, bq=bq, bk=bk)
        # Explicit XLU transposes of the (bq, bk) tiles, then plain NN matmuls on the MXU.
        prec = dot_precision(q.dtype)
        dv_sc[...] += lax.dot_general(p.T.astype(do.dtype), do, _NN, precision=prec, preferred_element_type=jnp.float32)  # P^T dO
        dk_sc[...] += lax.dot_general(ds.T.astype(q.dtype), q, _NN, precision=prec, preferred_element_type=jnp.float32)  # dS^T Q

    if causal:
        pl.when(_on_or_below_diag(qi, bq, ki, bk))(_step)
    else:
        _step()

    @pl.when(qi == nq - 1)
    def _finish():
        dk_ref[...] = (dk_sc[...] * scale).astype(dk_ref.dtype)
        dv_ref[...] = dv_sc[...].astype(dv_ref.dtype)


def _bwd_dq_kernel(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, dq_ref, dq_sc, *, scale, causal, bq, bk, nk):
    qi, ki = pl.program_id(2), pl.program_id(3)  # q block outer, kv block inner

    @pl.when(ki == 0)
    def _init():
        dq_sc[...] = jnp.zeros(dq_sc.shape, jnp.float32)

    def _step():
        q, k, v, do = q_ref[...], k_ref[...], v_ref[...], do_ref[...]
        _, ds = _recompute_p_ds(q, k, v, do, lse_ref[...], delta_ref[...], scale=scale, causal=causal, qi=qi, ki=ki, bq=bq, bk=bk)
        dq_sc[...] += lax.dot_general(ds.astype(k.dtype), k, _NN, precision=dot_precision(k.dtype), preferred_element_type=jnp.float32)  # dS K

    if causal:
        pl.when(_on_or_below_diag(qi, bq, ki, bk))(_step)
    else:
        _step()

    @pl.when(ki == nk - 1)
    def _finish():
        dq_ref[...] = (dq_sc[...] * scale).astype(dq_ref.dtype)


def _flash_backward(q, k, v, do, lse_rep, delta_rep, *, causal: bool, scale: float, bs: BlockSizes):
    b, h, sq, d = q.shape
    sk = k.shape[2]
    bq, bk = bs.block_q_bwd, bs.block_k_bwd
    nq, nk = sq // bq, sk // bk
    sems = compiler_params(("parallel", "parallel", "parallel", "arbitrary"))

    # ---- dK, dV: grid (b, h, kv blocks, q blocks) ----
    def kv_fixed(bi, hi, ki, qi):
        return (bi, hi, ki, 0)

    def q_sweep(bi, hi, ki, qi):
        if causal:  # blocks above the diagonal come first; point them at the first valid q block
            first_valid = (ki * bk) // bq
            qi = lax.select(_on_or_below_diag(qi, bq, ki, bk), qi, first_valid)
        return (bi, hi, qi, 0)

    dk, dv = pl.pallas_call(
        functools.partial(_bwd_dkv_kernel, scale=scale, causal=causal, bq=bq, bk=bk, nq=nq),
        grid=(b, h, nk, nq),
        in_specs=[
            pl.BlockSpec((None, None, bq, d), q_sweep),
            pl.BlockSpec((None, None, bk, d), kv_fixed),
            pl.BlockSpec((None, None, bk, d), kv_fixed),
            pl.BlockSpec((None, None, bq, d), q_sweep),
            pl.BlockSpec((None, None, bq, NUM_LANES), q_sweep),
            pl.BlockSpec((None, None, bq, NUM_LANES), q_sweep),
        ],
        out_specs=[pl.BlockSpec((None, None, bk, d), kv_fixed)] * 2,
        out_shape=[jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype)],
        scratch_shapes=[pltpu.VMEM((bk, d), jnp.float32), pltpu.VMEM((bk, d), jnp.float32)],
        compiler_params=sems,
        name="ferrite_flash_attention_bwd_dkv",
    )(q, k, v, do, lse_rep, delta_rep)

    # ---- dQ: grid (b, h, q blocks, kv blocks) ----
    def q_fixed(bi, hi, qi, ki):
        return (bi, hi, qi, 0)

    def kv_sweep(bi, hi, qi, ki):
        if causal:
            ki = lax.select(_on_or_below_diag(qi, bq, ki, bk), ki, 0)
        return (bi, hi, ki, 0)

    (dq,) = pl.pallas_call(
        functools.partial(_bwd_dq_kernel, scale=scale, causal=causal, bq=bq, bk=bk, nk=nk),
        grid=(b, h, nq, nk),
        in_specs=[
            pl.BlockSpec((None, None, bq, d), q_fixed),
            pl.BlockSpec((None, None, bk, d), kv_sweep),
            pl.BlockSpec((None, None, bk, d), kv_sweep),
            pl.BlockSpec((None, None, bq, d), q_fixed),
            pl.BlockSpec((None, None, bq, NUM_LANES), q_fixed),
            pl.BlockSpec((None, None, bq, NUM_LANES), q_fixed),
        ],
        out_specs=[pl.BlockSpec((None, None, bq, d), q_fixed)],
        out_shape=[jax.ShapeDtypeStruct(q.shape, q.dtype)],
        scratch_shapes=[pltpu.VMEM((bq, d), jnp.float32)],
        compiler_params=sems,
        name="ferrite_flash_attention_bwd_dq",
    )(q, k, v, do, lse_rep, delta_rep)
    return dq, dk, dv


# --------------------------------------------------------------------------
# custom_vjp glue and public API
# --------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _flash(q, k, v, causal, scale, bs):
    out, _ = _flash_forward(q, k, v, causal=causal, scale=scale, bs=bs)
    return out


def _flash_fwd(q, k, v, causal, scale, bs):
    out, lse_rep = _flash_forward(q, k, v, causal=causal, scale=scale, bs=bs)
    return out, (q, k, v, out, lse_rep)


def _flash_bwd(causal, scale, bs, res, do):
    q, k, v, out, lse_rep = res
    delta = jnp.sum(do.astype(jnp.float32) * out.astype(jnp.float32), axis=-1)  # (b, h, s)
    delta_rep = jnp.broadcast_to(delta[..., None], lse_rep.shape)
    return _flash_backward(q, k, v, do, lse_rep, delta_rep, causal=causal, scale=scale, bs=bs)


_flash.defvjp(_flash_fwd, _flash_bwd)


def _validate(q, k, v, causal, bs):
    if not (q.ndim == 4 and k.shape == v.shape and q.shape[:2] == k.shape[:2] and q.shape[3] == k.shape[3]):
        raise ValueError(f"expected q,k,v as [b,h,s,d] with matching b,h,d; got {q.shape} {k.shape} {v.shape}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v must share a dtype")
    d = q.shape[3]
    if d % NUM_LANES and d > NUM_LANES:
        raise ValueError(f"head_dim must be <= {NUM_LANES} or a multiple of it; got {d}")
    if causal and q.shape[2] != k.shape[2]:
        raise ValueError("causal attention needs equal q and k sequence lengths")
    for name, blk, dim, mult in [
        ("block_q", bs.block_q, q.shape[2], platform.SUBLANES),
        ("block_k", bs.block_k, k.shape[2], NUM_LANES),
        ("block_q_bwd", bs.block_q_bwd, q.shape[2], platform.SUBLANES),
        ("block_k_bwd", bs.block_k_bwd, k.shape[2], NUM_LANES),
    ]:
        if blk % mult or dim % blk:
            raise ValueError(f"{name}={blk} must be a multiple of {mult} and divide the sequence length {dim}")


def flash_attention(q, k, v, *, causal: bool = True, scale: float | None = None, block_sizes: BlockSizes | None = None):
    """Memory-efficient attention ``softmax(q k^T * scale) v`` with exact numerics.

    Differentiable through ``jax.grad`` (custom VJP with Pallas backward kernels).
    Shapes ``[batch, heads, seq, head_dim]``; sequence lengths must be multiples of
    the block sizes (pad upstream if needed).
    """
    bs = block_sizes or BlockSizes()
    _validate(q, k, v, causal, bs)
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return _flash(q, k, v, bool(causal), float(scale), bs)


def flash_attention_reference(q, k, v, *, causal: bool = True, scale: float | None = None):
    """Materialised-scores reference in f32, same contract as :func:`flash_attention`."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    s = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
    if causal:
        sq, sk = q.shape[2], k.shape[2]
        mask = jnp.tril(jnp.ones((sq, sk), dtype=bool), k=sk - sq)
        s = jnp.where(mask, s, MASK_VALUE)
    p = jax.nn.softmax(s, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", p, v.astype(jnp.float32)).astype(q.dtype)


def attention_flops(b: int, h: int, sq: int, sk: int, d: int, *, causal: bool, backward: bool = False) -> int:
    """MXU FLOPs for one forward (and optionally backward) pass, counting skipped blocks as free."""
    frac = 0.5 if causal else 1.0
    fwd = 4 * b * h * sq * sk * d * frac  # QK^T and PV, 2 flops per MAC
    if not backward:
        return int(fwd)
    # backward recomputes S (1 matmul) and does dP, dV, dK, dQ (4 matmuls) = 5 matmuls vs 2 in forward
    return int(fwd * 2.5)
