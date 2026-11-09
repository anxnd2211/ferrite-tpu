"""Fused residual-add + RMSNorm as a Pallas TPU kernel, with a custom VJP.

Transformer blocks do ``h = x + residual; y = rmsnorm(h) * w`` twice per layer.
The kernel reads x and residual once, writes both h (the new residual stream)
and y, and keeps the statistics in f32 regardless of the storage dtype. It is
memory-bound (arithmetic intensity well under 1 FLOP/byte), so the roofline
target is HBM bandwidth, not the MXU.

The backward pass is left to XLA on purpose: it is a handful of element-wise
ops and reductions that XLA fuses well, and a hand-written kernel would not
move the step time. The custom_vjp exists so ``jax.grad`` can see through the
``pallas_call`` in the forward pass.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ferrite import platform
from ferrite.kernels._compat import compiler_params, round_up

_LANES = platform.LANES


def _lane_tile(x: jax.Array, n: int) -> jax.Array:
    """Broadcast a (rows, 128) lane-replicated statistic to (rows, n)."""
    reps, rem = divmod(n, _LANES)
    if rem:
        raise ValueError(f"feature dim {n} must be a multiple of {_LANES}")
    return x if reps == 1 else jnp.tile(x, (1, reps))


def _inv_rms(h: jax.Array, eps: float, true_dim: int) -> jax.Array:
    """1/sqrt(mean(h^2) + eps) as a (rows, 128) lane-replicated f32 tile."""
    ss = jnp.sum(h * h, axis=-1, keepdims=True)  # (rows, 1)
    inv = lax.rsqrt(ss * (1.0 / true_dim) + eps)
    return jnp.broadcast_to(inv, (h.shape[0], _LANES))


def _rmsnorm_kernel(x_ref, w_ref, o_ref, *, eps: float, true_dim: int):
    h = x_ref[...].astype(jnp.float32)
    inv = _lane_tile(_inv_rms(h, eps, true_dim), h.shape[1])
    o_ref[...] = (h * inv * w_ref[...].astype(jnp.float32)).astype(o_ref.dtype)


def _add_rmsnorm_kernel(x_ref, r_ref, w_ref, h_ref, o_ref, *, eps: float, true_dim: int):
    h = x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)
    h_ref[...] = h.astype(h_ref.dtype)
    inv = _lane_tile(_inv_rms(h, eps, true_dim), h.shape[1])
    o_ref[...] = (h * inv * w_ref[...].astype(jnp.float32)).astype(o_ref.dtype)


def _rows_block(n_rows: int, block_rows: int | None) -> int:
    if block_rows is None:
        block_rows = 256 if n_rows >= 256 else round_up(n_rows, platform.SUBLANES)
    if block_rows % platform.SUBLANES:
        raise ValueError(f"block_rows must be a multiple of {platform.SUBLANES}")
    return block_rows


def _call(kernel, arrays, n_out, *, rows, dim, block_rows, out_dtype, eps):
    """Shared pallas_call plumbing: pad rows/features, run, slice back."""
    true_dim = dim
    bm = _rows_block(rows, block_rows)
    rows_p, dim_p = round_up(rows, bm), round_up(dim, _LANES)
    padded = []
    for a in arrays:
        if a.ndim == 1:  # weight -> (1, dim_p)
            a = jnp.pad(a, (0, dim_p - dim))[None, :]
        elif (rows_p, dim_p) != (rows, dim):
            a = jnp.pad(a, ((0, rows_p - rows), (0, dim_p - dim)))
        padded.append(a)
    row_spec = pl.BlockSpec((bm, dim_p), lambda i: (i, 0))
    w_spec = pl.BlockSpec((1, dim_p), lambda i: (0, 0))
    in_specs = [w_spec if a.shape[0] == 1 else row_spec for a in padded]
    itemsize = jnp.dtype(out_dtype).itemsize
    outs = pl.pallas_call(
        functools.partial(kernel, eps=eps, true_dim=true_dim),
        grid=(rows_p // bm,),
        in_specs=in_specs,
        out_specs=[row_spec] * n_out,
        out_shape=[jax.ShapeDtypeStruct((rows_p, dim_p), out_dtype)] * n_out,
        compiler_params=compiler_params(("parallel",)),
        cost_estimate=pl.CostEstimate(
            flops=rows_p * dim_p * (3 + 2 * (len(arrays) - 1)),
            transcendentals=rows_p,
            bytes_accessed=sum(a.size * jnp.dtype(a.dtype).itemsize for a in padded) + n_out * rows_p * dim_p * itemsize,
        ),
        name="ferrite_rmsnorm" if n_out == 1 else "ferrite_add_rmsnorm",
    )(*padded)
    if (rows_p, dim_p) != (rows, dim):
        outs = [o[:rows, :dim] for o in outs]
    return outs


# --------------------------------------------------------------------------
# custom_vjp wrappers (forward = Pallas kernel, backward = XLA)
# --------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _norm(x2d, w, eps, block_rows):
    (y,) = _call(_rmsnorm_kernel, [x2d, w], 1, rows=x2d.shape[0], dim=x2d.shape[1], block_rows=block_rows, out_dtype=x2d.dtype, eps=eps)
    return y


def _norm_fwd(x2d, w, eps, block_rows):
    return _norm(x2d, w, eps, block_rows), (x2d, w)


def _norm_bwd(eps, block_rows, res, dy):
    x2d, w = res
    dx, dw = _rmsnorm_bwd_math(x2d.astype(jnp.float32), w.astype(jnp.float32), dy.astype(jnp.float32), eps)
    return dx.astype(x2d.dtype), dw.astype(w.dtype)


_norm.defvjp(_norm_fwd, _norm_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _add_norm(x2d, r2d, w, eps, block_rows):
    h, y = _call(_add_rmsnorm_kernel, [x2d, r2d, w], 2, rows=x2d.shape[0], dim=x2d.shape[1], block_rows=block_rows, out_dtype=x2d.dtype, eps=eps)
    return h, y


def _add_norm_fwd(x2d, r2d, w, eps, block_rows):
    h, y = _add_norm(x2d, r2d, w, eps, block_rows)
    return (h, y), (h, w)


def _add_norm_bwd(eps, block_rows, res, cts):
    h, w = res
    dh_direct, dy = cts
    dh, dw = _rmsnorm_bwd_math(h.astype(jnp.float32), w.astype(jnp.float32), dy.astype(jnp.float32), eps)
    dh = dh + dh_direct.astype(jnp.float32)
    return dh.astype(h.dtype), dh.astype(h.dtype), dw.astype(w.dtype)


_add_norm.defvjp(_add_norm_fwd, _add_norm_bwd)


def _rmsnorm_bwd_math(h, w, dy, eps):
    """Analytic RMSNorm backward in f32. y = h * inv * w, inv = rsqrt(mean(h^2) + eps)."""
    inv = lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + eps)
    xhat = h * inv
    dw = jnp.sum(dy * xhat, axis=0)
    dxhat = dy * w
    dh = inv * (dxhat - xhat * jnp.mean(dxhat * xhat, axis=-1, keepdims=True))
    return dh, dw


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def rmsnorm(x, weight, *, residual=None, eps: float = 1e-6, block_rows: int | None = None):
    """RMSNorm over the last axis of ``x``.

    With ``residual`` given, returns ``(h, y)`` where ``h = x + residual`` is the
    new residual stream and ``y = rmsnorm(h) * weight``; otherwise returns ``y``.
    """
    dim = x.shape[-1]
    if weight.shape != (dim,):
        raise ValueError(f"weight must have shape ({dim},), got {weight.shape}")
    lead = x.shape[:-1]
    x2d = x.reshape(-1, dim)
    if residual is None:
        return _norm(x2d, weight, float(eps), block_rows).reshape(*lead, dim)
    if residual.shape != x.shape:
        raise ValueError("residual must match x")
    h, y = _add_norm(x2d, residual.reshape(-1, dim), weight, float(eps), block_rows)
    return h.reshape(*lead, dim), y.reshape(*lead, dim)


def rmsnorm_reference(x, weight, *, residual=None, eps: float = 1e-6):
    """Plain-JAX reference with identical f32 numerics."""
    h = x.astype(jnp.float32) + (0 if residual is None else residual.astype(jnp.float32))
    y = h * lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + eps) * weight.astype(jnp.float32)
    if residual is None:
        return y.astype(x.dtype)
    return h.astype(x.dtype), y.astype(x.dtype)
