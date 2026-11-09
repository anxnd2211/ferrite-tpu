"""Try candidate matmul forms inside Pallas TPU kernels and report which ones Mosaic lowers.

Used to pin down a 'failed to legalize tpu.matmul' seen only for bf16 inputs.
"""

from __future__ import annotations

import functools
import traceback

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

NT = (((1,), (1,)), ((), ()))
NN = (((1,), (0,)), ((), ()))
F32 = jnp.float32
BF16 = jnp.bfloat16


def run(name, fn):
    try:
        jax.block_until_ready(fn())
        print(f"PASS  {name}")
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " | ")
        print(f"FAIL  {name}: {type(e).__name__}: {msg[:700]}")


def simple(kernel, *arrays, out_dtype=F32, out_shape=(128, 128)):
    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(out_shape, out_dtype))(*arrays)


q = jax.random.normal(jax.random.key(0), (128, 128), F32).astype(BF16)
k = jax.random.normal(jax.random.key(1), (128, 128), F32).astype(BF16)
v = jax.random.normal(jax.random.key(2), (128, 128), F32).astype(BF16)


def k_nt(q_ref, k_ref, o_ref):
    o_ref[...] = lax.dot_general(q_ref[...], k_ref[...], NT, preferred_element_type=F32)


def k_nt_prec_none(q_ref, k_ref, o_ref):
    o_ref[...] = lax.dot_general(q_ref[...], k_ref[...], NT, precision=None, preferred_element_type=F32)


def k_nt_prec_default(q_ref, k_ref, o_ref):
    o_ref[...] = lax.dot_general(q_ref[...], k_ref[...], NT, precision=lax.Precision.DEFAULT, preferred_element_type=F32)


def k_transpose_nn(q_ref, k_ref, o_ref):
    o_ref[...] = jnp.dot(q_ref[...], k_ref[...].T, preferred_element_type=F32)


def k_nn(q_ref, k_ref, o_ref):
    o_ref[...] = jnp.dot(q_ref[...], k_ref[...], preferred_element_type=F32)


def k_pv(q_ref, k_ref, v_ref, o_ref):
    s = lax.dot_general(q_ref[...], k_ref[...], NT, preferred_element_type=F32)
    p = jnp.exp(s - jnp.max(s, axis=1, keepdims=True))
    o_ref[...] = jnp.dot(p.astype(BF16), v_ref[...], preferred_element_type=F32)


def k_pv_f32(q_ref, k_ref, v_ref, o_ref):
    s = lax.dot_general(q_ref[...], k_ref[...], NT, preferred_element_type=F32)
    p = jnp.exp(s - jnp.max(s, axis=1, keepdims=True))
    o_ref[...] = jnp.dot(p, v_ref[...].astype(F32), preferred_element_type=F32)


def k_pv_prec_none(q_ref, k_ref, v_ref, o_ref):
    s = lax.dot_general(q_ref[...], k_ref[...], NT, preferred_element_type=F32)
    p = jnp.exp(s - jnp.max(s, axis=1, keepdims=True))
    o_ref[...] = jnp.dot(p.astype(BF16), v_ref[...], precision=None, preferred_element_type=F32)


def k_f32_out_bf16(q_ref, k_ref, o_ref):
    o_ref[...] = lax.dot_general(q_ref[...], k_ref[...], NT, preferred_element_type=F32).astype(BF16)


print(jax.__version__, jax.devices())
run("NT bf16 -> f32", lambda: simple(k_nt, q, k))
run("NT bf16 -> f32, precision=None explicit", lambda: simple(k_nt_prec_none, q, k))
run("NT bf16 -> f32, precision=DEFAULT explicit", lambda: simple(k_nt_prec_default, q, k))
run("k.T then NN bf16 -> f32", lambda: simple(k_transpose_nn, q, k))
run("NN bf16 -> f32", lambda: simple(k_nn, q, k))
run("NT + exp + (p bf16) @ v", lambda: simple(k_pv, q, k, v))
run("NT + exp + (p f32) @ v f32", lambda: simple(k_pv_f32, q, k, v))
run("NT + exp + (p bf16) @ v, precision=None explicit", lambda: simple(k_pv_prec_none, q, k, v))
run("NT bf16 -> f32 -> bf16 store", lambda: simple(k_f32_out_bf16, q, k, out_dtype=BF16))

from ferrite.kernels.flash_attention import BlockSizes, flash_attention  # noqa: E402

qq = jax.random.normal(jax.random.key(0), (1, 1, 256, 128), F32)
run("ferrite fwd f32 jit", lambda: jax.jit(lambda a: flash_attention(a, a, a, causal=True))(qq))
run("ferrite fwd bf16 jit", lambda: jax.jit(lambda a: flash_attention(a, a, a, causal=True))(qq.astype(BF16)))
run("ferrite fwd bf16 eager", lambda: flash_attention(qq.astype(BF16), qq.astype(BF16), qq.astype(BF16), causal=True))
run("ferrite fwd bf16 non-causal jit", lambda: jax.jit(lambda a: flash_attention(a, a, a, causal=False))(qq.astype(BF16)))
run("ferrite fwd+bwd f32 jit", lambda: jax.jit(jax.grad(lambda a: jnp.sum(flash_attention(a, a, a, causal=True))))(qq))

try:
    from jax.experimental.pallas.ops.tpu import flash_attention as fa

    bs = fa.BlockSizes(block_q=128, block_k_major=128, block_k=128, block_b=1, block_q_major_dkv=128, block_k_major_dkv=128, block_k_dkv=128, block_q_dkv=128, block_k_major_dq=128, block_k_dq=128, block_q_dq=128)
    run("official fwd bf16", lambda: jax.jit(lambda a: fa.flash_attention(a, a, a, causal=True, sm_scale=0.088, block_sizes=bs))(qq.astype(BF16)))
except Exception as e:  # noqa: BLE001
    print("official import failed:", e)
