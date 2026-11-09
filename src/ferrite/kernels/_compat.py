"""Small compatibility layer across recent JAX releases."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas import tpu as pltpu

# jax < 0.6 called this TPUCompilerParams.
CompilerParams = getattr(pltpu, "CompilerParams", None) or getattr(pltpu, "TPUCompilerParams")


def compiler_params(dimension_semantics, **kw):
    return CompilerParams(dimension_semantics=tuple(dimension_semantics), **kw)


def dot_precision(dtype):
    """f32 operands get a true fp32 MXU contraction; bf16 operands use the single-pass default.

    Without this, an f32 matmul on TPU is one bf16 pass (DEFAULT precision), which is
    fine for training but makes f32 reference tests meaningless.
    """
    return lax.Precision.HIGHEST if jnp.dtype(dtype) == jnp.float32 else None


def round_up(x: int, m: int) -> int:
    return -(-x // m) * m
