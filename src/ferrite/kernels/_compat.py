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
    """f32 operands get a true fp32 MXU contraction; everything else the single-pass default.

    Always explicit, never ``None``: inside a kernel ``precision=None`` inherits the
    process-wide ``jax_default_matmul_precision``, and a caller who set that to
    "highest" (as the test-suite does for its f32 references) would make Mosaic try an
    fp32 contraction on bf16 operands, which it cannot legalize.
    """
    return lax.Precision.HIGHEST if jnp.dtype(dtype) == jnp.float32 else lax.Precision.DEFAULT


def round_up(x: int, m: int) -> int:
    return -(-x // m) * m
