"""A small pre-norm, decoder-only transformer in plain JAX with pluggable kernels.

Each of attention, normalisation and the linear layers can be routed through
the Pallas kernels in :mod:`ferrite.kernels` or through XLA, so the same model
gives an apples-to-apples comparison. Parameters live in f32; activations and
matmuls use ``cfg.dtype`` (bf16 by default); every reduction that matters
(softmax statistics, RMS statistics, loss) is in f32.

The residual stream is handled with the fused add+RMSNorm kernel: the add that
closes one sub-layer is fused into the norm that opens the next, so a layer
reads and writes the residual stream once per sub-layer instead of twice.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp

from ferrite import platform
from ferrite.kernels.flash_attention import BlockSizes, flash_attention
from ferrite.kernels.matmul import matmul
from ferrite.kernels.rmsnorm import rmsnorm, rmsnorm_reference


@dataclasses.dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 256  # byte-level tokens; 256 is also lane aligned
    seq_len: int = 256
    d_model: int = 512
    n_head: int = 4
    n_layer: int = 4
    d_ff: int = 2048
    attention: str = "pallas"  # "pallas" | "xla"
    norm: str = "pallas"  # "pallas" | "xla"
    linear: str = "xla"  # "xla" | "pallas"
    dtype: Any = jnp.bfloat16
    eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head

    def replace(self, **kw) -> "GPTConfig":
        return dataclasses.replace(self, **kw)

    def attention_blocks(self, seq_len: int | None = None) -> BlockSizes | None:
        """Tile sizes for a sequence of ``seq_len``, or None when the Pallas kernel cannot tile it.

        The key block must be a multiple of 128 lanes, so sequences shorter than
        128 (or not a multiple of 128) fall back to XLA attention.
        """
        s = seq_len or self.seq_len
        if s % 128:
            return None
        t = min(platform.default_tile(), s)
        while s % t:
            t -= 128
        return BlockSizes(t, t, t, t)


def init_params(key: jax.Array, cfg: GPTConfig) -> dict:
    k_emb, k_pos, *k_layers = jax.random.split(key, 2 + cfg.n_layer)
    std = 0.02
    out_std = std / math.sqrt(2 * cfg.n_layer)  # GPT-2 style scaling of residual projections
    d, f = cfg.d_model, cfg.d_ff

    def layer(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        return {
            "ln1": jnp.ones((d,), jnp.float32),
            "wqkv": std * jax.random.normal(k1, (d, 3 * d), jnp.float32),
            "wo": out_std * jax.random.normal(k2, (d, d), jnp.float32),
            "ln2": jnp.ones((d,), jnp.float32),
            "w1": std * jax.random.normal(k3, (d, f), jnp.float32),
            "w2": out_std * jax.random.normal(k4, (f, d), jnp.float32),
        }

    return {
        "tok_emb": std * jax.random.normal(k_emb, (cfg.vocab_size, d), jnp.float32),
        "pos_emb": std * jax.random.normal(k_pos, (cfg.seq_len, d), jnp.float32),
        "layers": [layer(k) for k in k_layers],
        "ln_f": jnp.ones((d,), jnp.float32),
    }


def param_count(params) -> int:
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))


def _linear(x, w, cfg: GPTConfig):
    """[..., in] @ [in, out] in cfg.dtype with f32 accumulation."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1]).astype(cfg.dtype)
    w = w.astype(cfg.dtype)
    if cfg.linear == "pallas":
        y = matmul(x2, w)
    else:
        y = jnp.dot(x2, w, preferred_element_type=jnp.float32).astype(cfg.dtype)
    return y.reshape(*lead, w.shape[1])


def _norm(x, w, residual, cfg: GPTConfig):
    if cfg.norm == "pallas":
        return rmsnorm(x, w, residual=residual, eps=cfg.eps)
    return rmsnorm_reference(x, w, residual=residual, eps=cfg.eps)


def _attention(q, k, v, cfg: GPTConfig):
    """q, k, v: [B, H, S, hd] -> [B, H, S, hd], causal."""
    blocks = cfg.attention_blocks(q.shape[2]) if cfg.attention == "pallas" else None
    if blocks is not None:
        return flash_attention(q, k, v, causal=True, block_sizes=blocks)
    # XLA baseline: jax.nn.dot_product_attention wants [B, S, H, hd].
    t = lambda a: jnp.swapaxes(a, 1, 2)
    out = jax.nn.dot_product_attention(t(q), t(k), t(v), is_causal=True, implementation="xla")
    return t(out)


def forward(params, tokens, cfg: GPTConfig):
    """tokens [B, S] int32 -> logits [B, S, vocab] f32."""
    b, s = tokens.shape
    if s > cfg.seq_len:
        raise ValueError(f"sequence {s} longer than cfg.seq_len={cfg.seq_len}")
    h = (params["tok_emb"][tokens] + params["pos_emb"][:s]).astype(cfg.dtype)  # residual stream
    pending = None  # sub-layer output whose residual add is fused into the next norm
    for layer in params["layers"]:
        if pending is None:
            y = _norm(h, layer["ln1"], None, cfg)
        else:
            h, y = _norm(pending, layer["ln1"], h, cfg)
        qkv = _linear(y, layer["wqkv"], cfg).reshape(b, s, 3, cfg.n_head, cfg.head_dim)
        q, k, v = (jnp.transpose(qkv[:, :, i], (0, 2, 1, 3)) for i in range(3))  # [B, H, S, hd]
        a = jnp.transpose(_attention(q, k, v, cfg), (0, 2, 1, 3)).reshape(b, s, cfg.d_model)
        a = _linear(a, layer["wo"], cfg)
        h, y2 = _norm(a, layer["ln2"], h, cfg)
        pending = _linear(jax.nn.gelu(_linear(y2, layer["w1"], cfg), approximate=True), layer["w2"], cfg)
    _, y = _norm(pending, params["ln_f"], h, cfg)
    logits = jnp.dot(y.reshape(-1, cfg.d_model), params["tok_emb"].T.astype(cfg.dtype), preferred_element_type=jnp.float32)
    return logits.reshape(b, s, cfg.vocab_size)


def loss_fn(params, tokens, targets, cfg: GPTConfig):
    """Mean next-token cross-entropy in f32."""
    logits = forward(params, tokens, cfg)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    return jnp.mean(nll)


def flops_per_token(cfg: GPTConfig, n_params_nonembed: int) -> dict:
    """Analytic FLOPs per token (forward; training = 3x) split into matmul and attention parts.

    Matmul: 2 FLOPs per parameter in the dense layers plus the tied LM head.
    Attention: QK^T and PV are each 2*S*d per head per token; causal skipping halves it.
    """
    dense = 2 * (n_params_nonembed + cfg.vocab_size * cfg.d_model)
    attn = 4 * cfg.seq_len * cfg.d_model * cfg.n_layer * 0.5
    return {"dense": dense, "attention": attn, "total": dense + attn}
