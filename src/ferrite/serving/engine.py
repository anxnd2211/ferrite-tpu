"""Ahead-of-time compiled greedy decoding with a static KV cache.

Serving on TPU is a fight against recompilation and dynamic shapes:

* Prompts are right-padded to one of a few **buckets**; each bucket's prefill is
  compiled once, ahead of time, with ``jit(...).lower(...).compile()``. The
  engine then calls the compiled executables directly, so no shape seen at
  serving time can trigger a compile.
* The KV cache is a fixed ``[layers, batch, heads, max_len, head_dim]`` buffer
  updated in place (``donate_argnums``), and the decode step is one compiled
  function regardless of position.
* Prefill is compute-bound and runs the Pallas flash-attention kernel; decode
  is memory-bound (one query row against the whole cache) and stays in XLA,
  where fusion already does the right thing. That split is deliberate and the
  benchmark measures both halves (TTFT vs inter-token latency).
* The decode step can be serialised with ``jax.export`` to StableHLO and loaded
  back without the Python model code, which is how a kernel-free serving binary
  would consume it.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ferrite.model.gpt import GPTConfig, _attention, _linear, _norm


@dataclasses.dataclass(frozen=True)
class ServeConfig:
    batch: int = 1
    max_len: int = 512
    buckets: tuple[int, ...] = (64, 128, 256)
    prefill_kernels: str = "pallas"  # "pallas" | "xla" for attention + norms in prefill


def bucket_for(length: int, buckets: tuple[int, ...]) -> int:
    for b in buckets:
        if length <= b:
            return b
    raise ValueError(f"prompt length {length} exceeds largest bucket {buckets[-1]}")


# ---------------------------------------------------------------- model pieces


def _block_prefill(layer, h, pending, cfg: GPTConfig):
    """One transformer block over a full sequence, returning (h, pending, k, v)."""
    b, s, _ = h.shape
    if pending is None:
        y = _norm(h, layer["ln1"], None, cfg)
    else:
        h, y = _norm(pending, layer["ln1"], h, cfg)
    qkv = _linear(y, layer["wqkv"], cfg).reshape(b, s, 3, cfg.n_head, cfg.head_dim)
    q, k, v = (jnp.transpose(qkv[:, :, i], (0, 2, 1, 3)) for i in range(3))
    a = jnp.transpose(_attention(q, k, v, cfg), (0, 2, 1, 3)).reshape(b, s, cfg.d_model)
    a = _linear(a, layer["wo"], cfg)
    h, y2 = _norm(a, layer["ln2"], h, cfg)
    pending = _linear(jax.nn.gelu(_linear(y2, layer["w1"], cfg), approximate=True), layer["w2"], cfg)
    return h, pending, k, v


def prefill(params, tokens, lengths, cfg: GPTConfig, max_len: int):
    """tokens [B, L] (right-padded), lengths [B] -> (next-token logits [B, V], k_cache, v_cache).

    Caches are [layers, B, H, max_len, hd]; positions >= length hold padding
    garbage that decode never attends to and overwrites in order.
    """
    b, s = tokens.shape
    h = (params["tok_emb"][tokens] + params["pos_emb"][:s]).astype(cfg.dtype)
    pending, ks, vs = None, [], []
    for layer in params["layers"]:
        h, pending, k, v = _block_prefill(layer, h, pending, cfg)
        pad = ((0, 0), (0, 0), (0, max_len - s), (0, 0))
        ks.append(jnp.pad(k, pad))
        vs.append(jnp.pad(v, pad))
    _, y = _norm(pending, params["ln_f"], h, cfg)
    last = jnp.take_along_axis(y, (lengths - 1)[:, None, None], axis=1)[:, 0]  # [B, D]
    logits = jnp.dot(last, params["tok_emb"].T.astype(cfg.dtype), preferred_element_type=jnp.float32)
    return logits, jnp.stack(ks), jnp.stack(vs)


def _cache_attention(q, k_cache, v_cache, pos, scale):
    """q [B, H, hd]; caches [B, H, max_len, hd]; pos [B] = index of the current token."""
    s = jnp.einsum("bhd,bhkd->bhk", q.astype(jnp.float32), k_cache.astype(jnp.float32)) * scale
    idx = jnp.arange(k_cache.shape[2])[None, :]
    s = jnp.where(idx <= pos[:, None], s[:, :, :], -1e30)[..., None, :] if False else jnp.where((idx <= pos[:, None])[:, None, :], s, -1e30)
    p = jax.nn.softmax(s, axis=-1)
    return jnp.einsum("bhk,bhkd->bhd", p, v_cache.astype(jnp.float32)).astype(q.dtype)


def _write_cache(cache, new, pos):
    """cache [B, H, max_len, hd], new [B, H, hd], pos [B] -> cache with new written at pos."""

    def one(c, n, p):  # c [H, max_len, hd]
        return lax.dynamic_update_slice(c, n[:, None, :].astype(c.dtype), (0, p, 0))

    return jax.vmap(one)(cache, new, pos)


def decode_step(params, token, pos, k_cache, v_cache, cfg: GPTConfig):
    """One greedy decode step. token [B] int32 at position pos [B] -> (logits [B, V], caches)."""
    b = token.shape[0]
    h = (params["tok_emb"][token] + params["pos_emb"][pos]).astype(cfg.dtype)  # [B, D]
    scale = 1.0 / np.sqrt(cfg.head_dim)
    pending = None
    new_k, new_v = [], []
    for i, layer in enumerate(params["layers"]):
        if pending is None:
            y = _norm(h, layer["ln1"], None, cfg)
        else:
            h, y = _norm(pending, layer["ln1"], h, cfg)
        qkv = _linear(y, layer["wqkv"], cfg).reshape(b, 3, cfg.n_head, cfg.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [B, H, hd]
        kc = _write_cache(k_cache[i], k, pos)
        vc = _write_cache(v_cache[i], v, pos)
        new_k.append(kc)
        new_v.append(vc)
        a = _cache_attention(q, kc, vc, pos, scale).reshape(b, cfg.d_model)
        a = _linear(a, layer["wo"], cfg)
        h, y2 = _norm(a, layer["ln2"], h, cfg)
        pending = _linear(jax.nn.gelu(_linear(y2, layer["w1"], cfg), approximate=True), layer["w2"], cfg)
    _, y = _norm(pending, params["ln_f"], h, cfg)
    logits = jnp.dot(y, params["tok_emb"].T.astype(cfg.dtype), preferred_element_type=jnp.float32)
    return logits, jnp.stack(new_k), jnp.stack(new_v)


# ---------------------------------------------------------------- engine


class Engine:
    """Precompiles prefill per bucket and one decode step; generates greedily."""

    def __init__(self, params, model_cfg: GPTConfig, serve_cfg: ServeConfig):
        if max(serve_cfg.buckets) > model_cfg.seq_len or serve_cfg.max_len > model_cfg.seq_len:
            raise ValueError("buckets and max_len must fit within the model's trained seq_len (positional table)")
        self.params = params
        self.cfg = model_cfg
        self.serve = serve_cfg
        self.prefill_cfg = model_cfg if serve_cfg.prefill_kernels == "pallas" else model_cfg.replace(attention="xla", norm="xla")
        self.decode_cfg = model_cfg.replace(attention="xla", norm="xla", linear="xla")
        self.compile_seconds: dict[str, float] = {}
        self._prefill: dict[int, Any] = {}
        b = serve_cfg.batch
        i32 = jnp.int32
        for bucket in serve_cfg.buckets:
            t0 = time.perf_counter()
            fn = jax.jit(lambda p, t, l, _c=self.prefill_cfg: prefill(p, t, l, _c, serve_cfg.max_len))
            self._prefill[bucket] = fn.lower(params, jax.ShapeDtypeStruct((b, bucket), i32), jax.ShapeDtypeStruct((b,), i32)).compile()
            self.compile_seconds[f"prefill_{bucket}"] = time.perf_counter() - t0
        cache_shape = jax.ShapeDtypeStruct((model_cfg.n_layer, b, model_cfg.n_head, serve_cfg.max_len, model_cfg.head_dim), model_cfg.dtype)
        t0 = time.perf_counter()
        self._decode_jit = jax.jit(lambda p, t, pos, kc, vc, _c=self.decode_cfg: decode_step(p, t, pos, kc, vc, _c), donate_argnums=(3, 4))
        self._decode = self._decode_jit.lower(params, jax.ShapeDtypeStruct((b,), i32), jax.ShapeDtypeStruct((b,), i32), cache_shape, cache_shape).compile()
        self.compile_seconds["decode"] = time.perf_counter() - t0
        self.n_compiles = len(self.compile_seconds)

    # -- public API ---------------------------------------------------------

    def prefill(self, prompts: list[list[int]]):
        """Returns (first token [B], pos [B], caches). Prompts are right-padded to a bucket."""
        b = self.serve.batch
        if len(prompts) != b:
            raise ValueError(f"engine compiled for batch {b}, got {len(prompts)} prompts")
        lengths = np.array([len(p) for p in prompts], np.int32)
        bucket = bucket_for(int(lengths.max()), self.serve.buckets)
        tokens = np.zeros((b, bucket), np.int32)
        for i, p in enumerate(prompts):
            tokens[i, : len(p)] = p
        logits, kc, vc = self._prefill[bucket](self.params, jnp.asarray(tokens), jnp.asarray(lengths))
        return jnp.argmax(logits, axis=-1).astype(jnp.int32), jnp.asarray(lengths), kc, vc

    def decode(self, token, pos, kc, vc):
        logits, kc, vc = self._decode(self.params, token, pos, kc, vc)
        return jnp.argmax(logits, axis=-1).astype(jnp.int32), kc, vc

    def generate(self, prompts: list[list[int]], max_new_tokens: int) -> tuple[np.ndarray, dict]:
        """Greedy generation; returns generated tokens [B, max_new_tokens] and timing stats."""
        if max(len(p) for p in prompts) + max_new_tokens > self.serve.max_len:
            raise ValueError("prompt + new tokens exceed max_len")
        t0 = time.perf_counter()
        token, pos, kc, vc = self.prefill(prompts)
        token = jax.block_until_ready(token)
        ttft = time.perf_counter() - t0
        out = [np.asarray(token)]
        step_times = []
        for _ in range(max_new_tokens - 1):
            t = time.perf_counter()
            token, kc, vc = self.decode(token, pos, kc, vc)
            token = jax.block_until_ready(token)
            step_times.append(time.perf_counter() - t)
            pos = pos + 1
            out.append(np.asarray(token))
        total = time.perf_counter() - t0
        stats = {
            "ttft_ms": ttft * 1e3,
            "itl_ms_median": float(np.median(step_times)) * 1e3 if step_times else float("nan"),
            "itl_ms_p99": float(np.percentile(step_times, 99)) * 1e3 if step_times else float("nan"),
            "tokens_per_second": len(prompts) * max_new_tokens / total,
            "total_s": total,
            "n_compiles": self.n_compiles,
        }
        return np.stack(out, axis=1), stats

    # -- export -------------------------------------------------------------

    def export_decode(self) -> bytes:
        """Serialise the decode step (StableHLO + shardings) with jax.export."""
        from jax import export

        b = self.serve.batch
        i32 = jnp.int32
        cache_shape = jax.ShapeDtypeStruct((self.cfg.n_layer, b, self.cfg.n_head, self.serve.max_len, self.cfg.head_dim), self.cfg.dtype)
        params_abs = jax.tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), self.params)
        exp = export.export(self._decode_jit)(params_abs, jax.ShapeDtypeStruct((b,), i32), jax.ShapeDtypeStruct((b,), i32), cache_shape, cache_shape)
        return exp.serialize()

    @staticmethod
    def load_exported(blob: bytes):
        from jax import export

        return export.deserialize(blob)


def reference_greedy(params, cfg: GPTConfig, prompt: list[int], max_new_tokens: int) -> np.ndarray:
    """Slow but obviously-correct greedy decoding by re-running the full model each step."""
    from ferrite.model.gpt import forward

    toks = list(prompt)
    out = []
    for _ in range(max_new_tokens):
        logits = forward(params, jnp.asarray([toks], jnp.int32), cfg)
        nxt = int(jnp.argmax(logits[0, -1]))
        out.append(nxt)
        toks.append(nxt)
    return np.array(out)
