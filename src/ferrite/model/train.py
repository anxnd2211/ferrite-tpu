"""Data-parallel training over all local TPU cores with hand-written AdamW.

Sharding uses ``jax.sharding`` directly: parameters and optimiser state are
replicated, the batch is sharded along a 1-D ``data`` mesh axis, and ``jax.jit``
inserts the gradient all-reduce. Step time is measured with
``block_until_ready`` after a warm-up, and model FLOPs utilisation (MFU) is
reported against the chip peaks from :mod:`ferrite.platform`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ferrite import platform
from ferrite.model import data as data_lib
from ferrite.model.gpt import GPTConfig, flops_per_token, init_params, loss_fn, param_count


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    steps: int = 200
    batch: int = 64
    lr: float = 6e-4
    warmup: int = 20
    weight_decay: float = 0.1
    clip: float = 1.0
    seed: int = 0
    log_every: int = 10
    data: str = "tinyshakespeare"
    out: str | None = None
    save_params: str | None = None


# ---------------------------------------------------------------- optimiser


def adamw_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": jax.tree_util.tree_map(jnp.zeros_like, params), "step": jnp.zeros((), jnp.int32)}


def lr_at(step, cfg: TrainConfig):
    warm = cfg.lr * (step + 1) / max(1, cfg.warmup)
    progress = jnp.clip((step - cfg.warmup) / max(1, cfg.steps - cfg.warmup), 0.0, 1.0)
    cosine = 0.1 * cfg.lr + 0.9 * cfg.lr * 0.5 * (1 + jnp.cos(jnp.pi * progress))
    return jnp.where(step < cfg.warmup, warm, cosine)


def adamw_update(params, grads, state, cfg: TrainConfig, b1=0.9, b2=0.95, eps=1e-8):
    step = state["step"] + 1
    lr = lr_at(step, cfg)
    gnorm = jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in jax.tree_util.tree_leaves(grads)))
    scale = jnp.minimum(1.0, cfg.clip / (gnorm + 1e-6))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
    bc1, bc2 = 1 - b1**step, 1 - b2**step

    def upd(p, m, v):
        decay = cfg.weight_decay if p.ndim >= 2 else 0.0  # no decay on norm gains
        return p - lr * ((m / bc1) / (jnp.sqrt(v / bc2) + eps) + decay * p)

    new_params = jax.tree_util.tree_map(upd, params, m, v)
    return new_params, {"m": m, "v": v, "step": step}, gnorm


# ---------------------------------------------------------------- training


def make_mesh() -> Mesh:
    return Mesh(np.array(jax.devices()), ("data",))


def build_train_step(model_cfg: GPTConfig, train_cfg: TrainConfig, mesh: Mesh):
    replicated = NamedSharding(mesh, P())
    sharded = NamedSharding(mesh, P("data"))

    def step(params, opt, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y, model_cfg)
        params, opt, gnorm = adamw_update(params, grads, opt, train_cfg)
        return params, opt, loss, gnorm

    jitted = jax.jit(
        step,
        in_shardings=(replicated, replicated, sharded, sharded),
        out_shardings=(replicated, replicated, replicated, replicated),
        donate_argnums=(0, 1),
    )
    return jitted, replicated, sharded


def train(model_cfg: GPTConfig, train_cfg: TrainConfig, *, verbose: bool = True) -> dict[str, Any]:
    if train_cfg.batch % jax.device_count():
        raise ValueError(f"batch {train_cfg.batch} must be divisible by {jax.device_count()} devices")
    mesh = make_mesh()
    corpus = data_lib.load(train_cfg.data)
    train_data, _ = data_lib.split(corpus)
    rng = np.random.default_rng(train_cfg.seed)

    params = init_params(jax.random.key(train_cfg.seed), model_cfg)
    n_params = param_count(params)
    n_nonembed = n_params - int(params["tok_emb"].size) - int(params["pos_emb"].size)
    opt = adamw_init(params)
    step_fn, replicated, sharded = build_train_step(model_cfg, train_cfg, mesh)
    params = jax.device_put(params, replicated)
    opt = jax.device_put(opt, replicated)

    def batch():
        x, y = data_lib.get_batch(rng, train_data, train_cfg.batch, model_cfg.seq_len)
        return jax.device_put(x, sharded), jax.device_put(y, sharded)

    # compile (and time the compile) on the first batch
    x, y = batch()
    t0 = time.perf_counter()
    compiled = step_fn.lower(params, opt, x, y).compile()
    compile_s = time.perf_counter() - t0
    cost = compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, (list, tuple)) else cost
    xla_flops = float(cost.get("flops", float("nan"))) if cost else float("nan")

    tokens_per_step = train_cfg.batch * model_cfg.seq_len
    fpt = flops_per_token(model_cfg, n_nonembed)
    train_flops_per_step = 3 * fpt["total"] * tokens_per_step  # fwd + bwd (2x fwd)
    peaks = platform.peaks()
    peak_flops = peaks.bf16_flops * jax.device_count() if peaks else None

    losses, step_times, gnorms = [], [], []
    warm = min(5, train_cfg.steps // 4)
    for i in range(train_cfg.steps):
        x, y = batch()
        t = time.perf_counter()
        params, opt, loss, gnorm = step_fn(params, opt, x, y)
        loss = float(jax.block_until_ready(loss))
        dt = time.perf_counter() - t
        losses.append(loss)
        gnorms.append(float(gnorm))
        if i >= warm:
            step_times.append(dt)
        if verbose and (i % train_cfg.log_every == 0 or i == train_cfg.steps - 1):
            print(f"step {i:5d} loss {loss:.4f} gnorm {float(gnorm):.3f} step {dt * 1e3:.1f} ms")

    med = float(np.median(step_times)) if step_times else float("nan")
    result = {
        "platform": platform.describe(),
        "model": {**dataclasses.asdict(model_cfg), "dtype": jnp.dtype(model_cfg.dtype).name, "params": n_params, "params_nonembed": n_nonembed},
        "train": dataclasses.asdict(train_cfg),
        "compile_seconds": compile_s,
        "step_ms_median": med * 1e3,
        "step_ms_min": float(np.min(step_times)) * 1e3 if step_times else float("nan"),
        "tokens_per_second": tokens_per_step / med if step_times else float("nan"),
        "train_flops_per_step_analytic": train_flops_per_step,
        "train_flops_per_step_xla_cost_analysis": xla_flops,
        "achieved_tflops": train_flops_per_step / med / 1e12 if step_times else float("nan"),
        "mfu": (train_flops_per_step / med / peak_flops) if (step_times and peak_flops) else None,
        "loss_first": losses[0],
        "loss_last": float(np.mean(losses[-10:])),
        "loss_curve": losses[:: max(1, len(losses) // 50)],
    }
    if train_cfg.out:
        os.makedirs(os.path.dirname(train_cfg.out) or ".", exist_ok=True)
        with open(train_cfg.out, "w") as f:
            json.dump(result, f, indent=2)
    if train_cfg.save_params:
        os.makedirs(os.path.dirname(train_cfg.save_params) or ".", exist_ok=True)
        flat = {"/".join(map(str, path)): np.asarray(leaf) for path, leaf in _flatten(params)}
        np.savez(train_cfg.save_params, **flat)
    return result


def _flatten(tree):
    out = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        keys = [getattr(p, "key", getattr(p, "idx", None)) for p in path]
        out.append((keys, leaf))
    return out


def load_params(path: str, cfg: GPTConfig) -> dict:
    """Inverse of the ``save_params`` layout in :func:`train`."""
    raw = np.load(path)
    params = {"layers": [dict() for _ in range(cfg.n_layer)]}
    for k, v in raw.items():
        parts = k.split("/")
        if parts[0] == "layers":
            params["layers"][int(parts[1])][parts[2]] = jnp.asarray(v)
        else:
            params[parts[0]] = jnp.asarray(v)
    return params


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the small transformer with Pallas or XLA kernels.")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--attention", choices=["pallas", "xla"], default="pallas")
    ap.add_argument("--norm", choices=["pallas", "xla"], default="pallas")
    ap.add_argument("--linear", choices=["pallas", "xla"], default="xla")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--data", choices=["tinyshakespeare", "synthetic"], default="tinyshakespeare")
    ap.add_argument("--out", default=None, help="results JSON path")
    ap.add_argument("--save-params", default=None, help=".npz checkpoint path")
    a = ap.parse_args(argv)
    mcfg = GPTConfig(seq_len=a.seq, d_model=a.d_model, n_head=a.heads, n_layer=a.layers, d_ff=a.d_ff, attention=a.attention, norm=a.norm, linear=a.linear, dtype=jnp.dtype(a.dtype))
    tcfg = TrainConfig(steps=a.steps, batch=a.batch, lr=a.lr, data=a.data, out=a.out, save_params=a.save_params)
    with platform.interpret_mode():
        res = train(mcfg, tcfg)
    print(json.dumps({k: v for k, v in res.items() if k != "loss_curve"}, indent=2, default=str))


if __name__ == "__main__":
    main()
