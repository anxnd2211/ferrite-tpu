# ferrite-tpu

Pallas TPU kernels, a sharded JAX transformer trained with them, roofline-checked
profiling and ahead-of-time-compiled serving, in one small repository. It goes from
a hand-written MXU kernel to tokens-per-second on a decode loop, and every claim in
it is backed by a test or a benchmark script rather than a sentence.

[![ci](https://github.com/anxnd2211/ferrite-tpu/actions/workflows/ci.yml/badge.svg)](https://github.com/anxnd2211/ferrite-tpu/actions/workflows/ci.yml)

```
                 ┌──────────────── kernels (Pallas / Mosaic) ────────────────┐
  q k v ───────▶ │ flash_attention: fwd + dQ + dKV kernels, online softmax,   │
                 │   causal block skipping in compute AND in the DMA schedule │
  x, residual ─▶ │ rmsnorm: fused add + norm, two outputs, f32 statistics     │
  x @ w ───────▶ │ matmul: bf16 blocks, f32 VMEM accumulator, megacore grid   │
                 └───────────────┬────────────────────────────┬──────────────┘
                                 │ custom_vjp                 │
        ┌────────────────────────▼───────────┐   ┌────────────▼──────────────────────┐
        │ model/gpt.py + model/train.py       │   │ serving/engine.py                  │
        │ pre-norm decoder, each component    │   │ bucketed prefill (Pallas), static  │
        │ switchable pallas|xla; data-parallel │   │ KV cache, XLA decode, AOT-compiled │
        │ NamedSharding, hand-written AdamW   │   │ executables, jax.export round trip │
        └────────────────────────┬───────────┘   └────────────┬──────────────────────┘
                                 │                            │
        ┌────────────────────────▼────────────────────────────▼──────────────────────┐
        │ profiling/: timing, roofline against JAX's own per-chip peak table,         │
        │ optimised-HLO inspection (fusions, tpu_custom_call, collectives), traces     │
        └─────────────────────────────────────────────────────────────────────────────┘
```

## What is in it

| Piece | File | Notes |
|---|---|---|
| Causal flash attention, forward and backward | `src/ferrite/kernels/flash_attention.py` | Three Pallas kernels (fwd, dQ, dK/dV), FlashAttention-2 recipe, `jax.custom_vjp` |
| Fused residual-add + RMSNorm | `src/ferrite/kernels/rmsnorm.py` | One read of the residual stream per sub-layer; analytic backward left to XLA on purpose |
| Blocked matmul | `src/ferrite/kernels/matmul.py` | bf16 in, f32 accumulate in VMEM scratch; ragged shapes padded exactly; differentiable |
| Small decoder-only transformer | `src/ferrite/model/gpt.py` | Attention, norm and linear layers each routed to `pallas` or `xla` for like-for-like comparison |
| Data-parallel training | `src/ferrite/model/train.py` | `jax.sharding` mesh over all cores, hand-written AdamW with clipping and cosine schedule, MFU |
| Serving engine | `src/ferrite/serving/engine.py` | Prompt buckets, static `[L,B,H,T,d]` KV cache, `lower().compile()` ahead of time, `jax.export` |
| Profiling | `src/ferrite/profiling/` | `timeit`, roofline bookkeeping, HLO report, `jax.profiler` trace context manager |
| Tests | `tests/` | 38 tests; run on CPU through Pallas **TPU interpret mode**, including the data-race detector |
| Real-TPU run | `notebooks/kaggle_tpu.ipynb` | Tests, benchmarks, training, serving and a trace on a Kaggle/Colab TPU |

## Kernel design notes

**Flash attention.** Grid `(batch, heads, q_blocks, kv_blocks)`; the last axis is
`arbitrary` because it carries the running softmax state, the first three are
`parallel` so Mosaic can spread them over both TensorCores on v4/v5p. Row
statistics `m` and `l` live in `(block_q, 128)` f32 VMEM scratch, lane-replicated,
which is the layout the TPU vector unit wants for per-row scalars; they are tiled
across lanes with `jnp.tile` when a tile is wider than 128. `P @ V` accumulates
unnormalised in an f32 `(block_q, head_dim)` scratch and is divided by `l` once at
the last key block instead of at every step. Causal masking is an iota compare,
but the saving comes from skipping: the kernel body sits under
`pl.when(block touches the diagonal or below)`, and the K/V `index_map` points
skipped blocks at the block that will be needed next, so the software pipeline
issues no DMA for them. Both halves matter; skipping only the compute still
pays the HBM traffic. Backward recomputes `P` blockwise from the saved
log-sum-exp and `delta = rowsum(dO * O)`: one kernel sweeps query blocks for
each key block to accumulate `dK, dV`, another sweeps key blocks for each query
block for `dQ`, so no kernel needs atomics. Masked scores use a large finite
negative instead of `-inf` so no row can produce `inf - inf`.

**Fused add + RMSNorm.** A transformer block adds a sub-layer output to the
residual stream and immediately normalises it. The kernel takes `x` and
`residual`, writes both `h = x + residual` and `rmsnorm(h) * w`, and does the
statistics in f32 whatever the storage dtype. It is memory-bound by a wide
margin (about 0.6 FLOP per byte against a ridge point of several hundred), so
its roofline target is HBM bandwidth and the comparison that matters is bytes
moved, not FLOPs. The backward is a few element-wise ops and reductions that
XLA fuses well; writing it in Pallas would not move the step time, so the
`custom_vjp` backward is plain JAX.

**Matmul.** The textbook blocked kernel: `(M/bm, N/bn, K/bk)` grid, f32
accumulator in VMEM, output written on the last K step. Block defaults are
derived from the MXU width reported by JAX (`128` through v5e, `256` on v6e),
so the same code picks sane tiles on every generation. f32 operands request a
true fp32 contraction (`Precision.HIGHEST`); bf16 operands take the single-pass
default, so a reference test in f32 means what it says. XLA's `dot` is the
baseline, and the kernel is not expected to beat it; it exists to make the
tiling and pipelining model explicit and to anchor the roofline tooling on a
kernel whose FLOPs and bytes are known exactly.

**Model and training.** The residual add that closes one sub-layer is fused
into the norm that opens the next, so the residual stream is read and written
once per sub-layer rather than twice. Parameters stay in f32, activations and
matmuls in bf16, every reduction that matters in f32. Training uses
`NamedSharding` directly: replicated parameters and optimiser state, batch
sharded over a 1-D `data` mesh, gradient all-reduce inserted by `jit`. MFU is
reported against the chip's peak from the same hardware table the roofline uses.

**Serving.** Prompts are right-padded into a few length buckets and each bucket's
prefill is compiled once with `jit(...).lower(...).compile()`; the engine then
calls compiled executables directly, so no serving-time shape can trigger a
compile (the benchmark records the compile count, which is `len(buckets) + 1`).
Prefill is compute-bound and runs the Pallas attention kernel; decode is one
query row against the whole cache, memory-bound, and stays in XLA where fusion
already does the right thing. The decode step round-trips through `jax.export`
so a serving binary could load it without the Python model code.

## How it is verified

- Every kernel is tested against a materialised-scores reference in f32 and bf16,
  for several block sizes, head dims (64, 128, 256), causal and non-causal.
- Backward kernels are checked against `jax.grad` of the reference; gradients of
  the full model agree between the Pallas and XLA paths.
- `flash_attention` on the first 256 tokens equals the first 256 rows of a
  512-token run (prefix invariance), which catches off-by-one errors in the
  causal skip logic.
- The forward + backward pass runs under the interpreter's **data-race detector**
  (`InterpretParams(detect_races=True)`).
- The serving engine reproduces greedy decoding from the slow full-recompute
  reference token for token, and the exported decode step returns identical logits.
- Benchmark scripts are smoke-tested in CI (`--quick`) so the real-TPU notebook
  does not discover syntax errors.

All of this runs on a laptop CPU in about half a minute through Pallas TPU
interpret mode, which simulates HBM/VMEM, DMAs and semaphores. Interpret mode
cannot catch Mosaic lowering errors, so the notebook runs the same test-suite on
the real chip first.

## Results

Numbers are produced by the scripts, written to `results/<device>/*.json`, and
rendered into `results/<device>/REPORT.md` by `benchmarks/make_report.py`.
Nothing in a report is typed by hand.

**Status:** pending the first real-TPU run (`notebooks/kaggle_tpu.ipynb`). This
README will link the report here once it exists. CPU interpret-mode timings are
deliberately excluded because they measure the interpreter, not the kernel.

## Roofline method

For each kernel the benchmark knows its FLOPs (from the shapes) and the bytes it
must move (operands read once, outputs written once), times it with
`block_until_ready`, and compares against the chip's peak bf16 FLOP/s and HBM
bandwidth. The peaks come from `jax.experimental.pallas.tpu.get_tpu_info()`, the
same table JAX itself uses, so no marketing numbers are hand-copied. A kernel is
memory-bound when its arithmetic intensity is below the ridge point
(`peak_flops / peak_bandwidth`), compute-bound otherwise, and "percent of roof"
is measured against whichever limit applies. Causal attention counts skipped
blocks as free, which is what makes the skip logic show up as achieved TFLOP/s.

## Running it

```bash
pip install -e ".[dev]"        # CPU: kernels run through TPU interpret mode
pytest -n auto                  # 38 tests, ~30 s on a laptop
FERRITE_DETECT_RACES=1 pytest tests/test_flash_attention.py -k race

python benchmarks/bench_kernels.py --quick          # smoke test; meaningful only on TPU
python -m ferrite.model.train --steps 20 --batch 8 --seq 128 --layers 1 --d-model 128 --heads 1 --d-ff 256 --data synthetic
python benchmarks/bench_serve.py --quick
```

On a TPU (`pip install -U "jax[tpu]"`), the same commands run the real kernels;
`notebooks/kaggle_tpu.ipynb` strings them together and zips the results. Each
stage is its own process because a TPU chip is owned by exactly one process:
a notebook kernel that has imported JAX locks every subprocess out with
`ABORTED: The TPU is already in use`.

## Decisions worth arguing about

- *Decode stays in XLA.* A Pallas decode-attention kernel is the obvious next
  step, but at this model size decode is dominated by weight reads, and the
  measured split between prefill (compute) and decode (bandwidth) is more
  informative than a kernel that would not change the inter-token latency.
- *Lane-replicated statistics.* Keeping `m`, `l`, `lse` and `delta` as
  `(rows, 128)` costs 128x the HBM for those small arrays in exchange for
  layouts Mosaic handles without relayout copies. For attention the extra
  traffic is under 1% of the Q/K/V bytes.
- *Analytic RMSNorm backward in XLA.* See above: the kernel is for the forward
  fusion; a backward kernel would be code without a measurable payoff.
- *No framework.* Flax/Optax would save lines but hide the sharding and the
  optimiser, which are part of what this repository is meant to show.

## Limitations and next steps

- Sequence lengths must be multiples of the block size (128 by default); the
  model falls back to XLA attention for shorter sequences rather than padding.
- No segment IDs / packed sequences, no sliding window, no soft-capping.
- Backward kernels use one tile size for both axes; a `block_q != block_k`
  backward is supported by the validation but not tuned.
- Next: a Pallas paged-attention decode kernel with a `[num_pages, page_size]`
  cache, int8 weight-only quantisation with a fused dequant epilogue in the
  matmul, and a tensor-parallel variant of the MLP with `shard_map`.

## Acknowledgements

Block-skipping and lane-tiling idioms follow `jax.experimental.pallas.ops.tpu`
(flash and splash attention); the backward decomposition follows the
FlashAttention-2 paper. All code here was written against the Pallas TPU docs
and tested from scratch.
