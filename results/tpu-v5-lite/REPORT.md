# Results: `tpu-v5-lite`

Platform: **TPU v5 lite** x1 devices, jax 0.12.0.1. Peak per device: 197.0 bf16 TFLOP/s, 820 GB/s (ridge 240 FLOP/byte).

## Matmul (bf16 in, f32 acc)

| shape | pass | impl | ms | TFLOP/s | GB/s | bound | % of roof |
|---|---|---|---:|---:|---:|---|---:|
| 1024x1024x1024 | fwd | pallas | 0.162 | 13.3 | 39 | compute | 7 |
| 1024x1024x1024 | fwd | xla | 0.162 | 13.2 | 39 | compute | 7 |
| 2048x2048x2048 | fwd | pallas | 0.320 | 53.7 | 79 | compute | 27 |
| 2048x2048x2048 | fwd | xla | 0.289 | 59.4 | 87 | compute | 30 |
| 4096x4096x4096 | fwd | pallas | 1.189 | 115.6 | 85 | compute | 59 |
| 4096x4096x4096 | fwd | xla | 0.939 | 146.4 | 107 | compute | 74 |
| 8192x8192x8192 | fwd | pallas | 7.634 | 144.0 | 53 | compute | 73 |
| 8192x8192x8192 | fwd | xla | 6.158 | 178.5 | 65 | compute | 91 |

## Fused add + RMSNorm

| shape | pass | impl | ms | TFLOP/s | GB/s | bound | % of roof |
|---|---|---|---:|---:|---:|---|---:|
| 8192x1024 | fwd | pallas | 0.403 | 0.1 | 166 | memory | 20 |
| 8192x1024 | fwd | xla | 0.394 | 0.1 | 170 | memory | 21 |
| 32768x2048 | fwd | pallas | 1.087 | 0.3 | 494 | memory | 60 |
| 32768x2048 | fwd | xla | 1.470 | 0.2 | 365 | memory | 45 |
| 65536x4096 | fwd | pallas | 3.619 | 0.4 | 593 | memory | 72 |
| 65536x4096 | fwd | xla | 5.083 | 0.3 | 422 | memory | 52 |

## Causal flash attention

| shape | pass | impl | ms | TFLOP/s | GB/s | bound | % of roof |
|---|---|---|---:|---:|---:|---|---:|
| 4x8x1024x128 | fwd | pallas | 0.964 | 8.9 | 35 | compute | 5 |
| 4x8x1024x128 | fwd | xla | 0.662 | 13.0 | 51 | compute | 7 |
| 4x8x1024x128 | fwd | jax-official-pallas | 0.953 | 9.0 | 35 | compute | 5 |
| 4x8x1024x128 | fwd+bwd | pallas | 2.233 | 13.5 | 45 | compute | 7 |
| 4x8x1024x128 | fwd+bwd | xla | 1.349 | 22.3 | 75 | compute | 11 |
| 4x8x2048x128 | fwd | pallas | 3.014 | 11.4 | 22 | compute | 6 |
| 4x8x2048x128 | fwd | xla | 1.827 | 18.8 | 37 | compute | 10 |
| 4x8x2048x128 | fwd | jax-official-pallas | 2.932 | 11.7 | 23 | compute | 6 |
| 4x8x2048x128 | fwd+bwd | pallas | 8.222 | 14.6 | 24 | compute | 7 |
| 4x8x2048x128 | fwd+bwd | xla | 5.513 | 21.8 | 37 | compute | 11 |
| 2x8x4096x128 | fwd | pallas | 5.548 | 12.4 | 12 | compute | 6 |
| 2x8x4096x128 | fwd | xla | 8.135 | 8.4 | 8 | compute | 4 |
| 2x8x4096x128 | fwd | jax-official-pallas | 5.432 | 12.7 | 12 | compute | 6 |
| 2x8x4096x128 | fwd+bwd | pallas | 15.366 | 15.7 | 13 | compute | 8 |
| 2x8x4096x128 | fwd+bwd | xla | 14.981 | 16.1 | 13 | compute | 8 |
| 1x8x8192x128 | fwd | pallas | 10.415 | 13.2 | 6 | compute | 7 |
| 1x8x8192x128 | fwd | xla | 225.022 | 0.6 | 0 | compute | 0 |
| 1x8x8192x128 | fwd | jax-official-pallas | 10.221 | 13.4 | 7 | compute | 7 |
| 1x8x8192x128 | fwd+bwd | pallas | 28.785 | 16.7 | 7 | compute | 8 |
| 1x8x8192x128 | fwd+bwd | xla | 238.514 | 2.0 | 1 | compute | 1 |

Flash-attention forward HLO: 1 Pallas custom call(s), 0 XLA fusions.

## Training: `train_pallas.json`

13.0M params, 4 layers, d_model 512, seq 512, batch 64 over 1 devices; attention=pallas, norm=pallas, linear=xla, dtype=bfloat16.

- step time (median): **45.5 ms**, 720100 tokens/s, 59.5 TFLOP/s achieved, MFU 30.2%
- loss 5.724 -> 1.919 in 300 steps; compile 11.6 s

## Training: `train_pallas_all.json`

13.0M params, 4 layers, d_model 512, seq 512, batch 64 over 1 devices; attention=pallas, norm=pallas, linear=pallas, dtype=bfloat16.

- step time (median): **59.8 ms**, 547906 tokens/s, 45.3 TFLOP/s achieved, MFU 23.0%
- loss 5.724 -> 1.920 in 300 steps; compile 8.1 s

## Training: `train_xla.json`

13.0M params, 4 layers, d_model 512, seq 512, batch 64 over 1 devices; attention=xla, norm=xla, linear=xla, dtype=bfloat16.

- step time (median): **47.3 ms**, 692541 tokens/s, 57.2 TFLOP/s achieved, MFU 29.0%
- loss 5.724 -> 1.917 in 300 steps; compile 15.8 s

## Serving (greedy, static KV cache, AOT-compiled)

13.0M params, max_len 512.

| batch | prefill kernels | TTFT ms | ITL ms (median) | ITL ms (p99) | tokens/s | compiles | exported decode (KB) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | pallas | 0.73 | 0.41 | 0.45 | 1462 | 4 | 17 |
| 1 | xla | 0.73 | 0.42 | 0.47 | 1454 | 4 | 17 |
| 8 | pallas | 1.03 | 0.75 | 0.89 | 7704 | 4 | 17 |
| 8 | xla | 0.90 | 0.76 | 0.84 | 7710 | 4 | 17 |
| 32 | pallas | 1.90 | 2.08 | 2.26 | 13665 | 4 | 17 |
| 32 | xla | 1.99 | 2.15 | 2.23 | 13098 | 4 | 17 |
