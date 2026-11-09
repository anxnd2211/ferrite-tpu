"""Turn results/<device>/*.json into results/<device>/REPORT.md."""

from __future__ import annotations

import glob
import json
import os
import sys


def _load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _fmt(x, nd=1):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def report(d: str) -> str:
    out = [f"# Results: `{os.path.basename(d)}`\n"]
    k = _load(d, "kernels.json")
    if k:
        p = k["platform"]
        out.append(f"Platform: **{p.get('device_kind')}** x{p.get('num_devices')} devices, jax {p.get('jax_version')}. "
                   f"Peak per device: {_fmt(p.get('peak_bf16_tflops'))} bf16 TFLOP/s, {_fmt(p.get('peak_hbm_gbps'), 0)} GB/s "
                   f"(ridge {_fmt(p.get('ridge_point_flop_per_byte'), 0)} FLOP/byte).\n")
        if k.get("quick"):
            out.append("> quick/CPU smoke run: timings are not meaningful.\n")
        for section, title in [("matmul", "Matmul (bf16 in, f32 acc)"), ("rmsnorm", "Fused add + RMSNorm"), ("attention", "Causal flash attention")]:
            out.append(f"## {title}\n")
            out.append("| shape | pass | impl | ms | TFLOP/s | GB/s | bound | % of roof |\n|---|---|---|---:|---:|---:|---|---:|")
            for r in k[section]:
                out.append(f"| {'x'.join(map(str, r['shape']))} | {r.get('pass', 'fwd')} | {r['impl']} | {_fmt(r['ms'], 3)} | {_fmt(r['achieved_tflops'])} | {_fmt(r['achieved_gbps'], 0)} | {r['bound']} | {_fmt(r['percent_of_roof'], 0)} |")
            out.append("")
        h = k.get("hlo_flash_attention_fwd", {})
        out.append(f"Flash-attention forward HLO: {h.get('n_pallas_calls', 0)} Pallas custom call(s), {h.get('n_fusions', 0)} XLA fusions.\n")
    for name in sorted(glob.glob(os.path.join(d, "train*.json"))):
        t = json.load(open(name))
        out.append(f"## Training: `{os.path.basename(name)}`\n")
        m, tr = t["model"], t["train"]
        out.append(f"{m['params'] / 1e6:.1f}M params, {m['n_layer']} layers, d_model {m['d_model']}, seq {m['seq_len']}, batch {tr['batch']} over {t['platform']['num_devices']} devices; "
                   f"attention={m['attention']}, norm={m['norm']}, linear={m['linear']}, dtype={m['dtype']}.\n")
        out.append(f"- step time (median): **{_fmt(t['step_ms_median'])} ms**, {_fmt(t['tokens_per_second'], 0)} tokens/s, {_fmt(t['achieved_tflops'])} TFLOP/s achieved, MFU {_fmt(None if t['mfu'] is None else 100 * t['mfu'])}%")
        out.append(f"- loss {_fmt(t['loss_first'], 3)} -> {_fmt(t['loss_last'], 3)} in {tr['steps']} steps; compile {_fmt(t['compile_seconds'])} s\n")
    s = _load(d, "serve.json")
    if s:
        out.append("## Serving (greedy, static KV cache, AOT-compiled)\n")
        out.append(f"{s['model']['params'] / 1e6:.1f}M params, max_len {s['model']['seq_len']}.\n")
        out.append("| batch | prefill kernels | TTFT ms | ITL ms (median) | ITL ms (p99) | tokens/s | compiles | exported decode (KB) |\n|---:|---|---:|---:|---:|---:|---:|---:|")
        for r in s["runs"]:
            out.append(f"| {r['batch']} | {r['prefill_kernels']} | {_fmt(r['ttft_ms'], 2)} | {_fmt(r['itl_ms_median'], 2)} | {_fmt(r['itl_ms_p99'], 2)} | {_fmt(r['tokens_per_second'], 0)} | {r['n_compiles']} | {r['exported_decode_bytes'] / 1024:.0f} |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    root = argv[0] if argv else (sys.argv[1] if len(sys.argv) > 1 else "results")
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        text = report(d)
        with open(os.path.join(d, "REPORT.md"), "w") as f:
            f.write(text)
        print(text)


if __name__ == "__main__":
    main()
