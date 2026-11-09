"""Look at what XLA did with a function: fusions, custom calls, collectives, cost."""

from __future__ import annotations

import os
import re

import jax


def _cost(compiled) -> dict:
    try:
        c = compiled.cost_analysis()
    except Exception:
        return {}
    if isinstance(c, (list, tuple)):
        c = c[0] if c else {}
    return {k: float(v) for k, v in dict(c or {}).items() if isinstance(v, (int, float))}


def _memory(compiled) -> dict:
    try:
        m = compiled.memory_analysis()
    except Exception:
        return {}
    out = {}
    for k in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "generated_code_size_in_bytes"):
        if hasattr(m, k):
            out[k] = int(getattr(m, k))
    return out


def compile_report(fn, *args, name: str = "fn", dump_dir: str | None = None, **kw) -> dict:
    """Lower + compile ``fn`` and summarise the optimised HLO.

    Returns counts of fusions, custom calls (Pallas kernels appear as
    ``tpu_custom_call``), collectives, the XLA cost model's FLOP/byte estimate
    and the memory analysis. Optionally writes the StableHLO and optimised HLO
    text to ``dump_dir``.
    """
    lowered = jax.jit(fn).lower(*args, **kw)
    compiled = lowered.compile()
    hlo = compiled.as_text()
    report = {
        "name": name,
        "n_fusions": len(re.findall(r"\bfusion\b|\bfusion\.\d+\b|^\s*fused_computation", hlo, re.M)),
        "n_custom_calls": len(re.findall(r"custom-call\(", hlo)),
        "n_pallas_calls": len(re.findall(r'custom_call_target="tpu_custom_call"', hlo)),
        "n_all_reduce": len(re.findall(r"\ball-reduce(?:-start)?\(", hlo)),
        "n_all_gather": len(re.findall(r"\ball-gather(?:-start)?\(", hlo)),
        "n_reduce_scatter": len(re.findall(r"\breduce-scatter\(", hlo)),
        "n_copies": len(re.findall(r"\bcopy\(", hlo)),
        "cost": _cost(compiled),
        "memory": _memory(compiled),
        "hlo_lines": hlo.count("\n"),
    }
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, f"{name}.stablehlo.mlir"), "w") as f:
            f.write(lowered.as_text())
        with open(os.path.join(dump_dir, f"{name}.optimized.hlo"), "w") as f:
            f.write(hlo)
        report["dump_dir"] = dump_dir
    return report
