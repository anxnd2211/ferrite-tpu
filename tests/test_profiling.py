import jax
import jax.numpy as jnp

from ferrite import platform
from ferrite.kernels.matmul import matmul, matmul_flops
from ferrite.profiling import analyze, compile_report, format_table, timeit


def test_roofline_math_off_device():
    peaks = platform.peaks("TPU_V5E")
    assert peaks.cores == 1 and peaks.ridge_point > 100
    p = analyze("mm", flops=2 * 1024**3, bytes_=3 * 1024 * 1024 * 2, seconds=20e-6, peaks=peaks)
    assert p.bound == "compute" and 0 < p.percent_of_roof < 1000
    q = analyze("norm", flops=4 * 1024 * 1024, bytes_=3 * 1024 * 1024 * 2, seconds=20e-6, peaks=peaks)
    assert q.bound == "memory"
    assert "| kernel |" in format_table([p, q])


def test_timeit_and_compile_report(tmp_path):
    x = jnp.ones((256, 256), jnp.bfloat16)
    t = timeit(jax.jit(lambda a: matmul(a, a)), x, warmup=1, iters=2)
    assert t.median_s > 0 and t.iters == 2
    rep = compile_report(lambda a: matmul(a, a), x, name="mm", dump_dir=str(tmp_path))
    assert rep["hlo_lines"] > 0 and (tmp_path / "mm.optimized.hlo").exists()
    assert matmul_flops(256, 256, 256) == 2 * 256**3


def test_trace_writes_a_profile(tmp_path):
    from ferrite.profiling import trace

    with trace(str(tmp_path / "t")):
        jax.block_until_ready(jnp.ones((64, 64)) @ jnp.ones((64, 64)))
    files = [p for p in (tmp_path / "t").rglob("*") if p.is_file()]
    assert files, "profiler produced no files"
