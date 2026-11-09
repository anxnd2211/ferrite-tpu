"""Hardware detection, interpret-mode switching and roofline peaks.

The kernels in :mod:`ferrite.kernels` are written against the Pallas TPU
(Mosaic) backend. On a machine without a TPU they are executed through Pallas
TPU interpret mode, which simulates HBM/VMEM, DMAs and semaphores on the CPU and
can detect data races. That is what the test-suite and CI use.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
from typing import Iterator

import jax
from jax.experimental.pallas import tpu as pltpu

LANES = 128  # TPU vector lane width; last block dim should be a multiple of this.
SUBLANES = 8  # second-to-last block dim should be a multiple of this (f32).


def backend() -> str:
    """'tpu', 'gpu' or 'cpu' for the default JAX backend."""
    return jax.default_backend()


def is_tpu() -> bool:
    return backend() == "tpu"


def device_kind() -> str:
    """Human readable device name, e.g. 'TPU v5 lite' or 'cpu'."""
    return jax.devices()[0].device_kind


def num_devices() -> int:
    return jax.device_count()


def chip_info() -> pltpu.TpuInfo | None:
    """Per-TensorCore hardware info for the attached TPU, or None off-TPU.

    The numbers (peak bf16 FLOP/s, HBM bandwidth, VMEM size, MXU width) come from
    JAX's own hardware table, so the roofline uses the same peaks the compiler
    assumes rather than hand-copied marketing figures.
    """
    if not is_tpu():
        return None
    try:
        return pltpu.get_tpu_info()
    except Exception:  # pragma: no cover - very old libtpu
        return None


def chip_info_for(chip: str, cores_per_device: int = 1) -> pltpu.TpuInfo:
    """Hardware info for a named chip, e.g. 'TPU_V5E', usable off-TPU."""
    return pltpu.get_tpu_info_for_chip(getattr(pltpu.ChipVersion, chip), cores_per_device)


def mxu_width() -> int:
    """MXU systolic-array width: 128 up to v5, 256 on v6e/v7x. Used for tile defaults."""
    info = chip_info()
    return int(info.mxu_column_size) if info is not None else 128


def default_tile() -> int:
    """Default square tile for matmul-shaped kernels (MXU width, but >= 128)."""
    return max(LANES, mxu_width())


@dataclasses.dataclass(frozen=True)
class Peaks:
    """Per-device peak numbers for the roofline, summed over TensorCores."""

    name: str
    bf16_flops: float  # FLOP/s
    hbm_bytes_per_s: float
    hbm_bytes: float
    vmem_bytes: float
    cores: int

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOP/byte) at which compute and memory limits meet."""
        return self.bf16_flops / self.hbm_bytes_per_s


def peaks(chip: str | None = None) -> Peaks | None:
    """Roofline peaks for the attached TPU chip, or for ``chip`` if given."""
    info = chip_info_for(chip) if chip else chip_info()
    if info is None:
        return None
    cores = int(info.num_cores)
    return Peaks(
        name=str(info.chip_version.name if hasattr(info.chip_version, "name") else info.chip_version),
        bf16_flops=float(info.bf16_ops_per_second) * cores,
        hbm_bytes_per_s=float(info.mem_bw_bytes_per_second) * cores,
        hbm_bytes=float(info.hbm_capacity_bytes) * cores,
        vmem_bytes=float(info.vmem_capacity_bytes),
        cores=cores,
    )


def interpret_params() -> pltpu.InterpretParams:
    """InterpretParams used off-TPU. FERRITE_DETECT_RACES=1 turns on the race detector."""
    detect = os.environ.get("FERRITE_DETECT_RACES", "0") == "1"
    return pltpu.InterpretParams(detect_races=detect)


@contextlib.contextmanager
def interpret_mode(force: bool | None = None) -> Iterator[bool]:
    """Run Pallas TPU kernels through the CPU interpreter when no TPU is attached.

    Yields True when interpret mode is active. On a TPU this is a no-op unless
    ``force=True``.
    """
    use = (not is_tpu()) if force is None else force
    if not use:
        yield False
        return
    with pltpu.force_tpu_interpret_mode(interpret_params()):
        yield True


def describe() -> dict:
    """Small dict describing the platform; written into every results file."""
    out = {
        "backend": backend(),
        "device_kind": device_kind(),
        "num_devices": num_devices(),
        "jax_version": jax.__version__,
    }
    p = peaks()
    if p is not None:
        out.update(
            chip=p.name,
            cores_per_device=p.cores,
            peak_bf16_tflops=p.bf16_flops / 1e12,
            peak_hbm_gbps=p.hbm_bytes_per_s / 1e9,
            vmem_mib=p.vmem_bytes / 2**20,
            ridge_point_flop_per_byte=p.ridge_point,
        )
    return out
