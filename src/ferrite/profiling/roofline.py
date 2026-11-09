"""Roofline bookkeeping: achieved FLOP/s and bytes/s against chip peaks.

A kernel is memory-bound when its arithmetic intensity (FLOP per HBM byte) is
below the chip's ridge point (peak FLOP/s / peak bytes/s); otherwise it is
compute-bound. The "percent of roof" is measured against whichever limit
applies, which is the number that says how much headroom a kernel has left.
"""

from __future__ import annotations

import dataclasses

from ferrite import platform


@dataclasses.dataclass(frozen=True)
class RooflinePoint:
    name: str
    flops: float
    bytes: float
    seconds: float
    peak_flops: float | None
    peak_bytes_per_s: float | None

    @property
    def intensity(self) -> float:
        return self.flops / max(self.bytes, 1.0)

    @property
    def achieved_tflops(self) -> float:
        return self.flops / self.seconds / 1e12

    @property
    def achieved_gbps(self) -> float:
        return self.bytes / self.seconds / 1e9

    @property
    def ridge(self) -> float | None:
        if not (self.peak_flops and self.peak_bytes_per_s):
            return None
        return self.peak_flops / self.peak_bytes_per_s

    @property
    def bound(self) -> str:
        r = self.ridge
        if r is None:
            return "unknown"
        return "memory" if self.intensity < r else "compute"

    @property
    def roof_seconds(self) -> float | None:
        """Time the roofline model says this work needs at peak."""
        if not (self.peak_flops and self.peak_bytes_per_s):
            return None
        return max(self.flops / self.peak_flops, self.bytes / self.peak_bytes_per_s)

    @property
    def percent_of_roof(self) -> float | None:
        rs = self.roof_seconds
        return None if rs is None else 100.0 * rs / self.seconds

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "flops": self.flops,
            "bytes": self.bytes,
            "seconds": self.seconds,
            "intensity_flop_per_byte": self.intensity,
            "achieved_tflops": self.achieved_tflops,
            "achieved_gbps": self.achieved_gbps,
            "bound": self.bound,
            "percent_of_roof": self.percent_of_roof,
        }


def analyze(name: str, flops: float, bytes_: float, seconds: float, peaks: platform.Peaks | None = None, devices: int = 1) -> RooflinePoint:
    peaks = peaks or platform.peaks()
    return RooflinePoint(
        name,
        float(flops),
        float(bytes_),
        float(seconds),
        peaks.bf16_flops * devices if peaks else None,
        peaks.hbm_bytes_per_s * devices if peaks else None,
    )


def format_table(points: list[RooflinePoint]) -> str:
    head = "| kernel | time (ms) | TFLOP/s | GB/s | FLOP/byte | bound | % of roof |\n|---|---:|---:|---:|---:|---|---:|"
    rows = []
    for p in points:
        pct = "n/a" if p.percent_of_roof is None else f"{p.percent_of_roof:.0f}%"
        rows.append(f"| {p.name} | {p.seconds * 1e3:.3f} | {p.achieved_tflops:.1f} | {p.achieved_gbps:.0f} | {p.intensity:.1f} | {p.bound} | {pct} |")
    return "\n".join([head, *rows])
