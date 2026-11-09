"""Shared helpers for the benchmark scripts."""

from __future__ import annotations

import json
import os
import re

from ferrite import platform


def device_slug() -> str:
    return re.sub(r"[^a-z0-9]+", "-", platform.device_kind().lower()).strip("-") or platform.backend()


def results_dir(root: str = "results") -> str:
    d = os.path.join(root, device_slug())
    os.makedirs(d, exist_ok=True)
    return d


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"wrote {path}")
