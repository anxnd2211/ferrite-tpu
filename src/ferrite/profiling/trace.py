"""Capture a device trace viewable in TensorBoard (xprof) or Perfetto."""

from __future__ import annotations

import contextlib
import os

import jax


@contextlib.contextmanager
def trace(logdir: str, create_perfetto_link: bool = False):
    """``with trace("traces/run1"): ...`` records host and device activity.

    On TPU the trace shows each Pallas kernel as a ``tpu_custom_call`` op with
    its measured duration next to XLA's fusions, which is the ground truth the
    roofline numbers are checked against. View with
    ``tensorboard --logdir traces`` (needs ``tensorboard-plugin-profile``), or
    upload the ``.xplane.pb`` to https://ui.perfetto.dev.
    """
    os.makedirs(logdir, exist_ok=True)
    jax.profiler.start_trace(logdir, create_perfetto_link=create_perfetto_link)
    try:
        yield logdir
    finally:
        jax.profiler.stop_trace()
