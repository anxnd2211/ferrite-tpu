"""Every test runs the Pallas TPU kernels through TPU interpret mode when no TPU is attached."""

import os

import jax
import pytest

from ferrite import platform

os.environ.setdefault("FERRITE_DETECT_RACES", "0")
# f32 reference matmuls must be true f32 on TPU (default is one bf16 pass); the kernels
# make the same choice per operand dtype, see ferrite.kernels._compat.dot_precision.
jax.config.update("jax_default_matmul_precision", "highest")


@pytest.fixture(autouse=True)
def _interpret_mode():
    with platform.interpret_mode():
        yield


@pytest.fixture
def key():
    return jax.random.key(0)


def pytest_report_header(config):
    return f"ferrite backend={platform.backend()} device={platform.device_kind()} jax={jax.__version__}"
