"""ferrite: Pallas TPU kernels, sharded training and AOT serving for a small transformer.

Everything here runs on CPU through Pallas TPU interpret mode (for tests) and on
real TPUs for the numbers. See README.md for the design notes and results.
"""

from ferrite import platform
from ferrite.kernels.flash_attention import flash_attention
from ferrite.kernels.matmul import matmul
from ferrite.kernels.rmsnorm import rmsnorm

__all__ = ["flash_attention", "matmul", "rmsnorm", "platform"]
__version__ = "0.1.0"
