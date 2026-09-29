"""FlyWheel: attention kernels for TPU, implemented in JAX Pallas."""

from .flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

__version__ = "0.1.0"

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
]
