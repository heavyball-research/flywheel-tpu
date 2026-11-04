"""FlyWheel: attention kernels for TPU, implemented in JAX Pallas."""

from .softmax_attention import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

__version__ = "0.4.1"

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
]
