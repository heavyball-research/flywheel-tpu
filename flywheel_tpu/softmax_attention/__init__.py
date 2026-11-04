"""Softmax attention: the flash-attention API on the TPU Pallas kernels."""

from flywheel_tpu.softmax_attention.decode import flash_attn_with_kvcache
from flywheel_tpu.softmax_attention.prefill import (
    flash_attn_func,
    flash_attn_varlen_func,
)

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
]
