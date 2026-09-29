"""Fused conv1d + GDN / KDA linear-attention kernels for TPU."""

from .gdn.wrapper import fused_conv1d_gdn
from .kda.wrapper import fused_conv1d_kda

__all__ = ["fused_conv1d_gdn", "fused_conv1d_kda"]
