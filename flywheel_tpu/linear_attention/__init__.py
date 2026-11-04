"""Fused conv1d + GDN / KDA linear-attention kernels for TPU."""

from flywheel_tpu.linear_attention.gdn.wrapper import (
    fused_conv1d_gdn,
    fused_conv1d_gdn_replay,
)
from flywheel_tpu.linear_attention.kda.wrapper import (
    fused_conv1d_kda,
    fused_conv1d_kda_replay,
)

__all__ = [
    "fused_conv1d_gdn",
    "fused_conv1d_gdn_replay",
    "fused_conv1d_kda",
    "fused_conv1d_kda_replay",
]
