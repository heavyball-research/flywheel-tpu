"""Vendored KDA chunked forward baseline; re-exports the public API of kda.kda."""

from kda.kda import (
    align_up,
    assert_shape,
    assert_shape_or_none,
    cdiv,
    chunk_gated_delta_rule_fwd_h,
    chunk_kda_fwd,
    chunk_kda_fwd_o_gk,
    chunk_local_cumsum_vector,
    exp,
    exp2,
    get_interpret,
    kda_fwd_intra,
    kda_gate_chunk_cumsum,
    pallas_kda_gate_cumsum,
    prepare_chunk_indices,
    prepare_lens,
)

__all__ = [
    "align_up",
    "assert_shape",
    "assert_shape_or_none",
    "cdiv",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_kda_fwd",
    "chunk_kda_fwd_o_gk",
    "chunk_local_cumsum_vector",
    "exp",
    "exp2",
    "get_interpret",
    "kda_fwd_intra",
    "kda_gate_chunk_cumsum",
    "pallas_kda_gate_cumsum",
    "prepare_chunk_indices",
    "prepare_lens",
]
