"""Exact top-k of score rows on SparseCore, alone or fused with a page table
or a per-row offset.

The result is the positions of each row's k largest scores, in no particular
order. Ties at the k-th largest are broken deterministically but unspecified.
Scores must not be NaN within a row's length; -inf is an ordinary score.
"""

import functools
import math

import jax
import jax.numpy as jnp

from flywheel_tpu.topk.kernels.sparse_core_topk import (
    TILE_COLUMNS,
    TILE_ROWS,
    OutputTransform,
    ScoreLayout,
    TopkGeometry,
    check_k,
    sparse_core_vmem_words,
    topk_geometry,
    topk_pallas,
    worker_major_rows,
)

# Score dtypes the kernel reads.
SCORE_DTYPES = (jnp.float32, jnp.bfloat16)
# The largest page size whose page shift and offsets stay in int32.
MAX_PAGE_SIZE_LOG2 = 30
# Rank of flat scores: (T, N / TILE_COLUMNS, TILE_COLUMNS).
FLAT_SCORES_NDIM = 3
KERNEL_NAME = "topk"


@functools.partial(jax.jit, static_argnames=("k", "return_values"))
def top_k(
    scores: jax.Array,
    k: int,
    *,
    row_lengths: jax.Array | None = None,
    return_values: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Positions of each row's k largest scores, any order, -1 padded.

    Args:
      scores: (T, N) float32 or bfloat16; or flat, (T, N / TILE_COLUMNS,
        TILE_COLUMNS) float32 with N a multiple of TILE_ROWS * TILE_COLUMNS,
        read in place. Row t is valid on [0, row_lengths[t]); what lies
        after is never read as a score.
      k: positions per row, from 1 to the chip's limit that check_k names;
        k may exceed N. For flat scores, a multiple of the SparseCore lane
        count.
      row_lengths: (T,) int32, clamped to [0, N]; None for N everywhere.
      return_values: also return the selected scores.

    Returns:
      (T, k) int32 indices: the positions of the min(k, row_lengths[t])
      largest scores of row t in any order, then -1. With return_values,
      (values, indices): values (T, k) in the scores' dtype, the score at
      each index and -inf at -1.
    """
    if scores.ndim == FLAT_SCORES_NDIM:
        check_flat_operands(scores, row_lengths)
    else:
        check_operands(scores, row_lengths)
    indices = run_topk(scores, k, row_lengths)
    if not return_values:
        return indices
    rows = scores.reshape(scores.shape[0], -1)
    values = jnp.take_along_axis(rows, jnp.maximum(indices, 0), axis=1)
    return (
        jnp.where(indices >= 0, values, -jnp.inf).astype(scores.dtype),
        indices,
    )


@functools.partial(jax.jit, static_argnames=("k", "page_size"))
def top_k_page_table_transform(
    scores: jax.Array,
    page_table: jax.Array,
    row_lengths: jax.Array,
    k: int,
    *,
    row_to_batch: jax.Array | None = None,
    page_size: int = 1,
) -> jax.Array:
    """top_k's positions translated to physical rows through a page table.

    Position p of row t becomes page_table[b, p // page_size] * page_size +
    p % page_size, b = row_to_batch[t] (t if None); -1 stays -1.

    Args:
      scores: (T, N) float32 or bfloat16, as top_k takes them.
      page_table: (B, P) int32, P * page_size >= N. Entries a selected
        position reads must be non-negative, and physical rows must fit in
        int32; neither is checked.
      row_lengths: (T,) int32, clamped to [0, N].
      k: positions per row, as top_k takes it.
      row_to_batch: (T,) int32 table row of each score row, clamped to
        [0, B - 1]; None requires B == T.
      page_size: positions per table entry, a power of two up to 2**30.

    Returns:
      (T, k) int32 physical rows in any order, then -1.
    """
    check_operands(scores, row_lengths)
    num_rows, num_columns = scores.shape
    page_shift = check_page_table(
        page_table, row_to_batch, num_rows, num_columns, page_size
    )
    if row_to_batch is None:
        row_to_batch = jnp.arange(num_rows, dtype=jnp.int32)
    batches = jnp.clip(row_to_batch, 0, page_table.shape[0] - 1)
    return run_topk(
        scores,
        k,
        row_lengths,
        OutputTransform.PAGE_TABLE,
        row_values=batches,
        page_table=page_table,
        page_shift=page_shift,
    )


@functools.partial(jax.jit, static_argnames=("k",))
def top_k_ragged_transform(
    scores: jax.Array, offsets: jax.Array, row_lengths: jax.Array, k: int
) -> jax.Array:
    """top_k's positions plus a per-row offset: p of row t becomes p +
    offsets[t]; -1 stays -1.

    Args:
      scores: (T, N) float32 or bfloat16, as top_k takes them.
      offsets: (T,) int32.
      row_lengths: (T,) int32, clamped to [0, N].
      k: positions per row, as top_k takes it.

    Returns:
      (T, k) int32 offset positions in any order, then -1.
    """
    check_operands(scores, row_lengths)
    check_row_vector(offsets, scores.shape[0], "offsets")
    return run_topk(
        scores, k, row_lengths, OutputTransform.OFFSET, row_values=offsets
    )


def run_topk(
    scores: jax.Array,
    k: int,
    row_lengths: jax.Array | None,
    transform: OutputTransform = OutputTransform.NONE,
    *,
    row_values: jax.Array | None = None,
    page_table: jax.Array | None = None,
    page_shift: int = 0,
) -> jax.Array:
    """The kernel's (T, k) output on checked operands. row_values are the
    transform's per-row values: offsets, or table rows of a page table."""
    words, geometry = scores_geometry(
        scores,
        k,
        sparse_core_vmem_words(),
        has_row_lengths=row_lengths is not None,
        transform=transform,
        page_shift=page_shift,
        pages_per_row=0 if page_table is None else page_table.shape[1],
    )
    tables = []
    if page_table is not None:
        row_values, table = page_table_operands(
            page_table, row_values, geometry
        )
        tables.append(table)
    per_row = [
        worker_major_rows(values, geometry)
        for values in (row_lengths, row_values)
        if values is not None
    ]
    return topk_pallas((*per_row, *tables, words), geometry, KERNEL_NAME)


def page_table_operands(
    page_table: jax.Array, batches: jax.Array, geometry: TopkGeometry
) -> tuple[jax.Array, jax.Array]:
    """Row bases and table for the kernel: batch rows of a tile-padded table
    when a table row fits its buffer, else flat offsets into the flat table."""
    if geometry.holds_table_row:
        padding = geometry.table_words - page_table.shape[1]
        if padding:
            page_table = jnp.pad(page_table, ((0, 0), (0, padding)))
        return batches, page_table
    return batches * page_table.shape[1], page_table.reshape(-1)


def flat_topk_geometry(
    scores: jax.Array, row_lengths: jax.Array, k: int, vmem_words: int
) -> TopkGeometry:
    """Kernel geometry for flat scores within vmem_words of a subcore's VMEM,
    for a caller that runs topk_kernel in its own SparseCore call."""
    check_flat_operands(scores, row_lengths)
    _, geometry = scores_geometry(scores, k, vmem_words, has_row_lengths=True)
    return geometry


def scores_geometry(
    scores: jax.Array,
    k: int,
    vmem_words: int,
    *,
    has_row_lengths: bool,
    transform: OutputTransform = OutputTransform.NONE,
    page_shift: int = 0,
    pages_per_row: int = 0,
) -> tuple[jax.Array, TopkGeometry]:
    """The scores as the kernel reads them, and the kernel's checked geometry
    for them within vmem_words of a subcore's VMEM."""
    words, layout = kernel_scores(scores)
    geometry = topk_geometry(
        scores.shape[0],
        math.prod(words.shape[1:]),
        k,
        vmem_words,
        layout=layout,
        num_columns=math.prod(scores.shape[1:]),
        has_row_lengths=has_row_lengths,
        transform=transform,
        page_shift=page_shift,
        pages_per_row=pages_per_row,
    )
    check_k(geometry)
    if layout is ScoreLayout.FLAT and k % geometry.lanes:
        raise ValueError(
            f"k must be a multiple of {geometry.lanes} for flat scores;"
            f" got {k}."
        )
    return words, geometry


def kernel_scores(scores: jax.Array) -> tuple[jax.Array, ScoreLayout]:
    """The scores as the kernel reads them, and their layout: flat scores in
    place; else the columns padded to a multiple of TILE_COLUMNS, bfloat16
    read in row pairs, or widened to float32 when T is odd."""
    if scores.ndim == FLAT_SCORES_NDIM:
        return scores, ScoreLayout.FLAT
    num_rows, num_columns = scores.shape
    padding = -num_columns % TILE_COLUMNS
    if padding:
        scores = jnp.pad(scores, ((0, 0), (0, padding)))
    if scores.dtype == jnp.bfloat16 and num_rows % 2 == 0:
        return scores, ScoreLayout.ROW_PAIRS
    return scores.astype(jnp.float32), ScoreLayout.ROWS


def check_operands(scores: jax.Array, row_lengths: jax.Array | None) -> None:
    if scores.ndim != 2 or 0 in scores.shape:
        raise ValueError(
            f"scores must be a non-empty (T, N); got {scores.shape}."
        )
    if scores.dtype not in SCORE_DTYPES:
        raise ValueError(
            f"scores must be float32 or bfloat16; got {scores.dtype}."
        )
    if row_lengths is not None:
        check_row_vector(row_lengths, scores.shape[0], "row_lengths")


def check_flat_operands(
    scores: jax.Array, row_lengths: jax.Array | None
) -> None:
    if (
        scores.ndim != FLAT_SCORES_NDIM
        or 0 in scores.shape
        or scores.shape[2] != TILE_COLUMNS
        or scores.shape[1] % TILE_ROWS
        or scores.dtype != jnp.float32
    ):
        raise ValueError(
            f"flat scores must be a non-empty (T, N / {TILE_COLUMNS},"
            f" {TILE_COLUMNS}) float32 with N a multiple of"
            f" {TILE_ROWS * TILE_COLUMNS}; got {scores.shape} {scores.dtype}."
        )
    if row_lengths is not None:
        check_row_vector(row_lengths, scores.shape[0], "row_lengths")


def check_row_vector(values: jax.Array, num_rows: int, name: str) -> None:
    if values.shape != (num_rows,) or values.dtype != jnp.int32:
        raise ValueError(
            f"{name} must be ({num_rows},) int32; got {values.shape}"
            f" {values.dtype}."
        )


def check_page_table(
    page_table: jax.Array,
    row_to_batch: jax.Array | None,
    num_rows: int,
    num_columns: int,
    page_size: int,
) -> int:
    """Fails unless the page table operands meet the contract; returns
    log2(page_size)."""
    page_shift = page_size.bit_length() - 1
    if page_size < 1 or page_size != 1 << page_shift:
        raise ValueError(f"page_size must be a power of two; got {page_size}.")
    if page_shift > MAX_PAGE_SIZE_LOG2:
        raise ValueError(
            f"page_size must be at most 2**{MAX_PAGE_SIZE_LOG2}; got"
            f" {page_size}."
        )
    if page_table.ndim != 2 or page_table.dtype != jnp.int32:
        raise ValueError(
            f"page_table must be (B, P) int32; got {page_table.shape}"
            f" {page_table.dtype}."
        )
    num_batches, pages_per_row = page_table.shape
    if pages_per_row * page_size < num_columns:
        raise ValueError(
            f"page_table's {pages_per_row} pages of {page_size} cover fewer"
            f" than the scores' {num_columns} columns."
        )
    if row_to_batch is None and num_batches != num_rows:
        raise ValueError(
            f"page_table must have {num_rows} rows without row_to_batch; got"
            f" {num_batches}."
        )
    if row_to_batch is not None:
        check_row_vector(row_to_batch, num_rows, "row_to_batch")
    return page_shift
