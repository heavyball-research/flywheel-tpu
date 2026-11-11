"""SparseCore scatter of new latent tokens into the split DSA cache."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.sparse_attention.dsa.kernels.gather import (
    IGNORED_ROW,
    NOPE_WORDS,
    ROPE_TOKENS_PER_ROW,
    ROPE_WORDS,
    sparse_core_mesh,
)

ROPE_ROW_SHIFT = ROPE_TOKENS_PER_ROW.bit_length() - 1
SEM_NOPE = 0
SEM_ROPE = 1
NUM_SEM_ROWS = 2


def dsa_cache_write_kernel(
    rows_hbm: jax.Array,
    nope_hbm: jax.Array,
    rope_hbm: jax.Array,
    nope_cache_ref: jax.Array,
    rope_cache_ref: jax.Array,
    row_vmem: jax.Array,
    nope_vmem: jax.Array,
    rope_vmem: jax.Array,
    sems: jax.Array,
    *,
    tokens_per_worker: int,
    num_workers: int,
    num_subcores: int,
) -> None:
    """Each worker writes its run of tokens a block of num_lanes tokens at a
    time: token i's nope row to cache row rows[i], its rope words into the
    16-word slot of rope row rows[i] // 8; a token of row IGNORED_ROW writes
    nothing."""
    block_tokens = row_vmem.shape[0]
    worker = lax.axis_index("core") * num_subcores + lax.axis_index("subcore")
    first_token = worker * tokens_per_worker
    num_blocks = jnp.where(
        worker < num_workers, tokens_per_worker // block_tokens, 0
    )

    @pl.loop(0, num_blocks)
    def _block(block):
        start = first_token + block * block_tokens
        pltpu.sync_copy(rows_hbm.at[pl.ds(start, block_tokens)], row_vmem)
        pltpu.sync_copy(nope_hbm.at[pl.ds(start, block_tokens)], nope_vmem)
        pltpu.sync_copy(rope_hbm.at[pl.ds(start, block_tokens)], rope_vmem)
        rows = row_vmem[...]
        copies = []
        for token in range(block_tokens):
            row = rows[token]
            rope_slot = (row & (ROPE_TOKENS_PER_ROW - 1)) * ROPE_WORDS
            copies.append((
                row != IGNORED_ROW,
                pltpu.make_async_copy(
                    nope_vmem.at[pl.ds(token, 1)],
                    nope_cache_ref.at[pl.ds(row, 1)],
                    sems.at[SEM_NOPE, token],
                ),
                pltpu.make_async_copy(
                    rope_vmem.at[pl.ds(token, 1)],
                    rope_cache_ref.at[
                        pl.ds(lax.shift_right_logical(row, ROPE_ROW_SHIFT), 1),
                        pl.ds(rope_slot, ROPE_WORDS),
                    ],
                    sems.at[SEM_ROPE, token],
                ),
            ))
        for is_written, *token_copies in copies:

            @pl.when(is_written)
            def _start(token_copies=token_copies):
                for copy in token_copies:
                    copy.start()

        for is_written, *token_copies in copies:

            @pl.when(is_written)
            def _wait(token_copies=token_copies):
                for copy in token_copies:
                    copy.wait()


def dsa_cache_write_pallas(
    rows: jax.Array,
    nope_words: jax.Array,
    rope_words: jax.Array,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """The scatter's SparseCore call on checked operands.

    Args:
      rows: (T,) int32 cache row of each new token, IGNORED_ROW for padding.
      nope_words: (T, NOPE_WORDS) int32 banded nope rows.
      rope_words: (T, ROPE_WORDS) int32 banded rope rows.
      nope_cache: (num_rows, NOPE_WORDS) int32.
      rope_cache: (num_rows / 8, NOPE_WORDS) int32.

    Returns:
      (nope_cache, rope_cache) with the tokens written.
    """
    num_tokens = rows.shape[0]
    sparse_core = pltpu.get_tpu_info().sparse_core
    block_tokens = sparse_core.num_lanes
    num_blocks = pl.cdiv(num_tokens, block_tokens)
    num_workers = min(
        sparse_core.num_cores * sparse_core.num_subcores, num_blocks
    )
    tokens_per_worker = pl.cdiv(num_blocks, num_workers) * block_tokens
    num_padding = num_workers * tokens_per_worker - num_tokens
    rows = jnp.pad(rows, (0, num_padding), constant_values=IGNORED_ROW)
    nope_words = jnp.pad(nope_words, ((0, num_padding), (0, 0)))
    rope_words = jnp.pad(rope_words, ((0, num_padding), (0, 0)))
    nope_ref = jax.new_ref(nope_cache)
    rope_ref = jax.new_ref(rope_cache)
    pl.kernel(
        functools.partial(
            dsa_cache_write_kernel,
            tokens_per_worker=tokens_per_worker,
            num_workers=num_workers,
            num_subcores=sparse_core.num_subcores,
        ),
        out_type=(),
        mesh=sparse_core_mesh(),
        scratch_types=(
            pltpu.VMEM((block_tokens,), jnp.int32),
            pltpu.VMEM((block_tokens, NOPE_WORDS), jnp.int32),
            pltpu.VMEM((block_tokens, ROPE_WORDS), jnp.int32),
            pltpu.SemaphoreType.DMA((NUM_SEM_ROWS, block_tokens)),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        name="dsa_cache_write",
    )(rows, nope_words, rope_words, nope_ref, rope_ref)
    return jax.freeze(nope_ref), jax.freeze(rope_ref)
