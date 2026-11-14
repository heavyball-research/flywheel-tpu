"""mla_cache_write against write_chunks bit for bit: the chunks' rows hold
(new_kv_c, new_k_pe, 0) and every other row, NaN rows included, keeps its
bits."""

import jax
import jax.numpy as jnp
import pytest

from flywheel_tpu.softmax_attention.kernels.mla_cache_write import (
    mla_cache_write_pallas,
)
from flywheel_tpu.softmax_attention.kernels.mla_decode import (
    cache_packing,
    cache_tile_rows,
)
from flywheel_tpu.softmax_attention.mla_cache_write import mla_cache_write
from flywheel_tpu.softmax_attention.mla_decode import (
    FP8_CACHE_DTYPE,
    mla_cache_shape,
)
from tests.softmax_attention.mla_reference import (
    MlaPrefillParam,
    check_kv_cache,
    logical_cache,
    make_prefill_case,
    write_chunks,
)

INTERPRET = jax.default_backend() != "tpu"
# Note (david): the write never reads q, so one head keeps its draw cheap.
NUM_HEADS = 1


def cache_geometry(is_fp8: bool) -> tuple[int, int]:
    """(tokens a cache word holds, tokens a cache tile holds)."""
    dtype = FP8_CACHE_DTYPE if is_fp8 else jnp.dtype(jnp.bfloat16)
    return cache_packing(dtype), cache_tile_rows(dtype)


def boundary_param(is_fp8: bool, page_size: int) -> MlaPrefillParam:
    """One request per (start, end) pair of a page, a tile, a word and an
    inside-word boundary, the starts in the second page and the ends past
    them, a page end the next page boundary; padding rows fill the last
    tile."""
    packing, tile_rows = cache_geometry(is_fp8)
    # Note (david): from a tile boundary that is no page boundary, these put
    # a position on the tile, on a word and inside a word.
    word_offsets = (0, packing, packing + 1)
    starts = [page_size] + [
        page_size + tile_rows + offset for offset in word_offsets
    ]
    ends = [2 * page_size] + [
        page_size + 3 * tile_rows + offset for offset in word_offsets
    ]
    pairs = [(start, end) for start in starts for end in ends]
    q_lens = tuple(end - start for start, end in pairs)
    return MlaPrefillParam(
        q_lens=q_lens,
        prefix_lens=tuple(start for start, _ in pairs),
        h_q=NUM_HEADS,
        page_size=page_size,
        is_fp8=is_fp8,
        padded_tokens=tile_rows - sum(q_lens) % tile_rows,
    )


def small_param(is_fp8: bool, page_size: int) -> MlaPrefillParam:
    """Chunks of a few tokens: a first chunk of one token, single tokens
    inside a word, at a page's last row and at a page's first, two tokens
    inside a tile, three across a tile boundary, an inactive request with a
    prefix and one without, and a first chunk of whole pages."""
    packing, tile_rows = cache_geometry(is_fp8)
    chunks = (
        (1, 0),
        (1, tile_rows + packing + 1),
        (1, page_size - 1),
        (1, page_size),
        (2, tile_rows + 1),
        (0, 50),
        (3, 2 * tile_rows - 2),
        (0, 0),
        (2 * page_size, 0),
    )
    return MlaPrefillParam(
        q_lens=tuple(q_len for q_len, _ in chunks),
        prefix_lens=tuple(prefix_len for _, prefix_len in chunks),
        h_q=NUM_HEADS,
        page_size=page_size,
        is_fp8=is_fp8,
        seed=1,
    )


# Note (david): long chunks span two groups and start inside a word; the fp8
# cell's padding fills its last tile, so only the bf16 cell pads its tokens.
LONG_PARAMS = [
    MlaPrefillParam(
        q_lens=(5, 1500, 64),
        prefix_lens=(3, 37, 0),
        h_q=NUM_HEADS,
        page_size=page_size,
        is_fp8=is_fp8,
        seed=2,
        padded_tokens=padded_tokens,
    )
    for is_fp8, page_size, padded_tokens in (
        (False, 64, 3),
        (True, 512, 31),
    )
]
# Note (david): tiny groups, so a chunk spans many groups and a group holds
# pieces of several requests.
SMALL_GROUP_CELLS = [
    (
        MlaPrefillParam(
            q_lens=(1, 2, 0, 300, 17, 1, 40),
            prefix_lens=(0, 7, 9, 61, 3, 64, 100),
            h_q=NUM_HEADS,
            is_fp8=is_fp8,
            seed=3,
            padded_tokens=9,
        ),
        group_tiles,
    )
    for is_fp8 in (False, True)
    for group_tiles in (1, 3)
]
WRITE_CELLS = (
    [
        (make_param(is_fp8, page_size), None)
        for make_param in (boundary_param, small_param)
        for is_fp8 in (False, True)
        for page_size in (64, 512)
    ]
    + [(param, None) for param in LONG_PARAMS]
    + SMALL_GROUP_CELLS
)


def cell_id(cell: tuple[MlaPrefillParam, int | None]) -> str:
    param, group_tiles = cell
    return "-".join([
        "q" + "_".join(map(str, param.q_lens)),
        "p" + "_".join(map(str, param.prefix_lens)),
        "fp8" if param.is_fp8 else "bf16",
        f"ps{param.page_size}",
        f"pad{param.padded_tokens}",
        *([] if group_tiles is None else [f"group{group_tiles}"]),
    ])


@pytest.mark.parametrize("cell", WRITE_CELLS, ids=cell_id)
def test_mla_cache_write(cell):
    param, group_tiles = cell
    case = make_prefill_case(param)
    num_pages, page_size, _ = case.kv_cache.shape
    operands = (
        jnp.asarray(case.new_kv_c),
        jnp.asarray(case.new_k_pe),
        jnp.asarray(
            case.kv_cache.reshape(
                mla_cache_shape(num_pages, page_size, case.kv_cache.dtype)
            )
        ),
        jnp.asarray(case.cu_q_lens),
        jnp.asarray(case.kv_lens),
        jnp.asarray(case.block_table),
    )
    if group_tiles is None:
        kv_cache = mla_cache_write(*operands, interpret=INTERPRET)
    else:
        _, tile_rows = cache_geometry(param.is_fp8)
        kv_cache = mla_cache_write_pallas(
            *operands,
            group_tokens=group_tiles * tile_rows,
            interpret=INTERPRET,
        )
    assert check_kv_cache(
        "kv_cache", logical_cache(kv_cache), write_chunks(case)
    )
