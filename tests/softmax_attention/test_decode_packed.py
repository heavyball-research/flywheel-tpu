# Copyright 2026 heavyball-research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Packed decode: several (row, block) items per straight-line kernel step
(a 4-element KV-cache config, items_per_step > 1), checked against the NumPy
reference of test_paged and against the one-item kernel.

The rows span one to four 256-token blocks, so steps hold several rows, a row
spans several steps, and the last step is padded whenever the item count is
not a multiple of items_per_step.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu import flash_attn_with_kvcache
from flywheel_tpu.softmax_attention import dispatch
from flywheel_tpu.softmax_attention.kernels import decode as decode_kernel
from tests.softmax_attention.test_paged import (
    INTERPRET,
    assert_extend_matches,
    ragged_reference,
    random_ragged_case,
)

CAPACITY = 1024
BLOCK_KV = 256
BLOCK_KV_COMPUTE = 128
# Note (david): 0 and 1022 are the extremes a 1024-token row can append to;
# 255 / 256 and 511 / 512 sit on each side of a block edge.
PREFIXES = (0, 300, 1, 1022, 255, 256, 511, 700)


def merge_pages(kc, vc):
    """The merged cache of K and V pages: each head's K row then its V row,
    [k0, v0, k1, v1, ...], an fp8 cache's odd head count padded by a zero
    head."""
    num_pages, page_size, kv_heads, head_dim = kc.shape
    pad = ((0, 0), (0, 0), (0, kv_heads % 2 if kc.dtype.itemsize == 1 else 0))
    kc, vc = (jnp.pad(pages, pad + ((0, 0),)) for pages in (kc, vc))
    return jnp.stack((kc, vc), axis=3).reshape(
        num_pages, page_size, -1, head_dim
    )


def split_pages(cache, kv_heads):
    """merge_pages' inverse: the (K pages, V pages) of a merged cache."""
    num_pages, page_size, cache_heads, head_dim = cache.shape
    pairs = cache.reshape(num_pages, page_size, cache_heads // 2, 2, head_dim)
    return pairs[:, :, :kv_heads, 0, :], pairs[:, :, :kv_heads, 1, :]


@pytest.fixture(name="force_config")
def fixture_force_config(monkeypatch):
    # Note (david): the config is read while tracing and is not part of any jit
    # cache key, so a call at the same shapes would reuse the kernel an
    # earlier config built; every change, and the teardown, drops the caches.
    def _force(config):
        jax.clear_caches()
        monkeypatch.setattr(
            dispatch, "get_tuned_kvcache_config", lambda **_: config
        )

    yield _force
    jax.clear_caches()


def decode_case(heads, kv_heads, head_dim, dtype, prefixes=PREFIXES):
    q, kc, vc, k, v, cu, lengths, table = random_ragged_case(
        lengths=(1,) * len(prefixes),
        prefixes=prefixes,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        capacity=CAPACITY,
        padding=0,
    )
    kc, vc, k, v = (x.astype(dtype) for x in (kc, vc, k, v))
    return q, kc, vc, k, v, cu, lengths, table


def run_decode(case, *, append, **kwargs):
    """flash_attn_with_kvcache's decode on the merged pool; (out, lse,
    updated_k, updated_v) in test_paged's layout."""
    q, kc, vc, k, v, _, lengths, table = case
    kv_heads = kc.shape[2]
    out, lse, cache = flash_attn_with_kvcache(
        q[:, None],
        merge_pages(kc, vc),
        None,
        k[:, None] if append else None,
        v[:, None] if append else None,
        block_table=table,
        cache_seqlens=lengths,
        nheads_k=kv_heads,
        causal=True,
        return_softmax_lse=True,
        interpret=INTERPRET,
        **kwargs,
    )
    return (out[:, 0], lse[:, :, 0].T, *split_pages(cache, kv_heads))


def expected_of(case, *, append, num_active=None, **kwargs):
    q, kc, vc, k, v, cu, lengths, table = case
    if num_active is not None:
        q, k, v = (x[:num_active] for x in (q, k, v))
        cu, lengths = cu[: num_active + 1], lengths[:num_active]
    expected = ragged_reference(
        q,
        kc,
        vc,
        k if append else None,
        v if append else None,
        cu,
        lengths,
        table,
        causal=True,
        **kwargs,
    )
    return expected, cu, lengths


@pytest.mark.parametrize("items_per_step", [2, 3, 8])
@pytest.mark.parametrize(
    "heads,kv_heads,head_dim,dtype",
    [
        (4, 1, 256, jnp.float8_e4m3fn),
        (8, 2, 128, jnp.float8_e4m3fn),
        (8, 2, 128, jnp.bfloat16),
        (16, 2, 128, jnp.bfloat16),
    ],
)
def test_packed_decode_matches_reference(
    force_config, items_per_step, heads, kv_heads, head_dim, dtype
):
    # Note (david): (4, 1) fp8 unpacks one head per fragment, (8, 2) fp8 rides
    # a word's head pair through one fragment, (8, 2) bf16 folds two KV heads
    # into one 8-row query tile, and (16, 2) runs two head groups.
    force_config((BLOCK_KV, BLOCK_KV_COMPUTE, 2, items_per_step))
    case = decode_case(heads, kv_heads, head_dim, dtype)
    expected, cu, lengths = expected_of(case, append=True)
    actual = run_decode(case, append=True)
    assert_extend_matches(actual, expected, cu, lengths)


@pytest.mark.parametrize("window_left", [None, 200])
def test_packed_decode_window_sinks(force_config, window_left):
    force_config((BLOCK_KV, BLOCK_KV_COMPUTE, 2, 3))
    case = decode_case(8, 2, 128, jnp.bfloat16)
    sinks = jnp.asarray(
        np.random.default_rng(3).uniform(-2.0, 8.0, 8), jnp.float32
    )
    expected, cu, lengths = expected_of(
        case, append=True, window_left=window_left, sinks=sinks
    )
    actual = run_decode(
        case,
        append=True,
        window_size=(-1 if window_left is None else window_left, 0),
        learnable_sink=sinks,
    )
    assert_extend_matches(actual, expected, cu, lengths)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
def test_packed_decode_read_only_empty_row(force_config, dtype):
    # Note (david): without an append, row 0 holds no key, so its step scores
    # only masked keys; it must still return out = 0 and lse = -inf and leave
    # the next row's state untouched.
    force_config((BLOCK_KV, BLOCK_KV_COMPUTE, 2, 4))
    case = decode_case(4, 1, 256, dtype)
    expected, cu, lengths = expected_of(case, append=False)
    actual = run_decode(case, append=False)
    assert_extend_matches(actual, expected, cu, lengths)


def test_packed_decode_num_active(force_config):
    force_config((BLOCK_KV, BLOCK_KV_COMPUTE, 2, 3))
    num_active = 5
    case = decode_case(4, 1, 256, jnp.float8_e4m3fn)
    expected, cu, lengths = expected_of(
        case, append=True, num_active=num_active
    )
    # Note (david): single-token decode leaves the out and lse of rows past
    # num_active unspecified (flash_attn_with_kvcache's contract), so only the
    # active rows and the caches, which they must leave untouched, are checked.
    out, lse, updated_k, updated_v = run_decode(
        case, append=True, num_active=num_active
    )
    assert_extend_matches(
        (out[:num_active], lse[:, :num_active], updated_k, updated_v),
        expected,
        cu,
        lengths,
    )


@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize("heads,kv_heads,seqlen_q", [(32, 8, 4), (8, 1, 16)])
def test_packed_verify_window(force_config, heads, kv_heads, seqlen_q, append):
    # Note (david): a verify window folds its tokens into the query heads and
    # bounds them causally through the kernel's row key offsets.
    force_config((BLOCK_KV, BLOCK_KV_COMPUTE, 2, 3))
    q, kc, vc, k, v, cu, lengths, table = random_ragged_case(
        lengths=(seqlen_q,) * 4,
        prefixes=(128, 256 - seqlen_q + 1, 127, 200),
        heads=heads,
        kv_heads=kv_heads,
        padding=0,
    )
    batch = table.shape[0]
    num_active = 3
    active_rows = num_active * seqlen_q
    if append:
        reference_k, reference_v = k[:active_rows], v[:active_rows]
        window_k, window_v = (
            rows.reshape(batch, seqlen_q, *rows.shape[1:]) for rows in (k, v)
        )
    else:
        reference_k = reference_v = window_k = window_v = None
    expected = ragged_reference(
        q[:active_rows],
        kc,
        vc,
        reference_k,
        reference_v,
        cu[: num_active + 1],
        lengths[:num_active],
        table,
        causal=True,
    )
    out, lse, kv_cache = flash_attn_with_kvcache(
        q.reshape(batch, seqlen_q, *q.shape[1:]),
        merge_pages(kc, vc),
        None,
        window_k,
        window_v,
        block_table=table,
        cache_seqlens=lengths,
        num_active=num_active,
        causal=True,
        return_softmax_lse=True,
        interpret=INTERPRET,
    )
    actual = (
        out.reshape(q.shape),
        lse.transpose(1, 0, 2).reshape(heads, -1),
        *split_pages(kv_cache, kv_heads),
    )
    assert_extend_matches(
        actual, expected, cu[: num_active + 1], lengths[:num_active]
    )


@pytest.mark.parametrize("items_per_step", [2, 8])
def test_packed_decode_matches_one_item_kernel(force_config, items_per_step):
    case = decode_case(4, 1, 256, jnp.float8_e4m3fn)
    force_config((BLOCK_KV, BLOCK_KV_COMPUTE, 2, 1))
    one_item = run_decode(case, append=True)
    force_config((BLOCK_KV, BLOCK_KV_COMPUTE, 2, items_per_step))
    packed = run_decode(case, append=True)
    np.testing.assert_allclose(
        np.asarray(packed[0], np.float32),
        np.asarray(one_item[0], np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(packed[1], one_item[1], rtol=1e-3, atol=1e-3)
    for packed_pages, one_item_pages in zip(packed[2:], one_item[2:]):
        np.testing.assert_array_equal(
            np.asarray(packed_pages, np.float32),
            np.asarray(one_item_pages, np.float32),
        )


QWEN35_DECODE = dict(
    num_query_heads=4,
    num_kv_heads=1,
    head_dim=256,
    capacity=16128,
    kv_itemsize=1,
    page_size=256,
)


@pytest.mark.parametrize("device_name", ["TPU v6", "TPU v7", "cpu"])
def test_default_packed_config_stops_at_the_max_block(device_name):
    config = dispatch.default_packed_config(
        batch=64, device_name=device_name, **QWEN35_DECODE
    )
    items_per_step, max_block_kv = dispatch.PACKED_STEP_LIMITS.get(
        device_name, dispatch.DEFAULT_PACKED_STEP_LIMITS
    )
    assert config == (
        max_block_kv,
        max_block_kv,
        dispatch.DEFAULT_NUM_KV_STAGES,
        items_per_step,
    )


@pytest.mark.parametrize("device_name", ["TPU v6", "TPU v7"])
@pytest.mark.parametrize("batch", [1, 2, 3])
def test_default_packed_config_steps_no_wider_than_the_batch(
    device_name, batch
):
    config = dispatch.default_packed_config(
        batch=batch, device_name=device_name, **QWEN35_DECODE
    )
    if batch == 1:
        assert config is None
    else:
        assert config[3] == batch


def test_default_packed_config_fills_the_vmem_budget():
    # Note (david): 8 bf16 KV heads of 256 fill the budget below the max block.
    shape = dict(
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=256,
        kv_itemsize=2,
        batch=64,
    )
    config = dispatch.default_packed_config(
        capacity=16128, page_size=256, device_name="TPU v6", **shape
    )
    block_kv, block_kv_compute, num_kv_stages, items_per_step = config
    budget = dispatch.PACKED_VMEM_FRACTION * dispatch.vmem_limit_bytes()

    def vmem_bytes(block):
        return decode_kernel.kvcache_config_vmem_bytes(
            block, block, num_kv_stages, items_per_step, **shape
        )

    _, max_block_kv = dispatch.DEFAULT_PACKED_STEP_LIMITS
    assert block_kv_compute == block_kv < max_block_kv
    assert block_kv % dispatch.PACKED_BLOCK_KV_GRANULE == 0
    next_block_kv = block_kv + dispatch.PACKED_BLOCK_KV_GRANULE
    assert vmem_bytes(block_kv) <= budget < vmem_bytes(next_block_kv)


def test_validate_takes_a_paged_block_that_does_not_divide_capacity():
    # Note (david): 768 tokens are whole 256-token pages but not a divisor of
    # 64 pages; a paged table is padded to whole blocks.
    shape = dict(
        num_query_heads=4,
        num_kv_heads=1,
        head_dim=256,
        capacity=16384,
        kv_itemsize=1,
        batch=64,
    )
    config = (768, 768, 2, 8)
    assert (
        decode_kernel.validate_kvcache_config(config, page_size=256, **shape)
        == config
    )
    with pytest.raises(ValueError, match="multiple of page_size"):
        decode_kernel.validate_kvcache_config(
            (640, 640, 2, 8), page_size=256, **shape
        )
    with pytest.raises(ValueError, match="divisor of capacity"):
        decode_kernel.validate_kvcache_config(config, page_size=None, **shape)


def test_packed_config_rejects_bad_items_per_step(force_config):
    force_config((BLOCK_KV, BLOCK_KV_COMPUTE, 2, 0))
    case = decode_case(4, 1, 256, jnp.float8_e4m3fn)
    with pytest.raises(ValueError, match="items_per_step"):
        run_decode(case, append=True)
