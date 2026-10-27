"""A tuned-config lookup that misses on a TPU warns (warnings.warn), once per
key, and the kernel then builds with its default config."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu import flash_attn_with_kvcache, tuned_block_sizes
from flywheel_tpu.pallas import flash_fwd_kvcache
from tests.test_flash_attn import INTERPRET, generate_block_kvcache
from tests.test_util import merge_kv_heads, randn

# A TPU v7 decode entry, and the same shape at a batch with no entry.
TUNED_DECODE = dict(
    dtype=jnp.bfloat16,
    num_query_heads=32,
    num_kv_heads=4,
    head_dim=256,
    capacity=16384,
    batch=32,
    append=True,
    sliding_window=None,
    return_lse=False,
)
UNTUNED_DECODE = {**TUNED_DECODE, "batch": 48}


@pytest.fixture
def run_on(monkeypatch, recwarn):
    """run_on(device_name) makes the lookup see that device and returns a
    callable listing the fallback warnings raised since."""

    def pretend(device_name):
        monkeypatch.setattr(
            tuned_block_sizes, "get_device_name", lambda: device_name
        )
        tuned_block_sizes._warn_untuned_once.cache_clear()
        recwarn.clear()
        return lambda: [
            str(warning.message)
            for warning in recwarn
            if str(warning.message).startswith("flywheel_tpu:")
        ]

    return pretend


def test_untuned_key_warns_once_on_tpu(run_on):
    warnings = run_on("TPU v7")
    for _ in range(2):
        assert (
            tuned_block_sizes.get_tuned_kvcache_config(**UNTUNED_DECODE) is None
        )
    key = tuned_block_sizes.kvcache_config_key(**UNTUNED_DECODE)
    assert len(warnings()) == 1
    assert key in warnings()[0] and "TPU v7" in warnings()[0]


def test_tuned_key_does_not_warn(run_on):
    warnings = run_on("TPU v7")
    config = tuned_block_sizes.get_tuned_kvcache_config(**TUNED_DECODE)
    assert config == (4096, 2048, 2)
    assert not warnings()


def test_untuned_dense_key_warns_on_tpu(run_on):
    warnings = run_on("TPU v7")
    config = tuned_block_sizes.get_tuned_config(
        "fwd",
        token_major=False,
        num_heads=3,
        batch=None,
        seq_len=384,
        head_dim=128,
        head_dim_v=128,
        causal=True,
        max_seqlen=None,
        return_lse=False,
    )
    assert config is None
    assert len(warnings()) == 1 and "fwd|hm|bfloat16|h3|" in warnings()[0]


def test_untuned_key_off_tpu_does_not_warn(run_on):
    warnings = run_on("cpu")
    assert tuned_block_sizes.get_tuned_kvcache_config(**UNTUNED_DECODE) is None
    assert not warnings()


def test_tuned_block_kv_not_a_multiple_of_the_page_size_warns(
    monkeypatch, run_on
):
    warnings = run_on("TPU v7")
    monkeypatch.setattr(
        flash_fwd_kvcache, "get_tuned_kvcache_config", lambda **_: (128, 128, 2)
    )
    # The lookup runs while the kernel is traced.
    jax.clear_caches()
    rng = np.random.default_rng(0)
    _, _, block_table, k_pages, v_pages, _ = generate_block_kvcache(
        512, 256, 2, 4, 128, 128, rng, jnp.bfloat16
    )
    q = randn(rng, (2, 1, 16, 128))
    k, v = randn(rng, (2, 1, 4, 128)), randn(rng, (2, 1, 4, 128))
    out, _ = flash_attn_with_kvcache(
        q,
        merge_kv_heads(k_pages, v_pages),
        None,
        k,
        v,
        cache_seqlens=jnp.array([300, 511], jnp.int32),
        block_table=block_table,
        interpret=INTERPRET,
    )
    assert bool(jnp.all(jnp.isfinite(out.astype(jnp.float32))))
    assert len(warnings()) == 1
    assert (
        "block_kv 128 is not a multiple of the page size 256" in warnings()[0]
    )
