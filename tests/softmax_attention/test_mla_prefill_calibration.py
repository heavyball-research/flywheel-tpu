"""The MLA prefill workload prior, cost model, candidate domain, and config.

The prior, cost, and lattice tests are pure Python and run anywhere. The
config tests import jax and the kernel, so they skip when jax is not
installed.
"""

import math

import pytest

from flywheel_tpu.softmax_attention import (
    mla_prefill_calibration as calibration,
)

HEADS = 32
BLOCKS = (128, 256, 512, 1024, 2048)
GROUPS = (64, 128, 256, 512, 1024)
WIDTHS = (32, 64, 96, 128, 256)


def constants(**overrides):
    values = dict(
        c_blocks=0.00027,
        c_rows=0.0,
        c_qcopy=4.2e-7,
        c_item_rows=7.1e-7,
        block_efficiency=tuple((block, 2e-9) for block in BLOCKS),
        group_efficiency=tuple((group, 1.0) for group in GROUPS),
        block_q_efficiency=tuple((width, 1.0) for width in WIDTHS),
        block_qs=WIDTHS,
        large_kv_block_qs=(128, 256),
        group_rows_set=GROUPS,
        max_block_kv=2048,
        calibrated_on="test",
    )
    values.update(overrides)
    return calibration.MlaPrefillCostConstants(**values)


def workload(**overrides):
    values = dict(
        batch=4,
        num_tokens=8192,
        max_seqlen_q=2048,
        max_seqlen_k=2048,
        capacity=16384,
    )
    values.update(overrides)
    return calibration.MlaPrefillWorkload(**values)


def test_prior_weights_sum_to_one():
    assert math.isclose(
        sum(base.weight for base in calibration.WORKLOAD_PRIOR), 1.0
    )
    for base in calibration.WORKLOAD_PRIOR:
        assert math.isclose(
            sum(component.weight for component in base.components), 1.0
        )


def test_samples_sum_to_one_and_match_the_mean_chunk():
    for batch, tokens, max_q, max_k in [
        (4, 8192, 2048, 2048),
        (4, 2048, 512, 8192),
        (8, 2048, 256, 4096),
        (1, 16384, 16384, 16384),
        (2, 16384, 8192, 32768),
    ]:
        samples = calibration.workload_samples(
            workload(
                batch=batch,
                num_tokens=tokens,
                max_seqlen_q=max_q,
                max_seqlen_k=max_k,
                capacity=65536,
            )
        )
        assert math.isclose(sum(weight for weight, _, _ in samples), 1.0)
        mean_chunk = sum(weight * q for weight, q, _ in samples)
        assert math.isclose(mean_chunk, tokens / batch, rel_tol=1e-9)


def test_samples_respect_the_maxima_and_capacity():
    samples = calibration.workload_samples(
        workload(
            batch=4,
            num_tokens=2048,
            max_seqlen_q=512,
            max_seqlen_k=8192,
            capacity=4096,
        )
    )
    for _, q, n in samples:
        assert 0 < q <= 512
        assert q <= n <= 4096
        assert n <= 8192


def test_samples_are_deterministic():
    first = calibration.workload_samples(workload())
    second = calibration.workload_samples(workload())
    assert first == second


def test_a_mean_chunk_past_the_maximum_is_rejected():
    with pytest.raises(ValueError):
        calibration.workload_samples(
            workload(num_tokens=8192, batch=1, max_seqlen_q=2048)
        )
    with pytest.raises(ValueError):
        calibration.workload_samples(workload(batch=0))


def test_row_groups_for_is_the_kernel_domain():
    assert calibration.row_groups_for(128, 256, 32) == 16
    assert calibration.row_groups_for(256, 256, 32) == 32
    assert calibration.row_groups_for(64, 256, 32) == 8
    # group_rows must divide block_q * num_heads and the quotient must
    # divide block_q.
    assert calibration.row_groups_for(100, 256, 32) is None
    assert calibration.row_groups_for(128, 100, 32) is None
    assert calibration.row_groups_for(256, 8192, 32) == 1
    assert calibration.row_groups_for(256, 16384, 32) is None
    assert calibration.row_groups_for(0, 256, 32) is None


def test_request_terms_prices_the_live_tile_path():
    # A whole prompt of 2048 tokens in q blocks of 128: with block_kv 2048,
    # every q block is one live-tiled item, so the computed columns are
    # 256 * ceil(q_block_end / 256) and there is one item a block.
    blocks, columns, items, q_tokens = calibration.request_terms(
        128, 2048, 2048.0, 2048.0
    )
    expected = sum(
        256 * math.ceil(128 * (index + 1) / 256) for index in range(16)
    )
    assert blocks == 16
    assert columns == expected
    assert items == 16
    assert q_tokens == 2048


def test_request_terms_uses_full_blocks_below_one_tile():
    # block_kv 256 is not above one live tile, so no item can skip dead
    # tiles: every item computes its whole block.
    blocks, columns, items, q_tokens = calibration.request_terms(
        128, 256, 512.0, 512.0
    )
    assert blocks == 4
    assert items == 6
    assert columns == 4 * 256 + 2 * 256
    assert q_tokens == 512


def test_candidate_lattice_is_page_compatible_and_measured():
    lattice = calibration.candidate_lattice(
        page_size=512,
        max_seqlen_k=8192,
        capacity=16384,
        constants=constants(),
        num_heads=HEADS,
    )
    assert lattice
    for block_q, block_kv, row_groups in lattice:
        assert block_q in WIDTHS
        if block_kv >= 1024:
            assert block_q in (128, 256)
        assert block_kv % 128 == 0
        assert block_kv % 512 == 0 or 512 % block_kv == 0
        assert block_q % row_groups == 0
        group_rows = (block_q // row_groups) * HEADS
        assert group_rows in GROUPS


def test_candidate_lattice_keeps_only_measured_blocks():
    narrow = constants(block_efficiency=((128, 2e-9), (512, 2e-9)))
    lattice = calibration.candidate_lattice(
        page_size=128,
        max_seqlen_k=8192,
        capacity=16384,
        constants=narrow,
        num_heads=HEADS,
    )
    assert {block_kv for _, block_kv, _ in lattice} == {128, 512}


def test_width_block_kv_caps_are_respected():
    capped = constants(width_block_kv_caps=((256, 512),))
    lattice = calibration.candidate_lattice(
        page_size=128,
        max_seqlen_k=8192,
        capacity=16384,
        constants=capped,
        num_heads=HEADS,
    )
    wide = [cell for cell in lattice if cell[0] == 256]
    assert wide
    assert all(block_kv <= 512 for _, block_kv, _ in wide)
    assert any(block_kv == 512 for _, block_kv, _ in wide)
    assert any(block_kv > 512 for _, block_kv, _ in lattice)


def test_config_cost_follows_the_efficiency_factors():
    samples = calibration.workload_samples(workload())
    base = constants()
    cheap_256 = constants(
        block_efficiency=((128, 2e-9), (256, 1e-9), (512, 2e-9),
                          (1024, 2e-9), (2048, 2e-9))
    )
    cheap_1024 = constants(
        block_efficiency=((128, 2e-9), (256, 2e-9), (512, 2e-9),
                          (1024, 1e-9), (2048, 2e-9))
    )
    arguments = dict(samples=samples, num_heads=HEADS)
    assert calibration.config_cost(
        128, 256, 256, constants=cheap_256, **arguments
    ) < calibration.config_cost(
        128, 256, 256, constants=cheap_1024, **arguments
    )
    assert calibration.config_cost(
        128, 1024, 256, constants=cheap_1024, **arguments
    ) < calibration.config_cost(
        128, 1024, 256, constants=cheap_256, **arguments
    )
    # A group factor of one is cheaper than a factor of two.
    slow_group = constants(group_efficiency=tuple(
        (group, 2.0 if group == 256 else 1.0) for group in GROUPS
    ))
    assert calibration.config_cost(
        128, 512, 256, constants=base, **arguments
    ) < calibration.config_cost(
        128, 512, 256, constants=slow_group, **arguments
    )
    # A block_q factor of one is cheaper than a factor of two.
    slow_width = constants(
        block_q_efficiency=tuple(
            (width, 2.0 if width == 128 else 1.0) for width in WIDTHS
        )
    )
    assert calibration.config_cost(
        128, 512, 256, constants=base, **arguments
    ) < calibration.config_cost(
        128, 512, 256, constants=slow_width, **arguments
    )


def test_config_cost_terms_reconstruct_the_cost():
    samples = calibration.workload_samples(workload())
    base = constants()
    work, blocks, rows, q_tokens, item_rows = (
        calibration.config_cost_terms(
            128, 1024, 256, samples=samples, constants=base, num_heads=HEADS
        )
    )
    rebuilt = (
        work
        + base.c_blocks * blocks
        + base.c_rows * rows
        + base.c_qcopy * q_tokens
        + base.c_item_rows * item_rows
    )
    assert math.isclose(
        rebuilt,
        calibration.config_cost(
            128, 1024, 256, samples=samples, constants=base,
            num_heads=HEADS,
        ),
        rel_tol=1e-12,
    )


def test_missing_constants_are_the_dormant_state(monkeypatch):
    monkeypatch.setattr(calibration, "CALIBRATION", {})
    assert calibration.get_mla_prefill_cost_constants(
        "TPU v6e", "bfloat16"
    ) is None


def test_shipped_v6e_entry_covers_the_measured_domain():
    entry = calibration.get_mla_prefill_cost_constants(
        "TPU v6e", "bfloat16"
    )
    assert entry is not None
    assert entry.calibrated_on
    assert entry.max_block_kv == calibration.MAX_BLOCK_KV
    assert entry.max_block_kv == 2048
    # Every 128-token block through the measured top is priced; the 1536+
    # cliff is in the table rather than hidden behind a default.
    assert {
        block for block, _ in entry.block_efficiency
    } == set(range(128, 2049, 128))
    assert dict(entry.group_efficiency)[256] == 1.0
    assert set(entry.group_rows_set) == set(GROUPS)
    assert entry.width_block_kv_cap(256) == 512
    assert entry.width_block_kv_cap(192) == 640
    assert entry.width_block_kv_cap(128) is None

    lattice = calibration.candidate_lattice(
        page_size=128,
        max_seqlen_k=8192,
        capacity=16384,
        constants=entry,
        num_heads=HEADS,
    )
    widths_at = {
        block_kv: {block_q for block_q, block, _ in lattice
                   if block == block_kv}
        for block_kv in {block for _, block, _ in lattice}
    }
    assert 768 in widths_at and 128 in widths_at[768]
    # The width caps keep the slow sideways widths off the large blocks.
    assert widths_at[512] >= {128, 192, 256}
    assert not any(width > 128 for width in widths_at[1024])
    assert 192 not in widths_at[768]
    assert 192 not in widths_at[896]
    assert 256 not in widths_at[640]
    # The group domain stays complete, so the VMEM filter can trade rows.
    for block_q, block_kv, row_groups in lattice:
        assert (block_q // row_groups) * HEADS in GROUPS


def kernel_and_jnp():
    jnp = pytest.importorskip("jax.numpy")
    kernel = pytest.importorskip(
        "flywheel_tpu.softmax_attention.kernels.mla_prefill"
    )
    return kernel, jnp


def calibrated(monkeypatch, constants_value):
    kernel, jnp = kernel_and_jnp()
    monkeypatch.setitem(
        calibration.CALIBRATION, "TPU v6e",
        {"bfloat16": constants_value},
    )
    return kernel, jnp


def test_calibrated_config_is_feasible(monkeypatch):
    kernel, jnp = calibrated(
        monkeypatch,
        constants(
            block_qs=(32, 64, 128, 256),
            block_q_efficiency=tuple(
                (width, 1.0) for width in (32, 64, 128, 256)
            ),
        ),
    )
    config = kernel.default_mla_prefill_config(
        num_heads=HEADS,
        page_size=512,
        pages_per_seq=64,
        cache_dtype=jnp.bfloat16,
        return_lse=False,
        max_seqlen_q=2048,
        max_seqlen_k=8192,
        batch=4,
        num_tokens=8192,
    )
    assert isinstance(config, kernel.MlaPrefillConfig)
    assert config.block_q in (32, 64, 128, 256)
    assert config.block_kv in BLOCKS
    assert config.block_q % config.row_groups == 0
    group_rows = (config.block_q // config.row_groups) * HEADS
    assert group_rows in GROUPS
    vmem = kernel.mla_prefill_vmem_bytes(
        config,
        num_heads=HEADS,
        cache_dtype=jnp.bfloat16,
        return_lse=False,
    )
    assert vmem <= 0.75 * kernel.vmem_limit_bytes()


def test_calibrated_config_is_deterministic(monkeypatch):
    kernel, jnp = calibrated(monkeypatch, constants())
    arguments = dict(
        num_heads=HEADS,
        page_size=128,
        pages_per_seq=128,
        cache_dtype=jnp.bfloat16,
        return_lse=False,
        max_seqlen_q=1024,
        max_seqlen_k=8192,
        batch=2,
        num_tokens=2048,
    )
    first = kernel.default_mla_prefill_config(**arguments)
    second = kernel.default_mla_prefill_config(**arguments)
    assert first == second


def test_an_uncalibrated_call_keeps_the_legacy_config(monkeypatch):
    kernel, jnp = kernel_and_jnp()
    monkeypatch.delitem(calibration.CALIBRATION, "TPU v6e", raising=False)
    arguments = dict(
        num_heads=HEADS,
        page_size=512,
        pages_per_seq=64,
        cache_dtype=jnp.bfloat16,
        return_lse=False,
        max_seqlen_q=2048,
        max_seqlen_k=8192,
    )
    inherited = kernel.default_mla_prefill_config(**arguments)
    legacy = kernel._legacy_default_mla_prefill_config(**arguments)
    assert inherited == legacy
    # The legacy rule also serves a call that gives no batch or token count.
    assert kernel.default_mla_prefill_config(
        **arguments, batch=4, num_tokens=8192
    ) == legacy
