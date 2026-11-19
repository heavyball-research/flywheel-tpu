"""The MLA decode length prior, cost model, and its uncalibrated fallback.

The prior and cost tests are pure Python and run anywhere. The config tests
import jax and the kernel, so they skip when jax is not installed.
"""

import dataclasses
import math
import warnings

import pytest

from flywheel_tpu.softmax_attention import mla_decode_calibration as calibration


def kernel_and_jnp():
    jnp = pytest.importorskip("jax.numpy")
    kernel = pytest.importorskip(
        "flywheel_tpu.softmax_attention.kernels.mla_decode"
    )
    return kernel, jnp


def shape(jnp):
    return {
        "batch": 64,
        "page_size": 128,
        "pages_per_seq": 256,
        "num_query_rows": 128,
        "query_tokens": 1,
        "cache_dtype": jnp.bfloat16,
        "return_lse": False,
        "max_seqlen_k": 8192,
    }


def constants():
    return calibration.MlaDecodeCostConstants(
        r_item=50.0,
        r_step=50.0,
        block_efficiency=(),
        max_block_kv=2048,
        vmem_fraction=0.75,
        calibrated_on="test",
    )


def fixed(mean):
    return calibration.LengthBase(
        "fixed",
        1.0,
        mean,
        (calibration.LengthComponent(1.0, mean, cv=0.0),),
    )


def argmin_block(bases):
    best = min(
        (
            calibration.config_cost(
                block,
                items_per_step,
                batch=64,
                capacity=65536,
                constants=constants(),
                bases=bases,
            ),
            block,
            items_per_step,
        )
        for block in (128, 256, 512, 1024, 2048)
        for items_per_step in (1, 2, 4, 8)
    )
    return best[1]


def test_prior_weights_sum_to_one():
    assert math.isclose(
        sum(base.weight for base in calibration.WORKLOAD_PRIOR),
        1.0,
        rel_tol=1e-9,
    )


def test_prior_component_means_match_the_base_mean():
    for base in calibration.WORKLOAD_PRIOR:
        mean = sum(c.weight * c.mean for c in base.components)
        assert math.isclose(mean, base.mean, rel_tol=1e-9), base.name


def test_lognormal_quantiles_match_their_mean():
    quantiles = calibration.component_quantiles(
        calibration.LengthComponent(1.0, 1024.0, cv=1.0),
        capacity=10**9,
        count=512,
    )
    assert math.isclose(
        sum(quantiles) / len(quantiles), 1024.0, rel_tol=0.05
    )


def test_expected_items_of_a_fixed_length():
    assert fixed(600.0).expected_ceil(512, capacity=10**9, count=32) == 2.0


def test_expected_items_fall_as_the_block_grows():
    base = fixed(1000.0)
    values = [
        base.expected_ceil(block, capacity=10**9, count=32)
        for block in (128, 256, 512, 1024, 2048)
    ]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_short_lengths_prefer_a_smaller_block():
    assert argmin_block((fixed(256.0),)) < argmin_block((fixed(4096.0),))


def test_frequent_lengths_dominate_the_choice():
    short = fixed(256.0)
    long = fixed(4096.0)
    short_heavy = (
        calibration.LengthBase("short", 0.9, 256.0, short.components),
        calibration.LengthBase("long", 0.1, 4096.0, long.components),
    )
    long_heavy = (
        calibration.LengthBase("short", 0.1, 256.0, short.components),
        calibration.LengthBase("long", 0.9, 4096.0, long.components),
    )
    assert argmin_block(short_heavy) < argmin_block(long_heavy)


def test_item_work_fraction_holds_a_floor_and_ramps():
    assert math.isclose(calibration.item_work_fraction(0.1, 0.6), 0.6)
    assert math.isclose(calibration.item_work_fraction(0.5, 0.6), 0.6)
    assert math.isclose(calibration.item_work_fraction(0.75, 0.6), 0.8)
    assert math.isclose(calibration.item_work_fraction(1.0, 0.6), 1.0)
    assert calibration.item_work_fraction(0.1, 1.0) == 1.0


def test_expected_work_fraction_averages_over_a_requests_items():
    # Two items, one full and one half: (1 + 0.6) / 2.
    assert math.isclose(
        fixed(768.0).expected_work_fraction(
            512, 0.6, capacity=10**9, count=32
        ),
        0.8,
    )
    assert math.isclose(
        fixed(256.0).expected_work_fraction(
            512, 0.6, capacity=10**9, count=32
        ),
        0.6,
    )
    assert fixed(1024.0).expected_work_fraction(
        512, 0.6, capacity=10**9, count=32
    ) == 1.0


def test_a_floor_of_one_is_the_full_block_model():
    plain = constants()
    floored = dataclasses.replace(plain, block_floor=((512, 1.0),))
    arguments = {"batch": 64, "capacity": 65536, "bases": (fixed(1000.0),)}
    assert calibration.config_cost(
        512, 4, constants=plain, **arguments
    ) == calibration.config_cost(512, 4, constants=floored, **arguments)


def test_a_floor_cheapens_a_large_block_for_short_requests():
    plain = constants()
    floored = dataclasses.replace(
        plain,
        block_floor=(
            (128, 0.92),
            (256, 0.88),
            (512, 0.80),
            (1024, 0.70),
            (2048, 0.63),
        ),
    )
    arguments = {"batch": 64, "capacity": 65536, "bases": (fixed(256.0),)}
    plain_ratio = calibration.config_cost(
        1024, 8, constants=plain, **arguments
    ) / calibration.config_cost(256, 8, constants=plain, **arguments)
    floored_ratio = calibration.config_cost(
        1024, 8, constants=floored, **arguments
    ) / calibration.config_cost(256, 8, constants=floored, **arguments)
    assert floored_ratio < plain_ratio


def test_missing_constants_fall_back_to_the_legacy_config(monkeypatch):
    kernel, jnp = kernel_and_jnp()
    monkeypatch.setattr(
        kernel, "get_tuned_mla_config", lambda **kwargs: None
    )
    monkeypatch.setattr(
        kernel, "get_mla_cost_constants", lambda chip, dtype: None
    )
    args = shape(jnp)
    assert kernel.default_mla_config(**args) == (
        kernel._legacy_default_mla_config(**args)
    )


def test_calibrated_config_is_feasible(monkeypatch):
    kernel, jnp = kernel_and_jnp()
    monkeypatch.setattr(
        kernel, "get_tuned_mla_config", lambda **kwargs: None
    )
    monkeypatch.setattr(
        kernel, "get_mla_cost_constants", lambda chip, dtype: constants()
    )
    args = shape(jnp)
    config = kernel.default_mla_config(**args)
    assert config.block_kv % kernel.NUM_LANES == 0
    assert config.block_kv <= constants().max_block_kv
    assert 1 <= config.items_per_step <= kernel.MAX_ITEMS_PER_STEP
    budget = int(constants().vmem_fraction * kernel.vmem_limit_bytes())
    assert (
        kernel.mla_vmem_bytes(
            config,
            batch=args["batch"],
            num_query_rows=args["num_query_rows"],
            query_tokens=args["query_tokens"],
            cache_dtype=args["cache_dtype"],
            return_lse=args["return_lse"],
        )
        <= budget
    )
    smem_budget = kernel.SMEM_BUDGET_FRACTION * kernel.smem_limit_bytes()
    assert (
        kernel.mla_smem_bytes(
            config,
            batch=args["batch"],
            page_size=args["page_size"],
            pages_per_seq=args["pages_per_seq"],
        )
        <= smem_budget
    )


def test_calibrated_config_respects_the_smem_budget(monkeypatch):
    """A cheaper block whose schedule and table outgrow SMEM is skipped."""
    kernel, jnp = kernel_and_jnp()
    monkeypatch.setattr(
        kernel, "get_tuned_mla_config", lambda **kwargs: None
    )
    monkeypatch.setattr(
        kernel, "get_mla_cost_constants", lambda chip, dtype: constants()
    )
    monkeypatch.setattr(
        kernel, "config_cost", lambda block, items_per_step, **kwargs: block
    )
    args = dict(shape(jnp))
    args.update(pages_per_seq=1250, max_seqlen_k=160000)
    budget = kernel.SMEM_BUDGET_FRACTION * kernel.smem_limit_bytes()
    small = kernel.MlaDecodeConfig(128, 1)
    assert (
        kernel.mla_smem_bytes(
            small,
            batch=args["batch"],
            page_size=args["page_size"],
            pages_per_seq=args["pages_per_seq"],
        )
        > budget
    )
    config = kernel.default_mla_config(**args)
    assert config == kernel.MlaDecodeConfig(256, 1)


def test_mla_smem_bytes_counts_the_schedule_and_table():
    kernel, _ = kernel_and_jnp()
    # 64 requests of 1250 pages: 64 * 1250 schedule entries, then 64 * 1250
    # block-table words, then 64 kv_lens words.
    assert kernel.mla_smem_bytes(
        kernel.MlaDecodeConfig(128, 8),
        batch=64,
        page_size=128,
        pages_per_seq=1250,
    ) == 4 * (80000 + 1 + 64 + 80000)


def test_calibrated_config_follows_the_cost_model(monkeypatch):
    kernel, jnp = kernel_and_jnp()
    monkeypatch.setattr(
        kernel, "get_tuned_mla_config", lambda **kwargs: None
    )
    monkeypatch.setattr(
        kernel, "get_mla_cost_constants", lambda chip, dtype: constants()
    )
    block_cost = {128: 10.0, 256: 1.0, 512: 5.0, 1024: 20.0, 2048: 40.0}

    def cost(block, items_per_step, **kwargs):
        return block_cost[block] + items_per_step

    monkeypatch.setattr(kernel, "config_cost", cost)
    config = kernel.default_mla_config(**shape(jnp))
    assert (config.block_kv, config.items_per_step) == (256, 1)


def test_single_request_many_rows_skips_calibration(monkeypatch):
    kernel, jnp = kernel_and_jnp()
    monkeypatch.setattr(
        kernel, "get_tuned_mla_config", lambda **kwargs: None
    )

    def fail(chip, dtype):
        raise AssertionError("a single-request decode must not calibrate")

    monkeypatch.setattr(kernel, "get_mla_cost_constants", fail)
    args = dict(shape(jnp))
    args.update(batch=1, pages_per_seq=8, max_seqlen_k=1024)
    config = kernel.default_mla_config(**args)
    assert (config.block_kv, config.items_per_step) == (1024, 1)


def test_a_tuned_entry_skips_calibration(monkeypatch):
    kernel, jnp = kernel_and_jnp()
    monkeypatch.setattr(
        kernel, "get_tuned_mla_config", lambda **kwargs: (1024, 8)
    )

    def fail(chip, dtype):
        raise AssertionError("a tuned entry must not consult calibration")

    monkeypatch.setattr(kernel, "get_mla_cost_constants", fail)
    config = kernel.default_mla_config(**shape(jnp))
    assert config.block_kv % kernel.NUM_LANES == 0


def test_uncalibrated_dtype_warns_once(monkeypatch):
    monkeypatch.setattr(
        calibration,
        "CALIBRATION",
        {"TPU v6e": {"bfloat16": constants()}},
    )
    monkeypatch.setattr(calibration, "_WARNED", set())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert (
            calibration.get_mla_cost_constants("TPU v6e", "float8_e4m3fn")
            is None
        )
    assert len(caught) == 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibration.get_mla_cost_constants("TPU v6e", "float8_e4m3fn")
    assert not caught


def test_no_warning_while_the_table_is_dormant(monkeypatch):
    monkeypatch.setattr(calibration, "CALIBRATION", {})
    monkeypatch.setattr(calibration, "_WARNED", set())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert (
            calibration.get_mla_cost_constants("TPU v6e", "bfloat16") is None
        )
    assert not caught
