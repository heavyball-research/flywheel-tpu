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


@pytest.mark.parametrize(
    ("page_size", "expected"),
    (
        (16, tuple(range(128, 2049, 128))),
        (32, tuple(range(128, 2049, 128))),
        (48, (384, 768, 1152, 1536, 1920)),
        (64, tuple(range(128, 2049, 128))),
        (96, (384, 768, 1152, 1536, 1920)),
        (128, tuple(range(128, 2049, 128))),
        (160, (640, 1280, 1920)),
        (192, (384, 768, 1152, 1536, 1920)),
        (256, (128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048)),
        (384, (128, 384, 768, 1152, 1536, 1920)),
        (512, (128, 256, 512, 1024, 1536, 2048)),
        (768, (128, 256, 384, 768, 1536)),
        (1024, (128, 256, 512, 1024, 2048)),
    ),
)
def test_legal_block_kv_is_the_kernels_full_lattice(page_size, expected):
    blocks = calibration.legal_block_kv(page_size, 2048)
    assert blocks == expected
    # B_min: 128 when the page is a lane multiple, else lcm(page, 128), as
    # long as that is not past the maximum.
    assert blocks[0] == (
        128 if page_size % 128 == 0 else math.lcm(page_size, 128)
    )
    assert all(block % 128 == 0 for block in blocks)
    assert all(
        block % page_size == 0 or page_size % block == 0
        for block in blocks
    )


def test_legal_block_kv_is_empty_when_the_smallest_block_is_too_large():
    # lcm(1000, 128) = 16000 is past a 2048-token ceiling.
    assert calibration.legal_block_kv(1000, 2048) == ()


def test_legal_block_kv_rejects_a_nonpositive_page():
    with pytest.raises(ValueError):
        calibration.legal_block_kv(0, 2048)


def page_constants():
    return calibration.MlaDecodeCostConstants(
        r_item=0.0,
        r_step=0.0,
        block_efficiency=((256, 0.80), (512, 0.75)),
        max_block_kv=2048,
        base_page=128,
        page_efficiency=((1024, 512, 0.70),),
    )


def test_page_efficiency_overrides_the_base_table():
    constants_ = page_constants()
    assert constants_.efficiency_with_source(512, 1024) == (0.70, "page")
    assert constants_.efficiency_with_source(512, 128) == (0.75, "base")
    assert constants_.efficiency_with_source(512, None) == (0.75, "base")


def test_an_unmeasured_page_is_a_labeled_fallback_not_an_interpolation():
    constants_ = page_constants()
    # 256 was fitted at the base page 128 and has no page-1024 entry.
    assert constants_.efficiency_with_source(256, 1024) == (
        0.80,
        "fallback",
    )
    # 384 is in neither table: no neighbor is interpolated, it costs 1.0.
    assert constants_.efficiency_with_source(384, 1024) == (1.0, "unlisted")
    assert constants_.efficiency_with_source(384, 128) == (1.0, "unlisted")


def test_config_cost_prices_the_page_override():
    constants_ = page_constants()
    arguments = {
        "batch": 64,
        "capacity": 65536,
        "constants": constants_,
        "bases": (fixed(1024.0),),
    }
    base_page_cost = calibration.config_cost(
        512, 8, page_size=128, **arguments
    )
    override_page_cost = calibration.config_cost(
        512, 8, page_size=1024, **arguments
    )
    assert math.isclose(
        override_page_cost / base_page_cost, 0.70 / 0.75, rel_tol=1e-9
    )


def stepped_constants():
    return calibration.MlaDecodeCostConstants(
        r_item=0.0,
        r_step=50.0,
        block_efficiency=((256, 0.80), (512, 0.75)),
        max_block_kv=2048,
        base_page=128,
        block_floor=((512, 0.60),),
        page_r_step=((1024, 300.0),),
        page_floor=((1024, 512, 0.90),),
    )


def test_page_r_step_overrides_the_base_scalar():
    constants_ = stepped_constants()
    assert constants_.r_step_for_page(1024) == 300.0
    assert constants_.r_step_for_page(512) == 50.0
    assert constants_.r_step_for_page(None) == 50.0


def test_page_floor_overrides_the_base_floor():
    constants_ = stepped_constants()
    assert constants_.floor_fraction(512, 1024) == 0.90
    assert constants_.floor_fraction(512, 128) == 0.60
    assert constants_.floor_fraction(512, None) == 0.60
    # No (page, block) entry and no base entry: fully charged.
    assert constants_.floor_fraction(256, 1024) == 1.0


def test_config_cost_prices_the_page_r_step():
    constants_ = stepped_constants()
    # 64 requests of exactly 1024 tokens make 2 full 512-token items each:
    # 128 items, 16 steps at items_per_step=8, and no floor is charged.
    arguments = {
        "batch": 64,
        "capacity": 65536,
        "constants": constants_,
        "bases": (fixed(1024.0),),
    }
    base_cost = calibration.config_cost(512, 8, page_size=128, **arguments)
    page_cost = calibration.config_cost(512, 8, page_size=1024, **arguments)
    assert math.isclose(
        page_cost - base_cost, 16 * (300.0 - 50.0), rel_tol=1e-9
    )


def test_config_cost_prices_the_page_floor():
    constants_ = dataclasses.replace(stepped_constants(), page_r_step=())
    # 64 requests of 768 tokens: one full 512 block and one half block, so
    # the expected work is (1 + f) / 2 and the floor is charged.
    arguments = {
        "batch": 64,
        "capacity": 65536,
        "constants": constants_,
        "bases": (fixed(768.0),),
    }
    base_cost = calibration.config_cost(512, 8, page_size=128, **arguments)
    page_cost = calibration.config_cost(512, 8, page_size=1024, **arguments)
    # 128 items x 512 tokens x efficiency 0.75 x work (0.95 - 0.80).
    assert math.isclose(
        page_cost - base_cost, 0.75 * 512 * 128 * 0.15, rel_tol=1e-9
    )


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
    assert config == kernel.MlaDecodeConfig(256, 3)


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
    block_cost = {
        128: 10.0,
        256: 1.0,
        384: 100.0,
        512: 5.0,
        640: 100.0,
        1024: 20.0,
        2048: 40.0,
    }

    def cost(block, items_per_step, **kwargs):
        return block_cost.get(block, 100.0) + items_per_step

    monkeypatch.setattr(kernel, "config_cost", cost)
    config = kernel.default_mla_config(**shape(jnp))
    # Items 1 and 2 are pruned when the cap allows 4, so 256's cheapest is 3.
    assert (config.block_kv, config.items_per_step) == (256, 3)


def _record_pricing(monkeypatch, kernel, constants_):
    calls = []

    def cost(block, items_per_step, **kwargs):
        calls.append((block, items_per_step))
        return 1.0

    monkeypatch.setattr(kernel, "get_tuned_mla_config", lambda **kwargs: None)
    monkeypatch.setattr(
        kernel, "get_mla_cost_constants", lambda chip, dtype: constants_
    )
    monkeypatch.setattr(kernel, "mla_vmem_bytes", lambda *args, **kwargs: 0)
    monkeypatch.setattr(kernel, "mla_smem_bytes", lambda *args, **kwargs: 0)
    monkeypatch.setattr(kernel, "config_cost", cost)
    return calls


def test_items_1_and_2_are_pruned_when_the_cap_allows_four(monkeypatch):
    kernel, jnp = kernel_and_jnp()
    calls = _record_pricing(monkeypatch, kernel, constants())
    kernel.default_mla_config(**shape(jnp))
    blocks = {block for block, _ in calls}
    assert 384 in blocks and 640 in blocks
    # TARGET_STEP_TOKENS // 2048 = 8, so every candidate's cap is >= 4.
    assert all(items >= 3 for _, items in calls)
    assert (2048, 8) in calls
    assert not any(items in (1, 2) for _, items in calls)


def test_items_1_and_2_are_kept_when_the_cap_is_shorter(monkeypatch):
    kernel, jnp = kernel_and_jnp()
    constants_ = dataclasses.replace(constants(), max_block_kv=8192)
    calls = _record_pricing(monkeypatch, kernel, constants_)
    kernel.default_mla_config(**shape(jnp))
    # 16384 // 8192 = 2, so the cap is 2 and the rule keeps 1..2.
    assert {items for block, items in calls if block == 8192} == {1, 2}
    # 16384 // 4096 = 4 keeps 3..4; below 4096 the cap is 4 or more.
    assert {items for block, items in calls if block == 4096} == {3, 4}
    assert all(items >= 3 for block, items in calls if block <= 4096)


def test_heuristic_can_choose_a_non_power_of_two_block_and_item(monkeypatch):
    """The candidate set is the kernel's lattice, not powers of two."""
    kernel, jnp = kernel_and_jnp()
    wanted = (384, 5)

    def cost(block, items_per_step, **kwargs):
        return 0.0 if (block, items_per_step) == wanted else 1.0

    monkeypatch.setattr(kernel, "get_tuned_mla_config", lambda **kwargs: None)
    monkeypatch.setattr(
        kernel, "get_mla_cost_constants", lambda chip, dtype: constants()
    )
    monkeypatch.setattr(kernel, "config_cost", cost)
    config = kernel.default_mla_config(**shape(jnp))
    assert (config.block_kv, config.items_per_step) == wanted


def test_heuristic_prices_blocks_on_the_calls_page(monkeypatch):
    kernel, jnp = kernel_and_jnp()
    seen = []

    def cost(block, items_per_step, **kwargs):
        seen.append(kwargs.get("page_size"))
        return 1.0

    monkeypatch.setattr(kernel, "get_tuned_mla_config", lambda **kwargs: None)
    monkeypatch.setattr(
        kernel, "get_mla_cost_constants", lambda chip, dtype: constants()
    )
    monkeypatch.setattr(kernel, "mla_vmem_bytes", lambda *args, **kwargs: 0)
    monkeypatch.setattr(kernel, "mla_smem_bytes", lambda *args, **kwargs: 0)
    monkeypatch.setattr(kernel, "config_cost", cost)
    kernel.default_mla_config(**shape(jnp))
    assert seen and set(seen) == {128}


def test_unlisted_blocks_are_not_candidates(monkeypatch):
    """A block with no measured constants is not priced at placeholder 1.0."""
    kernel, jnp = kernel_and_jnp()
    constants_ = dataclasses.replace(
        constants(), block_efficiency=((256, 0.5),)
    )
    calls = []

    def cost(block, items_per_step, **kwargs):
        calls.append(block)
        # Make every unlisted block look cheaper, so only the filter keeps
        # the heuristic from choosing one.
        return 0.0 if block != 256 else 1.0

    monkeypatch.setattr(kernel, "get_tuned_mla_config", lambda **kwargs: None)
    monkeypatch.setattr(
        kernel, "get_mla_cost_constants", lambda chip, dtype: constants_
    )
    monkeypatch.setattr(kernel, "mla_vmem_bytes", lambda *args, **kwargs: 0)
    monkeypatch.setattr(kernel, "mla_smem_bytes", lambda *args, **kwargs: 0)
    monkeypatch.setattr(kernel, "config_cost", cost)
    config = kernel.default_mla_config(**shape(jnp))
    assert config.block_kv == 256
    assert set(calls) == {256}


def test_a_dormant_constants_table_keeps_every_block(monkeypatch):
    """With no measured entries at all, the lattice is priced as before."""
    kernel, jnp = kernel_and_jnp()
    calls = _record_pricing(monkeypatch, kernel, constants())
    kernel.default_mla_config(**shape(jnp))
    assert 384 in {block for block, _ in calls}
    assert 2048 in {block for block, _ in calls}


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


def test_calibration_wins_over_a_tuned_entry(monkeypatch):
    """The calibrated model replaces the tuned table where constants exist."""
    kernel, jnp = kernel_and_jnp()
    monkeypatch.setattr(
        kernel, "get_mla_cost_constants", lambda chip, dtype: constants()
    )
    monkeypatch.setattr(
        kernel, "config_cost", lambda block, items_per_step, **kwargs: block
    )

    def fail(**kwargs):
        raise AssertionError("calibration must win over the tuned table")

    monkeypatch.setattr(kernel, "get_tuned_mla_config", fail)
    config = kernel.default_mla_config(**shape(jnp))
    assert (config.block_kv, config.items_per_step) == (128, 3)


def test_a_tuned_entry_serves_an_uncalibrated_dtype(monkeypatch):
    kernel, jnp = kernel_and_jnp()
    monkeypatch.setattr(
        kernel, "get_tuned_mla_config", lambda **kwargs: (1024, 8)
    )
    monkeypatch.setattr(
        kernel, "get_mla_cost_constants", lambda chip, dtype: None
    )
    config = kernel.default_mla_config(**shape(jnp))
    assert config.block_kv == 1024


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
