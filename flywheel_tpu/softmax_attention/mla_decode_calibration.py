"""Host-side cost constants and length prior for MLA decode block sizing.

A build cannot know the batch's sequence lengths: they are runtime data. What
it can know is the shape (batch, heads, dim, dtype, page size) and the device.
The heuristic therefore picks the block that minimizes the *weighted average
time* over a prior of length distributions, not the time of one length.

The prior below is a placeholder mix informed by published traces: input
lengths are heavy-tailed (ServeGen fits a Pareto/log-normal mixture; BurstGPT
reports a median near 218 tokens and a P95 near 2147). No paper prescribes a
universal mixture; the work of one deployment differs from the next, so
WORKLOAD_PRIOR is meant to be replaced by that deployment's own statistics.

The cost model converts every term to computed columns:

  C = efficiency(B) * work(B) * steps * items_per_step * B
      + r_item * steps * items_per_step
      + r_step * steps

work(B) is the prior-average fraction of a full block's per-item work. On
v6e an item that holds x = valid / block of its block still pays a floor
fraction floor(B), and the rest of the work ramps in over the upper half of
the block: g(x) = floor(B) + (1 - floor(B)) * clamp(2x - 1, 0, 1). floor(B)
is 1.0 where unlisted, which is the old full-block model; a calibrated entry
carries the measured floor per candidate block.

r_item and r_step are the per-item and per-step fixed costs in column
equivalents. They are hardware constants: CALIBRATION holds one entry per
(chip, cache dtype), the calibration script measures them, and a build with no
entry falls back to the kernel's max_seqlen-based rule with a warning.

Kernel time identifies only the per-item cost k(B) = efficiency(B) * B +
r_item and the per-step cost r_step, not the two terms of k(B) separately:
every computed item slot is exactly B columns wide. Cost predictions are
unchanged by moving cost between efficiency(B) and r_item, so the calibration
fixes r_item = 0 and folds the item's whole fixed cost into efficiency(B).

A batch sums many independent requests, so its item count concentrates near
batch * E[ceil(w / B)]; the cost evaluates the step count at that mean rather
than the exact expectation of ceil(items / items_per_step). A batch of one
request is the case where the two differ.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
import warnings


_NORMAL = statistics.NormalDist()


def item_work_fraction(
    valid_fraction: float, floor_fraction: float
) -> float:
    """The work of an item holding x of its block, in full blocks.

    Measured on v6e, an item pays floor_fraction even at a tiny valid length,
    and the remaining work ramps in linearly over the upper half of the block.
    A floor of 1.0 is the full-block model.
    """
    if floor_fraction >= 1.0:
        return 1.0
    ramp = min(max(2.0 * valid_fraction - 1.0, 0.0), 1.0)
    return floor_fraction + (1.0 - floor_fraction) * ramp


def component_quantiles(
    component: LengthComponent, *, capacity: int, count: int = 128
) -> tuple[float, ...]:
    """A component's lengths at the midpoints of count equal-mass bins.

    Clipped to [1, capacity], the largest length the block table can hold.
    """
    if component.pareto_alpha:
        alpha = component.pareto_alpha
        if alpha <= 1.0:
            raise ValueError(f"Pareto alpha must exceed 1; got {alpha}.")
        scale = component.mean * (alpha - 1.0) / alpha
        values = [
            scale * (1.0 - (index + 0.5) / count) ** (-1.0 / alpha)
            for index in range(count)
        ]
    else:
        variance = math.log1p(component.cv**2)
        location = math.log(component.mean) - 0.5 * variance
        spread = math.sqrt(variance)
        values = [
            math.exp(
                location
                + spread * _NORMAL.inv_cdf((index + 0.5) / count)
            )
            for index in range(count)
        ]
    return tuple(min(max(value, 1.0), float(capacity)) for value in values)


@dataclasses.dataclass(frozen=True)
class LengthComponent:
    """One log-normal or Pareto piece of a length distribution.

    A Pareto piece carries pareto_alpha > 0 and reads mean as its mean;
    otherwise the piece is log-normal with mean and cv. A zero cv is a fixed
    length.
    """

    weight: float
    mean: float
    cv: float = 0.0
    pareto_alpha: float = 0.0

    def quantiles(
        self, *, capacity: int, count: int = 128
    ) -> tuple[float, ...]:
        return component_quantiles(self, capacity=capacity, count=count)


@dataclasses.dataclass(frozen=True)
class LengthBase:
    """One workload: a length distribution, its share of the traffic, and
    the mean the distribution is scaled to."""

    name: str
    weight: float
    mean: float
    components: tuple[LengthComponent, ...]

    def expected_ceil(
        self, block_kv: int, *, capacity: int, count: int = 128
    ) -> float:
        """E[ceil(len / block_kv)] under this workload's distribution."""
        if block_kv <= 0:
            raise ValueError(f"block_kv must be positive; got {block_kv}.")
        total = 0.0
        for component in self.components:
            quantiles = component.quantiles(capacity=capacity, count=count)
            per_component = (
                sum(math.ceil(length / block_kv) for length in quantiles)
                / count
            )
            total += component.weight * per_component
        return total

    def expected_work_fraction(
        self,
        block_kv: int,
        floor_fraction: float,
        *,
        capacity: int,
        count: int = 128,
    ) -> float:
        """E[g] over this workload's items: E[sum of g] / E[items].

        A request's items are full except its last, so the sum of its items'
        work is ceil(len / block) - 1 + g(remainder / block).
        """
        if block_kv <= 0:
            raise ValueError(f"block_kv must be positive; got {block_kv}.")
        if floor_fraction >= 1.0:
            return 1.0
        numerator = 0.0
        denominator = 0.0
        for component in self.components:
            quantiles = component.quantiles(capacity=capacity, count=count)
            component_numerator = 0.0
            component_denominator = 0.0
            for length in quantiles:
                items = max(math.ceil(length / block_kv), 1)
                remainder = length - (items - 1) * block_kv
                component_numerator += items - 1 + item_work_fraction(
                    remainder / block_kv, floor_fraction
                )
                component_denominator += items
            numerator += component.weight * component_numerator / count
            denominator += component.weight * component_denominator / count
        return numerator / denominator


@dataclasses.dataclass(frozen=True)
class MlaDecodeCostConstants:
    """The chip's fixed costs, in units of computed columns.

    block_efficiency holds per-block multipliers, 1.0 where unlisted; a block
    the Mosaic compiler turns into less efficient code gets a value above 1.
    block_floor holds the measured floor work fraction per block, 1.0 where
    unlisted. r_item only sets how the per-item cost is split between the
    block and the item: predictions use efficiency(block_kv) * block_kv *
    work + r_item, so a calibrated entry carries the measured item cost in
    efficiency and leaves r_item at 0.
    """

    r_item: float
    r_step: float
    block_efficiency: tuple[tuple[int, float], ...]
    max_block_kv: int
    vmem_fraction: float = 0.75
    calibrated_on: str = ""
    block_floor: tuple[tuple[int, float], ...] = ()

    def efficiency(self, block_kv: int) -> float:
        for block, value in self.block_efficiency:
            if block == block_kv:
                return value
        return 1.0

    def floor_fraction(self, block_kv: int) -> float:
        for block, value in self.block_floor:
            if block == block_kv:
                return value
        return 1.0


def _lognormal(weight: float, mean: float, cv: float) -> LengthComponent:
    return LengthComponent(weight, mean, cv=cv)


def _pareto(
    weight: float, mean: float, alpha: float
) -> LengthComponent:
    return LengthComponent(weight, mean, pareto_alpha=alpha)


# Percent of traffic each scale and shape carries in the default prior. The
# scale split follows the published traces' medians (short requests dominate);
# the shape split favors the moderate and heavy-tailed fits the traces report.
_SCALES = ((256.0, 0.40), (1024.0, 0.45), (4096.0, 0.15))
_SHAPES = (
    ("lognormal_cv0.5", 0.30, lambda mean: (
        _lognormal(1.0, mean, 0.5),
    )),
    ("lognormal_cv1.5", 0.40, lambda mean: (
        _lognormal(1.0, mean, 1.5),
    )),
    ("pareto_mix", 0.20, lambda mean: (
        _lognormal(0.75, 0.7 * mean, 0.8),
        _pareto(0.25, 1.9 * mean, 1.5),
    )),
    ("bimodal", 0.10, lambda mean: (
        _lognormal(0.7, 0.5 * mean, 0.5),
        _lognormal(0.3, 13.0 / 6.0 * mean, 0.8),
    )),
)


def _build_prior() -> tuple[LengthBase, ...]:
    bases = []
    for scale, scale_weight in _SCALES:
        for name, shape_weight, components in _SHAPES:
            bases.append(
                LengthBase(
                    f"{name}@{int(scale)}",
                    scale_weight * shape_weight,
                    scale,
                    components(scale),
                )
            )
    return tuple(bases)


WORKLOAD_PRIOR = _build_prior()


def config_cost(
    block_kv: int,
    items_per_step: int,
    *,
    batch: int,
    capacity: int,
    constants: MlaDecodeCostConstants,
    bases: tuple[LengthBase, ...] = WORKLOAD_PRIOR,
    quantile_count: int = 128,
) -> float:
    """The prior-weighted cost of one (block, items per step), in columns."""
    if items_per_step < 1:
        raise ValueError(
            f"items_per_step must be positive; got {items_per_step}."
        )
    efficiency = constants.efficiency(block_kv)
    floor_fraction = constants.floor_fraction(block_kv)
    total = 0.0
    for base in bases:
        items = batch * base.expected_ceil(
            block_kv, capacity=capacity, count=quantile_count
        )
        steps = max(1, math.ceil(items / items_per_step))
        work = base.expected_work_fraction(
            block_kv,
            floor_fraction,
            capacity=capacity,
            count=quantile_count,
        )
        total += base.weight * (
            efficiency * work * steps * items_per_step * block_kv
            + constants.r_item * steps * items_per_step
            + constants.r_step * steps
        )
    return total


# Filled by the calibration script; empty while the feature is dormant.
# v3: fitted 2026-11-19 from 211 cells on TPU v6e -- the uniform 16k/32k
# grid (batch 32/128), the short 512-2048 grid (batch 32/128), and the
# block x length floor grid (batch 64, valid lengths 64-8192, 128-page
# table). R^2 0.982, MAPE 12.3%. block_floor is the measured per-item
# work floor at each block.
CALIBRATION: dict[str, dict[str, MlaDecodeCostConstants]] = {
    "TPU v6e": {
        "bfloat16": MlaDecodeCostConstants(
            r_item=0.0,
            r_step=263.712,
            block_efficiency=(
                (128, 1.0),
                (256, 0.787104),
                (512, 0.755948),
                (1024, 0.742238),
                (2048, 0.756332),
            ),
            max_block_kv=2048,
            vmem_fraction=0.75,
            calibrated_on="TPU v6e, JAX 0.11.0, 2026-11-19",
            block_floor=(
                (128, 0.920766),
                (256, 0.881365),
                (512, 0.802515),
                (1024, 0.702709),
                (2048, 0.626567),
            ),
        ),
    },
}
_WARNED: set[tuple[str, str]] = set()


def _warn_once(chip: str, cache_dtype: str) -> None:
    key = (chip, cache_dtype)
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(
        f"MLA decode has no calibrated cost constants for {chip}"
        f" / {cache_dtype}; the tuned table or the max_seqlen-based rule"
        " picks the block."
        " Run benchmarks/softmax_attention/calibrate_mla_decode_config.py"
        " on this device to add an entry.",
        stacklevel=3,
    )


def get_mla_cost_constants(
    chip: str, cache_dtype: str
) -> MlaDecodeCostConstants | None:
    """The chip's cost constants, or None for the legacy fallback.

    No warning is issued while CALIBRATION is empty: that is the feature's
    dormant state, not a missing entry.
    """
    table = CALIBRATION.get(chip)
    constants = None if table is None else table.get(cache_dtype)
    if constants is None and CALIBRATION:
        _warn_once(chip, cache_dtype)
    return constants
