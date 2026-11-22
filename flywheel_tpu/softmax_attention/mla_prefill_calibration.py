"""Host-side cost model and length prior for MLA prefill block sizing.

A prefill call's operand shapes give the batch B, the total query tokens T,
the longest chunk Qmax, the longest context Kmax, and the page size. They do
not give the individual (q_len, kv_len) pairs: those are runtime data. The
heuristic therefore prices a candidate (block_q, block_kv, row_groups) at the
prior-weighted average time over a workload prior, and picks the least cost.

Workload prior
--------------
A request is drawn as a context length w from the decode length prior (a
mixture of log-normal and Pareto pieces with published-trace scales). One
scale s is fitted per call so that E[min(s * w, Qmax)] = T / B, the batch's
mean chunk; then

    q = min(s * w, Qmax)
    n = min(s * w, Kmax, capacity)        (and n at least q)

With a mean chunk below Qmax this mixes short whole prompts with shorter
partial chunks, and with every request a full chunk it sends the context
distribution up to Kmax. The prior is a deployment assumption, not a fit:
replace WORKLOAD_PRIOR with the deployment's own length statistics.

Cost model
----------
Per request and q block, the kernel walks one item per block_kv of context:

    F = n - q + j * block_q                 first key position of the block
    E = F + q_count                          one past its last key
    m = max(ceil(E / block_kv), 1)           items
    cols(k) = block_kv                       k < m - 1
            = 256 * ceil(span / 256)         the last item, live-tile path
            = block_kv                       the last item, otherwise
    span = E - (m - 1) * block_kv

Only the last item can have dead tiles, and it computes live columns only
when block_kv is a multiple of 256 above 256; the model prices that path.
The fitted time is

    T = p(block_q) * h(group_rows) * e(block_kv) * W
        + c_blocks * blocks + c_rows * rows
        + c_qcopy * q_tokens + c_item_rows * item_rows

where W sums block_q * num_heads * cols over the request's q blocks, rows
sums block_q over the q blocks, q_tokens sums the block's real query tokens
times num_heads, and item_rows sums m * block_q * num_heads. p and h are
measured efficiency factors pinned at p(128) = h(256) = 1, and e is the
per-computed-column cost of a block. The factors are separate because v6e
makes them measurable separately: a width or group change that alters no
column count still changes the time.

The calibration measures each factor on one page and prices every page with
the same table: page size changes only which block_kv values are legal, and
the measured cost of a config is the same on pages 128/256/512/1024.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

from ..core.tpu import NUM_LANES
from .mla_decode_calibration import (
    WORKLOAD_PRIOR,
    LengthBase,
    legal_block_kv,
)


# The kernel's live-tile width: only a block_kv that is a whole number of
# these tiles, above one tile, can skip its dead tiles.
LIVE_TILE_COLUMNS = 256
# The largest block_kv the calibration measured on v6e.
MAX_BLOCK_KV = 2048
# The group_rows values the calibration measured; the fastest is 256.
GROUP_ROWS_SET = (64, 128, 256, 512, 1024)


@dataclasses.dataclass(frozen=True)
class MlaPrefillWorkload:
    """The batch's static view: its mean chunk and its longest lengths."""

    batch: int
    num_tokens: int
    max_seqlen_q: int
    max_seqlen_k: int
    capacity: int

    @property
    def mean_chunk(self) -> float:
        return self.num_tokens / self.batch


def workload_samples(
    workload: MlaPrefillWorkload,
    *,
    bases: tuple[LengthBase, ...] = WORKLOAD_PRIOR,
    count: int = 8,
) -> tuple[tuple[float, float, float], ...]:
    """(weight, q_len, kv_len) samples whose mean chunk is T / batch.

    Each base's component contributes count equal-mass quantiles. One scale
    serves both lengths: it is solved by bisection so that the samples' mean
    min(s * w, Qmax) equals the batch's mean chunk, then q and n clip the
    same scaled draws. The samples are deterministic and their weights sum to
    one.
    """
    if workload.batch < 1 or workload.num_tokens < 1:
        raise ValueError(
            "a prefill workload needs a positive batch and token count;"
            f" got {workload.batch} and {workload.num_tokens}."
        )
    if workload.mean_chunk > workload.max_seqlen_q:
        raise ValueError(
            f"the mean chunk {workload.mean_chunk} exceeds max_seqlen_q"
            f" {workload.max_seqlen_q}."
        )
    draws = []
    for base in bases:
        if base.weight <= 0.0:
            continue
        # Quantiles of the unscaled component, clipped far past any real
        # capacity; the call's Qmax and Kmax clip them below.
        far = 1 << 40
        for component in base.components:
            quantiles = component.quantiles(capacity=far, count=count)
            share = base.weight * component.weight / count
            for length in quantiles:
                draws.append((share, length))
    if not draws:
        raise ValueError("the workload prior has no positive-weight base.")
    total = sum(share for share, _ in draws)
    draws = [(share / total, length) for share, length in draws]
    target = workload.mean_chunk

    def clipped_mean(log_scale):
        return sum(
            share * min(math.exp(log_scale) * length, workload.max_seqlen_q)
            for share, length in draws
        )

    low, high = -40.0, 40.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if clipped_mean(middle) < target:
            low = middle
        else:
            high = middle
    log_scale = 0.5 * (low + high)
    scale = math.exp(log_scale)
    samples = []
    for share, length in draws:
        scaled = scale * length
        n = min(scaled, workload.max_seqlen_k, workload.capacity)
        q = min(scaled, workload.max_seqlen_q, n)
        samples.append((share, q, n))
    return tuple(samples)


def request_terms(
    block_q: int, block_kv: int, q_len: float, kv_len: float
) -> tuple[int, float, float, float]:
    """(blocks, columns, items, q_tokens) of one request.

    columns counts block_q * num_heads * computed columns summed over the
    request's q blocks; q_tokens counts real query tokens times heads. Both
    are per request; the caller multiplies by the constants.
    """
    blocks = int(math.ceil(q_len / block_q))
    columns = 0.0
    items = 0.0
    q_tokens = 0.0
    for index in range(blocks):
        q_count = min(block_q, q_len - index * block_q)
        first = kv_len - q_len + index * block_q
        end = first + q_count
        num_items = max(int(math.ceil(end / block_kv)), 1)
        items += num_items
        columns += (num_items - 1) * block_kv
        span = end - (num_items - 1) * block_kv
        if block_kv % LIVE_TILE_COLUMNS == 0 and block_kv > LIVE_TILE_COLUMNS:
            columns += LIVE_TILE_COLUMNS * int(
                math.ceil(span / LIVE_TILE_COLUMNS)
            )
        else:
            columns += block_kv
        q_tokens += q_count
    return blocks, columns, items, q_tokens


def request_cost(
    block_q: int,
    block_kv: int,
    group_rows: int,
    q_len: float,
    kv_len: float,
    constants: MlaPrefillCostConstants,
    num_heads: int,
) -> float:
    """One request's fitted time under the cost constants."""
    blocks, columns, items, q_tokens = request_terms(
        block_q, block_kv, q_len, kv_len
    )
    work = block_q * num_heads * columns
    efficiency = (
        constants.block_q_factor(block_q)
        * constants.group_rows_factor(group_rows)
        * constants.block_kv_efficiency(block_kv)
    )
    return (
        efficiency * work
        + constants.c_blocks * blocks
        + constants.c_rows * blocks * block_q
        + constants.c_qcopy * q_tokens * num_heads
        + constants.c_item_rows * items * block_q * num_heads
    )


def config_cost(
    block_q: int,
    block_kv: int,
    group_rows: int,
    *,
    samples: tuple[tuple[float, float, float], ...],
    constants: MlaPrefillCostConstants,
    num_heads: int,
) -> float:
    """The prior-weighted mean request time of one config."""
    return sum(
        weight
        * request_cost(
            block_q, block_kv, group_rows, q_len, kv_len, constants, num_heads
        )
        for weight, q_len, kv_len in samples
    )


def config_cost_terms(
    block_q: int,
    block_kv: int,
    group_rows: int,
    *,
    samples: tuple[tuple[float, float, float], ...],
    constants: MlaPrefillCostConstants,
    num_heads: int,
) -> tuple[float, float, float, float, float]:
    """The prior-weighted mean of each cost term, for calibration reports.

    Returns (work, blocks, rows, q_tokens, item_rows), each already
    multiplied by the config's efficiency factors where the model does.
    """
    work = blocks_total = rows = q_tokens = item_rows = 0.0
    for weight, q_len, kv_len in samples:
        blocks, columns, items, tokens = request_terms(
            block_q, block_kv, q_len, kv_len
        )
        work += weight * (
            constants.block_q_factor(block_q)
            * constants.group_rows_factor(group_rows)
            * constants.block_kv_efficiency(block_kv)
            * block_q
            * num_heads
            * columns
        )
        blocks_total += weight * blocks
        rows += weight * blocks * block_q
        q_tokens += weight * tokens * num_heads
        item_rows += weight * items * block_q * num_heads
    return work, blocks_total, rows, q_tokens, item_rows


@dataclasses.dataclass(frozen=True)
class MlaPrefillCostConstants:
    """The chip's calibrated prefill costs and candidate domain.

    block_efficiency holds measured per-column costs per block_kv;
    group_efficiency and block_q_efficiency hold measured time factors per
    group_rows and block_q, 1.0 where unlisted. A config whose block_kv is
    unlisted is not a candidate while block_efficiency lists any block: its
    1.0 would be a placeholder, not a measurement.

    block_qs lists the measured widths; at block_kv >= large_kv_from only
    large_kv_block_qs are candidates, because v6e made a block_q that is not
    a whole number of 128-row lane tiles up to 2.5x slower there. group_rows
    lists the measured groups. width_block_kv_caps bounds a width's
    block_kv: v6e ran block_q=256 at block_kv 640 and 768 2.7x and 2.8x
    slower than at 512 on page 128 even though both fit VMEM, so 256 stops
    at 512; block_q=192 is fine at 640 (1.2x its 512 time) but 2.4x and 2.5x
    slower at 768 and 896, so it stops at 640. The 128-wide blocks serve the
    longer contexts.
    """

    c_blocks: float
    c_rows: float
    c_qcopy: float
    c_item_rows: float
    block_efficiency: tuple[tuple[int, float], ...]
    group_efficiency: tuple[tuple[int, float], ...]
    block_q_efficiency: tuple[tuple[int, float], ...]
    block_qs: tuple[int, ...]
    large_kv_block_qs: tuple[int, ...]
    group_rows_set: tuple[int, ...]
    width_block_kv_caps: tuple[tuple[int, int], ...] = ()
    max_block_kv: int = MAX_BLOCK_KV
    large_kv_from: int = 1024
    vmem_fraction: float = 0.75
    calibrated_on: str = ""

    def block_kv_efficiency(self, block_kv: int) -> float:
        return dict(self.block_efficiency).get(block_kv, 1.0)

    def block_kv_is_listed(self, block_kv: int) -> bool:
        return any(block == block_kv for block, _ in self.block_efficiency)

    def group_rows_factor(self, group_rows: int) -> float:
        return dict(self.group_efficiency).get(group_rows, 1.0)

    def group_rows_is_listed(self, group_rows: int) -> bool:
        return any(group == group_rows for group, _ in self.group_efficiency)

    def block_q_factor(self, block_q: int) -> float:
        return dict(self.block_q_efficiency).get(block_q, 1.0)

    def block_q_is_listed(self, block_q: int) -> bool:
        return any(block == block_q for block, _ in self.block_q_efficiency)

    def block_q_candidates(self, block_kv: int) -> tuple[int, ...]:
        """The widths the calibration allows at this block_kv."""
        if block_kv >= self.large_kv_from:
            return self.large_kv_block_qs
        return self.block_qs

    def width_block_kv_cap(self, block_q: int) -> int | None:
        """The largest block_kv this width may take, or None."""
        return dict(self.width_block_kv_caps).get(block_q)


def row_groups_for(
    block_q: int, group_rows: int, num_heads: int
) -> int | None:
    """The row_groups that gives this group_rows, or None when impossible.

    The kernel needs group_rows = (block_q / row_groups) * num_heads with
    row_groups a positive divisor of block_q.
    """
    if group_rows <= 0 or block_q <= 0 or num_heads <= 0:
        return None
    if (block_q * num_heads) % group_rows:
        return None
    row_groups = block_q * num_heads // group_rows
    if row_groups < 1 or block_q % row_groups:
        return None
    return row_groups


def candidate_lattice(
    *,
    page_size: int,
    max_seqlen_k: int,
    capacity: int,
    constants: MlaPrefillCostConstants,
    num_heads: int,
) -> tuple[tuple[int, int, int], ...]:
    """Every (block_q, block_kv, row_groups) the constants allow.

    The block_kv lattice is the kernel's own up to the calibration's largest
    measured block and the call's longest context, pruned to measured block
    costs; the width and group are intersected with the measured domain. VMEM
    feasibility is the caller's filter: it needs the kernel's byte formula.
    """
    top = min(
        constants.max_block_kv,
        max(NUM_LANES, _next_pow2(min(max_seqlen_k, capacity))),
    )
    lattice = []
    for block_kv in legal_block_kv(page_size, top):
        if constants.block_efficiency and not constants.block_kv_is_listed(
            block_kv
        ):
            continue
        for block_q in constants.block_q_candidates(block_kv):
            if (
                constants.block_q_efficiency
                and not constants.block_q_is_listed(block_q)
            ):
                continue
            width_cap = constants.width_block_kv_cap(block_q)
            if width_cap is not None and block_kv > width_cap:
                continue
            for group_rows in constants.group_rows_set:
                if (
                    constants.group_efficiency
                    and not constants.group_rows_is_listed(group_rows)
                ):
                    continue
                row_groups = row_groups_for(block_q, group_rows, num_heads)
                if row_groups is not None:
                    lattice.append((block_q, block_kv, row_groups))
    return tuple(lattice)


def _next_pow2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


# Shipped table: 103 measured TPU v6e bf16 cells (10 page/query/context
# workload groups, 128/256/512/1024-token pages, q_len = kv_len and chunked
# q_len < kv_len, 512..16384 tokens). The linear/e/p terms are fitted on the
# group_rows=256 cells only: selection keeps the lowest-efficiency
# VMEM-feasible group, and 256 is feasible for every (block_q, block_kv) the
# width caps leave, so those are the cells the decision is made on. The group
# table lists the measured clean ratios; 128 has a codegen pathology at two
# cells (3.3x) and is priced at its clean block_kv=512 ratio instead.
CALIBRATION: dict[str, dict[str, MlaPrefillCostConstants]] = {
    "TPU v6e": {
        "bfloat16": MlaPrefillCostConstants(
            c_blocks=0.000427566299,
            c_rows=0,
            c_qcopy=6.49528882e-07,
            c_item_rows=7.84661985e-07,
            block_efficiency=(
                (128, 0),
                (256, 1.82354526e-10),
                (384, 1.66833546e-09),
                (512, 1.42998973e-09),
                (640, 2.20455601e-09),
                (768, 1.77823624e-09),
                (896, 2.35554019e-09),
                (1024, 2.07543154e-09),
                (1152, 2.45145984e-09),
                (1280, 2.85572942e-09),
                (1408, 2.54394027e-09),
                (1536, 6.73091273e-09),
                (1664, 5.91868625e-09),
                (1792, 6.52013609e-09),
                (1920, 5.93003549e-09),
                (2048, 5.94766305e-09),
            ),
            group_efficiency=(
                (64, 5.85656332),
                (128, 1.31),
                (256, 1),
                (512, 1.04906293),
                (1024, 1.06337766),
            ),
            block_q_efficiency=(
                (32, 1.45449619),
                (64, 1.19949166),
                (96, 1.01134463),
                (128, 1),
                (192, 1.01181693),
                (256, 1.02307855),
            ),
            block_qs=(32, 64, 96, 128, 192, 256),
            large_kv_block_qs=(128, 256),
            group_rows_set=GROUP_ROWS_SET,
            width_block_kv_caps=((256, 512), (192, 640)),
            max_block_kv=MAX_BLOCK_KV,
            calibrated_on=(
                "TPU v6e bfloat16 prefill, pages 128/256/512/1024,"
                " 2026-11-21"
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
        f"MLA prefill has no calibrated cost constants for {chip}"
        f" / {cache_dtype}; the max_seqlen-based rule picks the config."
        " See docs/mla_prefill_block_heuristic.md to measure and add an entry.",
        stacklevel=3,
    )


def get_mla_prefill_cost_constants(
    chip: str, cache_dtype: str
) -> MlaPrefillCostConstants | None:
    """The chip's cost constants, or None for the legacy fallback.

    No warning is issued while CALIBRATION is empty: that is the feature's
    dormant state, not a missing entry.
    """
    table = CALIBRATION.get(chip)
    constants = None if table is None else table.get(cache_dtype)
    if constants is None and CALIBRATION:
        _warn_once(chip, cache_dtype)
    return constants
