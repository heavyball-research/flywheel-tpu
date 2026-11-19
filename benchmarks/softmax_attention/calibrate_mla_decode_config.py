"""Calibrate MLA decode's cost constants on the TPU it runs on.

The kernel's time is modeled as

  t = work * item_slots * block_kv * t_col * e(B)
      + item_slots * r_item + steps * r_step

with columns = item_slots * block_kv. Every item a step computes is exactly
block_kv columns wide, so a timed run identifies only the per-item cost
k(B) = e(B) * block_kv + r_item and the per-step cost r_step. The model's
ranking is unchanged by moving cost between those two, so the calibration
fixes r_item = 0, folds the item's fixed cost into e(B), and reports

  t_col      = k(REF_BLOCK) / REF_BLOCK        ms per computed column
  e(B)       = k(B) / (t_col * block_kv)
  r_step     = step_ms / t_col                 columns

over uniform lengths that every measured block divides, then fits k(B) and
step_ms by weighted least squares. With --floor-scan it also times a block x
length grid and fits the per-item work fraction g(x) = floor(B) +
(1 - floor(B)) * clamp(2x - 1, 0, 1), where x is the item's valid length over
its block: an item pays a floor even when it holds few tokens, and the rest
of its work ramps in over the upper half of the block. work is the prior's
average of g over a request's items. The fit multiplies each cell's slots by
its g, so e(B) stays the full-block cost. --validate times held-out length
patterns and reports the model's regret and rank correlation against the
measurements.

A cell whose schedule and block table outgrow the SMEM budget is skipped: past
about 60% of the 1 MiB SMEM the compiled step runs about 20x slower, so the
cell says nothing about the fast regime the constants describe. The skipped
cells and their SMEM fraction are printed and recorded.

Usage, on the TPU:
  cd ~/flywheel-tpu-new && PYTHONPATH=. ~/venv-flywheel-tpu/bin/python \
      benchmarks/softmax_attention/calibrate_mla_decode_config.py \
      --out benchmarks/softmax_attention/results/mla_calibration --validate
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import pathlib

import numpy as np

from flywheel_tpu.softmax_attention.mla_decode_calibration import (
    item_work_fraction,
)

BLOCKS = (128, 256, 512, 1024, 2048)
ITEMS = (1, 2, 4, 8)
# The calls run for milliseconds, so a call's dispatch overhead is a small
# part of what is timed; a batch of 8 at 2048 tokens runs for tens of
# microseconds, where it is not. Lengths are multiples of every block and of
# every items count, so a step never pads and the fit sees exact counts.
BATCHES = (32, 128)
# Two lengths make the item count vary at a fixed block, which separates the
# per-item and per-step costs more sharply than ips alone, and widen the
# block table from 128 to 256 pages, which checks that a table that still fits
# SMEM has no cost of its own.
LENGTHS = (16384, 32768)
PAGE_SIZE = 128
HEADS = 128
REF_BLOCK = 128
VMEM_FRACTION = 0.75
VALIDATE_PATTERNS = ("uniform", "spiky", "longtail", "pareto")
VALIDATE_BATCH = 64
VALIDATE_MEAN_LEN = 8192
# The width scan keeps the actual lengths short and pads only the block table,
# so it isolates the SMEM cliff from the cache's size and from the lengths.
# 2048 is expected to miss the 1 MiB SMEM entirely and fail to compile.
REGIME_WIDTHS = (512, 1024, 1100, 1200, 1300, 1380, 1600, 2048)
REGIME_CONFIGS = ((128, 8), (256, 8), (512, 8))
REGIME_BATCH = 64
REGIME_LENGTH = 8192
# The floor scan keeps the block table wide and varies only the valid length.
# Every cell at length <= block has the same item and step counts, so the
# floor and the full-block point are read at the same step count; a point at
# half a block or less exists for every candidate block.
FLOOR_BATCH = 64
FLOOR_ITEMS = 8
FLOOR_TABLE_PAGES = 128
FLOOR_LENGTHS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(
            "benchmarks/softmax_attention/results/mla_calibration"
        ),
        help="output directory for measurements.jsonl and fit.json.",
    )
    parser.add_argument(
        "--blocks",
        default=",".join(map(str, BLOCKS)),
        help="comma-separated block_kv candidates.",
    )
    parser.add_argument(
        "--items",
        default=",".join(map(str, ITEMS)),
        help="comma-separated items_per_step candidates.",
    )
    parser.add_argument(
        "--batches",
        default=",".join(map(str, BATCHES)),
        help="comma-separated batch sizes of the fit.",
    )
    parser.add_argument(
        "--lengths",
        default=",".join(map(str, LENGTHS)),
        help="comma-separated uniform lengths of the fit.",
    )
    parser.add_argument("--heads", type=int, default=HEADS)
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)
    parser.add_argument(
        "--dtype", default="bfloat16", help="cache dtype name."
    )
    parser.add_argument(
        "--reference-block",
        type=int,
        default=REF_BLOCK,
        help="block whose efficiency is 1.0.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="time held-out length patterns and report the model's fit.",
    )
    parser.add_argument(
        "--max-validate-cells",
        type=int,
        default=None,
        help="stop validation after N (pattern, block, items) cells.",
    )
    parser.add_argument(
        "--regime-scan",
        action="store_true",
        help="time the table-width scan that locates the SMEM cliff.",
    )
    parser.add_argument(
        "--floor-scan",
        action="store_true",
        help="time a block x length grid and fit the per-item floor fraction.",
    )
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="measure cells over the SMEM budget too (the fit and validation"
        " skip them by default).",
    )
    parser.add_argument(
        "--regime-widths",
        default=",".join(map(str, REGIME_WIDTHS)),
        help="comma-separated block-table widths, in pages, of the scan.",
    )
    return parser.parse_args(argv)


def ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(","))


def plan_points(blocks, items, batches, lengths):
    """The fit's cells: uniform lengths that block and items both divide."""
    points = []
    for batch in batches:
        for length in lengths:
            for block in blocks:
                if block > length or length % block:
                    continue
                per_block = batch * length // block
                for step_items in items:
                    if per_block % step_items:
                        continue
                    points.append(
                        {
                            "batch": batch,
                            "length": length,
                            "block": block,
                            "items": step_items,
                        }
                    )
    return points


def model_counts(lengths, block, items):
    """(items, steps) the kernel schedules for these lengths and config."""
    total = sum(max(-(-int(length) // block), 1) for length in lengths)
    return total, max(1, math.ceil(total / items))


def validate_weights(pattern, batch):
    """Weights of a validation pattern; "pareto" is local to this script."""
    from benchmarks.common.lengths import pattern_weights

    if pattern == "pareto":
        alpha = 1.5
        return [
            (1.0 - (index + 0.5) / batch) ** (-1.0 / alpha)
            for index in range(batch)
        ]
    return pattern_weights(pattern, batch)


def plan_regime_points(widths):
    """The width scan: short uniform lengths, a padded block table."""
    return [
        {
            "batch": REGIME_BATCH,
            "length": REGIME_LENGTH,
            "block": block,
            "items": items,
            "table_pages": width,
        }
        for width in widths
        for block, items in REGIME_CONFIGS
    ]


def plan_floor_points(blocks):
    """The block x length grid that identifies the floor fraction."""
    return [
        {
            "batch": FLOOR_BATCH,
            "length": length,
            "block": block,
            "items": FLOOR_ITEMS,
            "table_pages": FLOOR_TABLE_PAGES,
        }
        for block in blocks
        for length in FLOOR_LENGTHS
    ]


def cell_work_fraction(length, block, floor_fraction):
    """The work fraction of one uniform cell's items, in full blocks."""
    items = max(math.ceil(length / block), 1)
    remainder = length - (items - 1) * block
    return (
        items
        - 1
        + item_work_fraction(remainder / block, floor_fraction)
    ) / items


def lengths_work_fraction(lengths, block, floor_fraction):
    """The work fraction averaged over a batch's items, in full blocks."""
    total_work = 0.0
    total_items = 0
    for length in lengths:
        length = int(length)
        items = max(math.ceil(length / block), 1)
        remainder = length - (items - 1) * block
        total_work += items - 1 + item_work_fraction(
            remainder / block, floor_fraction
        )
        total_items += items
    return total_work / total_items


def fit_floor(records, blocks):
    """floor(B) from the floor cells and the full-block cell of each block.

    Both are read at the same step count, so the per-call cost cancels.
    """
    floors = []
    for block in blocks:
        ok = [
            record
            for record in records
            if record.get("status") == "ok" and record["block"] == block
        ]
        full = [record for record in ok if record["length"] == block]
        floor_cells = [
            record for record in ok if record["length"] <= block // 2
        ]
        if not full or not floor_cells:
            raise RuntimeError(
                f"block {block} needs a length-{block} cell and a cell at"
                f" half a block or less; got"
                f" {sorted({record['length'] for record in ok})}."
            )
        full_step = float(
            np.median(
                [record["median_ms"] / record["steps"] for record in full]
            )
        )
        floor_step = float(
            np.mean(
                [
                    record["median_ms"] / record["steps"]
                    for record in floor_cells
                ]
            )
        )
        floors.append(
            [block, min(max(floor_step / full_step, 0.0), 1.0)]
        )
    return floors


class Harness:
    """Builds one forced-config flash_mla_decode step, as the tuner does:
    the public jit is swapped for its unwrapped function and default_mla_config
    is pinned, so each build traces and compiles afresh."""

    def __init__(self, heads, page_size, dtype_name):
        import jax
        import jax.numpy as jnp

        from benchmarks.common.mla import SM_SCALE
        from benchmarks.softmax_attention import kvcache_scenarios as scenarios
        from flywheel_tpu.softmax_attention import mla_decode as public
        from flywheel_tpu.softmax_attention.kernels import mla_decode as kernel

        self.jax = jax
        self.jnp = jnp
        self.scenarios = scenarios
        self.public = public
        self.kernel = kernel
        self.sm_scale = SM_SCALE
        self.heads = heads
        self.page_size = page_size
        self.dtype = jnp.dtype(dtype_name)
        self.raw = public.flash_mla_decode.__wrapped__
        self.orig_default = public.default_mla_config
        self.generator = np.random.default_rng(0)

    def close(self):
        self.public.default_mla_config = self.orig_default

    def build(self, lengths, block, items, table_pages=None):
        """(step, cache) for one call at these per-request lengths.

        table_pages widens the block table past the requests' own pages, as a
        deployment's capacity bucket does; every row past a request's last page
        repeats that page, so the kernel never reads it.
        """
        import jax.numpy as jnp

        from flywheel_tpu.softmax_attention.kernels.mla_decode import (
            LATENT_DIM,
            ROPE_DIM,
        )
        from flywheel_tpu.softmax_attention.mla_decode import mla_cache_shape

        lens = np.asarray(lengths, np.int64)
        batch = int(lens.shape[0])
        own_pages = -(-lens // self.page_size)
        pages = int(own_pages.max())
        if table_pages is not None:
            if table_pages < pages:
                raise ValueError(
                    f"table_pages {table_pages} is below the batch's"
                    f" {pages} pages."
                )
            pages = int(table_pages)
        # Note: a pool of exactly the requests' own pages, and a table that
        # points each request at its slice and pads with its last page, as
        # kvcache_scenarios does.
        table = np.empty((batch, pages), np.int32)
        first = 0
        for index, count in enumerate(own_pages):
            table[index, :count] = np.arange(
                first, first + count, dtype=np.int32
            )
            table[index, count:] = first + count - 1
            first += int(count)
        cache = self.scenarios.fill_cache(
            mla_cache_shape(int(own_pages.sum()), self.page_size, self.dtype),
            self.dtype,
            self.generator,
        )
        q_nope = self.scenarios.draw(
            self.generator, jnp.bfloat16, batch, 1, self.heads, LATENT_DIM
        )
        q_pe = self.scenarios.draw(
            self.generator, jnp.bfloat16, batch, 1, self.heads, ROPE_DIM
        )
        new_kv_c = self.scenarios.draw(
            self.generator, self.dtype, batch, 1, LATENT_DIM
        )
        new_k_pe = self.scenarios.draw(
            self.generator, self.dtype, batch, 1, ROPE_DIM
        )
        kv_lens = jnp.asarray(lens.astype(np.int32))
        table = jnp.asarray(table)
        config = self.kernel.MlaDecodeConfig(block, items)
        # Note: flash_mla_decode resolves default_mla_config from its own
        # module's globals, so the public module is the one to pin.
        self.public.default_mla_config = lambda **kwargs: config
        max_seqlen_k = int(lens.max())
        raw = self.raw
        # Note: the same kv_lens on every call rewrites the same appended
        # token, so the cache chains as the timed state without growing.
        def call(cache):
            out, new_cache = raw(
                q_nope,
                q_pe,
                new_kv_c,
                new_k_pe,
                cache,
                kv_lens,
                table,
                sm_scale=self.sm_scale,
                causal=True,
                s_dtype=jnp.bfloat16,
                max_seqlen_k=max_seqlen_k,
            )
            return out, new_cache

        return self.jax.jit(call, donate_argnums=(0,)), cache

    def vmem_bytes(self, batch, block, items):
        return self.kernel.mla_vmem_bytes(
            self.kernel.MlaDecodeConfig(block, items),
            batch=batch,
            num_query_rows=self.heads,
            query_tokens=1,
            cache_dtype=self.dtype,
            return_lse=False,
        )

    def smem_fraction(self, batch, pages_per_seq, block, items):
        """The call's SMEM use as a fraction of the scoped limit."""
        return self.kernel.mla_smem_bytes(
            self.kernel.MlaDecodeConfig(block, items),
            batch=batch,
            page_size=self.page_size,
            pages_per_seq=pages_per_seq,
        ) / self.kernel.smem_limit_bytes()


def measure_point(harness, point, vmem_budget, *, include_slow=False):
    from benchmarks.common.timing import time_call

    record = dict(point)
    pages_per_seq = int(
        point.get("table_pages")
        or -(-point["length"] // harness.page_size)
    )
    record["pages_per_seq"] = pages_per_seq
    record["smem_fraction"] = harness.smem_fraction(
        point["batch"], pages_per_seq, point["block"], point["items"]
    )
    if (
        not include_slow
        and record["smem_fraction"] > harness.kernel.SMEM_BUDGET_FRACTION
    ):
        # Note: past about 60% of SMEM the compiled step runs ~20x slower, so
        # the cell says nothing about the cost model's fast regime.
        record["status"] = "skipped_smem"
        return record
    if harness.vmem_bytes(
        point["batch"], point["block"], point["items"]
    ) > vmem_budget:
        record["status"] = "skipped_vmem"
        return record
    lengths = np.full(point["batch"], point["length"], np.int64)
    try:
        step, cache = harness.build(
            lengths,
            point["block"],
            point["items"],
            table_pages=point.get("table_pages"),
        )
        timing, _ = time_call(step, cache)
    except Exception as error:  # pylint: disable=broad-except
        record["status"] = "run_error"
        record["error"] = f"{type(error).__name__}: {error}"[:300]
        return record
    finally:
        harness.jax.clear_caches()
    item_count, steps = model_counts(lengths, point["block"], point["items"])
    record.update(
        {
            "status": "ok",
            "items_total": item_count,
            "slots": steps * point["items"],
            "steps": steps,
            "median_ms": timing["median_ms"],
            "samples_ms": timing["samples_ms"],
            "compile_s": timing["compile_s"],
        }
    )
    return record


def r_squared(design, times, solution):
    """1 - SSE/SST of the fit on these rows."""
    predicted = design @ solution
    return 1.0 - float(
        np.sum((predicted - times) ** 2)
        / max(np.sum((times - times.mean()) ** 2), 1e-30)
    )


def fit_constants(records, reference_block, *, floor=()):
    """k(B) and step_ms from the ok records, then the constants in columns.

    Each cell's slots are multiplied by its items' work fraction g, so the
    fitted per-item cost stays the full-block one. floor is a sequence of
    (block, fraction) pairs; a block not listed is fully charged.
    """
    floor_by_block = {int(block): value for block, value in floor}
    ok = [record for record in records if record["status"] == "ok"]
    blocks = sorted({record["block"] for record in ok})
    if len(blocks) < 2:
        raise RuntimeError(
            f"the fit needs at least two measured blocks; got {blocks}."
        )
    for block in blocks:
        # Note: items a step scales with block_kv, so only steps separate the
        # per-item and per-step costs at one block.
        step_counts = {
            record["steps"]
            for record in ok
            if record["block"] == block
        }
        if len(step_counts) < 2:
            raise RuntimeError(
                f"block {block} has {len(step_counts)} distinct step counts;"
                " the per-block and per-step costs need two."
            )
    index = {block: column for column, block in enumerate(blocks)}
    design = np.zeros((len(ok), len(blocks) + 1))
    times = np.zeros(len(ok))
    for row, record in enumerate(ok):
        work = cell_work_fraction(
            record["length"],
            record["block"],
            floor_by_block.get(record["block"], 1.0),
        )
        design[row, index[record["block"]]] = record["slots"] * work
        design[row, -1] = record["steps"]
        times[row] = record["median_ms"]
    weight = 1.0 / np.maximum(times, 1e-9)
    solution, *_ = np.linalg.lstsq(
        design * weight[:, None], times * weight, rcond=None
    )
    per_item_ms = {
        block: float(solution[index[block]]) for block in blocks
    }
    step_ms = float(solution[-1])
    if step_ms < 0.0:
        print(f"warning: fitted step cost {step_ms:.6g} ms < 0; clamping.")
        step_ms = 0.0
    reference = (
        reference_block if reference_block in per_item_ms else blocks[0]
    )
    column_ms = per_item_ms[reference] / reference
    if column_ms <= 0.0:
        raise RuntimeError(
            f"the reference block {reference} has a non-positive per-column"
            f" cost ({column_ms:.6g} ms); the run is unusable."
        )
    by_group = {}
    for field in ("batch", "length"):
        for value in sorted({record[field] for record in ok}):
            rows = [
                row
                for row, record in enumerate(ok)
                if record[field] == value
            ]
            if len(rows) > len(blocks) + 1:
                by_group[f"{field}={value}"] = r_squared(
                    design[rows], times[rows], solution
                )
    predicted = design @ solution
    mape = float(np.mean(np.abs((predicted - times) / times)))
    return {
        "reference_block": reference,
        "column_ms": column_ms,
        "per_item_ms": {str(block): value for block, value in
                        per_item_ms.items()},
        "step_ms": step_ms,
        "r_item": 0.0,
        "r_step": step_ms / column_ms,
        "block_efficiency": [
            [block, per_item_ms[block] / (column_ms * block)]
            for block in blocks
        ],
        "block_floor": [
            [block, floor_by_block.get(block, 1.0)] for block in blocks
        ],
        "max_block_kv": max(blocks),
        "vmem_fraction": VMEM_FRACTION,
        "r_squared": r_squared(design, times, solution),
        "mape": mape,
        "r_squared_by_group": by_group,
    }


def _ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while (
            end + 1 < len(order)
            and values[order[end + 1]] == values[order[index]]
        ):
            end += 1
        rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[order[position]] = rank
        index = end + 1
    return ranks


def correlation(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.size < 2 or left.std() == 0.0 or right.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def validate(
    harness, fit, blocks, items, max_cells=None, *, include_slow=False
):
    """Measured vs predicted on held-out length patterns."""
    from benchmarks.common.lengths import split_budget
    from benchmarks.common.timing import time_call

    per_item = {
        int(block): value
        for block, value in fit["per_item_ms"].items()
    }
    blocks = [block for block in blocks if block in per_item]
    step_ms = fit["step_ms"]
    floor_by_block = {
        int(block): value for block, value in fit.get("block_floor", ())
    }
    results = []
    cells = 0
    for pattern in VALIDATE_PATTERNS:
        lengths = np.asarray(
            split_budget(
                validate_weights(pattern, VALIDATE_BATCH),
                VALIDATE_BATCH * VALIDATE_MEAN_LEN,
            ),
            np.int64,
        )
        measured = {}
        predicted_of = {}
        for block in blocks:
            for step_items in items:
                if max_cells is not None and cells >= max_cells:
                    break
                cells += 1
                pages_per_seq = -(-int(lengths.max()) // harness.page_size)
                smem_fraction = harness.smem_fraction(
                    int(lengths.shape[0]), pages_per_seq, block, step_items
                )
                if (
                    not include_slow
                    and smem_fraction > harness.kernel.SMEM_BUDGET_FRACTION
                ):
                    print(
                        f"  validate {pattern} {block}/{step_items}:"
                        f" skipped (SMEM {100 * smem_fraction:.0f}%)"
                    )
                    continue
                item_count, steps = model_counts(
                    lengths, block, step_items
                )
                work = lengths_work_fraction(
                    lengths, block, floor_by_block.get(block, 1.0)
                )
                predicted = (
                    steps * step_items * per_item[block] * work
                    + steps * step_ms
                )
                try:
                    step, cache = harness.build(
                        lengths, block, step_items
                    )
                    timing, _ = time_call(step, cache)
                except Exception as error:  # pylint: disable=broad-except
                    print(
                        f"  validate {pattern} {block}/{step_items}:"
                        f" {type(error).__name__}: {error}"[:200]
                    )
                    continue
                finally:
                    harness.jax.clear_caches()
                measured[(block, step_items)] = timing["median_ms"]
                predicted_of[(block, step_items)] = predicted
                results.append(
                    {
                        "pattern": pattern,
                        "block": block,
                        "items": step_items,
                        "items_total": item_count,
                        "steps": steps,
                        "median_ms": timing["median_ms"],
                        "predicted_ms": predicted,
                        "pages_per_seq": pages_per_seq,
                        "smem_fraction": smem_fraction,
                    }
                )
            if max_cells is not None and cells >= max_cells:
                break
        if not measured:
            continue
        best_measured = min(measured, key=measured.get)
        best_predicted = min(measured, key=predicted_of.get)
        regret = measured[best_predicted] / measured[best_measured] - 1.0
        rho = correlation(
            list(predicted_of.values()), list(measured.values())
        )
        worst_cell = max(
            measured,
            key=lambda cell: abs(measured[cell] / predicted_of[cell] - 1.0),
        )
        worst = abs(measured[worst_cell] / predicted_of[worst_cell] - 1.0)
        mape = sum(
            abs(measured[cell] / predicted_of[cell] - 1.0)
            for cell in measured
        ) / len(measured)
        print(
            f"  {pattern}: best measured {best_measured}"
            f" ({measured[best_measured]:.4f} ms), model {best_predicted}"
            f" ({predicted_of[best_predicted]:.4f} ms), regret"
            f" {regret * 100:.1f}%, rank rho {rho:.3f}, MAPE"
            f" {mape * 100:.1f}%, worst cell {worst_cell} off"
            f" {worst * 100:.0f}%"
        )
    return results


def constant_snippet(device, dtype_name, fit, calibrated_on):
    efficiency = ", ".join(
        f"({block}, {value:.6g})"
        for block, value in fit["block_efficiency"]
    )
    floor = ", ".join(
        f"({block}, {value:.6g})"
        for block, value in fit.get("block_floor", ())
    )
    return (
        "CALIBRATION = {\n"
        f'    "{device}": {{\n'
        f'        "{dtype_name}": MlaDecodeCostConstants(\n'
        f"            r_item={fit['r_item']:.6g},\n"
        f"            r_step={fit['r_step']:.6g},\n"
        f"            block_efficiency=({efficiency}),\n"
        f"            max_block_kv={fit['max_block_kv']},\n"
        f"            vmem_fraction={fit['vmem_fraction']:.6g},\n"
        f'            calibrated_on="{calibrated_on}",\n'
        f"            block_floor=({floor}),\n"
        "        ),\n"
        "    },\n"
        "}\n"
    )


def main(argv=None):
    args = parse_args(argv)
    import jax

    from flywheel_tpu.softmax_attention.dispatch import (
        get_device_variant_name,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    blocks = ints(args.blocks)
    items = ints(args.items)
    (args.out / "measurements.jsonl").write_text("")
    harness = Harness(args.heads, args.page_size, args.dtype)
    budget = VMEM_FRACTION * harness.kernel.vmem_limit_bytes()
    records = []
    try:
        points = plan_points(
            blocks, items, ints(args.batches), ints(args.lengths)
        )
        print(f"{len(points)} fit cells")
        for index, point in enumerate(points, 1):
            record = measure_point(
                harness, point, budget, include_slow=args.include_slow
            )
            records.append(record)
            with (args.out / "measurements.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if record["status"] == "ok":
                print(
                    f"[{index}/{len(points)}] b{point['batch']}"
                    f" s{point['length']} {point['block']}/"
                    f"{point['items']}: {record['median_ms']:.4f} ms"
                )
            else:
                print(
                    f"[{index}/{len(points)}] b{point['batch']}"
                    f" s{point['length']} {point['block']}/"
                    f"{point['items']}: {record['status']}"
                    + (f" ({record.get('error', '')})" if
                       record.get("error") else "")
                )
        floor = []
        floor_records = []
        if args.floor_scan:
            floor_points = plan_floor_points(blocks)
            print(f"{len(floor_points)} floor cells")
            (args.out / "floor.jsonl").write_text("")
            for index, point in enumerate(floor_points, 1):
                record = measure_point(harness, point, budget)
                floor_records.append(record)
                with (args.out / "floor.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                prefix = (
                    f"[floor {index}/{len(floor_points)}]"
                    f" b{point['batch']} s{point['length']}"
                    f" {point['block']}/{point['items']}:"
                )
                if record["status"] == "ok":
                    print(f"{prefix} {record['median_ms']:.4f} ms")
                else:
                    print(f"{prefix} {record['status']}")
            floor = fit_floor(floor_records, blocks)
            for block, value in floor:
                print(f"  floor({block}) = {value:.6g}")
            records = records + floor_records
        fit = fit_constants(records, args.reference_block, floor=floor)
        print(
            f"fit: column {fit['column_ms']:.6g} ms, step"
            f" {fit['step_ms']:.6g} ms, r_step"
            f" {fit['r_step']:.6g} columns, R^2"
            f" {fit['r_squared']:.4f}, MAPE {100 * fit['mape']:.2f}%"
        )
        for group, value in fit["r_squared_by_group"].items():
            print(f"  R^2 {group}: {value:.4f}")
        for block, value in fit["block_efficiency"]:
            print(f"  e({block}) = {value:.6g}")
        result = {
            "device": get_device_variant_name(),
            "dtype": args.dtype,
            "heads": args.heads,
            "page_size": args.page_size,
            "fit": fit,
        }
        if floor_records:
            result["floor_records"] = floor_records
        if args.validate:
            print("validation:")
            result["validation"] = validate(
                harness,
                fit,
                blocks,
                items,
                args.max_validate_cells,
                include_slow=args.include_slow,
            )
        if args.regime_scan:
            print("regime scan:")
            widths = ints(args.regime_widths)
            regime_points = plan_regime_points(widths)
            regime = []
            (args.out / "regime.jsonl").write_text("")
            for index, point in enumerate(regime_points, 1):
                record = measure_point(
                    harness, point, budget, include_slow=True
                )
                regime.append(record)
                with (args.out / "regime.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                prefix = (
                    f"[{index}/{len(regime_points)}] width"
                    f" {point['table_pages']} {point['block']}/"
                    f"{point['items']}:"
                )
                if record["status"] == "ok":
                    print(
                        f"{prefix} {record['median_ms']:.4f} ms,"
                        f" {1000 * record['median_ms'] / record['steps']:.2f}"
                        f" us/step, SMEM {100 * record['smem_fraction']:.1f}%"
                    )
                else:
                    print(
                        f"{prefix} {record['status']} (SMEM"
                        f" {100 * record['smem_fraction']:.1f}%)"
                        + (
                            f" {record.get('error', '')}"
                            if record.get("error")
                            else ""
                        )
                    )
            result["regime"] = regime
        (args.out / "fit.json").write_text(json.dumps(result, indent=2))
        snippet = constant_snippet(
            get_device_variant_name(),
            args.dtype,
            fit,
            f"{get_device_variant_name()}, JAX {jax.__version__}, "
            f"{datetime.date.today().isoformat()}",
        )
        (args.out / "constants.py").write_text(snippet)
        print(snippet)
    finally:
        harness.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
