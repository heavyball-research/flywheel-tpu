"""Measure per-item cost over a (block, valid length) grid.

The valid-length sweep showed that an item's cost is not proportional to its
request's valid length: at block 2048 the per-step time is flat at ~0.65 of
the full-block plateau until the valid length reaches ~block/2, then rises to
1.0. This grid repeats that shape at every candidate block so the floor
fraction f(B) can be fitted:

  cost(w) ~ full(B) * [f(B) + (1 - f(B)) * clamp(2w/B - 1, 0, 1)]

Length 64 is included so every block, including 128 and 256, has at least
one point below half a block. The padded table width is constant at 128
pages, so the per-step table load is the same in every cell.

Run on the TPU host, from the repo root:
  cd ~/flywheel-tpu-new && PYTHONPATH=. ~/venv-flywheel-tpu/bin/python \
      benchmarks/softmax_attention/test_mla_block_length_grid.py \
      --results benchmarks/softmax_attention/results/mla_calibration_v2
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

BATCH = 64
PAGE_SIZE = 128
ITEMS = 8
TABLE_PAGES = 128
BLOCKS = (128, 256, 512, 1024, 2048)
LENGTHS = (64, 128, 512, 1024, 2048, 4096, 8192)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        default=pathlib.Path(
            "benchmarks/softmax_attention/results/mla_calibration_v2"
        ),
        help="calibration directory holding fit.json (heads, dtype).",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(
            "benchmarks/softmax_attention/results/mla_block_length"
        ),
    )
    return parser.parse_args(argv)


def valid_fraction(length, block):
    """Mean valid tokens per item over a block: w / (ceil(w/B) * B)."""
    return length / ((-(-length // block)) * block)


def main(argv=None):
    from benchmarks.common.timing import time_call
    from benchmarks.softmax_attention.calibrate_mla_decode_config import (
        Harness,
        model_counts,
    )

    args = parse_args(argv)
    result = json.loads((args.results / "fit.json").read_text())
    harness = Harness(result["heads"], PAGE_SIZE, result["dtype"])
    records = []
    try:
        for block in BLOCKS:
            print(f"\nblock {block}, items {ITEMS}, batch {BATCH},"
                  f" table {TABLE_PAGES} pages")
            print(
                f"{'length':>7}{'valid':>8}{'items':>7}{'steps':>7}"
                f"{'median ms':>11}{'us/step':>10}{'us/item':>10}"
            )
            for length in LENGTHS:
                lengths = np.full(BATCH, length, np.int64)
                item_count, steps = model_counts(lengths, block, ITEMS)
                record = {
                    "block": block,
                    "length": length,
                    "valid_fraction": valid_fraction(length, block),
                    "batch": BATCH,
                    "items_per_step": ITEMS,
                    "table_pages": TABLE_PAGES,
                    "items_total": item_count,
                    "steps": steps,
                }
                try:
                    step, cache = harness.build(
                        lengths, block, ITEMS, table_pages=TABLE_PAGES
                    )
                    timing, _ = time_call(step, cache)
                except Exception as error:  # pylint: disable=broad-except
                    record["status"] = f"{type(error).__name__}: {error}"
                    records.append(record)
                    print(f"{length:>7}  {record['status'][:70]}")
                    continue
                finally:
                    harness.jax.clear_caches()
                record.update(
                    {
                        "status": "ok",
                        "median_ms": timing["median_ms"],
                        "per_step_us": 1e3
                        * timing["median_ms"]
                        / steps,
                        "per_item_us": 1e3
                        * timing["median_ms"]
                        / item_count,
                    }
                )
                records.append(record)
                print(
                    f"{length:>7}{record['valid_fraction']:>8.3f}"
                    f"{item_count:>7}{steps:>7}"
                    f"{timing['median_ms']:>11.4f}"
                    f"{record['per_step_us']:>10.2f}"
                    f"{record['per_item_us']:>10.2f}"
                )
    finally:
        harness.close()

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "block_length.json"
    path.write_text(json.dumps({"records": records}, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
