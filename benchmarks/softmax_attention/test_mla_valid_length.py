"""Does an item's cost track its request's valid length or the full block?

Fixes the batch, the padded table width (128 pages = 16384 tokens, constant
across every cell), and items per step, then sweeps a uniform per-request
length. Below block_kv every request is one item, so the item and step
counts stay constant and only the valid length changes: a per-step time that
rises with length means an item's work tracks the valid length; a flat one
means it tracks the full block.

The 1024 sweep is the control: if the plateau follows block_kv, the same
shape of curve should saturate at 1024 instead of 2048.

Run on the TPU host, from the repo root:
  cd ~/flywheel-tpu-new && PYTHONPATH=. ~/venv-flywheel-tpu/bin/python \
      benchmarks/softmax_attention/test_mla_valid_length.py \
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
SWEEPS = (
    (2048, (128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192)),
    (1024, (128, 512, 1024, 2048, 4096)),
)


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
            "benchmarks/softmax_attention/results/mla_valid_length"
        ),
    )
    return parser.parse_args(argv)


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
        for block, lengths_sweep in SWEEPS:
            print(
                f"\nblock {block}, items {ITEMS}, batch {BATCH},"
                f" table {TABLE_PAGES} pages"
                f" ({TABLE_PAGES * PAGE_SIZE} tokens)"
            )
            print(
                f"{'length':>7}{'items':>7}{'steps':>7}{'median ms':>11}"
                f"{'us/step':>10}{'us/item':>10}"
            )
            for length in lengths_sweep:
                lengths = np.full(BATCH, length, np.int64)
                item_count, steps = model_counts(lengths, block, ITEMS)
                record = {
                    "block": block,
                    "length": length,
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
                        "per_step_us": 1e3 * timing["median_ms"] / steps,
                        "per_item_us": 1e3
                        * timing["median_ms"]
                        / item_count,
                    }
                )
                records.append(record)
                print(
                    f"{length:>7}{item_count:>7}{steps:>7}"
                    f"{timing['median_ms']:>11.4f}"
                    f"{record['per_step_us']:>10.2f}"
                    f"{record['per_item_us']:>10.2f}"
                )
    finally:
        harness.close()

    print("\nper-step time relative to the full-block plateau:")
    for block, _ in SWEEPS:
        full = [
            record["per_step_us"]
            for record in records
            if record["status"] == "ok"
            and record["block"] == block
            and record["length"] >= block
            and record["length"] % block == 0
        ]
        if not full:
            continue
        plateau = float(np.median(full))
        print(f"  block {block}: plateau {plateau:.2f} us/step")
        for record in records:
            if record["status"] != "ok" or record["block"] != block:
                continue
            print(
                f"    length {record['length']:>5}"
                f"  {record['per_step_us'] / plateau:>6.3f}"
                f"  (valid fraction {min(record['length'], block) / block:.3f})"
            )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "valid_length.json"
    path.write_text(json.dumps({"records": records}, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
