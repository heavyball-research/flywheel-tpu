"""Refit the MLA decode cost constants with the measured per-item floor.

Reads saved measurement files, so it needs no TPU: the fit cells of earlier
runs, the short-range cells, and the block x length grid. It fits floor(B)
from the grid, then fits column, e(B) and r_step with every cell's slots
scaled by its items' work fraction g. Writes fit.json and constants.py.

Run from the repo root:
  PYTHONPATH=. .venv/bin/python \
      benchmarks/softmax_attention/refit_mla_floor.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib

from benchmarks.softmax_attention.calibrate_mla_decode_config import (
    constant_snippet,
    fit_constants,
    fit_floor,
)
from flywheel_tpu.softmax_attention.mla_decode_calibration import (
    legal_block_kv,
)

RESULTS = pathlib.Path("benchmarks/softmax_attention/results")
# The saved grids were measured at page 128, where every lane multiple is a
# whole number of pages.
BLOCKS = legal_block_kv(128, 2048)
DATASETS = {
    "v2": RESULTS / "mla_calibration_v2" / "measurements.jsonl",
    "short": RESULTS / "mla_calibration_short" / "measurements.jsonl",
    "floor": RESULTS / "mla_block_length" / "block_length.json",
}
EXTRA_FLOOR = RESULTS / "mla_block_length" / "block_256_full.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="v2,short,floor",
        help="comma-separated measurement sets to pool.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=RESULTS / "mla_calibration_v3",
    )
    parser.add_argument("--reference-block", type=int, default=128)
    return parser.parse_args(argv)


def load_records(name):
    path = DATASETS[name]
    if path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    else:
        records = json.loads(path.read_text())["records"]
    for record in records:
        record.setdefault("status", "ok")
        if "slots" not in record:
            record["slots"] = record["steps"] * record["items_per_step"]
    return records


def main(argv=None):
    args = parse_args(argv)
    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    records = []
    for name in names:
        records.extend(load_records(name))
    floor_records = load_records("floor")
    if EXTRA_FLOOR.exists():
        floor_records.append(json.loads(EXTRA_FLOOR.read_text()))
    floor = fit_floor(floor_records, BLOCKS)

    fit = fit_constants(records, args.reference_block, floor=floor)
    print(f"datasets: {', '.join(names)} ({len(records)} cells)")
    print(
        f"fit: column {fit['column_ms']:.6g} ms, step {fit['step_ms']:.6g}"
        f" ms, r_step {fit['r_step']:.6g}, R^2 {fit['r_squared']:.4f},"
        f" MAPE {100 * fit['mape']:.2f}%"
    )
    for block, value in fit["block_floor"]:
        print(f"  floor({block}) = {value:.6g}")
    for block, value in fit["block_efficiency"]:
        print(f"  e({block}) = {value:.6g}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fit.json").write_text(json.dumps({"fit": fit}, indent=2))
    snippet = constant_snippet(
        "TPU v6e",
        "bfloat16",
        fit,
        f"TPU v6e, JAX 0.11.0, {datetime.date.today().isoformat()}",
    )
    (args.out / "constants.py").write_text(snippet)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
