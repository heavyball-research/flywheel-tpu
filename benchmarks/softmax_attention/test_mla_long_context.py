"""Measure the cost model at 64k and 128k context lengths.

The fitted e(B) constants come from uniform 16k/32k cells. This grid checks
the extrapolation: batch 32/64, lengths 65536/131072, blocks 512/1024/2048,
items 4/8. Each cell is timed and compared with the model's prediction from
the v2 fit, and each (batch, length) group reports the measured best, the
model's best, its regret, and what the three rules pick.

Run on the TPU host, from the repo root:
  cd ~/flywheel-tpu-new && PYTHONPATH=. ~/venv-flywheel-tpu/bin/python \
      benchmarks/softmax_attention/test_mla_long_context.py \
      --results benchmarks/softmax_attention/results/mla_calibration_v2
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

BATCHES = (32, 64)
LENGTHS = (65536, 131072)
# 128 and 256 are what the short-scaled default prior picks, so the grid must
# measure them to score that choice rather than only the large-block basin.
BLOCKS = (128, 256, 512, 1024, 2048)
ITEMS = (4, 8)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        default=pathlib.Path(
            "benchmarks/softmax_attention/results/mla_calibration_v2"
        ),
        help="calibration directory holding fit.json.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(
            "benchmarks/softmax_attention/results/mla_long_context"
        ),
    )
    return parser.parse_args(argv)


def correlation(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.size < 2 or left.std() == 0.0 or right.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def main(argv=None):
    import jax.numpy as jnp

    from benchmarks.softmax_attention.analyze_mla_prior import (
        LONG_SCALES,
        as_tuple,
        constants_from_fit,
        patched_kernel,
        prior_with_scales,
    )
    from benchmarks.softmax_attention.calibrate_mla_decode_config import (
        Harness,
        measure_point,
    )
    from flywheel_tpu.softmax_attention import (
        mla_decode_calibration as calibration,
    )
    from flywheel_tpu.softmax_attention.kernels import mla_decode as kernel

    args = parse_args(argv)
    result = json.loads((args.results / "fit.json").read_text())
    fit = result["fit"]
    constants = constants_from_fit(fit)
    dtype = jnp.dtype(result["dtype"])
    heads = result["heads"]
    page_size = result["page_size"]
    per_item = {int(b): v for b, v in fit["per_item_ms"].items()}
    step_ms = fit["step_ms"]
    vmem_budget = constants.vmem_fraction * kernel.vmem_limit_bytes()

    harness = Harness(heads, page_size, result["dtype"])
    records = []
    try:
        for batch in BATCHES:
            for length in LENGTHS:
                for block in BLOCKS:
                    for items in ITEMS:
                        point = {
                            "batch": batch,
                            "length": length,
                            "block": block,
                            "items": items,
                        }
                        record = measure_point(
                            harness, point, vmem_budget
                        )
                        if record["status"] == "ok":
                            record["predicted_ms"] = (
                                record["slots"] * per_item[block]
                                + record["steps"] * step_ms
                            )
                        records.append(record)
                        prefix = (
                            f"b{batch} s{length} {block}/{items}:"
                        )
                        if record["status"] == "ok":
                            print(
                                f"{prefix} {record['median_ms']:.3f} ms,"
                                f" predicted"
                                f" {record['predicted_ms']:.3f} ms,"
                                f" SMEM {100 * record['smem_fraction']:.0f}%"
                            )
                        else:
                            print(f"{prefix} {record['status']}")
    finally:
        harness.close()

    print("\ngroup summary:")
    print(
        f"{'batch':>5}{'length':>8}{'best':>12}{'best ms':>10}"
        f"{'model regret':>13}{'rho':>7}{'MAPE':>7}"
        f"  choices (default / long / legacy)"
    )
    groups = []
    for batch in BATCHES:
        for length in LENGTHS:
            ok = [
                record
                for record in records
                if record["status"] == "ok"
                and record["batch"] == batch
                and record["length"] == length
            ]
            if not ok:
                continue
            measured = {
                (r["block"], r["items"]): r["median_ms"] for r in ok
            }
            predicted = {
                (r["block"], r["items"]): r["predicted_ms"] for r in ok
            }
            best = min(measured, key=measured.get)
            model_best = min(measured, key=predicted.get)
            regret = measured[model_best] / measured[best] - 1.0
            rho = correlation(
                list(predicted.values()), list(measured.values())
            )
            mape = float(
                np.mean(
                    [
                        abs(measured[cell] / predicted[cell] - 1.0)
                        for cell in measured
                    ]
                )
            )
            pages = -(-length // page_size)
            shape = {
                "batch": batch,
                "page_size": page_size,
                "pages_per_seq": pages,
                "num_query_rows": heads,
                "query_tokens": 1,
                "cache_dtype": dtype,
                "return_lse": False,
                "max_seqlen_k": length,
            }
            choices = {}
            for name, prior in (
                ("default", calibration.WORKLOAD_PRIOR),
                ("long", prior_with_scales(LONG_SCALES)),
            ):
                with patched_kernel(kernel, constants, prior) as choose:
                    choices[name] = list(as_tuple(choose(**shape)))
            choices["legacy"] = list(
                as_tuple(
                    kernel._legacy_default_mla_config(**shape, tuned=None)
                )
            )
            row = {
                "batch": batch,
                "length": length,
                "best": list(best),
                "best_ms": measured[best],
                "model_best": list(model_best),
                "model_regret": regret,
                "rho": rho,
                "mape": mape,
                "choices": choices,
                "measured_ms": {
                    f"{b}/{i}": value
                    for (b, i), value in measured.items()
                },
                "predicted_ms": {
                    f"{b}/{i}": value
                    for (b, i), value in predicted.items()
                },
            }
            groups.append(row)
            print(
                f"{batch:>5}{length:>8}{str(best):>12}"
                f"{measured[best]:>10.3f}{100 * regret:>12.1f}%"
                f"{rho:>7.3f}{100 * mape:>6.1f}%"
                f"  {choices['default']} / {choices['long']}"
                f" / {choices['legacy']}"
            )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "long_context.json").write_text(
        json.dumps({"records": records, "groups": groups}, indent=2)
    )
    print(f"wrote {args.out / 'long_context.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
