"""Compare length priors and the legacy rule on measured validation cells.

Reads a calibration result directory (fit.json, regime.jsonl) and, for each
validation pattern and each table width, reports what each rule picks and how
far the pick is from the best measured config:

  default  the WORKLOAD_PRIOR shipped with the heuristic
  long     the same shapes on longer scales (512 / 2048 / 8192)
  legacy   the max_seqlen-based rule the calibrated path replaces

Every choice goes through the kernel's real default_mla_config, with the
fitted constants and the chosen prior swapped in, so the SMEM and VMEM guards
apply exactly as they will in production. A pick that the validation did not
measure is reported as "unmeasured" rather than scored.

Run on the TPU host, from the repo root:
  cd ~/flywheel-tpu-new && PYTHONPATH=. ~/venv-flywheel-tpu/bin/python \
      benchmarks/softmax_attention/analyze_mla_prior.py \
      --results benchmarks/softmax_attention/results/mla_calibration_v2
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib

import numpy as np

from flywheel_tpu.softmax_attention import mla_decode_calibration as calibration

LONG_SCALES = ((512.0, 0.40), (2048.0, 0.45), (8192.0, 0.15))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        default=pathlib.Path(
            "benchmarks/softmax_attention/results/mla_calibration_v2"
        ),
        help="directory holding fit.json and regime.jsonl.",
    )
    parser.add_argument("--heads", type=int, default=128)
    return parser.parse_args(argv)


def constants_from_fit(fit):
    return calibration.MlaDecodeCostConstants(
        r_item=fit["r_item"],
        r_step=fit["r_step"],
        block_efficiency=tuple(
            (int(block), value) for block, value in fit["block_efficiency"]
        ),
        max_block_kv=fit["max_block_kv"],
        vmem_fraction=fit["vmem_fraction"],
        calibrated_on="analysis",
        block_floor=tuple(
            (int(block), value)
            for block, value in fit.get("block_floor", ())
        ),
    )


def prior_with_scales(scales):
    bases = []
    for scale, scale_weight in scales:
        for name, shape_weight, components in calibration._SHAPES:
            bases.append(
                calibration.LengthBase(
                    f"{name}@{int(scale)}",
                    scale_weight * shape_weight,
                    scale,
                    components(scale),
                )
            )
    return tuple(bases)


@contextlib.contextmanager
def patched_kernel(kernel, constants, prior):
    """default_mla_config with no tuned entry, these constants, this prior."""
    saved = (
        kernel.config_cost,
        kernel.get_tuned_mla_config,
        kernel.get_mla_cost_constants,
    )

    def config_cost(block_kv, items_per_step, **kwargs):
        return calibration.config_cost(
            block_kv, items_per_step, bases=prior, **kwargs
        )

    kernel.config_cost = config_cost
    kernel.get_tuned_mla_config = lambda **kwargs: None
    kernel.get_mla_cost_constants = lambda chip, name: constants
    try:
        yield kernel.default_mla_config
    finally:
        (
            kernel.config_cost,
            kernel.get_tuned_mla_config,
            kernel.get_mla_cost_constants,
        ) = saved


def as_tuple(config):
    return (int(config.block_kv), int(config.items_per_step))


def items_per_step(record):
    """The record's item count per step, from either record layout.

    A run before the field fix stored the batch's total item count in
    "items"; the scheduled step count recovers the per-step count exactly.
    """
    if "items_total" in record:
        return record["items"]
    return record["items"] // record["steps"]


def measured_matrix(records):
    """{(block, items): ms} of the ok records, and their shared pages."""
    measured = {
        (record["block"], items_per_step(record)): record["median_ms"]
        for record in records
        if record.get("status", "ok") == "ok"
    }
    return measured


def score(measured, choice):
    """(chosen ms, best ms, regret) or None when the choice was not measured."""
    best = min(measured, key=measured.get)
    if choice not in measured:
        return None
    return measured[choice], measured[best], measured[choice] / measured[best] - 1.0


def report_validation(kernel, constants, priors, result, heads, dtype):
    from benchmarks.common.lengths import split_budget
    from benchmarks.softmax_attention.calibrate_mla_decode_config import (
        VALIDATE_BATCH,
        VALIDATE_MEAN_LEN,
        validate_weights,
    )

    records = result["validation"]
    patterns = []
    for record in records:
        if record["pattern"] not in patterns:
            patterns.append(record["pattern"])
    page_size = result["page_size"]
    print("validation (batch 64, mean 8192, natural table width):")
    print(
        f"{'pattern':<10}{'rule':<9}{'choice':<12}{'chosen ms':>10}"
        f"{'best ms':>9}{'regret':>9}"
    )
    rows = []
    for pattern in patterns:
        lengths = np.asarray(
            split_budget(
                validate_weights(pattern, VALIDATE_BATCH),
                VALIDATE_BATCH * VALIDATE_MEAN_LEN,
            ),
            np.int64,
        )
        pages_per_seq = -(-int(lengths.max()) // page_size)
        max_seqlen_k = int(lengths.max())
        measured = measured_matrix(
            [record for record in records if record["pattern"] == pattern]
        )
        if not measured:
            continue
        best = min(measured, key=measured.get)
        row = {
            "pattern": pattern,
            "best": list(best),
            "best_ms": measured[best],
            "rules": {},
        }
        shape = {
            "batch": VALIDATE_BATCH,
            "page_size": page_size,
            "pages_per_seq": pages_per_seq,
            "num_query_rows": heads,
            "query_tokens": 1,
            "cache_dtype": dtype,
            "return_lse": False,
            "max_seqlen_k": max_seqlen_k,
        }
        for name, prior in priors.items():
            with patched_kernel(kernel, constants, prior) as choose:
                choice = as_tuple(choose(**shape))
            row["rules"][name] = {
                "choice": list(choice),
                "score": score(measured, choice),
            }
        legacy = as_tuple(kernel._legacy_default_mla_config(**shape, tuned=None))
        row["rules"]["legacy"] = {
            "choice": list(legacy),
            "score": score(measured, legacy),
        }
        for name, entry in row["rules"].items():
            choice = tuple(entry["choice"])
            scored = entry["score"]
            if scored is None:
                print(
                    f"{pattern:<10}{name:<9}{str(choice):<12}"
                    f"{'-':>10}{measured[best]:>9.4f}{'unmeasured':>9}"
                )
            else:
                chosen, best_ms, regret = scored
                print(
                    f"{pattern:<10}{name:<9}{str(choice):<12}"
                    f"{chosen:>10.4f}{best_ms:>9.4f}"
                    f"{100 * regret:>8.1f}%"
                )
        rows.append(row)
        print()
    return rows


def report_regime(kernel, constants, priors, path, result, heads, dtype):
    if not path.exists():
        return []
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    page_size = result["page_size"]
    widths = []
    for record in records:
        if record["table_pages"] not in widths:
            widths.append(record["table_pages"])
    print("regime scan (batch 64, uniform 8192, padded table):")
    print(
        f"{'width':<7}{'rule':<9}{'choice':<12}{'chosen ms':>10}"
        f"{'best ms':>9}{'regret':>9}"
    )
    rows = []
    for width in widths:
        measured = measured_matrix(
            [record for record in records if record["table_pages"] == width]
        )
        if not measured:
            print(f"{width:<7}(no measured config)")
            continue
        best = min(measured, key=measured.get)
        row = {
            "width": width,
            "available": [list(cell) for cell in measured],
            "best": list(best),
            "best_ms": measured[best],
            "rules": {},
        }
        shape = {
            "batch": records[0]["batch"],
            "page_size": page_size,
            "pages_per_seq": width,
            "num_query_rows": heads,
            "query_tokens": 1,
            "cache_dtype": dtype,
            "return_lse": False,
            "max_seqlen_k": records[0]["length"],
        }
        for name, prior in priors.items():
            with patched_kernel(kernel, constants, prior) as choose:
                choice = as_tuple(choose(**shape))
            row["rules"][name] = {
                "choice": list(choice),
                "score": score(measured, choice),
            }
        legacy = as_tuple(kernel._legacy_default_mla_config(**shape, tuned=None))
        row["rules"]["legacy"] = {
            "choice": list(legacy),
            "score": score(measured, legacy),
        }
        for name, entry in row["rules"].items():
            choice = tuple(entry["choice"])
            scored = entry["score"]
            if scored is None:
                print(
                    f"{width:<7}{name:<9}{str(choice):<12}"
                    f"{'-':>10}{measured[best]:>9.4f}{'unmeasured':>9}"
                )
            else:
                chosen, best_ms, regret = scored
                print(
                    f"{width:<7}{name:<9}{str(choice):<12}"
                    f"{chosen:>10.4f}{best_ms:>9.4f}"
                    f"{100 * regret:>8.1f}%"
                )
        rows.append(row)
        print()
    return rows


def main(argv=None):
    args = parse_args(argv)
    import jax.numpy as jnp

    from flywheel_tpu.softmax_attention.kernels import mla_decode as kernel

    result = json.loads((args.results / "fit.json").read_text())
    constants = constants_from_fit(result["fit"])
    dtype = jnp.dtype(result["dtype"])
    priors = {
        "default": calibration.WORKLOAD_PRIOR,
        "long": prior_with_scales(LONG_SCALES),
    }
    print(
        f"constants: column {result['fit']['column_ms']:.6g} ms,"
        f" step {result['fit']['step_ms']:.6g} ms,"
        f" r_step {result['fit']['r_step']:.6g},"
        f" R^2 {result['fit']['r_squared']:.4f}"
    )
    print()
    validation = report_validation(
        kernel, constants, priors, result, args.heads, dtype
    )
    regime = report_regime(
        kernel,
        constants,
        priors,
        args.results / "regime.jsonl",
        result,
        args.heads,
        dtype,
    )
    (args.results / "prior_analysis.json").write_text(
        json.dumps(
            {"validation": validation, "regime": regime},
            indent=2,
        )
    )
    print(f"wrote {args.results / 'prior_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
