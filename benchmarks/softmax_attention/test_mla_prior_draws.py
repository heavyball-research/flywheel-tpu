"""Time the heuristic's pick on batches drawn from the shipped prior.

Each draw is one 64-request batch sampled i.i.d. from WORKLOAD_PRIOR, clipped
to a 16k-token block table. Every candidate config is timed on every draw, so
each draw has a *measured* best config and the rules' regrets are realized,
not predicted. The summary also reports the best single config in hindsight
over all draws, which is what a static choice can hope to match.

Run on the TPU host, from the repo root:
  cd ~/flywheel-tpu-new && PYTHONPATH=. ~/venv-flywheel-tpu/bin/python \
      benchmarks/softmax_attention/test_mla_prior_draws.py \
      --results benchmarks/softmax_attention/results/mla_calibration_v2
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

from flywheel_tpu.softmax_attention import mla_decode_calibration as calibration

DRAWS = 5
BATCH = 64
PAGE_SIZE = 128
CAPACITY_TOKENS = 16384
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
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(
            "benchmarks/softmax_attention/results/mla_prior_draws"
        ),
    )
    return parser.parse_args(argv)


def sample_lengths(prior, rng, batch, capacity):
    """batch i.i.d. lengths from the mixture, clipped to [1, capacity]."""
    bases = rng.choice(
        len(prior), size=batch, p=[base.weight for base in prior]
    )
    lengths = []
    for base_index in bases:
        base = prior[base_index]
        components = base.components
        component = components[
            rng.choice(
                len(components), p=[c.weight for c in components]
            )
        ]
        if component.pareto_alpha:
            alpha = component.pareto_alpha
            scale = component.mean * (alpha - 1.0) / alpha
            value = scale * (1.0 - rng.random()) ** (-1.0 / alpha)
        else:
            spread = math.sqrt(math.log1p(component.cv**2))
            location = math.log(component.mean) - 0.5 * spread**2
            value = math.exp(location + spread * rng.standard_normal())
        lengths.append(min(max(value, 1.0), float(capacity)))
    return np.asarray(lengths, np.int64)


def correlation(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.size < 2 or left.std() == 0.0 or right.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def main(argv=None):
    import jax.numpy as jnp

    from benchmarks.common.timing import time_call
    from benchmarks.softmax_attention.analyze_mla_prior import (
        LONG_SCALES,
        as_tuple,
        constants_from_fit,
        patched_kernel,
        prior_with_scales,
    )
    from benchmarks.softmax_attention.calibrate_mla_decode_config import (
        Harness,
        model_counts,
    )
    from flywheel_tpu.softmax_attention.kernels import mla_decode as kernel

    args = parse_args(argv)
    result = json.loads((args.results / "fit.json").read_text())
    fit = result["fit"]
    constants = constants_from_fit(fit)
    dtype = jnp.dtype(result["dtype"])
    heads = result["heads"]
    page_size = result["page_size"]
    pages = CAPACITY_TOKENS // page_size
    per_item = {int(b): v for b, v in fit["per_item_ms"].items()}
    step_ms = fit["step_ms"]

    shape = {
        "batch": BATCH,
        "page_size": page_size,
        "pages_per_seq": pages,
        "num_query_rows": heads,
        "query_tokens": 1,
        "cache_dtype": dtype,
        "return_lse": False,
        "max_seqlen_k": CAPACITY_TOKENS,
    }
    rules = {}
    for name, prior in (
        ("default", calibration.WORKLOAD_PRIOR),
        ("long", prior_with_scales(LONG_SCALES)),
    ):
        with patched_kernel(kernel, constants, prior) as choose:
            rules[name] = as_tuple(choose(**shape))
    rules["legacy"] = as_tuple(
        kernel._legacy_default_mla_config(**shape, tuned=None)
    )
    print(
        f"draws: {args.draws} x batch {BATCH}, capacity"
        f" {CAPACITY_TOKENS}, candidate grid {len(BLOCKS) * len(ITEMS)}"
    )
    print(f"choices at this shape: {rules}")

    harness = Harness(heads, page_size, result["dtype"])
    vmem_budget = constants.vmem_fraction * harness.kernel.vmem_limit_bytes()
    draws = []
    try:
        for seed in range(args.draws):
            rng = np.random.default_rng(seed)
            lengths = sample_lengths(
                calibration.WORKLOAD_PRIOR, rng, BATCH, CAPACITY_TOKENS
            )
            measured = {}
            predicted = {}
            for block in BLOCKS:
                for items in ITEMS:
                    if harness.vmem_bytes(BATCH, block, items) > vmem_budget:
                        continue
                    try:
                        step, cache = harness.build(
                            lengths, block, items, table_pages=pages
                        )
                        timing, _ = time_call(step, cache)
                    except Exception as error:  # pylint: disable=broad-except
                        print(
                            f"seed {seed} {block}/{items}:"
                            f" {type(error).__name__}: {error}"[:200]
                        )
                        continue
                    finally:
                        harness.jax.clear_caches()
                    measured[(block, items)] = timing["median_ms"]
                    _, steps = model_counts(lengths, block, items)
                    predicted[(block, items)] = (
                        steps * items * per_item[block] + steps * step_ms
                    )
            best = min(measured, key=measured.get)
            row = {
                "seed": seed,
                "mean_length": float(lengths.mean()),
                "max_length": int(lengths.max()),
                "best": list(best),
                "best_ms": measured[best],
                "measured_ms": {
                    f"{b}/{i}": value for (b, i), value in measured.items()
                },
                "predicted_ms": {
                    f"{b}/{i}": value for (b, i), value in predicted.items()
                },
                "rho": correlation(
                    list(predicted.values()), list(measured.values())
                ),
                "rules": {},
            }
            for name, choice in rules.items():
                if choice not in measured:
                    row["rules"][name] = {"choice": list(choice)}
                    continue
                row["rules"][name] = {
                    "choice": list(choice),
                    "ms": measured[choice],
                    "regret": measured[choice] / measured[best] - 1.0,
                }
            draws.append(row)
            print(
                f"seed {seed}: mean {lengths.mean():.0f}, max"
                f" {lengths.max()}, best {best} {measured[best]:.4f} ms,"
                f" rho {row['rho']:.3f}"
            )
            for name, entry in row["rules"].items():
                if "regret" in entry:
                    print(
                        f"  {name:<8}{str(tuple(entry['choice'])):<12}"
                        f"{entry['ms']:.4f} ms  regret"
                        f" {100 * entry['regret']:.1f}%"
                    )
    finally:
        harness.close()

    print("\nsummary over draws:")
    print(
        f"{'rule':<10}{'choice':<12}{'mean regret':>12}{'worst':>9}"
    )
    summary = {}
    for name, choice in rules.items():
        regrets = [
            row["rules"][name]["regret"]
            for row in draws
            if "regret" in row["rules"][name]
        ]
        if not regrets:
            continue
        summary[name] = {
            "choice": list(choice),
            "mean_regret": float(np.mean(regrets)),
            "worst_regret": float(np.max(regrets)),
        }
        print(
            f"{name:<10}{str(choice):<12}{100 * np.mean(regrets):>11.1f}%"
            f"{100 * np.max(regrets):>8.1f}%"
        )
    cells = sorted(
        set.intersection(*(set(row["measured_ms"]) for row in draws))
    )
    mean_time = {
        cell: float(
            np.mean([row["measured_ms"][cell] for row in draws])
        )
        for cell in cells
    }
    static_best = min(mean_time, key=mean_time.get)
    static_regrets = [
        row["measured_ms"][static_best] / row["best_ms"] - 1.0
        for row in draws
    ]
    print(
        f"best single config in hindsight: {static_best}, mean"
        f" regret {100 * np.mean(static_regrets):.1f}%"
    )
    summary["hindsight_best"] = {
        "choice": static_best,
        "mean_regret": float(np.mean(static_regrets)),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "draws.json").write_text(
        json.dumps({"draws": draws, "summary": summary}, indent=2)
    )
    print(f"wrote {args.out / 'draws.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
