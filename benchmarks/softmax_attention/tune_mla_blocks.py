"""Tune flywheel_mla decode/verify block configs on kvcache_scenarios cells.

One cell is one serving shape (impl, scenario, heads, batch, seq, dtype) run
through the same input and cache construction as kvcache_scenarios.py. For
each cell, candidates (target_block_kv, max_items_per_step) are written into a
temporary copy of tuned_configs.json; default_mla_config fits each target to
the actual (block_kv, items_per_step), duplicate fits are measured once, and
the top candidates are re-measured with the benchmark's full timing protocol.

The kernel's own jit caches are keyed on shapes, not on the tuned table, so
building two candidates in one process would silently reuse the first
compiled kernel. The tuner therefore swaps flash_mla_decode for its
__wrapped__ function, leaving the builder's outer jax.jit as the only jit:
each candidate gets a fresh trace, and its compiled step can be re-used for
the finalist re-measurement.

Usage, on the TPU:
  PYTHONPATH=. ~/venv-flywheel-tpu/bin/python \
      benchmarks/softmax_attention/tune_mla_blocks.py \
      --cells /tmp/mla_cells_missing.jsonl --out results/mla_tune

  ... --collect  # merge summary.jsonl into tuned_configs.json (TPU v6)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pathlib
import statistics
import tempfile
import time
import traceback
import warnings

BLOCKS = (512, 1024, 2048, 4096, 8192, 16384)
ITEMS = (1, 2, 4, 8)
DEVICE = "TPU v6"
REPO = pathlib.Path(__file__).resolve().parents[2]
TABLE_PATH = REPO / "flywheel_tpu" / "softmax_attention" / "tuned_configs.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        type=pathlib.Path,
        help="JSONL of cells: scenario, heads, batch, seq, kv_dtype, "
        "decode_tokens, verify_tokens, priority.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("benchmarks/softmax_attention/results/mla_tune"),
        help="output directory for per-cell JSONL and summary.jsonl.",
    )
    parser.add_argument(
        "--blocks",
        default=",".join(map(str, BLOCKS)),
        help="comma-separated target_block_kv candidates.",
    )
    parser.add_argument(
        "--items",
        default=",".join(map(str, ITEMS)),
        help="comma-separated max_items_per_step candidates.",
    )
    parser.add_argument("--top", type=int, default=3, help="finalists to re-measure.")
    parser.add_argument("--limit", type=int, default=None, help="stop after N cells.")
    parser.add_argument("--force", action="store_true", help="redo recorded cells.")
    parser.add_argument(
        "--max-screen-s",
        type=float,
        default=0.25,
        help="target seconds per screening round.",
    )
    parser.add_argument(
        "--max-screen-repeats", type=int, default=3, help="screening rounds."
    )
    parser.add_argument(
        "--collect", action="store_true", help="merge summaries into the table."
    )
    parser.add_argument(
        "--table", type=pathlib.Path, default=TABLE_PATH, help="tune table to edit."
    )
    parser.add_argument("--device", default=DEVICE, help="table device section.")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the cells and exit."
    )
    return parser.parse_args(argv)


def table_key(row):
    """The mla| key of a cell row, exactly as dispatch builds it."""
    from flywheel_tpu.softmax_attention.config import next_pow2

    new_tokens = row["verify_tokens"] if row["scenario"] == "verify" else 1
    seq = row["seq"]
    return (
        f"mla|{row['kv_dtype']}|r{new_tokens * row['heads']}"
        f"|b{next_pow2(row['batch'])}|kv{max(512, next_pow2(seq))}"
    )


def cell_from_row(row):
    from benchmarks.softmax_attention.kvcache_scenarios import Cell

    return Cell(
        impl="flywheel_mla",
        scenario=row["scenario"],
        heads=row["heads"],
        heads_k=1,
        head_dim=512,
        seq=row["seq"],
        chunk=1024,
        batch=row["batch"],
        decode_tokens=row["decode_tokens"],
        page_size=row.get("page_size", 1024),
        verify_tokens=row.get("verify_tokens", 1),
        dtype=row["kv_dtype"],
        kv_dtype=row["kv_dtype"],
        pattern=row.get("pattern", "uniform"),
    )


class Tuner:
    def __init__(self, args):
        warnings.filterwarnings(
            "ignore", message="flywheel_tpu: no usable tuned config"
        )
        import jax

        from benchmarks.softmax_attention import kvcache_scenarios as scenarios
        import flywheel_tpu.softmax_attention.dispatch as dispatch
        import flywheel_tpu.softmax_attention.mla_decode as mla
        from flywheel_tpu.softmax_attention.kernels import mla_decode as kernel

        self.jax = jax
        self.scenarios = scenarios
        self.dispatch = dispatch
        self.mla = mla
        self.kernel = kernel
        self.args = args
        self.base_table = json.loads(args.table.read_text())
        self.orig_table_path = dispatch.TUNED_CONFIGS_PATH
        self.jitted = mla.flash_mla_decode
        self.raw = mla.flash_mla_decode.__wrapped__
        # The builder imports the module attribute inside the call, so this
        # patch makes its outer jit trace every candidate afresh.
        mla.flash_mla_decode = self.raw
        self.hbm = jax.devices()[0].memory_stats().get("bytes_limit")

    def close(self):
        self.mla.flash_mla_decode = self.jitted
        self.dispatch.TUNED_CONFIGS_PATH = self.orig_table_path
        self.dispatch.load_table.cache_clear()

    def write_table(self, key, value):
        """A fresh temporary table with the cell's key set (value None removes
        it), and the dispatch path pointing at it."""
        table = copy.deepcopy(self.base_table)
        section = table.setdefault(self.args.device, {})
        if value is None:
            section.pop(key, None)
        else:
            section[key] = list(value)
        fd, name = tempfile.mkstemp(prefix="mla_tune_", suffix=".json")
        os.close(fd)
        path = pathlib.Path(name)
        path.write_text(json.dumps(table))
        self.dispatch.TUNED_CONFIGS_PATH = path
        self.dispatch.load_table.cache_clear()
        self.dispatch._warn_untuned_once.cache_clear()
        return path

    def fitted(self, cell, table_path):
        """default_mla_config's actual config for the table's entry."""
        import jax.numpy as jnp

        new_tokens = cell.verify_tokens if cell.scenario == "verify" else 1
        max_seqlen_k = int(cell.seq_lens.max())
        self.dispatch.TUNED_CONFIGS_PATH = table_path
        self.dispatch.load_table.cache_clear()
        return self.kernel.default_mla_config(
            batch=cell.batch,
            page_size=cell.page_size,
            pages_per_seq=-(-max_seqlen_k // cell.page_size),
            num_query_rows=new_tokens * cell.heads,
            query_tokens=new_tokens,
            cache_dtype=jnp.dtype(cell.kv_dtype),
            return_lse=False,
            max_seqlen_k=max_seqlen_k,
        )

    def build(self, cell, inputs, table_path):
        self.dispatch.TUNED_CONFIGS_PATH = table_path
        self.dispatch.load_table.cache_clear()
        return self.scenarios.build_flywheel_mla_decode(cell, inputs, None)

    def screen(self, step, fresh):
        """Median ms over short rounds, and the compile seconds."""
        jax = self.jax
        start = time.perf_counter()
        out, state = jax.block_until_ready(step(fresh()))
        compile_s = time.perf_counter() - start
        start = time.perf_counter()
        out, state = jax.block_until_ready(step(state))
        one_call_s = time.perf_counter() - start
        if one_call_s >= 5.0:
            iters, repeats, warmup = 1, 2, 0
        else:
            iters = max(1, math.ceil(self.args.max_screen_s / one_call_s))
            repeats, warmup = self.args.max_screen_repeats, 1
        for _ in range(warmup):
            out, state = step(state)
        jax.block_until_ready((out, state))
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(iters):
                out, state = step(state)
            jax.block_until_ready(out)
            samples.append((time.perf_counter() - start) * 1e3 / iters)
        return statistics.median(samples), compile_s

    def run_cell(self, row, index, total):
        from benchmarks.common.timing import time_call

        cell = cell_from_row(row)
        key = table_key(row)
        out_dir = self.args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        cell_path = out_dir / (cell.key.replace("/", "_") + ".jsonl")
        rows = []

        def write(record):
            rows.append(record)
            with cell_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

        if self.hbm and cell.cache_bytes > 0.9 * self.hbm:
            summary = {
                "kind": "summary",
                "cell": cell.key,
                "table_key": key,
                "status": "skipped_hbm",
                "cache_gib": round(cell.cache_bytes / 2**30, 2),
            }
            write(summary)
            print(f"[{index}/{total}] {cell.key}: skipped, cache too large")
            return summary

        inputs = self.scenarios.make_inputs(cell)
        existing = self.base_table.get(self.args.device, {}).get(key)
        targets = []
        if existing is not None:
            targets.append(("existing", tuple(existing)))
        targets.append(("heuristic", None))
        for block in self.args.blocks.split(","):
            for items in self.args.items.split(","):
                targets.append(("grid", (int(block), int(items))))

        measured = {}  # actual config -> (target source, target, ms, compile_s)
        temp_paths = []
        steps = {}
        try:
            for source, target in targets:
                if target is None:
                    path = self.write_table(key, None)
                else:
                    path = self.write_table(key, target)
                temp_paths.append(path)
                try:
                    config = self.fitted(cell, path)
                except Exception as error:  # pylint: disable=broad-except
                    write({
                        "kind": "candidate",
                        "cell": cell.key,
                        "table_key": key,
                        "source": source,
                        "target": None if target is None else list(target),
                        "status": "fit_error",
                        "error": f"{type(error).__name__}: {error}"[:300],
                    })
                    continue
                actual = (config.block_kv, config.items_per_step)
                if actual in measured:
                    write({
                        "kind": "candidate",
                        "cell": cell.key,
                        "table_key": key,
                        "source": source,
                        "target": None if target is None else list(target),
                        "actual": list(actual),
                        "status": "duplicate",
                    })
                    continue
                try:
                    step, fresh, info = self.build(cell, inputs, path)
                    ms, compile_s = self.screen(step, fresh)
                except Exception as error:  # pylint: disable=broad-except
                    write({
                        "kind": "candidate",
                        "cell": cell.key,
                        "table_key": key,
                        "source": source,
                        "target": None if target is None else list(target),
                        "actual": list(actual),
                        "status": "run_error",
                        "error": f"{type(error).__name__}: {error}"[:300],
                    })
                    continue
                measured[actual] = (source, target, ms, compile_s, step, fresh, info)
                write({
                    "kind": "candidate",
                    "cell": cell.key,
                    "table_key": key,
                    "source": source,
                    "target": None if target is None else list(target),
                    "actual": list(actual),
                    "median_ms": ms,
                    "compile_s": compile_s,
                    "steps": cell.decode_tokens,
                    "status": "ok",
                })

            if not measured:
                summary = {
                    "kind": "summary",
                    "cell": cell.key,
                    "table_key": key,
                    "status": "no_candidate",
                }
                write(summary)
                print(f"[{index}/{total}] {cell.key}: no candidate")
                return summary

            ranked = sorted(measured, key=lambda config: measured[config][2])
            final = {}
            for config in ranked[: self.args.top]:
                source, target, ms, compile_s, step, fresh, info = measured[config]
                timing, _ = time_call(step, fresh())
                final[config] = timing["median_ms"]
                write({
                    "kind": "finalist",
                    "cell": cell.key,
                    "table_key": key,
                    "source": source,
                    "target": None if target is None else list(target),
                    "actual": list(config),
                    "median_ms": timing["median_ms"],
                    "samples_ms": timing["samples_ms"],
                    "status": "ok",
                })

            baseline_actual = None
            baseline_ms = None
            baseline_source = None
            baseline_target = None
            for config, (src, tgt, ms, _, _, _, _) in measured.items():
                if src in ("existing", "heuristic"):
                    baseline_actual = config
                    baseline_source = src
                    baseline_target = tgt
                    break
            if baseline_actual is not None and baseline_actual not in final:
                # Re-measure the baseline with the full protocol, so the
                # speedup is not a screening-vs-full comparison.
                step, fresh = measured[baseline_actual][4], measured[baseline_actual][5]
                timing, _ = time_call(step, fresh())
                final[baseline_actual] = timing["median_ms"]
                write({
                    "kind": "finalist",
                    "cell": cell.key,
                    "table_key": key,
                    "source": baseline_source,
                    "target": None
                    if baseline_target is None
                    else list(baseline_target),
                    "actual": list(baseline_actual),
                    "median_ms": timing["median_ms"],
                    "samples_ms": timing["samples_ms"],
                    "baseline": True,
                    "status": "ok",
                })
            if baseline_actual is not None:
                baseline_ms = final[baseline_actual]

            best = min(final, key=final.get)
            source, target, _, _, _, _, _ = measured[best]
            ordered = sorted(final, key=final.get)
            second = ordered[1] if len(ordered) > 1 else None
            summary = {
                "kind": "summary",
                "cell": cell.key,
                "table_key": key,
                "cell_row": row,
                "best_target": None if target is None else list(target),
                "best_actual": list(best),
                "best_ms": final[best],
                "second_actual": None if second is None else list(second),
                "second_ms": None if second is None else final[second],
                "margin": None
                if second is None
                else round(final[second] / final[best] - 1, 6),
                "baseline_target": None
                if baseline_target is None
                else list(baseline_target),
                "baseline_actual": None
                if baseline_actual is None
                else list(baseline_actual),
                "baseline_ms": baseline_ms,
                "baseline_source": baseline_source,
                "speedup": None
                if not baseline_ms
                else round(baseline_ms / final[best], 4),
                "tested": sum(1 for r in rows if r["kind"] == "candidate" and r["status"] == "ok"),
                "status": "ok",
            }
            write(summary)
            print(
                f"[{index}/{total}] {cell.key}: best {best} "
                f"({final[best]:.4f} ms) vs {baseline_source} "
                f"{baseline_actual} ({baseline_ms:.4f} ms), "
                f"tested {summary['tested']}",
                flush=True,
            )
            return summary
        finally:
            self.jax.clear_caches()
            for path in temp_paths:
                try:
                    path.unlink()
                except OSError:
                    pass


def collect(args):
    """Merge summary.jsonl into args.table's device section."""
    summary_path = args.out / "summary.jsonl"
    table = json.loads(args.table.read_text())
    section = table.setdefault(args.device, {})
    best = {}
    for line in summary_path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") != "summary" or record.get("status") != "ok":
            continue
        key = record["table_key"]
        # A duplicate key from another shape: keep the fastest absolute cell,
        # which only matters when both summaries exist.
        if key not in best or record["best_ms"] < best[key]["best_ms"]:
            best[key] = record
    added, replaced, kept = [], [], []
    for key, record in sorted(best.items()):
        best_target = record.get("best_target")
        if best_target is None:
            best_target = record["best_actual"]
        value = [int(best_target[0]), int(best_target[1])]
        old = section.get(key)
        if old is None:
            section[key] = value
            added.append((key, value))
        elif record.get("speedup") is not None and record["speedup"] >= 1.02:
            section[key] = value
            replaced.append((key, old, value, record["speedup"]))
        else:
            kept.append((key, old, value, record.get("speedup")))
    backup = args.table.with_suffix(args.table.suffix + ".bak")
    backup.write_text(args.table.read_text())
    args.table.write_text(json.dumps(table, indent=2) + "\n")
    print(f"added {len(added)}, replaced {len(replaced)}, kept {len(kept)}")
    for item in added:
        print("  +", item)
    for item in replaced:
        print("  ~", item)
    for item in kept:
        print("  =", item)


def main(argv=None):
    args = parse_args(argv)
    if args.collect:
        collect(args)
        return 0
    rows = [
        json.loads(line)
        for line in args.cells.read_text().splitlines()
        if line.strip()
    ]
    if args.dry_run:
        print(f"{len(rows)} cells")
        for row in rows:
            print(json.dumps(row))
        return 0
    done = set()
    if not args.force and (args.out / "summary.jsonl").exists():
        for line in (args.out / "summary.jsonl").read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind") == "summary" and record.get("status") == "ok":
                done.add(record["cell"])
    tuner = Tuner(args)
    try:
        ran = 0
        for index, row in enumerate(rows, 1):
            from benchmarks.softmax_attention.kvcache_scenarios import Cell

            cell = cell_from_row(row)
            if cell.key in done:
                print(f"[{index}/{len(rows)}] {cell.key}: already done")
                continue
            if args.limit is not None and ran >= args.limit:
                break
            try:
                summary = tuner.run_cell(row, index, len(rows))
            except Exception:  # pylint: disable=broad-except
                print(f"[{index}/{len(rows)}] {cell.key}: CRASH")
                traceback.print_exc()
                summary = {
                    "kind": "summary",
                    "cell": cell.key,
                    "table_key": table_key(row),
                    "status": "error",
                    "error": traceback.format_exc()[-1000:],
                }
            with (args.out / "summary.jsonl").open("a") as handle:
                handle.write(json.dumps(summary) + "\n")
            ran += 1
    finally:
        tuner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
