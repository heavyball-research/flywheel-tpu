"""Refine the MLA tune dataset: fill coarse gaps, bisect sharp transitions.

Stage 1 fills the coarse batch/seq grid for six slices (fp8 h16/h32/h64 and
bf16 h32/h64/h128), including points whose key already existed and therefore
were not part of the main 432-cell run.

Stage 2 finds adjacent points whose best (block_kv, items) differs and whose
best-vs-second margin is at least --sharp (default 5%), then measures the
midpoint. It repeats for --levels rounds; a midpoint that lands on a block
different from both ends creates two new intervals, which the next round can
bisect too.

Stage 3 writes dataset.jsonl, transitions.jsonl and report.md. The tune table
is not modified: this run is data collection for the future MLA block
heuristic, not a runtime-table update.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import traceback

from benchmarks.softmax_attention import tune_mla_blocks as tune

# The six slices the main run's sharp intervals point at.
SLICES = (
    ("float8_e4m3fn", 16),
    ("float8_e4m3fn", 32),
    ("float8_e4m3fn", 64),
    ("bfloat16", 32),
    ("bfloat16", 64),
    ("bfloat16", 128),
)
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
SEQS = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
BATCH_SLICE_SEQ = 4096
SEQ_SLICE_BATCH = 64
SEQ_STEP = 256
REPO = pathlib.Path(__file__).resolve().parents[2]
MAIN_OUT = REPO / "benchmarks/softmax_attention/results/mla_tune"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-summary", type=pathlib.Path,
                        default=MAIN_OUT / "summary.jsonl")
    parser.add_argument("--main-out", type=pathlib.Path, default=MAIN_OUT)
    parser.add_argument("--out", type=pathlib.Path,
                        default=REPO / "benchmarks/softmax_attention/results/mla_tune_refine")
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument(
        "--scope",
        choices=("six", "all"),
        default="all",
        help="bisect only the six coarse slices, or every decode group.",
    )
    parser.add_argument("--sharp", type=float, default=0.05)
    parser.add_argument("--batch-res", type=int, default=1)
    parser.add_argument("--seq-res", type=int, default=SEQ_STEP)
    parser.add_argument("--max-cells", type=int, default=250)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--table", type=pathlib.Path, default=tune.TABLE_PATH)
    parser.add_argument("--device", default=tune.DEVICE)
    return parser.parse_args(argv)


def steps(batch, seq):
    """The main run's decode_tokens rule, so generated cells match its keys."""
    tokens = batch * seq
    if tokens <= 65536:
        return 128
    if tokens <= 524288:
        return 64
    if tokens <= 4194304:
        return 32
    return 8


def row(dtype, heads, batch, seq, tag):
    return {
        "priority": 0,
        "scenario": "decode",
        "heads": heads,
        "batch": batch,
        "seq": seq,
        "verify_tokens": 1,
        "kv_dtype": dtype,
        "decode_tokens": steps(batch, seq),
        "page_size": 1024,
        "tag": tag,
    }


def key_of(row):
    return tune.cell_from_row(row).key


def read_jsonl(path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def finalist_info(path):
    """(best_actual, best_ms, second_actual, second_ms, margin) of a cell."""
    finals = []
    if path.exists():
        for record in read_jsonl(path):
            if record.get("kind") == "finalist" and record.get("status") == "ok":
                finals.append(record)
    finals.sort(key=lambda record: record["median_ms"])
    if not finals:
        return None, None, None, None, 0.0
    best = finals[0]
    second = None
    for candidate in finals[1:]:
        if tuple(candidate["actual"]) != tuple(best["actual"]):
            second = candidate
            break
    if second is None:
        return (tuple(best["actual"]), best["median_ms"], None, None, 0.0)
    margin = second["median_ms"] / best["median_ms"] - 1
    return (
        tuple(best["actual"]),
        best["median_ms"],
        tuple(second["actual"]),
        second["median_ms"],
        margin,
    )


class Points:
    def __init__(self, main_summary, main_out, refine_summary, refine_out):
        self.main_out = main_out
        self.refine_out = refine_out
        self.points = {}
        self.attempted = set()
        for record in read_jsonl(main_summary):
            if record.get("kind") != "summary":
                continue
            key = record.get("cell")
            if record.get("status") != "ok" or not key:
                continue
            best, best_ms, second, second_ms, margin = finalist_info(
                main_out / (key + ".jsonl")
            )
            if best is None:
                best = tuple(record["best_actual"])
                best_ms = record["best_ms"]
            self.points[key] = {
                "summary": record,
                "block": best,
                "block_ms": best_ms,
                "second": second,
                "second_ms": second_ms,
                "margin": margin,
                "out": main_out,
                "stage": "main",
            }
        for record in read_jsonl(refine_summary):
            if record.get("kind") != "summary":
                continue
            key = record.get("cell")
            if not key:
                continue
            self.attempted.add(key)
            if record.get("status") != "ok":
                continue
            margin = record.get("margin")
            second = record.get("second_actual")
            second_ms = record.get("second_ms")
            if margin is None or second is None:
                best, best_ms, second2, second_ms2, margin2 = finalist_info(
                    refine_out / (key + ".jsonl")
                )
                if second is None:
                    second = second2
                if second_ms is None:
                    second_ms = second_ms2
                if margin is None:
                    margin = margin2
            self.points[key] = {
                "summary": record,
                "block": tuple(record["best_actual"]),
                "block_ms": record["best_ms"],
                "second": None if second is None else tuple(second),
                "second_ms": second_ms,
                "margin": 0.0 if margin is None else margin,
                "out": refine_out,
                "stage": record.get("stage", "refine"),
            }

    def add(self, record, stage):
        key = record["cell"]
        self.attempted.add(key)
        if record.get("status") != "ok":
            return
        self.points[key] = {
            "summary": record,
            "block": tuple(record["best_actual"]),
            "block_ms": record["best_ms"],
            "second": None
            if record.get("second_actual") is None
            else tuple(record["second_actual"]),
            "second_ms": record.get("second_ms"),
            "margin": 0.0 if record.get("margin") is None else record["margin"],
            "out": self.refine_out,
            "stage": stage,
        }


def coarse_rows():
    seen = set()
    rows = []
    for dtype, heads in SLICES:
        for batch in BATCHES:
            candidate = row(dtype, heads, batch, BATCH_SLICE_SEQ, "coarse_batch")
            key = key_of(candidate)
            if key not in seen:
                seen.add(key)
                rows.append(candidate)
        for seq in SEQS:
            candidate = row(dtype, heads, SEQ_SLICE_BATCH, seq, "coarse_seq")
            key = key_of(candidate)
            if key not in seen:
                seen.add(key)
                rows.append(candidate)
    return rows


def group_points(points, dtype, heads, axis, fixed):
    found = []
    for point in points.points.values():
        cell = point["summary"]["cell_row"]
        if cell.get("scenario") != "decode":
            continue
        if cell.get("kv_dtype") != dtype or cell.get("heads") != heads:
            continue
        if cell.get("verify_tokens", 1) != 1:
            continue
        if axis == "batch":
            if cell.get("seq") != fixed:
                continue
            found.append((cell["batch"], point))
        else:
            if cell.get("batch") != fixed:
                continue
            found.append((cell["seq"], point))
    found.sort(key=lambda item: item[0])
    return found


def slice_points(points, dtype, heads, axis):
    fixed = BATCH_SLICE_SEQ if axis == "batch" else SEQ_SLICE_BATCH
    return group_points(points, dtype, heads, axis, fixed)


def all_groups(points):
    """(dtype, heads, axis, fixed) for every decode group with >=2 points."""
    counts = collections.Counter()
    for point in points.points.values():
        cell = point["summary"]["cell_row"]
        if cell.get("scenario") != "decode":
            continue
        if cell.get("verify_tokens", 1) != 1:
            continue
        dtype, heads = cell["kv_dtype"], cell["heads"]
        counts[(dtype, heads, "batch", cell["seq"])] += 1
        counts[(dtype, heads, "seq", cell["batch"])] += 1
    for group, count in sorted(counts.items()):
        if count >= 2:
            yield group


def find_intervals(points, sharp, batch_res, seq_res, scope="all"):
    intervals = []
    seen = set()
    if scope == "six":
        groups = (
            (dtype, heads, axis,
             BATCH_SLICE_SEQ if axis == "batch" else SEQ_SLICE_BATCH)
            for dtype, heads in SLICES
            for axis in ("batch", "seq")
        )
    else:
        groups = all_groups(points)
    for dtype, heads, axis, fixed in groups:
        pts = group_points(points, dtype, heads, axis, fixed)
        for (a_value, a), (b_value, b) in zip(pts, pts[1:]):
            if a["block"] == b["block"]:
                continue
            strength = max(a["margin"], b["margin"])
            gap = b_value - a_value
            resolution = batch_res if axis == "batch" else seq_res
            if strength < sharp or gap <= resolution:
                continue
            key = (dtype, heads, axis, fixed, a_value, b_value)
            if key in seen:
                continue
            seen.add(key)
            intervals.append({
                "dtype": dtype,
                "heads": heads,
                "axis": axis,
                "fixed": fixed,
                "a": a_value,
                "b": b_value,
                "a_block": a["block"],
                "b_block": b["block"],
                "a_margin": a["margin"],
                "b_margin": b["margin"],
                "strength": strength,
            })
    return intervals


def midpoint_rows(intervals):
    rows = []
    for interval in intervals:
        if interval["axis"] == "batch":
            middle = (interval["a"] + interval["b"]) // 2
            if middle <= interval["a"] or middle >= interval["b"]:
                continue
            rows.append(row(
                interval["dtype"], interval["heads"], middle,
                interval["fixed"], "bisect_batch",
            ))
        else:
            middle = (interval["a"] + interval["b"]) // 2
            rounded = (middle // SEQ_STEP) * SEQ_STEP
            if interval["a"] < rounded < interval["b"]:
                middle = rounded
            if middle <= interval["a"] or middle >= interval["b"]:
                continue
            rows.append(row(
                interval["dtype"], interval["heads"], interval["fixed"],
                middle, "bisect_seq",
            ))
    return rows


def run_rows(tuner, points, rows, summary_path, stage, max_cells):
    todo = []
    seen = set()
    skipped = 0
    for candidate in rows:
        key = key_of(candidate)
        if key in points.points or key in points.attempted or key in seen:
            skipped += 1
            continue
        seen.add(key)
        todo.append((key, candidate))
    if max_cells is not None:
        todo = todo[:max_cells]
    print(
        "== %s: %d cells to run (%d skipped as already measured or duplicate)"
        % (stage, len(todo), skipped),
        flush=True,
    )
    for index, (key, candidate) in enumerate(todo, 1):
        try:
            summary = tuner.run_cell(candidate, index, len(todo))
        except Exception:  # pylint: disable=broad-except
            summary = {
                "kind": "summary",
                "cell": key,
                "table_key": tune.table_key(candidate),
                "cell_row": candidate,
                "status": "error",
                "error": traceback.format_exc()[-1000:],
            }
        summary["stage"] = stage
        with summary_path.open("a") as handle:
            handle.write(json.dumps(summary) + "\n")
        points.add(summary, stage)
        print(
            "  [%s %d/%d] %s -> %s"
            % (
                stage,
                index,
                len(todo),
                key,
                summary.get("status"),
            ),
            flush=True,
        )
    return len(todo)


def write_outputs(points, out_dir, intervals_left, scope):
    dataset_path = out_dir / "dataset.jsonl"
    with dataset_path.open("w") as handle:
        for key, point in sorted(points.points.items()):
            record = {
                "cell": key,
                "cell_row": point["summary"].get("cell_row"),
                "stage": point["stage"],
                "best_actual": list(point["block"]),
                "best_ms": point["block_ms"],
                "second_actual": None
                if point["second"] is None
                else list(point["second"]),
                "second_ms": point["second_ms"],
                "margin": point["margin"],
                "speedup": point["summary"].get("speedup"),
            }
            handle.write(json.dumps(record) + "\n")
    transitions_path = out_dir / "transitions.jsonl"
    with transitions_path.open("w") as handle:
        for dtype, heads, axis, fixed in all_groups(points):
            pts = group_points(points, dtype, heads, axis, fixed)
            for (a_value, a), (b_value, b) in zip(pts, pts[1:]):
                a_row = a["summary"].get("cell_row") or {}
                b_row = b["summary"].get("cell_row") or {}
                record = {
                    "dtype": dtype,
                    "heads": heads,
                    "axis": axis,
                    "fixed": fixed,
                    "a": a_value,
                    "b": b_value,
                    "a_decode_tokens": a_row.get("decode_tokens"),
                    "b_decode_tokens": b_row.get("decode_tokens"),
                    "a_block": list(a["block"]),
                    "b_block": list(b["block"]),
                    "a_margin": a["margin"],
                    "b_margin": b["margin"],
                    "strength": max(a["margin"], b["margin"]),
                    "changed": a["block"] != b["block"],
                }
                handle.write(json.dumps(record) + "\n")
    report_path = out_dir / "report.md"
    with report_path.open("w") as handle:
        handle.write("# MLA tune refinement report\n\n")
        handle.write("Bisection scope: %s\n\n" % scope)
        handle.write("## Remaining sharp intervals\n\n")
        if intervals_left:
            for interval in intervals_left:
                handle.write(
                    "- %s h%d %s (fixed %s) %s->%s: %s (m %.2f%%) -> %s "
                    "(m %.2f%%), strength %.2f%%\n"
                    % (
                        interval["dtype"],
                        interval["heads"],
                        interval["axis"],
                        interval["fixed"],
                        interval["a"],
                        interval["b"],
                        interval["a_block"],
                        interval["a_margin"] * 100,
                        interval["b_block"],
                        interval["b_margin"] * 100,
                        interval["strength"] * 100,
                    )
                )
        else:
            handle.write("None.\n")
        handle.write("\n## Step functions (six coarse slices)\n\n")
        for dtype, heads in SLICES:
            for axis in ("batch", "seq"):
                pts = slice_points(points, dtype, heads, axis)
                if not pts:
                    continue
                handle.write("### %s h%d %s\n\n" % (dtype, heads, axis))
                for value, point in pts:
                    cell_row = point["summary"].get("cell_row") or {}
                    handle.write(
                        "- %s %d (n%s): %s (%.4f ms, margin %.2f%%)\n"
                        % (
                            axis,
                            value,
                            cell_row.get("decode_tokens"),
                            point["block"],
                            point["block_ms"],
                            point["margin"] * 100,
                        )
                    )
                handle.write("\n")
    print("wrote %s, %s, %s" % (dataset_path, transitions_path, report_path))


def main(argv=None):
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.jsonl"
    points = Points(
        args.main_summary,
        args.main_out,
        summary_path,
        args.out,
    )
    print(
        "loaded %d main/refine points, %d attempted"
        % (len(points.points), len(points.attempted)),
        flush=True,
    )
    coarse = coarse_rows()
    if args.dry_run:
        print("coarse rows: %d" % len(coarse))
        print("missing coarse cells:")
        for candidate in coarse:
            key = key_of(candidate)
            if key not in points.points:
                print(" ", key)
        intervals = find_intervals(
            points, args.sharp, args.batch_res, args.seq_res, args.scope
        )
        print("sharp intervals: %d" % len(intervals))
        for interval in intervals:
            print(" ", interval)
        return 0

    tune_args = tune.parse_args([])
    tune_args.out = args.out
    tune_args.table = args.table
    tune_args.device = args.device
    tuner = tune.Tuner(tune_args)
    total = 0
    try:
        total += run_rows(
            tuner, points, coarse, summary_path, "coarse", args.max_cells - total
        )
        intervals = find_intervals(
            points, args.sharp, args.batch_res, args.seq_res, args.scope
        )
        for level in range(1, args.levels + 1):
            if total >= args.max_cells:
                print("max cells reached", flush=True)
                break
            print(
                "== bisect level %d: %d sharp intervals" % (level, len(intervals)),
                flush=True,
            )
            for interval in intervals:
                print("   %s" % interval, flush=True)
            mids = midpoint_rows(intervals)
            ran = run_rows(
                tuner,
                points,
                mids,
                summary_path,
                "bisect%d" % level,
                args.max_cells - total,
            )
            total += ran
            if ran == 0:
                break
            intervals = find_intervals(
                points, args.sharp, args.batch_res, args.seq_res, args.scope
            )
        intervals = find_intervals(
            points, args.sharp, args.batch_res, args.seq_res, args.scope
        )
        write_outputs(points, args.out, intervals, args.scope)
        print("TOTAL NEW CELLS %d" % total, flush=True)
    finally:
        tuner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
