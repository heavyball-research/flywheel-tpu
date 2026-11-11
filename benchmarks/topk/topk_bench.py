#!/usr/bin/env python3
"""Top-k of score rows on one device: flywheel's SparseCore top-k against
XLA's top-k ops.

    PYTHONPATH=. python benchmarks/topk/topk_bench.py --impl flywheel \\
        --suite dsa,sampling --output results/benchmark/topk/v7x.jsonl
    python benchmarks/topk/topk_bench.py --list
    python benchmarks/topk/topk_bench.py --table results/benchmark/topk/*.jsonl

Suites, each in float32 and bfloat16 on standard-normal scores:

  dsa       DSA indexer rows: T in {1, 128, 4096}, N in {8K, 32K, 128K}, k 2048;
            full rows, or causal prefill rows (row t of length N - T + 1 + t)
  sampling  vocabulary rows: T in {1, 32, 256, 512}, N in {128000, 151936,
            262144}, k in {1, 50, 1024}, full rows

Impls: flywheel (row_lengths for causal cells), lax_top_k and approx_max_k
(recall_target 0.95), both masked past each row's length.

Metrics, per call: kernel_us is device time of the jitted function's XLA
module, all ops included; sparse_core_us is the SparseCore calls alone;
wall_us is wall-clock of queued calls, host dispatch included; compile_s is
lower plus compile; hbm_fraction is the scores' bytes read once over
kernel_us at the HBM peak. Positions are checked against lax.top_k on the
first rows: exact as a set for flywheel and lax_top_k, recall for
approx_max_k. To compare flywheel versions, run once per checkout and join
the result files with --table.
"""

# jax loads only where a TPU run needs it, so --list and --table never claim
# the TPU.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import tempfile
import time

from benchmarks.common.records import recorded, write_record
from benchmarks.common.timing import time_call
from benchmarks.softmax_attention.segments import load as load_trace

IMPLS = ("flywheel", "lax_top_k", "approx_max_k")
APPROX_RECALL_TARGET = 0.95
DSA_TOPK = 2048
DSA_ROWS = (1, 128, 4096)
DSA_COLUMNS = (8192, 32768, 131072)
SAMPLING_ROWS = (1, 32, 256, 512)
SAMPLING_COLUMNS = (128000, 151936, 262144)
SAMPLING_TOPK = (1, 50, 1024)
DTYPES = ("float32", "bfloat16")
TRACE_CALLS = 5
CHECKED_ROWS = 8
# HBM peak per JAX device in bytes/s, by device_kind. A TPU7x chip is two
# devices.
HBM_PEAK = {"TPU v6 lite": 1638e9, "TPU7x": 7380e9 / 2}
# Trace rows of a call's module span and of each SparseCore's calls; a
# SparseCore call's done event spans the call.
TENSORCORE_PROCESS = "/device:TPU:0"
TENSORCORE_MODULES_ROW = "XLA Modules"
SPARSECORE_OPS_ROW = "Sparse Core Ops"
SPARSECORE_DONE_MARKER = ".call-done"
# Tracing every tile overflows the device trace buffer; one tile suffices.
TRACED_SPARSECORE_TILES = 1


@dataclasses.dataclass(frozen=True)
class Cell:
    suite: str
    num_rows: int
    num_columns: int
    k: int
    dtype: str
    lengths: str

    @property
    def key(self) -> str:
        return (
            f"{self.suite}-t{self.num_rows}-n{self.num_columns}-k{self.k}"
            f"-{self.dtype}-{self.lengths}"
        )


def suite_cells(suite: str) -> list[Cell]:
    if suite == "dsa":
        return [
            Cell(suite, t, n, DSA_TOPK, dtype, lengths)
            for dtype in DTYPES
            for lengths in ("full", "causal")
            for t in DSA_ROWS
            for n in DSA_COLUMNS
            if lengths == "full" or t > 1
        ]
    if suite == "sampling":
        return [
            Cell(suite, t, n, k, dtype, "full")
            for dtype in DTYPES
            for t in SAMPLING_ROWS
            for n in SAMPLING_COLUMNS
            for k in SAMPLING_TOPK
        ]
    raise ValueError(f"unknown suite {suite!r}")


def make_inputs(cell: Cell):
    """(scores, row lengths or None)."""
    import jax
    import jax.numpy as jnp

    key = jax.random.key(cell.num_rows * 7 + cell.num_columns)
    scores = jax.random.normal(
        key, (cell.num_rows, cell.num_columns), jnp.float32
    ).astype(cell.dtype)
    if cell.lengths == "full":
        return scores, None
    first = cell.num_columns - cell.num_rows + 1
    lengths = jnp.maximum(first + jnp.arange(cell.num_rows), 1)
    return scores, lengths.astype(jnp.int32)


def masked(scores, lengths):
    import jax.numpy as jnp

    if lengths is None:
        return scores
    columns = jnp.arange(scores.shape[1])[None, :]
    return jnp.where(columns < lengths[:, None], scores, -jnp.inf)


def impl_fn(impl: str, k: int):
    """f(scores, lengths) -> (T, k) int32 positions."""
    import jax

    if impl == "flywheel":
        from flywheel_tpu.topk import top_k

        return jax.jit(lambda s, n: top_k(s, k, row_lengths=n))
    if impl == "lax_top_k":
        return jax.jit(lambda s, n: jax.lax.top_k(masked(s, n), k)[1])
    if impl == "approx_max_k":
        return jax.jit(
            lambda s, n: jax.lax.approx_max_k(
                masked(s, n), k, recall_target=APPROX_RECALL_TARGET
            )[1]
        )
    raise ValueError(f"unknown impl {impl!r}")


def check_positions(impl, positions, scores, lengths, k) -> dict:
    """Exactness (or recall) of the first rows' positions against
    lax.top_k, ties either way."""
    import jax
    import numpy as np

    rows = min(CHECKED_ROWS, scores.shape[0])
    values = np.asarray(masked(scores, lengths)[:rows].astype("float32"))
    kth = np.asarray(jax.lax.top_k(values, k)[0][:, k - 1])
    found = 0
    exact = True
    for row in range(rows):
        got = np.asarray(positions[row])
        got = got[got >= 0]
        picked = values[row, got]
        above = np.sum(values[row] > kth[row])
        found += min(np.sum(picked >= kth[row]), k)
        exact &= len(np.unique(got)) == len(got) == k
        exact &= bool(np.all(picked >= kth[row]))
        exact &= int(np.sum(picked > kth[row])) == int(above)
    if impl == "approx_max_k":
        return {"recall": found / (rows * k)}
    return {"exact": bool(exact)}


def sparse_core_processes() -> list[str]:
    """Trace processes of the device's SparseCores, which run a call side
    by side."""
    from jax.experimental.pallas import tpu_sc as plsc

    num_cores = plsc.get_sparse_core_info().num_cores
    return [
        f"{TENSORCORE_PROCESS} SparseCore {core}" for core in range(num_cores)
    ]


def row_events(trace: dict, process: str, thread: str) -> list[dict]:
    """The complete events of one trace row."""
    processes, threads = {}, {}
    for event in trace["traceEvents"]:
        if event.get("ph") == "M" and event.get("name") == "process_name":
            processes[event["pid"]] = event["args"]["name"]
        elif event.get("ph") == "M" and event.get("name") == "thread_name":
            threads[(event["pid"], event["tid"])] = event["args"]["name"]
    return [
        event
        for event in trace["traceEvents"]
        if event.get("ph") == "X"
        and "dur" in event
        and processes.get(event["pid"]) == process
        and threads.get((event["pid"], event["tid"])) == thread
    ]


def trace_times(fn, args, directory: pathlib.Path) -> dict:
    """Device us per call: the module span, and the SparseCore calls."""
    import jax

    options = jax.profiler.ProfileOptions()
    options.advanced_configuration = {
        "tpu_num_sparse_core_tiles_to_trace": TRACED_SPARSECORE_TILES
    }
    jax.block_until_ready(fn(*args))
    with jax.profiler.trace(str(directory), profiler_options=options):
        for _ in range(TRACE_CALLS):
            out = fn(*args)
        jax.block_until_ready(out)
    trace = load_trace(directory)
    modules = row_events(trace, TENSORCORE_PROCESS, TENSORCORE_MODULES_ROW)
    if len(modules) != TRACE_CALLS:
        raise RuntimeError(
            f"the trace holds {len(modules)} module spans for {TRACE_CALLS}"
            " calls"
        )
    sparse_core = [
        sum(
            event["dur"]
            for event in row_events(trace, process, SPARSECORE_OPS_ROW)
            if SPARSECORE_DONE_MARKER in event["name"]
        )
        for process in sparse_core_processes()
    ]
    return {
        "kernel_us": sum(event["dur"] for event in modules) / TRACE_CALLS,
        "sparse_core_us": max(sparse_core) / TRACE_CALLS,
    }


def run_cell(cell: Cell, impl: str) -> dict:
    import jax

    scores, lengths = make_inputs(cell)
    fn = impl_fn(impl, cell.k)
    start = time.perf_counter()
    fn.lower(scores, lengths).compile()
    compile_s = time.perf_counter() - start
    positions = jax.block_until_ready(fn(scores, lengths))
    record = {"compile_s": compile_s}
    record.update(check_positions(impl, positions, scores, lengths, cell.k))
    timing, _ = time_call(lambda state: (fn(scores, lengths), state), 0)
    record["wall_us"] = timing["median_ms"] * 1e3
    with tempfile.TemporaryDirectory(prefix="topk_trace_") as directory:
        record.update(
            trace_times(fn, (scores, lengths), pathlib.Path(directory))
        )
    peak = HBM_PEAK[jax.devices()[0].device_kind]
    score_bytes = scores.size * scores.dtype.itemsize
    record["hbm_fraction"] = score_bytes / (record["kernel_us"] * 1e-6) / peak
    return record


def summary(key: str, record: dict) -> str:
    if "error" in record:
        return f"{key}: ERROR {record['error']}"
    check = (
        f"recall {record['recall']:.3f}"
        if "recall" in record
        else f"exact {record['exact']}"
    )
    return (
        f"{key}: kernel {record['kernel_us']:.1f} us (sc"
        f" {record['sparse_core_us']:.1f}) wall {record['wall_us']:.1f} us"
        f" compile {record['compile_s']:.2f} s hbm"
        f" {record['hbm_fraction']:.3f} {check}"
    )


def print_table(paths: list[str]) -> None:
    """Markdown table of kernel us per cell, one column per (file, impl)."""
    cells, columns = {}, []
    for path in paths:
        for line in pathlib.Path(path).read_text("utf-8").splitlines():
            record = json.loads(line)
            if record.get("status") != "ok":
                continue
            column = f"{pathlib.Path(path).stem}:{record['impl']}"
            if column not in columns:
                columns.append(column)
            cells.setdefault(record["cell_key"], {})[column] = record
    print("| cell | " + " | ".join(columns) + " |")
    print("|---|" + "---|" * len(columns))
    for key, by_column in cells.items():
        values = [
            f"{by_column[c]['kernel_us']:.1f}" if c in by_column else ""
            for c in columns
        ]
        print(f"| {key} | " + " | ".join(values) + " |")


def run(args) -> None:
    import jax

    device_kind = jax.devices()[0].device_kind
    print(device_kind, jax.__version__, flush=True)
    output = pathlib.Path(args.output)
    for suite in args.suite.split(","):
        for cell in suite_cells(suite):
            key = f"{args.impl}-{cell.key}"
            if recorded(output, key):
                continue
            record = {
                "key": key,
                "cell_key": cell.key,
                "impl": args.impl,
                "device_kind": device_kind,
                "cell": dataclasses.asdict(cell),
            }
            try:
                record.update(run_cell(cell, args.impl), status="ok")
            except Exception as error:  # pylint: disable=broad-exception-caught
                record.update(status="error", error=repr(error)[:400])
            print(summary(key, record), flush=True)
            write_record(output, record)
            jax.clear_caches()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", maxsplit=1)[0]
    )
    parser.add_argument("--impl", choices=IMPLS, default="flywheel")
    parser.add_argument("--suite", default="dsa,sampling")
    parser.add_argument("--output", help="JSONL; recorded cells are skipped")
    parser.add_argument("--list", action="store_true", help="list the cells")
    parser.add_argument("--table", nargs="+", metavar="JSONL")
    args = parser.parse_args()
    if args.list:
        for suite in args.suite.split(","):
            for cell in suite_cells(suite):
                print(cell.key)
        return
    if args.table:
        print_table(args.table)
        return
    if not args.output:
        parser.error("--output is required to run cells")
    run(args)


if __name__ == "__main__":
    main()
