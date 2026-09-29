#!/usr/bin/env python3
"""Split an attention block's device time into its segments, from one trace.

    PYTHONPATH=. uv run --no-sync python \\
        benchmarks/softmax_attention/segments.py \\
        results/benchmark/softmax_attention/traces

attention_block.py calls this on the trace it just wrote; run it by hand to
re-split a directory of traces. Each subdirectory is one cell.

The segments come from the same executable whose wall clock attention_block.py
reports, so they need no separately compiled variants and no assumption that
the parts add up. The `jit_block` event is the denominator, and only leaf ops
go into the segments, since the nesting would count them twice.

Ops are sorted by what they are, not by which jax.named_scope XLA filed them
under, because a fusion that spans two scopes carries only one of their names:

  attn_kernel  the Pallas call, named after the kernel
  layout       ops that only move bytes: copy, reshape, transpose, slice,
               concatenate, pad, bitcast, broadcast, dtype conversion
  qkv_proj     the remaining ops of the qkv_proj scope, the projection matmuls
  out_proj     the remaining ops of the out_proj scope
  other        anything left, reported so it is never silently dropped
"""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import re
from collections import defaultdict

SEGMENTS = ("qkv_proj", "layout", "attn_kernel", "out_proj", "other")
KERNEL_NAMES = ("flash_attn", "splash_mha", "RPAm", "ragged_paged")
MOVERS = ("copy", "reshape", "transpose", "slice", "concatenate", "pad",
          "bitcast", "broadcast_in_dim", "convert_element_type")


def load(directory):
    """The newest trace under directory, or None."""
    hits = sorted(pathlib.Path(directory).rglob("*.trace.json.gz"),
                  key=lambda path: path.stat().st_mtime)
    if not hits:
        return None
    with gzip.open(hits[-1], "rt") as handle:
        return json.load(handle)


def device_pids(trace):
    keep = set()
    for event in trace.get("traceEvents", []):
        if event.get("ph") == "M" and event.get("name") == "process_name":
            name = event.get("args", {}).get("name", "")
            if (re.search(r"TPU|Device|/device:", name, re.I)
                    and "Host" not in name):
                keep.add(event["pid"])
    return keep


def segment_of(event):
    name = event.get("name", "")
    if any(kernel in name for kernel in KERNEL_NAMES):
        return "attn_kernel"
    tf_op = (event.get("args") or {}).get("tf_op", "")
    primitive = tf_op.rstrip(":").rsplit("/", 1)[-1] if tf_op else ""
    if any(primitive.startswith(m) for m in MOVERS) or name.startswith("copy"):
        return "layout"
    if "/qkv_proj/" in tf_op:
        return "qkv_proj"
    if "/out_proj/" in tf_op:
        return "out_proj"
    return "layout" if not tf_op else "other"


def segments(trace):
    """Per-call ms of each segment and of the whole module, or None."""
    pids = device_pids(trace)
    per_segment = defaultdict(float)
    module_ms = 0.0
    calls = 0
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X" or "dur" not in event:
            continue
        if pids and event.get("pid") not in pids:
            continue
        if event.get("name", "").startswith("jit_"):
            module_ms += event["dur"] / 1e3
            calls += 1
            continue
        per_segment[segment_of(event)] += event["dur"] / 1e3
    if not calls:
        return None
    return {"calls": calls, "module_ms": module_ms / calls,
            "ms": {name: per_segment.get(name, 0.0) / calls
                   for name in SEGMENTS}}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", type=pathlib.Path)
    args = parser.parse_args(argv)

    for directory in sorted(args.traces.iterdir()):
        if not directory.is_dir():
            continue
        trace = load(directory)
        split = segments(trace) if trace is not None else None
        if split is None:
            print(f"{directory.name}: no trace")
            continue
        parts = "  ".join(f"{name} {split['ms'][name]:.3f}"
                          for name in SEGMENTS)
        print(f"{directory.name}: {split['module_ms']:.3f} ms | {parts}")


if __name__ == "__main__":
    main()
