#!/usr/bin/env python3
"""Print the kernel-level and block-level tables from attention_block.py runs.

    PYTHONPATH=. uv run --no-sync python \\
        benchmarks/softmax_attention/report.py \\
        results/benchmark/softmax_attention/benchmark.jsonl

kernel level  attention-kernel TFLOP/s of each impl, from the kernel segment
              of the trace, and flywheel's speedup over each baseline
block level   wall clock of the whole block split into its segments, and
              flywheel's speedup over each baseline

The kernel level needs runs made with --trace. When a cell was recorded more
than once, the last ok record wins.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

IMPLS = ("rpa", "splash", "flywheel")
SEGMENTS = ("qkv_proj", "layout", "attn_kernel", "out_proj")


def load(path):
    records = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "ok":
            records[record["key"]] = record
    return records.values()


def group(records):
    """(heads, heads_k, head_dim, mask, batch) -> seq -> impl -> record."""
    out = defaultdict(lambda: defaultdict(dict))
    for record in records:
        c = record["cell"]
        shape = (c["heads"], c["heads_k"], c["head_dim"], c["mask"],
                 c["batch"])
        out[shape][c["seq"]][c["impl"]] = record
    return out


def ratio(numerator, denominator):
    if not numerator or not denominator:
        return "-"
    return f"{numerator / denominator:.2f}x"


def kernel_table(shape, by_seq):
    impls = [impl for impl in IMPLS
             if any(impl in row for row in by_seq.values())]
    rivals = [impl for impl in impls if impl != "flywheel"]
    print(f"\nkernel level, TFLOP/s  {describe(shape)}")
    header = f"{'seq':>7} " + "".join(f"{impl:>10}" for impl in impls)
    header += "".join(f"{'vs ' + impl:>12}" for impl in rivals)
    print(header)
    for seq in sorted(by_seq):
        row = by_seq[seq]
        tflops = {impl: row[impl].get("kernel_tflops")
                  for impl in impls if impl in row}
        line = f"{seq:>7} " + "".join(
            f"{tflops[impl]:>10.0f}" if tflops.get(impl) else f"{'-':>10}"
            for impl in impls)
        line += "".join(f"{ratio(tflops.get('flywheel'), tflops.get(impl)):>12}"
                        for impl in rivals)
        print(line)


def block_table(shape, by_seq):
    impls = [impl for impl in IMPLS
             if any(impl in row for row in by_seq.values())]
    rivals = [impl for impl in impls if impl != "flywheel"]
    print(f"\nblock level, ms  {describe(shape)}")
    print(f"{'seq':>7} {'impl':>9} "
          + "".join(f"{name:>12}" for name in SEGMENTS)
          + f"{'total':>11}" + "".join(f"{'vs ' + impl:>12}"
                                        for impl in rivals))
    for seq in sorted(by_seq):
        row = by_seq[seq]
        ours = row.get("flywheel", {}).get("median_ms")
        for impl in impls:
            if impl not in row:
                continue
            record = row[impl]
            parts = record.get("segments_ms", {})
            line = f"{seq:>7} {impl:>9} " + "".join(
                f"{parts[name]:>12.3f}" if name in parts else f"{'-':>12}"
                for name in SEGMENTS)
            line += f"{record['median_ms']:>11.3f}"
            if impl == "flywheel":
                line += "".join(
                    f"{ratio(row.get(rival, {}).get('median_ms'), ours):>12}"
                    for rival in rivals)
            print(line)


def describe(shape):
    heads, heads_k, head_dim, mask, batch = shape
    return (f"(heads {heads}:{heads_k}, head_dim {head_dim}, {mask}, "
            f"batch {batch})")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records", type=pathlib.Path)
    args = parser.parse_args(argv)

    grouped = group(load(args.records))
    for shape in sorted(grouped, key=lambda s: (s[0], -s[1], s[2:])):
        kernel_table(shape, grouped[shape])
        block_table(shape, grouped[shape])


if __name__ == "__main__":
    main()
