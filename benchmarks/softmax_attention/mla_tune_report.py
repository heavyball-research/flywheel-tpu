"""Summarize an mla_tune run: progress, per-cell best blocks, speedups.

Usage:
  python benchmarks/softmax_attention/mla_tune_report.py \
      benchmarks/softmax_attention/results/mla_tune/summary.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=pathlib.Path)
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument("--csv", type=pathlib.Path)
    args = parser.parse_args()

    records = []
    for line in args.summary.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "summary":
            records.append(record)
    ok = [r for r in records if r.get("status") == "ok"]
    print(f"summaries {len(records)}, ok {len(ok)}")
    print("status", dict(collections.Counter(r.get("status") for r in records)))
    print(
        "priority",
        dict(
            collections.Counter(
                (r.get("cell_row") or {}).get("priority") for r in ok
            )
        ),
    )
    speedups = [r["speedup"] for r in ok if r.get("speedup")]
    if speedups:
        print(
            "speedup: median %.4f, >1.02 %d/%d, >1.10 %d/%d, max %.4f"
            % (
                statistics.median(speedups),
                sum(s > 1.02 for s in speedups),
                len(speedups),
                sum(s > 1.10 for s in speedups),
                len(speedups),
                max(speedups),
            )
        )
    blocks = collections.Counter(
        tuple(r["best_actual"]) for r in ok if r.get("best_actual")
    )
    print("best actual blocks:")
    for block, count in blocks.most_common():
        print(f"  {block}: {count}")

    rows = []
    for record in ok:
        row = record.get("cell_row") or {}
        rows.append((
            row.get("priority"),
            record.get("cell"),
            row.get("kv_dtype"),
            row.get("scenario"),
            row.get("heads"),
            row.get("batch"),
            row.get("seq"),
            row.get("verify_tokens"),
            record.get("table_key"),
            record.get("best_actual"),
            record.get("baseline_actual"),
            record.get("best_ms"),
            record.get("baseline_ms"),
            record.get("speedup"),
        ))
    rows.sort(key=lambda item: (item[0] or 9, item[1] or ""))
    print(
        "\npriority cell dtype scenario h b seq t key best baseline "
        "best_ms base_ms speedup"
    )
    for row in rows[-args.tail :]:
        print(
            f"{row[0]} {row[1]} {row[2]} {row[3]} h{row[4]} b{row[5]} "
            f"s{row[6]} t{row[7]} {row[8]} {row[9]} {row[10]} "
            f"{row[11]:.4f} {row[12]:.4f} {row[13]}"
        )
    if args.csv:
        with args.csv.open("w") as handle:
            handle.write(
                "priority,cell,dtype,scenario,heads,batch,seq,verify_tokens,"
                "table_key,best_block,best_items,baseline_block,"
                "baseline_items,best_ms,baseline_ms,speedup\n"
            )
            for row in rows:
                best = row[9] or [None, None]
                base = row[10] or [None, None]
                handle.write(
                    f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},"
                    f"{row[5]},{row[6]},{row[7]},{row[8]},"
                    f"{best[0]},{best[1]},{base[0]},{base[1]},"
                    f"{row[11]},{row[12]},{row[13]}\n"
                )
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
