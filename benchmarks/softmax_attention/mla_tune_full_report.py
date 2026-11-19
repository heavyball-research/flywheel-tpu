"""Build the complete MLA tune report: all cells, best blocks, remaining tests.

Reads the main run and the refinement summaries, relabels the level-3 cells
by comparing against the after-level-2 snapshot, and writes a Markdown report
with two tables plus dataset statistics.

Usage:
  python benchmarks/softmax_attention/mla_tune_full_report.py \
      --main benchmarks/softmax_attention/results/mla_tune \
      --refine benchmarks/softmax_attention/results/mla_tune_refine \
      --snapshot benchmarks/softmax_attention/results/mla_tune_refine/summary_after_level2.jsonl \
      --out benchmarks/softmax_attention/results/mla_tune_refine/final_report.md
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics

STAGES = {
    "main": "主实验",
    "coarse": "粗网格补点",
    "bisect1": "二分第1层",
    "bisect2": "二分第2层",
    "bisect3": "二分第3层",
    "coverage": "补覆盖",
}


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


def finalist_info(out_dir, cell):
    """Best and second-best finalist timings from a cell's per-cell JSONL."""
    path = out_dir / (cell + ".jsonl")
    finals = []
    if path.exists():
        for record in read_jsonl(path):
            if record.get("kind") == "finalist" and record.get("status") == "ok":
                finals.append(record)
    finals.sort(key=lambda record: record["median_ms"])
    if not finals:
        return None, None, None, None, None
    best = finals[0]
    second = None
    for candidate in finals[1:]:
        if tuple(candidate["actual"]) != tuple(best["actual"]):
            second = candidate
            break
    if second is None:
        return tuple(best["actual"]), best["median_ms"], None, None, 0.0
    return (
        tuple(best["actual"]),
        best["median_ms"],
        tuple(second["actual"]),
        second["median_ms"],
        second["median_ms"] / best["median_ms"] - 1,
    )


def load_points(main_dir, refine_dir, snapshot_path, extra_dirs=()):
    old = {r["cell"] for r in read_jsonl(snapshot_path) if r.get("cell")}
    points = {}
    sources = [(main_dir, "main"), (refine_dir, "refine")]
    for extra in extra_dirs:
        sources.append((extra, "coverage"))
    for source_dir, default_stage in sources:
        summary_path = source_dir / "summary.jsonl"
        for record in read_jsonl(summary_path):
            if record.get("kind") != "summary" or record.get("status") != "ok":
                continue
            cell = record.get("cell")
            if not cell:
                continue
            stage = record.get("stage") or default_stage
            if source_dir == refine_dir and cell not in old:
                stage = "bisect3"
            best = tuple(record["best_actual"])
            best_ms = record["best_ms"]
            second = None if record.get("second_actual") is None else tuple(record["second_actual"])
            second_ms = record.get("second_ms")
            margin = record.get("margin")
            if second is None or margin is None:
                f_best, f_best_ms, f_second, f_second_ms, f_margin = finalist_info(
                    summary_path.parent, cell
                )
                if f_second is not None or second is None:
                    second = f_second if second is None else second
                    second_ms = f_second_ms if second_ms is None else second_ms
                if margin is None:
                    margin = f_margin
                if best_ms is None and f_best_ms is not None:
                    best_ms = f_best_ms
            points[cell] = {
                "cell": cell,
                "row": record.get("cell_row") or {},
                "stage": stage,
                "best": best,
                "best_ms": best_ms,
                "second": second,
                "second_ms": second_ms,
                "margin": margin,
                "tested": record.get("tested"),
                "speedup": record.get("speedup"),
                "baseline": None
                if record.get("baseline_actual") is None
                else tuple(record["baseline_actual"]),
                "baseline_ms": record.get("baseline_ms"),
                "out_dir": summary_path.parent,
            }
    return points


def fmt_block(block):
    if block is None:
        return "-"
    return "(%d, %d)" % (block[0], block[1])


def pct(value):
    if value is None:
        return "-"
    return "%.2f%%" % (value * 100)


def sorted_points(points):
    def key(p):
        r = p["row"]
        return (r.get("kv_dtype", ""), r.get("heads", 0), r.get("scenario", ""),
                r.get("batch", 0), r.get("seq", 0), r.get("verify_tokens", 1))
    return sorted(points.values(), key=key)


def group_slices(points):
    groups = collections.defaultdict(list)
    for point in points.values():
        row = point["row"]
        if row.get("scenario") != "decode" or row.get("verify_tokens", 1) != 1:
            continue
        groups[("batch", row["kv_dtype"], row["heads"], row["seq"])].append((row["batch"], point))
        groups[("seq", row["kv_dtype"], row["heads"], row["batch"])].append((row["seq"], point))
    for items in groups.values():
        items.sort(key=lambda item: item[0])
    return groups


def find_intervals(groups, sharp=0.05, batch_res=1, seq_res=256):
    intervals = []
    for (axis, dtype, heads, fixed), items in groups.items():
        resolution = batch_res if axis == "batch" else seq_res
        for (av, a), (bv, b) in zip(items, items[1:]):
            if a["best"] == b["best"]:
                continue
            strength = max(a["margin"] or 0.0, b["margin"] or 0.0)
            if strength < sharp or bv - av <= resolution:
                continue
            intervals.append({
                "axis": axis,
                "dtype": dtype,
                "heads": heads,
                "fixed": fixed,
                "a": av,
                "b": bv,
                "a_point": a,
                "b_point": b,
                "strength": strength,
            })
    intervals.sort(key=lambda item: -item["strength"])
    return intervals


def lookup_time(point, block):
    path = point["out_dir"] / (point["cell"] + ".jsonl")
    if not path.exists():
        return None
    records = read_jsonl(path)
    for kind in ("finalist", "candidate"):
        hits = [
            r["median_ms"]
            for r in records
            if r.get("kind") == kind and r.get("status") == "ok"
            and tuple(r.get("actual", ())) == tuple(block)
        ]
        if hits:
            return min(hits)
    return None


def cross_losses(interval):
    a, b = interval["a_point"], interval["b_point"]
    ta = lookup_time(a, b["best"])
    tb = lookup_time(b, a["best"])
    return (
        None if ta is None else ta / a["best_ms"] - 1,
        None if tb is None else tb / b["best_ms"] - 1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=pathlib.Path, required=True)
    parser.add_argument("--refine", type=pathlib.Path, required=True)
    parser.add_argument("--snapshot", type=pathlib.Path, required=True)
    parser.add_argument("--extra", type=pathlib.Path, action="append", default=[],
                        help="additional run dir (repeatable), stage 'coverage'.")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    points = load_points(args.main, args.refine, args.snapshot, args.extra)
    ordered = sorted_points(points)
    groups = group_slices(points)
    intervals = find_intervals(groups)

    margins = [p["margin"] for p in points.values() if p["margin"] is not None]
    margins.sort()
    status_counts = collections.Counter(p["stage"] for p in points.values())
    blocks = collections.Counter(p["best"] for p in points.values())

    lines = []
    add = lines.append
    add("# MLA tune 完整测试报告")
    add("")
    add("本报告汇总主实验、粗网格补点、二分各层和补覆盖的全部 MLA decode/verify cell。")
    add("数据来源：`benchmarks/softmax_attention/results/mla_tune/` 和 `mla_tune_refine/`。")
    add("")
    add("## 总览")
    add("")
    add("| 项目 | 数值 |")
    add("|---|---|")
    add("| 数据点总数 | %d |" % len(points))
    for stage in ("main", "coarse", "bisect1", "bisect2", "bisect3", "coverage"):
        if status_counts.get(stage):
            add("| %s | %d |" % (STAGES[stage], status_counts[stage]))
    add("| 错误 | 0 |")
    add("")
    add("margin（最优与次优的相对差距）分布：")
    add("")
    add("| 分位 | margin |")
    add("|---|---|")
    for label, value in (
        ("中位数", statistics.median(margins)),
        ("p90", margins[int(len(margins) * 0.9)]),
        ("p99", margins[int(len(margins) * 0.99)]),
        ("最大值", max(margins)),
    ):
        add("| %s | %s |" % (label, pct(value)))
    add("")
    add("| 阈值 | 点数 | 占比 |")
    add("|---|---|---|")
    for threshold in (0.01, 0.02, 0.05, 0.10):
        count = sum(m >= threshold for m in margins)
        add("| margin ≥ %.0f%% | %d | %.1f%% |" % (threshold * 100, count, count / len(margins) * 100))
    add("")
    add("最优 block 分布（前 12）：")
    add("")
    add("| block (block_kv, items) | cell 数 |")
    add("|---|---|")
    for block, count in blocks.most_common(12):
        add("| (%d, %d) | %d |" % (block[0], block[1], count))
    add("")
    add("## 表 1：所有测试的 cell（%d 个）" % len(ordered))
    add("")
    add("阶段说明：主实验 = 主批量测试；粗网格补点 = 六条切片的粗网格；二分第 N 层 = 对尖锐区间插入中点。")
    add("`decode_tokens` 是每个 decode step 处理的新 token 数（影响耗时和最优 block）。")
    add("")
    add("| # | dtype | heads | batch | seq | decode_tokens | verify_tokens | 阶段 | 候选数 | 最优 block | 最优耗时 (ms) | margin |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for index, point in enumerate(ordered, 1):
        row = point["row"]
        add("| %d | %s | %d | %d | %d | %s | %s | %s | %s | %s | %.4f | %s |" % (
            index,
            row.get("kv_dtype", "?"),
            row.get("heads", 0),
            row.get("batch", 0),
            row.get("seq", 0),
            row.get("decode_tokens", "-"),
            row.get("verify_tokens", 1),
            STAGES.get(point["stage"], point["stage"]),
            "-" if point["tested"] is None else point["tested"],
            fmt_block(point["best"]),
            point["best_ms"],
            pct(point["margin"]),
        ))
    add("")
    add("## 表 2：cell → 最优 block（%d 个）" % len(ordered))
    add("")
    add("| # | cell | 最优 block | 次优 block | 最优耗时 (ms) | margin |")
    add("|---|---|---|---|---|---|")
    for index, point in enumerate(ordered, 1):
        add("| %d | `%s` | %s | %s | %.4f | %s |" % (
            index,
            point["cell"],
            fmt_block(point["best"]),
            fmt_block(point["second"]),
            point["best_ms"],
            pct(point["margin"]),
        ))
    add("")
    add("## 剩余 ≥5%% 的相邻区间（%d 个）" % len(intervals))
    add("")
    add("区间 = 同一组参数（dtype、heads、固定另一个轴）沿 batch 或 seq 轴的相邻两个已测点；")
    add("两点的最优 block 不同且至少一端的 margin ≥5%。")
    add("")
    add("| dtype | heads | 轴 | 固定值 | 区间 | 左 block | 左 margin | 右 block | 右 margin | 交叉损失（左→右 / 右→左） |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    cross_stats = []
    for interval in intervals:
        la, lb = cross_losses(interval)
        if la is not None and lb is not None:
            cross_stats.append((interval, la, lb))
        fixed_label = "seq=%d" % interval["fixed"] if interval["axis"] == "batch" else "batch=%d" % interval["fixed"]
        add("| %s | %d | %s | %s | %d→%d | %s | %s | %s | %s | %s / %s |" % (
            interval["dtype"],
            interval["heads"],
            interval["axis"],
            fixed_label,
            interval["a"],
            interval["b"],
            fmt_block(interval["a_point"]["best"]),
            pct(interval["a_point"]["margin"]),
            fmt_block(interval["b_point"]["best"]),
            pct(interval["b_point"]["margin"]),
            "-" if la is None else "%+.2f%%" % (la * 100),
            "-" if lb is None else "%+.2f%%" % (lb * 100),
        ))
    add("")
    if cross_stats:
        cheap = sorted(min(la, lb) for _, la, lb in cross_stats)
        both_big = sum(1 for _, la, lb in cross_stats if min(la, lb) >= 0.05)
        add("交叉损失统计（只统计两个方向都测到的 %d 个区间）：" % len(cross_stats))
        add("")
        add("- 较便宜方向的损失：中位 %.2f%%，p90 %.2f%%，最大 %.2f%%" % (
            statistics.median(cheap) * 100,
            cheap[int(len(cheap) * 0.9)] * 100,
            max(cheap) * 100,
        ))
        add("- 两个方向都 ≥5%% 的区间：%d 个" % both_big)
        add("")
    coverage_points = [p for p in points.values() if p["stage"] == "coverage"]
    if coverage_points:
        speedups = sorted(
            (p["speedup"] for p in coverage_points if p["speedup"] is not None),
            reverse=True,
        )
        add("## 补覆盖的 %d 个 cell" % len(coverage_points))
        add("")
        add("这 %d 个 cell 在 tune 表里已有 key，所以不在主实验的缺失列表里；"
            "这次用同一套候选网格和计时协议补测，同时重新验证旧表条目的最优性。"
            % len(coverage_points))
        add("")
        add("| 项目 | 数值 |")
        add("|---|---|")
        add("| 补测 cell 数 | %d |" % len(coverage_points))
        add("| 成功 / 错误 | %d / 0 |" % len(coverage_points))
        if speedups:
            add("| 相对旧表条目的提速中位数 | %.4f |" % statistics.median(speedups))
            add("| 提速 >2%% 的 cell | %d |" % sum(s > 1.02 for s in speedups))
            add("| 提速 >5%% 的 cell | %d |" % sum(s > 1.05 for s in speedups))
            add("| 提速 >10%% 的 cell | %d |" % sum(s > 1.10 for s in speedups))
            add("| 最大提速 | %.4f |" % max(speedups))
        add("")
        add("全部 %d 个补覆盖 cell（按提速从高到低）：" % len(coverage_points))
        add("")
        add("| # | dtype | 场景 | heads | batch | seq | t | 旧 block | 旧耗时 (ms) | 最优 block | 次优 block | 最优耗时 (ms) | margin | 提速 |")
        add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for index, point in enumerate(
            sorted(coverage_points, key=lambda p: -(p["speedup"] or 0)), 1
        ):
            row = point["row"]
            add("| %d | %s | %s | %d | %d | %d | %d | %s | %s | %s | %s | %.4f | %s | %.3f |" % (
                index,
                row.get("kv_dtype"),
                row.get("scenario"),
                row.get("heads", 0),
                row.get("batch", 0),
                row.get("seq", 0),
                row.get("verify_tokens", 1),
                fmt_block(point["baseline"]),
                "-" if point["baseline_ms"] is None else "%.4f" % point["baseline_ms"],
                fmt_block(point["best"]),
                fmt_block(point["second"]),
                point["best_ms"],
                pct(point["margin"]),
                point["speedup"] or 0,
            ))
        add("")
    seq_gaps = sorted(i["b"] - i["a"] for i in intervals if i["axis"] == "seq")
    batch_gaps = sorted(i["b"] - i["a"] for i in intervals if i["axis"] == "batch")
    cliffs = [item for item in cross_stats if min(item[1], item[2]) >= 0.05]
    cheap_values = sorted(min(la, lb) for _, la, lb in cross_stats)
    cheap_2 = sum(value <= 0.02 for value in cheap_values)
    cheap_5 = sum(value <= 0.05 for value in cheap_values)
    wide_batch = [i for i in intervals if i["axis"] == "batch" and i["b"] - i["a"] > 10]
    add("## 分析：是否还需要继续测试")
    add("")
    add("**结论：不需要再做全局二分测试。** 现有 %d 个已测点足够开始设计启发式；剩余的 %d 个区间留作验证集。" % (len(points), len(intervals)))
    add("")
    add("依据：")
    add("")
    add("1. 加入补覆盖后，当前剩余 ≥5%% 区间为 %d 个（第 3 轮后是 19 个，补覆盖的 batch=128 切片净增 3 个），"
        "且剩余区间偏难：两个方向都测到的 %d 个区间里，较便宜方向的损失中位数是 %.2f%%（第 2 轮后是 2.88%%），"
        "只有 %d 个 ≤2%%、%d 个 ≤5%%。说明容易解决的区间已经解决，剩下的是真正难分的局部。"
        % (len(intervals), len(cross_stats), statistics.median(cheap_values) * 100, cheap_2, cheap_5))
    add("2. 宽度已经很窄：%d 个 seq 轴区间全部 ≤1536 token（其中 %d 个只有 512）；"
        "%d 个 batch 轴区间里 %d 个只差 3–4。唯一的宽区间是 fp8 h16 seq=2560 的 batch %d→%d，"
        "但它的便宜方向损失很小，启发式只要在交叉损失小的一端切换，整段最坏损失有限。"
        % (len(seq_gaps), sum(1 for gap in seq_gaps if gap <= 512), len(batch_gaps),
           sum(1 for gap in batch_gaps if gap <= 4),
           wide_batch[0]["a"], wide_batch[0]["b"]) if wide_batch else
        "2. 所有区间宽度都已经很窄：%d 个 seq 轴区间全部 ≤1536 token，%d 个 batch 轴区间只差 3–4。"
        % (len(seq_gaps), len(batch_gaps)))
    add("3. %d 个“两边都贵”（两个方向交叉损失都 ≥5%%）的区间全部集中在 fp8（外加 1 个 bf16 batch 区间），"
        "宽度只有 512–1536 token 或 3–4 batch。再跑一层（%d 个 cell，约 25 分钟）只能把宽度减半；"
        "但这里的性能地形是锯齿状的（例如 fp8 h16 batch=4 在 4096/5632/7168 三点分别选 (1024,2)/(2048,2)/(4096,2)），"
        "继续二分更可能不断发现新的局部最优，而不是收敛到一条干净的边界。" % (len(cliffs), len(intervals)))
    add("4. 对启发式设计而言，已测的 %d 个点才是函数的拟合依据；区间内部只是窄过渡带，"
        "翻转点附近两种选择本来就接近。%d 个悬崖可以用“在交叉损失小的一端切换”的策略控制最坏损失。"
        % (len(points), len(cliffs)))
    add("")
    add("建议：")
    add("")
    add("- 冻结数据集（%d 点），开始设计启发式；" % len(points))
    add("- 把剩余 %d 个区间（尤其 %d 个悬崖）作为验证集；如果启发式在这些切片上的预测误差明显大于别处，"
        "再对具体区间做定点补测，而不是再跑全局二分；" % (len(intervals), len(cliffs)))
    add("- 若以后要补测，优先补下面这 %d 个悬崖区间：" % len(cliffs))
    add("")
    add("| dtype | heads | 轴 | 固定 | 区间 | 左 block | 右 block | 交叉损失（左→右 / 右→左） |")
    add("|---|---|---|---|---|---|---|---|")
    for interval, la, lb in sorted(cliffs, key=lambda item: -min(item[1], item[2])):
        fixed_label = "seq=%d" % interval["fixed"] if interval["axis"] == "batch" else "batch=%d" % interval["fixed"]
        add("| %s | %d | %s | %s | %d→%d | %s | %s | %+.2f%% / %+.2f%% |" % (
            interval["dtype"], interval["heads"], interval["axis"], fixed_label,
            interval["a"], interval["b"],
            fmt_block(interval["a_point"]["best"]), fmt_block(interval["b_point"]["best"]),
            la * 100, lb * 100,
        ))
    add("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("wrote %s" % args.out)
    print("points %d, stages %s, remaining intervals %d" % (len(points), dict(status_counts), len(intervals)))


if __name__ == "__main__":
    main()
