#!/usr/bin/env python3
"""Search one prefill cell's block config for one attention kernel.

    PYTHONPATH=. uv run --no-sync python \\
        benchmarks/softmax_attention/tune_blocks.py \\
        --impl rpa --heads 32 --heads-k 4 --head-dim 256 --seq 8192 \\
        --output results/tune/softmax_attention

    PYTHONPATH=. uv run --no-sync python \\
        benchmarks/softmax_attention/tune_blocks.py \\
        --collect results/tune/softmax_attention --device-tag v7x

The search times the kernel alone, on q, k and v already in the layout it
takes, with the wall clock of queued calls. It is a coordinate descent: from
each seed config (the impl's own default, and its v6e-tuned entry when there
is one) it sweeps one field at a time over its candidates, keeps the fastest,
and repeats until a pass gains under 1%. The fastest few are then timed again
with attention_block.py's protocol and the winner is recorded.

Every candidate is logged before it runs, so a config that aborts the process,
hangs (the candidate alarm kills it) or halts the TPU core is not lost: run
the same command again until it prints the best config. A candidate that took
the process down is tried once more, in case something else did, and skipped
after the second time.

--collect turns a directory of finished searches into the tables the block
benchmark reads: rpa_tuned_<tag>.json and splash_tuned_<tag>.json next to this
file, and this device's section of flywheel_tpu/tuned_configs.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import signal
import statistics
import sys
import time

from benchmarks.common.rpa import (DEFAULT_RPA_SOURCE, RPA_V3_GIT_SHA,
                                   load_rpa_v3, vllm_page_size)
from benchmarks.common.rpa import FIELDS as RPA_FIELDS
from benchmarks.common.splash import (DEFAULT_SPLASH_SOURCE, HEURISTIC,
                                      SPLASH_GIT_SHA, load_splash)
from benchmarks.common.splash import FIELDS as SPLASH_FIELDS
from benchmarks.common.timing import time_call
from benchmarks.softmax_attention.attention_block import (
    RPA_TABLE, SPLASH_TABLE, Cell, table_entry)

HERE = pathlib.Path(__file__).resolve().parent
FLYWHEEL_FIELDS = ("q_block", "kv_block", "q_cblock", "kv_cblock", "stages",
                   "head_fold", "qkv_layout", "transposed_pv")
PRUNE_FACTOR = 1.5
MAX_CRASHES = 2
MIN_PASS_GAIN = 0.01
HALT_EXIT = 75
MIN_ROUND_S = 0.2
REPEATS = 3
RERANK = 3


def divisors(values, block):
    return [value for value in values if value <= block and block % value == 0]


def snap(value, values, block):
    """value if it divides block, else the largest candidate under it that
    does."""
    fits = divisors(values, block)
    if value in fits or (value <= block and block % value == 0):
        return value
    below = [candidate for candidate in fits if candidate <= value]
    return max(below) if below else min(fits, default=block)


# ---- flywheel ---------------------------------------------------------------

class Flywheel:
    fields = FLYWHEEL_FIELDS
    # The fields a config's VMEM footprint grows with.
    size_fields = ("q_block", "kv_block", "q_cblock", "kv_cblock", "stages",
                   "head_fold")
    layout = "hm"

    def __init__(self, cell, args):
        self.cell = cell

    def axes(self):
        cell = self.cell
        # Note: 4096-token blocks, 64-token q tiles and 128-token kv tiles
        # were searched at 32K to 128K on v7x and are left out: each ran 1.6x
        # to 7x slower than the best and took 90 to 220 s to compile.
        blocks = divisors((256, 512, 1024, 2048), cell.seq)
        return {
            "q_block": blocks, "kv_block": blocks,
            "q_cblock": (128, 256, 512),
            "kv_cblock": (256, 384, 512, 1024),
            "stages": (2, 3, 4),
            "head_fold": (1, 2, 4, 8),
            "qkv_layout": ("head_dim_minor", "seq_minor"),
            "transposed_pv": (False, True),
        }

    def seeds(self):
        from flywheel_tpu.flash_attn_interface import default_block_sizes

        default = default_block_sizes(self.cell.seq, self.cell.seq)
        return [{"q_block": default.block_q, "kv_block": default.block_kv,
                 "q_cblock": default.block_q_compute,
                 "kv_cblock": default.block_kv_compute,
                 "stages": default.num_stages, "head_fold": 1,
                 "qkv_layout": "head_dim_minor", "transposed_pv": False}]

    def normalize(self, config, changed):
        cell, config = self.cell, dict(config)
        axes = self.axes()
        # block_q must hold at least two block_q_compute tiles.
        config["q_cblock"] = snap(
            min(config["q_cblock"], config["q_block"] // 2), axes["q_cblock"],
            config["q_block"])
        config["kv_cblock"] = snap(config["kv_cblock"], axes["kv_cblock"],
                                   config["kv_block"])
        # transposed_pv computes on SEQ_MINOR operands only.
        if changed == "transposed_pv" and config["transposed_pv"]:
            config["qkv_layout"] = "seq_minor"
        if config["qkv_layout"] != "seq_minor":
            config["transposed_pv"] = False
        # A dense fold needs MHA and the whole sequence in one physical block.
        if (cell.heads != cell.heads_k or config["q_block"] != cell.seq
                or config["kv_block"] != cell.seq
                or cell.heads % config["head_fold"]):
            config["head_fold"] = 1
        return config

    def build(self, config, qkv):
        import jax

        from flywheel_tpu.flash_attn_interface import fused_q_scale
        from flywheel_tpu.pallas.block_sizes import BlockSizes, QKVLayout
        from flywheel_tpu.pallas.flash_fwd import make_flash_attn_mha

        cell = self.cell
        kernel = make_flash_attn_mha(
            cell.batch * cell.heads, cell.seq, cell.seq, causal=True,
            block_sizes=BlockSizes(
                block_q=config["q_block"], block_kv=config["kv_block"],
                block_kv_compute=config["kv_cblock"],
                block_q_compute=config["q_cblock"],
                num_stages=config["stages"],
                qkv_layout=QKVLayout[config["qkv_layout"].upper()]),
            num_kv_heads=cell.batch * cell.heads_k,
            head_fold=config["head_fold"],
            transposed_pv=config["transposed_pv"],
            q_scale=float(fused_q_scale(1.0 / math.sqrt(cell.head_dim), 0.0)))
        call = jax.jit(kernel)
        return (lambda state: (call(*qkv), state)), (lambda: None)


# ---- splash -----------------------------------------------------------------

class Splash:
    fields = SPLASH_FIELDS
    size_fields = ("block_q", "block_kv", "block_kv_compute",
                   "num_stacked_q_heads")
    layout = "hm"

    def __init__(self, cell, args):
        self.cell = cell
        self.kernel, self.mask_lib = load_splash(args.splash_source)
        self.table = args.splash_table

    def axes(self):
        cell = self.cell
        group = cell.heads // cell.heads_k
        return {
            "block_q": divisors((256, 512, 1024, 2048, 4096), cell.seq),
            "block_kv": divisors((512, 1024, 2048, 4096), cell.seq),
            "block_kv_compute": (128, 256, 512, 1024, 2048),
            "num_stacked_q_heads": divisors((1, 2, 4, 8, 16, 32), group),
            "layout": ("hhh", "hsh", "hss", "hhs"),
            "scheduler": (1, 0),
            "diag_grid": (0, 2, 4),
        }

    def seeds(self):
        seeds = [dict(HEURISTIC)]
        tuned = table_entry(self.table, self.cell, per_mask=True)
        if tuned is not None:
            seeds.insert(0, tuned)
        # Note: at 64K and beyond on v7x the v6e entry runs out of VMEM and
        # the heuristic's 128-token blocks run out of SMEM for the mask
        # tables, so two mid-sized configs make sure the descent has a start.
        for blocks in ((2048, 2048, 1024, 1, "hhh", 1, 0),
                       (1024, 1024, 512, 1, "hhh", 1, 0)):
            seeds.append(dict(zip(SPLASH_FIELDS, blocks)))
        return [seed for seed in seeds
                if seed["block_q"] <= self.cell.seq
                and seed["block_kv"] <= self.cell.seq]

    def normalize(self, config, changed):
        config = dict(config)
        config["block_kv_compute"] = snap(
            config["block_kv_compute"], self.axes()["block_kv_compute"],
            config["block_kv"])
        return config

    def build(self, config, qkv):
        import jax
        import jax.numpy as jnp

        cell, kernel = self.cell, self.kernel
        layouts = {"h": kernel.QKVLayout.HEAD_DIM_MINOR,
                   "s": kernel.QKVLayout.SEQ_MINOR}
        diag_grid = int(config["diag_grid"])
        splash_config = kernel.SplashConfig(
            block_q=int(config["block_q"]),
            block_kv=int(config["block_kv"]),
            block_kv_compute=int(config["block_kv_compute"]),
            num_stacked_q_heads=int(config["num_stacked_q_heads"]),
            q_layout=layouts[config["layout"][0]],
            k_layout=layouts[config["layout"][1]],
            v_layout=layouts[config["layout"][2]],
            use_experimental_scheduler=bool(config["scheduler"]),
            qk_diag_skip=diag_grid > 0,
            sv_diag_skip=diag_grid > 0,
            qk_diag_grid=diag_grid or 2)
        mask = self.mask_lib.CausalMask((cell.seq, cell.seq))
        attention = kernel.make_splash_mha_single_device(mask,
                                                         config=splash_config)
        # Splash takes no softmax scale, so it goes into q, outside the clock.
        q, k, v = qkv
        q = (q.astype(jnp.float32)
             / math.sqrt(cell.head_dim)).astype(jnp.bfloat16)
        call = jax.jit(attention)
        return (lambda state: (call(q, k, v), state)), (lambda: None)


# ---- RPA v3 -----------------------------------------------------------------

class Rpa:
    fields = RPA_FIELDS
    size_fields = RPA_FIELDS
    layout = "tm"

    def __init__(self, cell, args):
        self.cell = cell
        self.rpa = load_rpa_v3(args.rpa_source)
        self.table = args.rpa_table
        self.page_size = vllm_page_size(cell.seq, cell.batch)
        self.pages_per_seq = -(-cell.seq // self.page_size)

    def axes(self):
        seq, page = self.cell.seq, self.page_size
        return {
            "bq_sz": [v for v in (64, 128, 256, 512, 1024, 2048) if v <= seq],
            "bkv_sz": [v for v in (256, 512, 1024, 2048, 4096)
                       if v <= seq and v % page == 0],
            "bq_csz": (32, 64, 128, 256, 512, 1024),
            "bkv_csz": [v for v in (128, 256, 512, 1024, 2048)
                        if v % page == 0],
        }

    def seeds(self):
        import jax.numpy as jnp

        cell = self.cell
        formula = self.rpa.get_default_block_sizes(
            jnp.bfloat16, jnp.bfloat16, cell.heads, cell.heads_k,
            cell.head_dim, self.page_size, cell.batch * cell.seq, cell.batch,
            self.pages_per_seq, case=self.rpa.RpaCase.MIXED)
        seeds = [{name: int(formula[name]) for name in RPA_FIELDS}]
        tuned = table_entry(self.table, cell, per_mask=False)
        if tuned is not None:
            seeds.insert(0, tuned)
        # Note: on v7x neither of those fits VMEM for 32 kv heads at head_dim
        # 256, so two small configs make sure the descent has a start.
        for blocks in ((256, 512, 128, 256), (128, 256, 64, 256)):
            seeds.append(dict(zip(RPA_FIELDS, blocks)))
        return seeds

    def normalize(self, config, changed):
        config, axes = dict(config), self.axes()
        config["bq_csz"] = snap(config["bq_csz"], axes["bq_csz"],
                                config["bq_sz"])
        config["bkv_csz"] = snap(config["bkv_csz"], axes["bkv_csz"],
                                 config["bkv_sz"])
        return config

    def cache_and_metadata(self, cache_shape):
        import jax.numpy as jnp
        import numpy as np

        cell = self.cell
        num_pages = cell.batch * self.pages_per_seq
        metadata = (
            jnp.full((cell.batch,), cell.seq, jnp.int32),
            jnp.arange(num_pages, dtype=jnp.int32),
            jnp.arange(cell.batch + 1, dtype=jnp.int32) * cell.seq,
            jnp.asarray(np.array([0, 0, cell.batch], np.int32)),
        )
        return (lambda: jnp.zeros(cache_shape, jnp.bfloat16)), metadata

    def build(self, config, qkv):
        import jax
        import jax.numpy as jnp

        cell, rpa = self.cell, self.rpa
        cache_shape = rpa.get_kv_cache_shape(
            cell.batch * self.pages_per_seq, self.page_size, cell.heads_k,
            cell.head_dim, jnp.bfloat16)
        fresh, metadata = self.cache_and_metadata(cache_shape)
        block_sizes = tuple(int(config[name]) for name in RPA_FIELDS)
        kernel = getattr(rpa.ragged_paged_attention, "__wrapped__",
                         rpa.ragged_paged_attention)
        scale = 1.0 / math.sqrt(cell.head_dim)

        def run(q, k, v, cache):
            return kernel(q, k, v, cache, *metadata, use_causal_mask=True,
                          sm_scale=scale, m_block_sizes=block_sizes)

        # The call donates the cache, so every timing run starts from a
        # fresh one.
        call = jax.jit(run, donate_argnums=(3,))
        return (lambda state: call(*qkv, state)), fresh


SEARCHERS = {"flywheel": Flywheel, "splash": Splash, "rpa": Rpa}


# ---- the search -------------------------------------------------------------

class Log:
    """The JSONL of one search: every candidate's start, then its result."""

    def __init__(self, path):
        self.path = path
        self.results = {}
        self.best = None
        started = None
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = json.dumps(record.get("config"))
                if record["event"] == "start":
                    started = key
                elif record["event"] == "result":
                    self.results[key] = record
                    started = None
                elif record["event"] == "best":
                    self.best = record
        if started is not None:
            # The process died inside this candidate: an abort or the alarm.
            crashes = self.results.get(started, {}).get("crashes", 0)
            self.write({"event": "result", "config": json.loads(started),
                        "status": "crashed", "crashes": crashes + 1})

    def write(self, record):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if record["event"] == "result":
            self.results[json.dumps(record["config"])] = record
        elif record["event"] == "best":
            self.best = record

    def ok(self):
        return [record for record in self.results.values()
                if record["status"] == "ok"]

    def fastest(self):
        """The fastest fully timed config; a pruned one is never the best."""
        return min((record for record in self.ok() if not record["pruned"]),
                   key=lambda record: record["ms"], default=None)

    def implied_oom(self, searcher, config):
        """A config that ran out of VMEM and is no larger than this one in any
        block field while equal in the rest, or None."""
        for record in self.results.values():
            # Note: only VMEM grows with the blocks; SMEM, which holds
            # Splash's mask tables, shrinks with them.
            if (record["status"] != "error"
                    or "RESOURCE_EXHAUSTED" not in record["error"]
                    or "vmem" not in record["error"].lower()):
                continue
            other = dict(zip(searcher.fields, record["config"]))
            if all(config[name] >= other[name] if name in searcher.size_fields
                   else config[name] == other[name]
                   for name in searcher.fields):
                return record["config"]
        return None


def make_qkv(cell, layout):
    """bf16 unit-normal q, k and v drawn on the device, in the kernel's
    layout."""
    import jax
    import jax.numpy as jnp

    def draw(key, heads):
        if layout == "hm":
            shape = (cell.batch * heads, cell.seq, cell.head_dim)
        else:
            shape = (cell.batch * cell.seq, heads, cell.head_dim)
        return jax.random.normal(key, shape, jnp.bfloat16)

    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    return jax.block_until_ready((draw(keys[0], cell.heads),
                                  draw(keys[1], cell.heads_k),
                                  draw(keys[2], cell.heads_k)))


def time_kernel(step, fresh, best_single_ms):
    """(ms per call, single-call ms, pruned): a candidate whose first warm call
    is already PRUNE_FACTOR over the best config's first warm call keeps that
    one sample. Both are single synchronous calls, so the fixed cost of a
    sync, which dominates a sub-millisecond kernel, cancels out."""
    import jax

    state = fresh()
    out, state = jax.block_until_ready(step(state))
    start = time.perf_counter()
    out, state = jax.block_until_ready(step(state))
    single_ms = (time.perf_counter() - start) * 1e3
    if (best_single_ms is not None
            and single_ms > PRUNE_FACTOR * best_single_ms):
        return single_ms, single_ms, True
    iters = max(1, math.ceil(MIN_ROUND_S * 1e3 / single_ms))
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        for _ in range(iters):
            out, state = step(state)
        jax.block_until_ready((out, state))
        samples.append((time.perf_counter() - start) * 1e3 / iters)
    return statistics.median(samples), single_ms, False


def evaluate(searcher, config, qkv, log, timeout):
    key = json.dumps([config[name] for name in searcher.fields])
    known = log.results.get(key)
    if known is not None and (known["status"] != "crashed"
                              or known["crashes"] >= MAX_CRASHES):
        return known
    values = json.loads(key)
    implied = log.implied_oom(searcher, config)
    if implied is not None:
        record = {"event": "result", "config": values, "status": "error",
                  "error": f"skipped: VMEM OOM implied by {implied}"}
        log.write(record)
        return record
    log.write({"event": "start", "config": values})
    best = log.fastest()
    signal.alarm(timeout)
    started = time.perf_counter()
    try:
        step, fresh = searcher.build(config, qkv)
        ms, single_ms, pruned = time_kernel(
            step, fresh, best["single_ms"] if best else None)
        record = {"event": "result", "config": values, "status": "ok",
                  "ms": ms, "single_ms": single_ms, "pruned": pruned}
    except Exception as error:
        message = next((line for line in str(error).splitlines()
                        if line.strip()), "")
        record = {"event": "result", "config": values, "status": "error",
                  "error": f"{type(error).__name__}: {message[:400]}"}
    finally:
        signal.alarm(0)
    record["elapsed_s"] = round(time.perf_counter() - started, 1)
    log.write(record)
    if record["status"] == "error" and "halted" in record["error"].lower():
        # The core is gone for this process; the next run starts on a fresh
        # one.
        print(f"  {key}: {record['error'][:90]}; exiting to restart",
              flush=True)
        os._exit(HALT_EXIT)
    if record["status"] == "ok":
        shown = (f"{record['ms']:.3f} ms"
                 + (" (pruned)" if record["pruned"] else ""))
    else:
        shown = record["error"][:90]
    print(f"  {key}: {shown}", flush=True)
    return record


def search(cell, args):
    import jax

    if jax.default_backend() != "tpu":
        raise RuntimeError(f"requires TPU; got {jax.default_backend()!r}.")
    log = Log(args.output / f"{cell.key}.jsonl")
    if log.best is not None and not args.force:
        print(f"{cell.key}: already tuned, {log.best['config']} "
              f"{log.best['ms']:.3f} ms", flush=True)
        return 0

    # Note: the default action of SIGALRM kills the process even while it is
    # blocked inside the runtime, which a Python handler could not do.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    searcher = SEARCHERS[cell.impl](cell, args)
    qkv = make_qkv(cell, searcher.layout)
    as_config = lambda record: dict(zip(searcher.fields, record["config"]))

    print(f"{cell.key}: seeds", flush=True)
    for seed in searcher.seeds():
        evaluate(searcher, searcher.normalize(seed, None), qkv, log,
                 args.candidate_timeout)
    if log.fastest() is None:
        print(f"{cell.key}: no seed config runs", flush=True)
        return 1

    for sweep in range(args.passes):
        before = log.fastest()["ms"]
        for axis, values in searcher.axes().items():
            base = as_config(log.fastest())
            print(f"{cell.key}: pass {sweep + 1}, {axis}", flush=True)
            for value in values:
                evaluate(searcher,
                         searcher.normalize({**base, axis: value}, axis),
                         qkv, log, args.candidate_timeout)
        # Another pass only pays when this one moved the best by more than
        # the timing noise between near-tied configs.
        if log.fastest()["ms"] > before * (1 - MIN_PASS_GAIN):
            break

    # The search's short rounds rank near-ties loosely, so the fastest few
    # are timed again with the block benchmark's protocol.
    finalists = sorted((record for record in log.ok() if not record["pruned"]),
                       key=lambda record: record["ms"])[:RERANK]
    reranked = []
    for record in finalists:
        signal.alarm(args.candidate_timeout)
        step, fresh = searcher.build(as_config(record), qkv)
        timing, _ = time_call(step, fresh())
        signal.alarm(0)
        reranked.append((timing["median_ms"], record["config"]))
        print(f"  rerank {record['config']}: {timing['median_ms']:.3f} ms",
              flush=True)
    ms, config = min(reranked)
    seed_ms = [log.results[json.dumps(
        [searcher.normalize(seed, None)[name] for name in searcher.fields])]
        for seed in searcher.seeds()]
    log.write({"event": "best", "key": cell.key,
               "cell": {"impl": cell.impl, "heads": cell.heads,
                        "heads_k": cell.heads_k, "head_dim": cell.head_dim,
                        "mask": cell.mask, "batch": cell.batch,
                        "seq": cell.seq},
               "fields": list(searcher.fields), "config": config, "ms": ms,
               "candidates": len(log.results),
               "seeds": [{"config": r["config"], "status": r["status"],
                          "ms": r.get("ms")} for r in seed_ms],
               "device": jax.devices()[0].device_kind})
    print(f"{cell.key}: best {config} {ms:.3f} ms after "
          f"{len(log.results)} candidates", flush=True)
    return 0


# ---- tables -----------------------------------------------------------------

def dump_table(header, configs, per_mask):
    """The layout of the v6e tables: one line per sequence length."""
    lines = ["{"]
    for name, value in header.items():
        lines.append(f"  {json.dumps(name)}: {json.dumps(value)},")
    lines.append('  "configs": {')
    heads = []
    for head, by_seq in configs.items():
        rows = [f"{json.dumps(str(seq))}: {json.dumps(by_seq[seq])}"
                for seq in sorted(by_seq)]
        if per_mask:
            body = ('      "causal": {\n'
                    + ",\n".join(f"        {row}" for row in rows)
                    + "\n      }")
        else:
            body = ",\n".join(f"      {row}" for row in rows)
        heads.append(f"    {json.dumps(head)}: {{\n{body}\n    }}")
    lines.append(",\n".join(heads))
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def collect(directory, tag):
    from flywheel_tpu import tuned_block_sizes

    best = []
    for path in sorted(directory.glob("*.jsonl")):
        log = Log(path)
        if log.best is not None:
            best.append(log.best)
    by_impl = {impl: [b for b in best if b["cell"]["impl"] == impl]
               for impl in SEARCHERS}
    devices = {b["device"] for b in best}
    if len(devices) != 1:
        raise ValueError(f"expected searches from one device, got {devices}.")
    device = devices.pop()

    def head_tables(records):
        configs = {}
        for record in records:
            cell = record["cell"]
            head = (f"q_head-{cell['heads']}_kv_head-{cell['heads_k']}"
                    f"_head-{cell['head_dim']}")
            configs.setdefault(head, {})[cell["seq"]] = record["config"]
        return configs

    written = []
    for impl, order_name, extra, per_mask in (
            ("rpa", "block_order", {"rpa_git_sha": RPA_V3_GIT_SHA}, False),
            ("splash", "config_order", {"splash_git_sha": SPLASH_GIT_SHA},
             True)):
        if not by_impl[impl]:
            continue
        header = {"device": device, "dtype": "bfloat16", **extra,
                  order_name: by_impl[impl][0]["fields"]}
        path = HERE / f"{impl}_tuned_{tag}.json"
        path.write_text(dump_table(header, head_tables(by_impl[impl]),
                                   per_mask))
        written.append((path, len(by_impl[impl])))

    if by_impl["flywheel"]:
        path = tuned_block_sizes.TUNED_CONFIGS_PATH
        table = json.loads(path.read_text())
        section = table.setdefault(tuned_block_sizes.get_device_name(), {})
        for record in by_impl["flywheel"]:
            cell = record["cell"]
            key = tuned_block_sizes.tuned_config_key(
                "fwd", token_major=False,
                num_heads=cell["batch"] * cell["heads"], batch=None,
                seq_len=cell["seq"], head_dim=cell["head_dim"], causal=True,
                max_seqlen=None, return_lse=False,
                num_kv_heads=cell["batch"] * cell["heads_k"])
            section[key] = record["config"]
        table[tuned_block_sizes.get_device_name()] = dict(sorted(
            section.items()))
        path.write_text(json.dumps(table, indent=2) + "\n")
        written.append((path, len(by_impl["flywheel"])))

    for path, count in written:
        print(f"wrote {count} cells to {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl", choices=tuple(SEARCHERS))
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--heads-k", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--seq", type=int, default=16384)
    parser.add_argument("--output", type=pathlib.Path,
                        help="directory of the search logs, one per cell")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--candidate-timeout", type=int, default=300,
                        help="seconds before a candidate is killed as hung")
    parser.add_argument("--force", action="store_true",
                        help="search a cell again that already has a best")
    parser.add_argument("--rpa-table", type=pathlib.Path, default=RPA_TABLE)
    parser.add_argument("--splash-table", type=pathlib.Path,
                        default=SPLASH_TABLE)
    parser.add_argument("--rpa-source", type=pathlib.Path,
                        default=DEFAULT_RPA_SOURCE)
    parser.add_argument("--splash-source", type=pathlib.Path,
                        default=DEFAULT_SPLASH_SOURCE)
    parser.add_argument("--collect", type=pathlib.Path,
                        help="write the tables from this directory of "
                             "finished searches instead of searching")
    parser.add_argument("--device-tag", default="v7x",
                        help="suffix of the tables --collect writes")
    args = parser.parse_args(argv)

    if args.collect is not None:
        collect(args.collect, args.device_tag)
        return 0
    if args.impl is None or args.output is None:
        parser.error("--impl and --output are required to search")
    cell = Cell(args.impl, args.heads, args.heads_k, args.head_dim, "causal",
                1, args.seq)
    return search(cell, args)


if __name__ == "__main__":
    sys.exit(main())
