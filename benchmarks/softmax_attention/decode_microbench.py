#!/usr/bin/env python3
"""Decode steps of served models on one device: flywheel against batched RPA.

    PYTHONPATH=. python benchmarks/softmax_attention/decode_microbench.py \\
        --torchtpu-source ~/vllm-torchtpu \\
        --output results/benchmark/decode_microbench/flywheel.json
    python benchmarks/softmax_attention/decode_microbench.py --list
    python benchmarks/softmax_attention/decode_microbench.py \\
        --table results/benchmark/decode_microbench/*.json

A case is one decode step of a served model at its per-device attention shape
(query heads, KV heads, head_dim, KV dtype; 256-token pages and
max_model_len 16384, as vllm-torchtpu serves), a number of sequences and
their cached lengths. CASES holds the cases: Qwen3.5-35B-A3B-FP8 at TP4 over
its serving runs' length mixes, batch sizes and fixed lengths; other served
models at TP4; and the decode cells of ForgeMegakernel (arXiv 2609.12379,
Table 3) on one device.

The step runs the model's attention layers back to back (at most
--max-layers of them), each with its own KV cache and all reading one step's
lengths and page table, so XLA merges batched RPA's schedule kernel across
the layers as it does in serving. Each layer's query depends on the layer
before, the step's metadata passes an optimization barrier (so the schedule
is not hoisted out of the timed loop), and both kernels append the step's
token. flywheel runs flash_attn_with_kvcache as vllm-torchtpu's flywheel
backend calls it for decode; batched RPA (load_batched_rpa, --torchtpu-source)
runs ragged_paged_attention as its CUSTOM backend does, with the decode
request distribution and its own default block sizes.

Times are device time per op from the profiler, in us per layer: flywheel's
kernel; batched RPA's decode kernel, its mixed kernel (launched in
decode-only steps too) and its schedule (once per step, spread over the
model's attention layers). loop_us is all device time of the timed loop per
layer, the query chaining included. The DMA floor is the KV bytes a layer
reads (the cache layout's padding included) at --hbm-tbps. Each kernel's
output is checked against an f32 reference over its own cache, and its
append byte for byte. To compare flywheel versions, run once per checkout
and join the result files with --table.
"""

# jax loads only where a TPU run needs it, so --list and --table never claim
# the TPU.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import collections
import dataclasses
import fnmatch
import functools
import glob
import json
import math
import os
import pathlib
import re
import zlib

import numpy as np

PAGE = 256
MAX_MODEL_LEN = 16384
MIN_MAX_SEQS = 128
NUM_REFERENCE_SEQS = 8


@dataclasses.dataclass(frozen=True)
class Case:
    """One decode step: a per-device attention shape and its sequences."""

    name: str
    source: str  # model, sharding, and where the batch and lengths come from
    hq: int  # query heads per device
    hk: int  # KV heads per device
    d: int  # head_dim
    dtype: str  # KV cache dtype: "fp8" or "bf16"
    layers: int  # full-attention layers per step in the model
    b: int  # sequences
    # ("fixed", L) | ("uniform", lo, hi) | ("skew", long, lo, hi)
    # | ("lognormal", median, sigma), in cached tokens before the step
    lens: tuple


QWEN35 = {"hq": 4, "hk": 1, "d": 256, "dtype": "fp8", "layers": 10}
Q35 = "Qwen3.5-35B-A3B-FP8 TP4"
U1K = ("uniform", 1024, 2048)  # 1k/1k decode, requests staggered
CASES = [
    # vllm-torchtpu's serving runs of Qwen3.5-35B-A3B-FP8: c64, 128 seqs max.
    Case(
        "prod_decode_trace",
        f"{Q35}; the decode-only trace: 64 prompts of 1024 started together,"
        " traced at ~1.2-1.5k cached tokens",
        **QWEN35,
        b=64,
        lens=("fixed", 1344),
    ),
    Case(
        "prod_1k1k",
        f"{Q35}; 1k/1k at c64, requests staggered",
        **QWEN35,
        b=64,
        lens=U1K,
    ),
    Case(
        "prod_8k1k",
        f"{Q35}; 8k/1k at c64 (input 7373 +-11%, output 922 +-11%)",
        **QWEN35,
        b=64,
        lens=("uniform", 6554, 9115),
    ),
    Case(
        "prod_wide",
        f"{Q35}; 1k/1k at range ratio 0.8",
        **QWEN35,
        b=64,
        lens=("uniform", 205, 3687),
    ),
    Case(
        "prod_skew",
        f"{Q35}; one 16k sequence among 63 short ones",
        **QWEN35,
        b=64,
        lens=("skew", 16000, 128, 1024),
    ),
    Case(
        "prod_lognormal",
        f"{Q35}; lognormal lengths, median 1024, sigma 1",
        **QWEN35,
        b=64,
        lens=("lognormal", 1024, 1.0),
    ),
]
CASES += [
    Case(
        f"batch_b{b}",
        f"{Q35}; {b} sequences at 1k/1k lengths",
        **QWEN35,
        b=b,
        lens=U1K,
    )
    for b in (1, 8, 16, 32, 128, 256)
]
CASES += [
    Case(
        f"fixed_L{length}",
        f"{Q35}; 64 sequences of {length} cached tokens",
        **QWEN35,
        b=64,
        lens=("fixed", length),
    )
    for length in (256, 4096, 16000)
]
# Other served models' per-device shapes, 64 sequences at 1k/1k lengths.
CASES += [
    Case(
        "q35_397b_tp4",
        "Qwen3.5-397B-A17B-FP8 TP4 (32/2 heads, KV replicated)",
        hq=8,
        hk=1,
        d=256,
        dtype="fp8",
        layers=15,
        b=64,
        lens=U1K,
    ),
    Case(
        "q35_27b_tp4",
        "Qwen3.5-27B TP4 (24/4 heads)",
        hq=6,
        hk=1,
        d=256,
        dtype="fp8",
        layers=16,
        b=64,
        lens=U1K,
    ),
    Case(
        "q3vl_30b_tp4",
        "Qwen3-VL-30B-A3B TP4 (32/4 heads)",
        hq=8,
        hk=1,
        d=128,
        dtype="bf16",
        layers=48,
        b=64,
        lens=U1K,
    ),
    Case(
        "llama2_13b_tp4",
        "Llama-2-13B TP4 (MHA, 40/40 heads)",
        hq=10,
        hk=10,
        d=128,
        dtype="bf16",
        layers=40,
        b=64,
        lens=U1K,
    ),
]
# ForgeMegakernel's decode cells (batch, context), one device, bf16 KV, each
# model's head counts from its config. Not included: its Llama-3.2-1B cell
# (b=1, S=96), head_dim 64, which vllm-torchtpu's flywheel backend does not
# decode.
CASES += [
    Case(
        f"paper_{tag}_b{b}_s{seq}",
        f"{model} TP1 ({hq}/{hk} heads); paper cell b={b}, S={seq}",
        hq=hq,
        hk=hk,
        d=128,
        dtype="bf16",
        layers=layers,
        b=b,
        lens=("fixed", seq),
    )
    for tag, model, hq, hk, layers, b, seq in [
        ("qwen3_0.6b", "Qwen3-0.6B", 16, 8, 28, 8, 2048),
        ("qwen3_0.6b", "Qwen3-0.6B", 16, 8, 28, 16, 1024),
        ("qwen3_0.6b", "Qwen3-0.6B", 16, 8, 28, 16, 4096),
        ("qwen3_4b_8b", "Qwen3-4B and Qwen3-8B", 32, 8, 36, 1, 128),
        ("qwen3_4b", "Qwen3-4B", 32, 8, 36, 8, 1024),
        ("qwen3_8b", "Qwen3-8B", 32, 8, 36, 1, 512),
        ("qwen3_8b", "Qwen3-8B", 32, 8, 36, 1, 2048),
        ("llama3.1_8b", "Llama-3.1-8B", 32, 8, 32, 4, 2048),
        ("llama2_13b", "Llama-2-13B-chat (MHA)", 40, 40, 40, 1, 128),
        ("minicpm5_1b", "MiniCPM5-1B", 16, 2, 24, 1, 128),
        (
            "minicpm4",
            "MiniCPM4 (MiniCPM4-8B's heads; the paper's 1B and 3B)",
            32,
            2,
            32,
            1,
            128,
        ),
    ]
]


def case_lens(case: Case) -> np.ndarray:
    """Cached tokens of each sequence before the step, seeded by the name."""
    rng = np.random.default_rng(zlib.crc32(case.name.encode()))
    kind, *a = case.lens
    if kind == "fixed":
        return np.full(case.b, a[0])
    if kind == "uniform":
        return rng.integers(a[0], a[1], case.b)
    if kind == "skew":
        lens = rng.integers(a[1], a[2], case.b)
        lens[rng.integers(case.b)] = a[0]
        return lens
    if kind == "lognormal":
        lens = np.round(rng.lognormal(math.log(a[0]), a[1], case.b))
        return np.clip(lens, 16, MAX_MODEL_LEN - 384).astype(int)
    raise ValueError(kind)


def lens_label(lens) -> str:
    kind, *a = lens
    if kind == "fixed":
        return f"{a[0]}"
    if kind == "uniform":
        return f"{a[0]}–{a[1] - 1}"
    if kind == "skew":
        return f"1×{a[0]} + {a[1]}–{a[2] - 1}"
    return f"lognormal, median {a[0]}"


def kv_rows(kernel: str, hk: int, dtype: str) -> tuple[list[int], list[int]]:
    """(K rows, V rows) of each KV head among one token's head rows. Batched
    RPA: [k0, v0, k1, v1, ...]. flywheel's merged cache: the same in bf16; in
    fp8, a head pair's [k0, k1, v0, v1] in one u32 word."""
    if kernel == "rpa" or dtype == "bf16":
        return [2 * h for h in range(hk)], [2 * h + 1 for h in range(hk)]
    return (
        [4 * (h // 2) + h % 2 for h in range(hk)],
        [4 * (h // 2) + 2 + h % 2 for h in range(hk)],
    )


class Step:
    """One decode step of a case: its metadata, per-layer caches, q/k/v."""

    def __init__(self, rpa, case: Case, num_layers: int, iters: int):
        import jax
        import jax.numpy as jnp
        from flywheel_tpu.softmax_attention.kv_layout import merged_cache_heads

        self.rpa, self.case, self.iters = rpa, case, iters
        self.dtype = jnp.float8_e4m3fn if case.dtype == "fp8" else jnp.bfloat16
        self.lens = case_lens(case)
        total = self.lens + 1
        rng = np.random.default_rng(zlib.crc32(case.name.encode()) + 1)
        self.pages_per_seq = MAX_MODEL_LEN // PAGE
        max_seqs = max(MIN_MAX_SEQS, case.b)
        need = -(-total // PAGE)
        self.num_pages = int(need.sum()) + 1
        # pages scattered over the pool, as after a long serving run
        perm = rng.permutation(self.num_pages)
        table = np.zeros((max_seqs, self.pages_per_seq), np.int32)
        start = 0
        for i, n in enumerate(need):
            table[i, :n] = perm[start : start + n]
            start += n
        self.table = table
        kv_lens = np.zeros(max_seqs, np.int32)
        kv_lens[: case.b] = total  # serving's seq_lens count the step's token
        cu_q_lens = np.zeros(max_seqs + 1, np.int32)
        cu_q_lens[1 : case.b + 1] = np.arange(1, case.b + 1)
        cu_q_lens[case.b + 1 :] = case.b
        self.meta = (
            jnp.asarray(kv_lens),
            jnp.asarray(table.reshape(-1)),
            jnp.asarray(cu_q_lens),
            jnp.asarray([case.b, case.b, case.b], jnp.int32),
            jnp.asarray(case.b, jnp.int32),
        )
        self.rows = merged_cache_heads(case.hk, self.dtype)
        self.fw_shape = (self.num_pages, PAGE, self.rows, case.d)
        self.rpa_shape = tuple(
            rpa.get_kv_cache_shape(
                self.num_pages, PAGE, case.hk, case.d, self.dtype
            )
        )
        if math.prod(self.rpa_shape) != math.prod(self.fw_shape):
            raise ValueError(
                f"cache sizes differ: {self.fw_shape} vs {self.rpa_shape}"
            )
        self.token_bytes = self.rows * case.d * jnp.dtype(self.dtype).itemsize
        keys = jax.random.split(
            jax.random.key(zlib.crc32(case.name.encode())), 4
        )
        base = jax.random.normal(keys[0], self.fw_shape, jnp.bfloat16)
        base = base.astype(self.dtype)
        self.caches = tuple(jnp.copy(base) for _ in range(num_layers))
        del base
        self.q = jax.random.normal(
            keys[1], (case.b, case.hq, case.d), jnp.bfloat16
        )
        kv_shape = (case.b, case.hk, case.d)
        self.k = jax.random.normal(keys[2], kv_shape, jnp.bfloat16).astype(
            self.dtype
        )
        self.v = jax.random.normal(keys[3], kv_shape, jnp.bfloat16).astype(
            self.dtype
        )
        self.scale = case.d**-0.5
        self.descale = 1.0 if case.dtype == "fp8" else None

    def layer(self, kernel, q, cache, meta):
        """One attention layer's decode: (output [b, hq, d], cache)."""
        from flywheel_tpu import flash_attn_with_kvcache

        kv_lens, page_indices, cu_q_lens, distribution, num_active = meta
        b = self.case.b
        if kernel == "fw":
            # as vllm-torchtpu's flywheel backend decodes: the step's K/V
            # appended in-kernel at the length before it
            out, cache = flash_attn_with_kvcache(
                q[:, None],
                cache,
                None,
                self.k[:, None],
                self.v[:, None],
                cache_seqlens=kv_lens[:b] - 1,
                block_table=page_indices[: b * self.pages_per_seq].reshape(
                    b, self.pages_per_seq
                ),
                num_active=num_active,
                softmax_scale=self.scale,
                k_descale=self.descale,
                v_descale=self.descale,
            )
            return out[:, 0], cache
        out, cache = self.rpa.ragged_paged_attention(
            q,
            self.k,
            self.v,
            cache.reshape(self.rpa_shape),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=self.scale,
            k_scale=self.descale,
            v_scale=self.descale,
        )
        return out[:b], cache.reshape(self.fw_shape)

    def loop(self, kernel):
        """The compiled timed loop: iters steps of every layer."""
        import jax
        import jax.numpy as jnp
        from jax import lax

        # flywheel_tpu jits internally, keyed on its arguments
        jax.clear_caches()

        @functools.partial(jax.jit, donate_argnums=(0,))
        def run(caches, q):
            def body(_, carry):
                caches, q, acc = carry
                # q varies per step, so the barrier's outputs do too: batched
                # RPA's schedule, built from the metadata, runs every step, as
                # in serving, not once per loop.
                meta, q = lax.optimization_barrier((self.meta, q))
                new = []
                for cache in caches:
                    out, cache = self.layer(kernel, q, cache, meta)
                    new.append(cache)
                    q = q.astype(jnp.float32) + 1e-3 * out.astype(jnp.float32)
                    q = q.astype(jnp.bfloat16)
                return tuple(new), q, acc + out[0, 0, 0].astype(jnp.float32)

            return lax.fori_loop(
                0, self.iters, body, (caches, q, jnp.float32(0))
            )

        return run.lower(self.caches, self.q).compile()

    def once(self, kernel):
        """One layer on a copy of layer 0's cache: (output, cache)."""
        import jax
        import jax.numpy as jnp

        jax.clear_caches()
        return jax.jit(lambda c: self.layer(kernel, self.q, c, self.meta))(
            jnp.copy(self.caches[0])
        )

    def append_ok(self, kernel, cache) -> bool:
        """Only the step's token slots changed, holding the step's K and V."""
        import jax.numpy as jnp

        def bits(x):
            itemsize = jnp.dtype(x.dtype).itemsize
            return x.view(jnp.uint8 if itemsize == 1 else jnp.uint16)

        b = self.case.b
        changed = jnp.any(bits(cache) != bits(self.caches[0]), axis=(2, 3))
        pos = self.lens[:b]
        pages, slots = self.table[np.arange(b), pos // PAGE], pos % PAGE
        expect = np.zeros((self.num_pages, PAGE), bool)
        expect[pages, slots] = True
        new = cache[jnp.asarray(pages), jnp.asarray(slots)]
        k_rows, v_rows = kv_rows(kernel, self.case.hk, self.case.dtype)
        return (
            bool(jnp.array_equal(changed, jnp.asarray(expect)))
            and bool(jnp.array_equal(bits(new[:, k_rows]), bits(self.k)))
            and bool(jnp.array_equal(bits(new[:, v_rows]), bits(self.v)))
        )

    def reference(self, kernel, cache):
        """f32 attention of the first sequences over a kernel's cache."""
        import jax
        import jax.numpy as jnp

        c = self.case
        rows = cache.reshape(self.num_pages * PAGE, self.rows, c.d)
        k_rows, v_rows = (
            jnp.asarray(r) for r in kv_rows(kernel, c.hk, c.dtype)
        )
        outs = []
        for i in range(min(NUM_REFERENCE_SEQS, c.b)):
            n = int(self.lens[i]) + 1
            pages = self.table[i, : -(-n // PAGE)]
            index = (pages[:, None] * PAGE + np.arange(PAGE)).reshape(-1)[:n]
            kv = rows[jnp.asarray(index)].astype(jnp.float32)
            q = self.q[i].astype(jnp.float32).reshape(c.hk, c.hq // c.hk, c.d)
            s = jnp.einsum("hgd,nhd->hgn", q, kv[:, k_rows]) * self.scale
            p = jax.nn.softmax(s, axis=-1)
            o = jnp.einsum("hgn,nhd->hgd", p, kv[:, v_rows])
            outs.append(o.reshape(c.hq, c.d))
        return jnp.stack(outs)


def op_class(name: str) -> str | None:
    if name.startswith("flash_attn_with_kvcache"):
        return "fw_kernel"
    if name.startswith("RPA"):
        return {"d": "rpa_decode", "m": "rpa_mixed"}.get(
            name[3:4], "rpa_other_kernel"
        )
    if name.startswith("rpa_metadata_schedule"):
        return "rpa_schedule"
    return None


def device_times(trace_dir: str):
    """Per op class on the first TPU: [total ns, count], and the timed loops'
    durations (ns) in run order."""
    import jax

    xplane = sorted(glob.glob(f"{trace_dir}/**/*.xplane.pb", recursive=True))[
        -1
    ]
    data = jax.profiler.ProfileData.from_file(xplane)
    plane = next(p for p in data.planes if p.name.startswith("/device:TPU:"))
    ops = collections.defaultdict(lambda: [0, 0])
    modules = []
    for line in plane.lines:
        if line.name == "XLA Ops":
            for event in line.events:
                # an op's event name is its HLO text: "%name.N = (...) ..."
                match = re.match(r"%?([^\s=]+)", event.name)
                cls = op_class(match.group(1)) if match else None
                if cls:
                    ops[cls][0] += event.duration_ns
                    ops[cls][1] += 1
        elif line.name == "XLA Modules":
            modules += [
                (event.start_ns, event.duration_ns)
                for event in line.events
                if event.name.startswith("jit_run")
            ]
    return ops, [duration for _, duration in sorted(modules)]


def table_entry(case: Case, dtype) -> list | None:
    """flywheel's tuned-table entry for the case's decode call (None: its
    default config), from the same key fields the kernel looks up."""
    from flywheel_tpu.softmax_attention import dispatch

    config = dispatch.get_tuned_kvcache_config(
        dtype=dtype,
        num_query_heads=case.hq,
        num_kv_heads=case.hk,
        head_dim=case.d,
        capacity=MAX_MODEL_LEN,
        batch=case.b,
        append=True,
        sliding_window=None,
        return_lse=False,
    )
    return None if config is None else list(config)


def run_case(rpa, case: Case, args, trace_root: str) -> dict:
    import jax
    import jax.numpy as jnp

    lens = case_lens(case)
    num_pages = int((-(-(lens + 1) // PAGE)).sum()) + 1
    itemsize = 1 if case.dtype == "fp8" else 2
    rows = 4 * -(-case.hk // 2) if case.dtype == "fp8" else 2 * case.hk
    layer_bytes = num_pages * PAGE * rows * case.d * itemsize
    budget_layers = int(args.mem_gb * 2**30 // layer_bytes)
    num_layers = max(1, min(case.layers, args.max_layers, budget_layers))
    step = Step(rpa, case, num_layers, args.iters)
    tokens = int((lens + 1).sum())
    res = {
        "case": dataclasses.asdict(case),
        "device": jax.devices()[0].device_kind,
        "lens_mean": float(lens.mean()),
        "layers_simulated": num_layers,
        "iters": args.iters,
        "dma_floor_us": tokens * step.token_bytes / (args.hbm_tbps * 1e6),
        "fw_table_entry": table_entry(case, step.dtype),
    }

    # Checks on layer 0. The two fp8 layouts order K and V rows differently
    # (same bytes per token), so each kernel is checked over its own cache.
    for kernel in args.kernels:
        try:
            out, cache = step.once(kernel)
        except Exception as e:  # pylint: disable=broad-exception-caught
            res[f"{kernel}_error"] = repr(e)[:400]
            continue
        ref = np.asarray(step.reference(kernel, cache))
        out = np.asarray(out.astype(jnp.float32))[: len(ref)]
        scale = float(np.max(np.abs(ref))) or 1.0
        res[f"{kernel}_rel_err"] = float(np.max(np.abs(out - ref))) / scale
        res[f"{kernel}_append_ok"] = step.append_ok(kernel, cache)
        del out, cache

    loops = {k: step.loop(k) for k in args.kernels if f"{k}_error" not in res}
    for run in loops.values():  # warm
        step.caches, step.q, acc = run(step.caches, step.q)
        acc.block_until_ready()
    trace_dir = os.path.join(trace_root, case.name)
    with jax.profiler.trace(trace_dir):
        for run in loops.values():
            step.caches, step.q, acc = run(step.caches, step.q)
            acc.block_until_ready()
    ops, modules = device_times(trace_dir)
    calls = args.iters * num_layers
    for i, kernel in enumerate(loops):
        if kernel == "fw":
            per = {"fw_kernel": ops["fw_kernel"][0] / 1e3 / calls}
            per["attention_us"] = per["fw_kernel"]
        else:
            per = {
                cls: ops[cls][0] / 1e3 / calls
                for cls in ("rpa_decode", "rpa_mixed", "rpa_other_kernel")
            }
            per["rpa_schedule_per_step"] = (
                ops["rpa_schedule"][0] / 1e3 / args.iters
            )
            per["rpa_schedule_ops_per_step"] = (
                ops["rpa_schedule"][1] / args.iters
            )
            # once per step in serving, spread over the attention layers
            per["rpa_schedule"] = per["rpa_schedule_per_step"] / case.layers
            per["attention_us"] = sum(
                per[cls]
                for cls in (
                    "rpa_decode",
                    "rpa_mixed",
                    "rpa_other_kernel",
                    "rpa_schedule",
                )
            )
        if i < len(modules):
            per["loop_us"] = modules[i] / 1e3 / calls
        res[kernel] = per
    if "fw" in res and "rpa" in res:
        res["fw_over_rpa"] = (
            res["fw"]["attention_us"] / res["rpa"]["attention_us"]
        )
    return res


def summary(name: str, r: dict) -> str:
    parts = [name + ":"]
    for kernel in ("fw", "rpa"):
        if kernel in r:
            p = r[kernel]
            s = f"{kernel} {p['attention_us']:.1f} us/layer"
            if kernel == "rpa":
                s += (
                    f" (decode {p['rpa_decode']:.1f}, mixed"
                    f" {p['rpa_mixed']:.1f}, schedule"
                    f" {p['rpa_schedule_per_step']:.1f}/step in"
                    f" {p['rpa_schedule_ops_per_step']:.0f} ops)"
                )
            parts.append(s + ";")
        elif f"{kernel}_error" in r:
            parts.append(f"{kernel} ERROR {r[kernel + '_error'][:120]};")
    if "fw_over_rpa" in r:
        parts.append(f"fw/rpa {r['fw_over_rpa']:.2f};")
    parts.append(
        f"floor {r['dma_floor_us']:.1f}; rel err fw"
        f" {r.get('fw_rel_err', float('nan')):.2g} rpa"
        f" {r.get('rpa_rel_err', float('nan')):.2g}; append ok fw"
        f" {r.get('fw_append_ok')} rpa {r.get('rpa_append_ok')}; table"
        f" {r.get('fw_table_entry')}; layers {r['layers_simulated']}"
    )
    return " ".join(parts)


def print_table(paths: list[str]) -> None:
    """Markdown table: batched RPA from the first file, flywheel from each."""
    runs = [
        (pathlib.Path(p).stem, json.loads(pathlib.Path(p).read_text("utf-8")))
        for p in paths
    ]
    print(
        "| case | heads (q/kv), head_dim, KV | seqs | cached tokens |"
        " batched RPA | "
        + " | ".join(f"flywheel {n}" for n, _ in runs)
        + " | DMA floor | notes |"
    )
    print("|---|---|---:|---|---:|" + "---:|" * len(runs) + "---:|---|")
    for name, r in runs[0][1].items():
        c = r["case"]
        rpa = r.get("rpa", {}).get("attention_us")
        cells, notes = [], []
        if r.get("rpa_error"):
            error = r["rpa_error"]
            notes.append(
                "batched RPA: "
                + ("out of VMEM" if "vmem" in error else error[:60])
            )
        for n, run in runs:
            rr = run.get(name, {})
            v = rr.get("fw", {}).get("attention_us")
            if v is None:
                cells.append("—")
            else:
                cells.append(f"{v:.1f}" + (f" ({v / rpa:.2f}x)" if rpa else ""))
            if rr.get("fw_error"):
                notes.append(f"flywheel {n}: {rr['fw_error'][:60]}")
            failed = [
                k
                for k in ("fw_append_ok", "rpa_append_ok")
                if rr.get(k) is False
            ]
            failed += [
                k
                for k in ("fw_rel_err", "rpa_rel_err")
                if (rr.get(k) or 0) > 0.05
            ]
            if failed:
                notes.append(f"CHECK FAILED ({n}): " + ", ".join(failed))
            if rr and rr.get("fw_table_entry") is None and "fw" in rr:
                notes.append(f"flywheel {n}: default config")
        rpa_cell = "—" if rpa is None else f"{rpa:.1f}"
        print(
            f"| {name} | {c['hq']}/{c['hk']}, {c['d']}, {c['dtype']} |"
            f" {c['b']} | {lens_label(c['lens'])} | {rpa_cell} | "
            + " | ".join(cells)
            + f" | {r.get('dma_floor_us', 0):.1f} | {'; '.join(notes)} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", maxsplit=1)[0]
    )
    parser.add_argument("--list", action="store_true", help="list the cases")
    parser.add_argument(
        "--table", nargs="+", metavar="JSON", help="markdown table of results"
    )
    parser.add_argument(
        "--output", help="result file (JSON); cases already in it are skipped"
    )
    parser.add_argument(
        "--cases", default="*", help="comma-separated globs of case names"
    )
    parser.add_argument("--kernels", default="fw,rpa")
    parser.add_argument(
        "--torchtpu-source",
        type=pathlib.Path,
        help="a vllm-torchtpu checkout for batched RPA (load_batched_rpa)",
    )
    parser.add_argument(
        "--iters", type=int, default=20, help="steps per timed loop"
    )
    parser.add_argument("--max-layers", type=int, default=10)
    parser.add_argument(
        "--mem-gb", type=float, default=24, help="KV cache budget of a step"
    )
    parser.add_argument(
        "--hbm-tbps",
        type=float,
        default=3.5,
        help="DMA rate of the floor (a v7x core's measured rate)",
    )
    parser.add_argument(
        "--trace", help="profiler trace directory (default: OUTPUT.traces)"
    )
    args = parser.parse_args()
    args.kernels = args.kernels.split(",")
    patterns = args.cases.split(",")
    cases = [
        c for c in CASES if any(fnmatch.fnmatch(c.name, p) for p in patterns)
    ]
    if args.list:
        for c in cases:
            print(
                f"{c.name:28s} {c.hq:2d}/{c.hk:<2d} d{c.d} {c.dtype:4s}"
                f" b{c.b:<3d} {lens_label(c.lens):22s} {c.layers:2d} layers "
                f" {c.source}"
            )
        return
    if args.table:
        print_table(args.table)
        return
    if not args.output:
        parser.error("--output is required to run cases")
    rpa = None
    if "rpa" in args.kernels:
        if args.torchtpu_source is None:
            parser.error("batched RPA needs --torchtpu-source")
        from benchmarks.common.rpa import load_batched_rpa

        rpa = load_batched_rpa(args.torchtpu_source)
    import jax

    print(jax.devices()[0].device_kind, jax.__version__, flush=True)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(output.read_text("utf-8")) if output.exists() else {}
    trace_root = args.trace or f"{args.output}.traces"
    for case in cases:
        if case.name in results:
            continue
        try:
            r = run_case(rpa, case, args, trace_root)
        except Exception as e:  # pylint: disable=broad-exception-caught
            r = {"case": dataclasses.asdict(case), "error": repr(e)[:600]}
            print(f"{case.name}: ERROR {repr(e)[:600]}", flush=True)
        else:
            print(summary(case.name, r), flush=True)
        results[case.name] = r
        output.write_text(json.dumps(results, indent=1, default=str), "utf-8")
        jax.clear_caches()


if __name__ == "__main__":
    main()
