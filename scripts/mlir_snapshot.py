"""Per-commit MLIR gate: snapshot the IR of every generated kernel.

A fixed set of representative configurations is lowered through the public
API (jax.vjp of flash_attn_func for the backward, plus the internal entry
points listed below). Every tpu_custom_call the lowered program reaches has
its serialized Mosaic module decoded back to text, and the XLA program around
the kernels (pads, schedule tables, rotary gathers, masks) is kept as well. A
commit that only moves code must leave every file byte-identical.

    PYTHONPATH=. uv run --no-sync python scripts/mlir_snapshot.py dump \
        --out results/mlir_baseline
    PYTHONPATH=. uv run --no-sync python scripts/mlir_snapshot.py compare \
        --baseline results/mlir_baseline --out /tmp/mlir_candidate

dump writes, per configuration, <config>.mlir (its kernels) and
<config>.wrapper.mlir (the lowered program, each kernel's body replaced by
its call index), plus index.json: the environment, the source the snapshot
was taken from, and per configuration the label of every call (kernel_name
and the frontend attributes, i.e. kernel_metadata) and every tuned-table
lookup with its outcome (hit, miss, or rejected with the reason). dump
refuses a flywheel_tpu/ with uncommitted changes, so a baseline always names
a commit. compare dumps afresh into --out from any tree, then exits 1 if any
configuration's kernels, wrapper, labels or tuned lookups differ from
--baseline, printing what differs and the head of a unified diff (SSA value
names shown as %_, since one inserted op renumbers every later value).
Lowering only: no kernel runs, but the builds read the chip (tuned configs,
VMEM size), so it runs on a TPU host.

Each configuration lowers from cleared caches (jax.clear_caches() and every
functools cache in flywheel_tpu), so its IR and its tuned lookups do not
depend on the configurations lowered before it.

Normalization, i.e. everything a pure code move can change:
- MLIR locations are dropped (the modules are printed without debug info),
  and with them source file paths, line numbers, named scopes and the Python
  argument names of the kernel refs.
- Every symbol a Mosaic module defines (its kernel function, named after the
  pallas_call name or the Python kernel function, and its index-map functions)
  is renamed to @sym0, @sym1, ... in definition order, references included.
  The wrapper's module and private functions (named after jitted Python
  functions) are renamed the same way; @main keeps its name.
- kernel_name and the frontend attributes move from the IR files to
  index.json, so compare reports a relabeled kernel apart from a changed one.
A module text that still holds a loc( or a .py path after this raises.
Everything else is kept verbatim: op order, attributes and types, and per
call the operand and result types, the custom call attributes (layouts,
aliases, side effects) and the backend config without its body. The wrapper
is not inlined, so wrapping a helper in jax.jit (or unwrapping one) shows up
as a wrapper difference.

Internal entry points (no public wrapper; a commit that moves them updates
the imports here): flash_attn_interface._flash_attn_func_pcp_causal_offset
(the runtime-offset kernel tpu-inference's PCP attention calls),
pallas.kv_cache_write.write_kv_cache_pages, and
core.tuning.lookup / _warn_untuned_once, which the tuned lookup record wraps.

Out of reach by construction: interpret=True (plain XLA, no
tpu_custom_call); the dense head_fold % 8 == 0 group lse copy, since a dense
head fold comes only from a tuned entry and no entry has return_lse; and
rotary_groups > 1 or a rotary batch other than 1, since prepare_rotary always
returns one shared table.
"""

import argparse
import base64
import dataclasses
import difflib
import functools
import hashlib
import importlib.metadata
import itertools
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
from jax.interpreters import mlir as jax_mlir
from jaxlib.mlir import ir, passmanager
from jaxlib.mosaic.python import tpu as mosaic_tpu

import flywheel_tpu
from flywheel_tpu import flash_attn_interface, linear_attention
from flywheel_tpu.core import tuning

# Note: the page writer has no public wrapper.
from flywheel_tpu.pallas.kv_cache_write import write_kv_cache_pages

FORMAT_VERSION = 2
INDEX_FILE = "index.json"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_DIR = "flywheel_tpu"
BF16 = jnp.bfloat16
F32 = jnp.float32
I32 = jnp.int32
FP8_E4M3 = jnp.float8_e4m3fn
FP8_E5M2 = jnp.float8_e5m2
# A symbol reference: @bare_id or @"quoted".
SYMBOL_REF = re.compile(r'@(?:"((?:[^"\\]|\\.)*)"|([A-Za-z_][\w$.]*))')
# Source information that must not survive normalization.
SOURCE_LEAK = re.compile(r"\bloc\(|\.py\b")
# A call section header of a config file; a repeated call names its first.
CALL_HEADER = re.compile(
    r"// ===== call (\d+)(?:: identical to call (\d+))? ====="
)
# The backend config a wrapper's tpu_custom_call holds in place of its kernel.
CALL_MARKER = re.compile(r"call (\d+)")
# An SSA value other than a kernel argument, anonymized in printed diffs.
SSA_VALUE = re.compile(r"%(?!arg\d+\b)[\w$.\-]+")
# A tpu_custom_call attribute that is a label, not part of the kernel.
LABEL_ATTRIBUTES = ("kernel_name", "mhlo.frontend_attributes")
# The custom call's identity and its serialized kernel, rendered apart.
SKIPPED_ATTRIBUTES = ("call_target_name", "backend_config")
# JAX flags that change what a program lowers to.
LOWERING_FLAGS = (
    "jax_enable_x64",
    "jax_default_matmul_precision",
    "jax_numpy_dtype_promotion",
    "jax_use_shardy_partitioner",
    "jax_pallas_enable_debug_checks",
    "jax_mosaic_allow_hlo",
    "jax_enable_checks",
)
# Environment variables that change what a program lowers to.
LOWERING_ENV_VARS = (
    "LIBTPU_INIT_ARGS",
    "XLA_FLAGS",
    "GDN_PROFILE_SCOPES",
    "KDA_PROFILE_SCOPES",
)
# Outcome of every tuned-table lookup of the lowering in progress, by key.
TUNED_LOOKUPS: dict[str, str] = {}


@dataclasses.dataclass(frozen=True)
class Case:
    """One configuration: jax.jit(entry) lowered at arrays, with static bound.

    api names the entry point for index.json; static values bind as keyword
    arguments, arrays as abstract keyword arguments.
    """

    name: str
    api: str
    entry: Callable[..., Any]
    arrays: dict[str, jax.ShapeDtypeStruct]
    static: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class KernelCall:
    """One tpu_custom_call: its labels and its normalized IR section."""

    labels: dict[str, str]
    section: str


@dataclasses.dataclass(frozen=True)
class Snapshot:
    """One configuration lowered: its kernel calls in execution order, the
    normalized wrapper program and its tuned lookups."""

    calls: list[KernelCall]
    wrapper: str
    tuned: dict[str, str]


def spec(shape: tuple[int, ...], dtype: Any = BF16) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape, dtype)


def merged_pool(
    num_pages: int, page_size: int, kv_heads: int, head_dim: int, dtype: Any
) -> jax.ShapeDtypeStruct:
    """A merged paged KV pool: bf16 holds 2 * kv_heads rows per token, fp8
    rounds them up to a multiple of 4 (an odd head count gets a zero pad)."""
    if dtype == BF16:
        cache_heads = 2 * kv_heads
    else:
        cache_heads = -(-2 * kv_heads // 4) * 4
    return spec((num_pages, page_size, cache_heads, head_dim), dtype)


def head_major_qkv(
    batch: int,
    heads: int,
    kv_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    head_dim: int,
    head_dim_v: int | None = None,
) -> dict[str, jax.ShapeDtypeStruct]:
    head_dim_v = head_dim if head_dim_v is None else head_dim_v
    return dict(
        q=spec((batch, heads, seqlen_q, head_dim)),
        k=spec((batch, kv_heads, seqlen_k, head_dim)),
        v=spec((batch, kv_heads, seqlen_k, head_dim_v)),
    )


def heads_outer_qkv(
    heads: int, kv_heads: int, batch: int, seqlen: int, head_dim: int
) -> dict[str, jax.ShapeDtypeStruct]:
    """Heads-outer operands: (nheads, batch, seqlen, head_dim)."""
    return dict(
        q=spec((heads, batch, seqlen, head_dim)),
        k=spec((kv_heads, batch, seqlen, head_dim)),
        v=spec((kv_heads, batch, seqlen, head_dim)),
    )


def token_major_qkv(
    batch: int, seqlen: int, heads: int, kv_heads: int, head_dim: int
) -> dict[str, jax.ShapeDtypeStruct]:
    """Token-major operands: (batch, seqlen, nheads * head_dim)."""
    return dict(
        q=spec((batch, seqlen, heads * head_dim)),
        k=spec((batch, seqlen, kv_heads * head_dim)),
        v=spec((batch, seqlen, kv_heads * head_dim)),
    )


def packed_qkv(
    heads: int, kv_heads: int, total: int, head_dim: int, num_seqs: int
) -> dict[str, jax.ShapeDtypeStruct]:
    return dict(
        q=spec((heads, total, head_dim)),
        k=spec((kv_heads, total, head_dim)),
        v=spec((kv_heads, total, head_dim)),
        cu_seqlens_q=spec((num_seqs + 1,), I32),
        cu_seqlens_k=spec((num_seqs + 1,), I32),
    )


def rotary_tables(seqlen: int, head_dim: int) -> dict[str, Any]:
    return dict(
        rotary_cos=spec((seqlen, head_dim // 2)),
        rotary_sin=spec((seqlen, head_dim // 2)),
    )


def sinks(heads: int) -> dict[str, jax.ShapeDtypeStruct]:
    return dict(learnable_sink=spec((heads,), F32))


def flash_attn_func_vjp(
    q: jax.Array, k: jax.Array, v: jax.Array, dout: jax.Array, **static: Any
) -> tuple[jax.Array, ...]:
    """(dq, dk, dv) of flash_attn_func at cotangent dout."""
    _, vjp_fn = jax.vjp(
        functools.partial(flywheel_tpu.flash_attn_func, **static), q, k, v
    )
    return vjp_fn(dout)


def dense_cases() -> list[Case]:
    """flash_attn_func forward configurations."""
    api = "flash_attn_func"
    entry = flywheel_tpu.flash_attn_func
    mha = head_major_qkv(2, 4, 4, 512, 512, 128)
    gqa = head_major_qkv(2, 8, 2, 512, 512, 128)
    long_mha = head_major_qkv(1, 4, 4, 1024, 1024, 128)
    rotary_gqa = head_major_qkv(1, 4, 2, 512, 512, 128) | rotary_tables(
        512, 128
    )
    return [
        Case("fwd_causal_mha_d128", api, entry, mha, dict(causal=True)),
        Case("fwd_causal_gqa_d128", api, entry, gqa, dict(causal=True)),
        Case("fwd_full_mha_d128", api, entry, mha, dict(causal=False)),
        # Note: q pads 100 -> 128 rows and the causal diagonal sits at 284.
        Case(
            "fwd_causal_rect_short_q",
            api,
            entry,
            head_major_qkv(1, 4, 4, 100, 384, 128),
            dict(causal=True),
        ),
        Case(
            "fwd_causal_mha_d64",
            api,
            entry,
            head_major_qkv(2, 4, 4, 512, 512, 64),
            dict(causal=True),
        ),
        # Note: 1024 flat heads of d64 at 512 hit the tuned table's seq_minor
        # layout with transposed PV and a 16-head fold.
        Case(
            "fwd_causal_mha_d64_tuned_seq_minor",
            api,
            entry,
            head_major_qkv(32, 32, 32, 512, 512, 64),
            dict(causal=True),
        ),
        # Note: 256 flat heads of d64 at 2048 hit seq_minor and transposed PV
        # with no head fold; Q and K rotate on that layout.
        Case(
            "fwd_causal_mha_d64_tuned_seq_minor_fold1_rotary",
            api,
            entry,
            head_major_qkv(8, 32, 32, 2048, 2048, 64) | rotary_tables(2048, 64),
            dict(causal=True),
        ),
        # Note: 512 flat heads of d128 at 512 hit a tuned 16-head fold.
        Case(
            "fwd_causal_mha_d128_tuned_fold",
            api,
            entry,
            head_major_qkv(16, 32, 32, 512, 512, 128),
            dict(causal=True),
        ),
        Case(
            "fwd_causal_gqa_d256",
            api,
            entry,
            head_major_qkv(1, 4, 2, 512, 512, 256),
            dict(causal=True),
        ),
        Case(
            "fwd_causal_mla_d192_v128",
            api,
            entry,
            head_major_qkv(1, 4, 4, 512, 512, 192, 128),
            dict(causal=True),
        ),
        Case(
            "fwd_causal_softcap",
            api,
            entry,
            mha,
            dict(causal=True, softcap=30.0),
        ),
        # Note: a left-bounded dense window runs the static band copies.
        Case(
            "fwd_window_band_causal",
            api,
            entry,
            long_mha,
            dict(causal=True, window_size=(256, -1)),
        ),
        Case(
            "fwd_window_band_two_sided",
            api,
            entry,
            long_mha,
            dict(window_size=(192, 64)),
        ),
        Case(
            "fwd_window_right_only",
            api,
            entry,
            long_mha,
            dict(window_size=(-1, 128)),
        ),
        # Note: the only window shape whose right bound is unbounded.
        Case(
            "fwd_window_left_only_full",
            api,
            entry,
            long_mha,
            dict(window_size=(256, -1)),
        ),
        Case(
            "fwd_causal_sink_gqa",
            api,
            entry,
            gqa | sinks(8),
            dict(causal=True),
        ),
        Case(
            "fwd_causal_sink_gqa_return_lse",
            api,
            entry,
            gqa | sinks(8),
            dict(causal=True, return_softmax_lse=True),
        ),
        Case(
            "fwd_causal_rotary_qk",
            api,
            entry,
            rotary_gqa,
            dict(causal=True),
        ),
        Case(
            "fwd_causal_rotary_q_only_interleaved",
            api,
            entry,
            rotary_gqa,
            dict(causal=True, rotary_k=False, rotary_interleaved=True),
        ),
        Case(
            "fwd_full_rotary_qk_interleaved_sink",
            api,
            entry,
            rotary_gqa | sinks(4),
            dict(rotary_interleaved=True),
        ),
        Case(
            "fwd_token_major_causal_gqa_d128",
            api,
            entry,
            token_major_qkv(2, 512, 8, 2, 128),
            dict(causal=True, token_major=True, head_dim=128),
        ),
        Case(
            "fwd_token_major_causal_mha_d64_paired",
            api,
            entry,
            token_major_qkv(2, 512, 4, 4, 64),
            dict(causal=True, token_major=True, head_dim=64),
        ),
        # Note: 16 heads of d128 per row at 512 hit a tuned 16-head lane fold;
        # Q and K rotate inside the fold.
        Case(
            "fwd_token_major_causal_mha_d128_tuned_fold16_rotary",
            api,
            entry,
            token_major_qkv(32, 512, 16, 16, 128) | rotary_tables(512, 128),
            dict(causal=True, token_major=True, head_dim=128),
        ),
        # Note: 32 heads of d64 per row at 1024 hit a tuned fold of 4 (two
        # lane tiles of paired heads) with 4 stages.
        Case(
            "fwd_token_major_full_mha_d64_tuned_fold4_stages4",
            api,
            entry,
            token_major_qkv(16, 1024, 32, 32, 64),
            dict(token_major=True, head_dim=64),
        ),
        Case(
            "fwd_token_major_window_softcap_sink_lse_gqa",
            api,
            entry,
            token_major_qkv(2, 512, 8, 2, 128) | sinks(8),
            dict(
                window_size=(128, 0),
                softcap=30.0,
                return_softmax_lse=True,
                token_major=True,
                head_dim=128,
            ),
        ),
        Case(
            "fwd_causal_return_lse",
            api,
            entry,
            mha,
            dict(causal=True, return_softmax_lse=True),
        ),
        Case(
            "fwd_heads_outer_causal",
            api,
            entry,
            heads_outer_qkv(4, 4, 2, 512, 128),
            dict(causal=True, heads_outer_fold=True),
        ),
        Case(
            "fwd_heads_outer_causal_rotary_q_only",
            api,
            entry,
            heads_outer_qkv(4, 4, 2, 512, 128) | rotary_tables(512, 128),
            dict(causal=True, rotary_k=False, heads_outer_fold=True),
        ),
        # Note: 3 q heads per kv head over 4 rows of 16K tokens hit a tuned
        # heads-outer GQA entry with 4 stages.
        Case(
            "fwd_heads_outer_gqa_d256_tuned_stages4_sink",
            api,
            entry,
            heads_outer_qkv(3, 1, 4, 16384, 256) | sinks(3),
            dict(causal=True, heads_outer_fold=True),
        ),
    ]


def runtime_offset_cases() -> list[Case]:
    """The PCP forward with a traced causal offset (the runtime-offset kernel
    tpu-inference calls); internal, so it is not reached from the API."""
    api = "_flash_attn_func_pcp_causal_offset"
    # pylint: disable-next=protected-access
    entry = flash_attn_interface._flash_attn_func_pcp_causal_offset
    offset = dict(causal_offset=spec((), I32))
    return [
        Case(
            "fwd_pcp_runtime_offset_gqa_lse",
            api,
            entry,
            head_major_qkv(1, 8, 2, 512, 2048, 128) | offset,
            dict(return_softmax_lse=True),
        ),
        Case(
            "fwd_pcp_runtime_offset_heads_outer",
            api,
            entry,
            heads_outer_qkv(8, 2, 1, 512, 128)
            | dict(
                k=spec((2, 1, 2048, 128)),
                v=spec((2, 1, 2048, 128)),
            )
            | offset,
            dict(heads_outer_fold=True),
        ),
    ]


def backward_cases() -> list[Case]:
    """jax.vjp of flash_attn_func: the lse forward plus the backward kernels."""
    api = "jax.vjp(flash_attn_func)"
    entry = flash_attn_func_vjp

    def _with_dout(
        qkv: dict[str, jax.ShapeDtypeStruct],
    ) -> dict[str, jax.ShapeDtypeStruct]:
        q_shape = qkv["q"].shape
        if len(q_shape) == 4:
            dout_shape = (*q_shape[:-1], qkv["v"].shape[-1])
        else:
            dout_shape = q_shape
        return qkv | dict(dout=spec(dout_shape))

    return [
        # Note: GQA folds each kv head's q heads in the general kernel.
        Case(
            "bwd_causal_gqa_d128",
            api,
            entry,
            _with_dout(head_major_qkv(1, 8, 2, 512, 512, 128)),
            dict(causal=True),
        ),
        # Note: square MHA in one 512 block runs the single-block fold kernel.
        Case(
            "bwd_causal_mha_single_block_fold",
            api,
            entry,
            _with_dout(head_major_qkv(2, 4, 4, 512, 512, 128)),
            dict(causal=True),
        ),
        Case(
            "bwd_full_mha_single_block_fold",
            api,
            entry,
            _with_dout(head_major_qkv(2, 4, 4, 512, 512, 128)),
            dict(causal=False),
        ),
        # Note: two 1024 q blocks keep MHA on the unfolded general kernel.
        Case(
            "bwd_full_mha_multi_block",
            api,
            entry,
            _with_dout(head_major_qkv(1, 2, 2, 2048, 2048, 128)),
            dict(causal=False),
        ),
        # Note: 128 flat heads at 2048 hit a tuned entry with 2048 blocks.
        Case(
            "bwd_full_mha_tuned_block2048",
            api,
            entry,
            _with_dout(head_major_qkv(1, 128, 128, 2048, 2048, 128)),
            dict(causal=False),
        ),
        Case(
            "bwd_causal_rect_gqa",
            api,
            entry,
            _with_dout(head_major_qkv(1, 4, 2, 256, 512, 128)),
            dict(causal=True),
        ),
        # Note: seqlen_q > seqlen_k gives a negative causal offset: the
        # leading q rows see no key, so the backward guards their lse.
        Case(
            "bwd_causal_q_longer_than_kv",
            api,
            entry,
            _with_dout(head_major_qkv(1, 4, 4, 512, 256, 128)),
            dict(causal=True),
        ),
        Case(
            "bwd_causal_mla_d192_v128",
            api,
            entry,
            _with_dout(head_major_qkv(1, 4, 4, 512, 512, 192, 128)),
            dict(causal=True),
        ),
        # Note: an 8-head group's f32 dq at 16K tokens passes half the VMEM
        # limit, so the backward runs as two q-row chunks.
        Case(
            "bwd_causal_gqa_q_chunked",
            api,
            entry,
            _with_dout(head_major_qkv(1, 8, 1, 16384, 16384, 128)),
            dict(causal=True),
        ),
        Case(
            "bwd_token_major_causal_mha_d64_paired",
            api,
            entry,
            _with_dout(token_major_qkv(2, 512, 4, 4, 64)),
            dict(causal=True, token_major=True, head_dim=64),
        ),
        # Note: square token-major MHA in one block runs the folded kernel.
        Case(
            "bwd_token_major_causal_mha_d128_folded",
            api,
            entry,
            _with_dout(token_major_qkv(2, 512, 4, 4, 128)),
            dict(causal=True, token_major=True, head_dim=128),
        ),
        Case(
            "bwd_token_major_causal_gqa_d128",
            api,
            entry,
            _with_dout(token_major_qkv(2, 512, 8, 2, 128)),
            dict(causal=True, token_major=True, head_dim=128),
        ),
    ]


def varlen_cases() -> list[Case]:
    """flash_attn_varlen_func over packed and paged K/V."""
    api = "flash_attn_varlen_func"
    entry = flywheel_tpu.flash_attn_varlen_func
    packed_bounds = dict(max_seqlen_q=512, max_seqlen_k=512)
    gqa = packed_qkv(8, 2, 1000, 128, 4)
    token_major_gqa = dict(
        q=spec((1000, 8 * 128)),
        k=spec((1000, 2 * 128)),
        v=spec((1000, 2 * 128)),
        cu_seqlens_q=spec((5,), I32),
        cu_seqlens_k=spec((5,), I32),
    )
    # Note: paged sequences of up to 4 pages of 128 tokens, 300 packed q rows.
    paged_static = dict(
        v=None, cu_seqlens_k=None, max_seqlen_q=128, max_seqlen_k=512
    )

    def _paged(
        heads: int, kv_heads: int, dtype: Any
    ) -> dict[str, jax.ShapeDtypeStruct]:
        return dict(
            q=spec((heads, 300, 128)),
            k=merged_pool(12, 128, kv_heads, 128, dtype),
            cu_seqlens_q=spec((4,), I32),
            block_table=spec((3, 4), I32),
            seqused_k=spec((3,), I32),
        )

    fp8_static = dict(k_descale=0.5, v_descale=0.25)
    return [
        Case(
            "varlen_causal_gqa",
            api,
            entry,
            gqa,
            packed_bounds | dict(causal=True),
        ),
        # Note: 2048-token buckets split each kv block into 4 compute tiles,
        # so the per-sequence copy runs its dead-tile loop.
        Case(
            "varlen_causal_gqa_tile_loop",
            api,
            entry,
            packed_qkv(8, 2, 8192, 128, 4),
            dict(max_seqlen_q=2048, max_seqlen_k=2048, causal=True),
        ),
        Case(
            "varlen_causal_rotary_qk",
            api,
            entry,
            gqa | rotary_tables(512, 128),
            packed_bounds | dict(causal=True),
        ),
        Case(
            "varlen_causal_rotary_q_only",
            api,
            entry,
            gqa | rotary_tables(512, 128),
            packed_bounds | dict(causal=True, rotary_k=False),
        ),
        Case(
            "varlen_window_return_lse",
            api,
            entry,
            gqa,
            packed_bounds | dict(window_size=(128, 0), return_softmax_lse=True),
        ),
        # Note: 1024-token buckets give two compute tiles per kv block, the
        # first peeled, with no tile loop.
        Case(
            "varlen_window_lse_two_tiles",
            api,
            entry,
            packed_qkv(8, 2, 4096, 128, 4),
            dict(
                max_seqlen_q=1024,
                max_seqlen_k=1024,
                window_size=(256, 0),
                return_softmax_lse=True,
            ),
        ),
        Case(
            "varlen_window_left_only",
            api,
            entry,
            gqa,
            packed_bounds | dict(window_size=(128, -1)),
        ),
        # Note: MHA on square 512 blocks takes the varlen fold table's 8.
        Case(
            "varlen_causal_mha_head_fold",
            api,
            entry,
            packed_qkv(16, 16, 2048, 128, 4),
            packed_bounds | dict(causal=True),
        ),
        Case(
            "varlen_causal_mha_head_fold_return_lse",
            api,
            entry,
            packed_qkv(16, 16, 2048, 128, 4),
            packed_bounds | dict(causal=True, return_softmax_lse=True),
        ),
        Case(
            "varlen_token_major_causal_gqa",
            api,
            entry,
            token_major_gqa,
            packed_bounds | dict(causal=True, token_major=True, head_dim=128),
        ),
        Case(
            "varlen_token_major_window_lse_gqa",
            api,
            entry,
            token_major_gqa,
            packed_bounds
            | dict(
                window_size=(128, 0),
                return_softmax_lse=True,
                token_major=True,
                head_dim=128,
            ),
        ),
        Case(
            "varlen_full_sink_softcap",
            api,
            entry,
            gqa | sinks(8),
            packed_bounds | dict(softcap=30.0),
        ),
        Case(
            "varlen_paged_bf16_causal_gqa",
            api,
            entry,
            _paged(8, 2, BF16),
            paged_static | dict(causal=True),
        ),
        Case(
            "varlen_paged_fp8_e4m3_causal_gqa",
            api,
            entry,
            _paged(8, 2, FP8_E4M3),
            paged_static | fp8_static | dict(causal=True, nheads_k=2),
        ),
        # Note: 3 kv heads pad to 8 merged fp8 rows.
        Case(
            "varlen_paged_fp8_e5m2_causal_odd_kv_heads",
            api,
            entry,
            _paged(6, 3, FP8_E5M2),
            paged_static | fp8_static | dict(causal=True, nheads_k=3),
        ),
        Case(
            "varlen_paged_window_rotary_q_num_active",
            api,
            entry,
            _paged(8, 2, BF16)
            | rotary_tables(512, 128)
            | dict(num_active=spec((), I32)),
            paged_static
            | dict(causal=True, window_size=(256, 0), rotary_k=False),
        ),
        Case(
            "varlen_paged_window_left_only",
            api,
            entry,
            _paged(8, 2, BF16),
            paged_static | dict(window_size=(128, -1)),
        ),
        Case(
            "varlen_paged_full_mha_softcap_sink",
            api,
            entry,
            _paged(4, 4, BF16) | sinks(4),
            paged_static | dict(softcap=30.0),
        ),
        # Note: 2048-token kv buckets stage 1024-token blocks of two compute
        # tiles; GQA 8:2 with sinks folds 4 q heads into each lse row.
        Case(
            "varlen_paged_causal_gqa_sink_lse_two_tiles",
            api,
            entry,
            dict(
                q=spec((8, 1024, 128)),
                k=merged_pool(64, 128, 2, 128, BF16),
                cu_seqlens_q=spec((4,), I32),
                block_table=spec((3, 16), I32),
                seqused_k=spec((3,), I32),
            )
            | sinks(8),
            dict(
                v=None,
                cu_seqlens_k=None,
                max_seqlen_q=512,
                max_seqlen_k=2048,
                causal=True,
                return_softmax_lse=True,
            ),
        ),
    ]


def kvcache_cases() -> list[Case]:
    """flash_attn_with_kvcache: decode, verify windows and packed appends."""
    api = "flash_attn_with_kvcache"
    entry = flywheel_tpu.flash_attn_with_kvcache
    decode_q = spec((4, 1, 8, 128))
    contiguous_gqa = dict(
        q=decode_q,
        k_cache=spec((4, 1024, 2, 128)),
        v_cache=spec((4, 1024, 2, 128)),
        cache_seqlens=spec((4,), I32),
    )
    fp8_static = dict(k_descale=0.5, v_descale=0.25)

    def _paged(
        q_shape: tuple[int, ...],
        dtype: Any,
        num_seqs: int = 4,
        kv_heads: int = 2,
    ) -> dict[str, jax.ShapeDtypeStruct]:
        return dict(
            q=spec(q_shape),
            k_cache=merged_pool(32, 128, kv_heads, 128, dtype),
            block_table=spec((num_seqs, 8), I32),
            cache_seqlens=spec((num_seqs,), I32),
        )

    def _new_kv(
        shape: tuple[int, ...], dtype: Any = BF16
    ) -> dict[str, jax.ShapeDtypeStruct]:
        return dict(k=spec(shape, dtype), v=spec(shape, dtype))

    return [
        Case(
            "kvcache_decode_contiguous_gqa",
            api,
            entry,
            contiguous_gqa,
            {},
        ),
        # Note: one kv head takes the head-major MQA view of the cache.
        Case(
            "kvcache_decode_contiguous_mqa_append",
            api,
            entry,
            dict(
                q=decode_q,
                k_cache=spec((4, 1024, 1, 128)),
                v_cache=spec((4, 1024, 1, 128)),
                cache_seqlens=spec((4,), I32),
            )
            | _new_kv((4, 1, 1, 128)),
            {},
        ),
        Case(
            "kvcache_decode_contiguous_window_lse",
            api,
            entry,
            contiguous_gqa,
            dict(window_size=(256, -1), return_softmax_lse=True),
        ),
        # Note: the tuned entry stages 2048 tokens in two 1024 compute tiles
        # over 4 stages, and the append writes the token-major cache in place.
        Case(
            "kvcache_decode_contiguous_mha_d256_append_tuned",
            api,
            entry,
            dict(
                q=spec((128, 1, 8, 256)),
                k_cache=spec((128, 2048, 8, 256)),
                v_cache=spec((128, 2048, 8, 256)),
                cache_seqlens=spec((128,), I32),
            )
            | _new_kv((128, 1, 8, 256)),
            {},
        ),
        # Note: 8 q heads per kv head load each kv head on its own.
        Case(
            "kvcache_decode_contiguous_gqa16x2_sink_lse",
            api,
            entry,
            dict(
                q=spec((4, 1, 16, 128)),
                k_cache=spec((4, 1024, 2, 128)),
                v_cache=spec((4, 1024, 2, 128)),
                cache_seqlens=spec((4,), I32),
            )
            | sinks(16),
            dict(return_softmax_lse=True),
        ),
        Case(
            "kvcache_decode_contiguous_gqa_d64",
            api,
            entry,
            dict(
                q=spec((4, 1, 8, 64)),
                k_cache=spec((4, 1024, 2, 64)),
                v_cache=spec((4, 1024, 2, 64)),
                cache_seqlens=spec((4,), I32),
            ),
            {},
        ),
        Case(
            "kvcache_decode_paged_bf16_append",
            api,
            entry,
            _paged((4, 1, 8, 128), BF16) | _new_kv((4, 1, 2, 128)),
            dict(v_cache=None),
        ),
        Case(
            "kvcache_decode_paged_fp8_e4m3",
            api,
            entry,
            _paged((4, 1, 8, 128), FP8_E4M3),
            dict(v_cache=None, nheads_k=2) | fp8_static,
        ),
        # Note: 2 q heads per odd kv head leave the fp8 pair-word fragment.
        Case(
            "kvcache_decode_paged_fp8_e5m2_odd_kv_heads_append",
            api,
            entry,
            _paged((4, 1, 6, 128), FP8_E5M2, kv_heads=3)
            | _new_kv((4, 1, 3, 128), FP8_E5M2),
            dict(v_cache=None, nheads_k=3) | fp8_static,
        ),
        # Note: 384-token pages give 768-token blocks of two 384 compute
        # tiles, so a left window starts mid-block.
        Case(
            "kvcache_decode_paged_page384_d256_window_lse",
            api,
            entry,
            dict(
                q=spec((4, 1, 8, 256)),
                k_cache=merged_pool(16, 384, 2, 256, BF16),
                block_table=spec((4, 4), I32),
                cache_seqlens=spec((4,), I32),
            )
            | _new_kv((4, 1, 2, 256)),
            dict(v_cache=None, window_size=(512, -1), return_softmax_lse=True),
        ),
        Case(
            "kvcache_decode_paged_softcap",
            api,
            entry,
            _paged((4, 1, 8, 128), BF16),
            dict(v_cache=None, softcap=30.0),
        ),
        Case(
            "kvcache_decode_paged_sink",
            api,
            entry,
            _paged((4, 1, 8, 128), BF16) | sinks(8),
            dict(v_cache=None),
        ),
        # Note: 4 tokens of 4 q heads per kv head fold into the decode kernel,
        # after the ragged append writes the window.
        Case(
            "kvcache_verify_window_append_causal",
            api,
            entry,
            _paged((4, 4, 8, 128), BF16)
            | _new_kv((4, 4, 2, 128))
            | dict(num_active=spec((), I32)),
            dict(v_cache=None, causal=True, return_softmax_lse=True),
        ),
        Case(
            "kvcache_verify_window_append_full",
            api,
            entry,
            _paged((4, 4, 8, 128), BF16) | _new_kv((4, 4, 2, 128)),
            dict(v_cache=None),
        ),
        Case(
            "kvcache_varlen_append_causal",
            api,
            entry,
            _paged((300, 8, 128), BF16, num_seqs=3)
            | _new_kv((300, 2, 128))
            | dict(cu_seqlens_q=spec((4,), I32)),
            dict(v_cache=None, causal=True, max_seqlen_q=128),
        ),
        Case(
            "kvcache_varlen_append_fp8_e4m3_causal",
            api,
            entry,
            _paged((300, 8, 128), FP8_E4M3, num_seqs=3)
            | _new_kv((300, 2, 128), FP8_E4M3)
            | dict(cu_seqlens_q=spec((4,), I32)),
            dict(v_cache=None, causal=True, max_seqlen_q=128, nheads_k=2)
            | fp8_static,
        ),
        # Note: a left window keeps a multi-token call off the decode kernel.
        Case(
            "kvcache_multitoken_window_fp8_e4m3",
            api,
            entry,
            _paged((2, 64, 8, 128), FP8_E4M3, num_seqs=2),
            dict(
                v_cache=None,
                causal=True,
                window_size=(128, -1),
                nheads_k=2,
            )
            | fp8_static,
        ),
    ]


def kv_write_cases() -> list[Case]:
    """write_kv_cache_pages, the host-sliced page writer."""
    api = "write_kv_cache_pages"

    def _write(kv_heads: int, dtype: Any) -> dict[str, jax.ShapeDtypeStruct]:
        return dict(
            k=spec((300, kv_heads, 128), dtype),
            v=spec((300, kv_heads, 128), dtype),
            kv_cache=merged_pool(6, 128, kv_heads, 128, dtype),
            slices=spec((3, 8), I32),
            num_slices=spec((1,), I32),
        )

    return [
        Case("kv_write_bf16", api, write_kv_cache_pages, _write(2, BF16), {}),
        Case(
            "kv_write_fp8_e4m3_odd_kv_heads",
            api,
            write_kv_cache_pages,
            _write(3, FP8_E4M3),
            {},
        ),
    ]


def linear_attention_case(
    name: str,
    kernel: str,
    *,
    replay_dtype: Any = None,
    num_spec_tokens: int = 0,
    has_conv_bias: bool = True,
    **static: Any,
) -> Case:
    """A fused conv1d GDN / KDA call over 256 tokens of 4 sequences.

    Both kernels of a call are always emitted: the BATCHED decode segment
    and the PER_SEQ prefill / mixed segment.
    """
    n_kq, n_v, d_k, d_v, kernel_size = 2, 4, 128, 128, 4
    num_tokens, num_seqs = 256, 4
    replay_window = 16
    dim = n_kq * d_k * 2 + n_v * d_v
    num_slots = 1 + num_seqs * (num_spec_tokens + 1)
    is_kda = kernel == "kda"
    # Note: KDA gates per key channel, GDN per value head.
    gate_width = n_v * d_k if is_kda else n_v
    arrays = dict(
        qkv=spec((num_tokens, dim)),
        b=spec((num_tokens, n_v)),
        conv_state=spec((num_slots, kernel_size - 1, dim), F32),
        conv_weight=spec((dim, 1, kernel_size), F32),
        a_log=spec((n_v,), F32),
        dt_bias=spec((gate_width,), F32),
        query_start_loc=spec((num_seqs + 1,), I32),
        state_indices=spec((num_seqs,), I32),
        distribution=spec((3,), I32),
        seq_lens=spec((num_seqs,), I32),
        read_state_indices=spec((num_seqs,), I32),
    )
    arrays["g" if is_kda else "a"] = spec((num_tokens, gate_width))
    static = dict(
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        num_spec_tokens=num_spec_tokens,
        **static,
    )
    if has_conv_bias:
        arrays["conv_bias"] = spec((dim,), F32)
    else:
        static["conv_bias"] = None
    if num_spec_tokens:
        arrays["read_offsets"] = spec((num_seqs,), I32)
    if replay_dtype is None:
        arrays["recurrent_state"] = spec((num_slots, n_v, d_k, d_v), F32)
        api = f"fused_conv1d_{kernel}"
    else:
        # Note: the replay layouts as replay_layout(n_kq, n_v, d_k, d_v) gives
        # them: GDN pads its per-head gate row to a whole lane tile.
        gate_dim = n_v * d_k if is_kda else -(-n_v // 128) * 128
        arrays |= dict(
            recurrent_state=spec((num_slots, 2, n_v, d_k // 2, d_v), I32),
            replay_state=spec(
                (num_slots, replay_window, n_v * d_v + n_kq * d_k),
                replay_dtype,
            ),
            replay_gates=spec((num_slots, replay_window, gate_dim), F32),
            replay_lens=spec((num_seqs,), I32),
        )
        static["replay_window"] = replay_window
        api = f"fused_conv1d_{kernel}_replay"
    return Case(name, api, getattr(linear_attention, api), arrays, static)


def linear_attention_cases() -> list[Case]:
    bf16_precision = dict(compute_precision=jnp.dtype(BF16))
    return [
        linear_attention_case("gdn_prefill_decode", "gdn"),
        linear_attention_case(
            "gdn_prefill_decode_bf16_precision", "gdn", **bf16_precision
        ),
        linear_attention_case(
            "gdn_spec_tokens_no_conv_bias",
            "gdn",
            num_spec_tokens=3,
            has_conv_bias=False,
        ),
        linear_attention_case("gdn_replay_f32", "gdn", replay_dtype=F32),
        linear_attention_case("gdn_replay_bf16", "gdn", replay_dtype=BF16),
        # Note: verify windows with replay checkpoints are a kernel of their
        # own, its replay ring sized by num_spec_tokens.
        linear_attention_case(
            "gdn_replay_spec_tokens_bf16",
            "gdn",
            replay_dtype=BF16,
            num_spec_tokens=3,
        ),
        linear_attention_case("kda_prefill_decode", "kda"),
        linear_attention_case(
            "kda_prefill_decode_bf16_precision_no_zero_init",
            "kda",
            zero_initialize_out=False,
            **bf16_precision,
        ),
        linear_attention_case(
            "kda_spec_tokens_lower_bound_beta_activated",
            "kda",
            num_spec_tokens=3,
            lower_bound=-5.0,
            beta_is_activated=True,
        ),
        linear_attention_case("kda_replay_f32", "kda", replay_dtype=F32),
        linear_attention_case("kda_replay_bf16", "kda", replay_dtype=BF16),
        linear_attention_case(
            "kda_replay_spec_tokens_f32",
            "kda",
            replay_dtype=F32,
            num_spec_tokens=3,
        ),
    ]


def all_cases() -> list[Case]:
    cases = [
        *dense_cases(),
        *runtime_offset_cases(),
        *backward_cases(),
        *varlen_cases(),
        *kvcache_cases(),
        *kv_write_cases(),
        *linear_attention_cases(),
    ]
    names = [case.name for case in cases]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate configuration names: {duplicates}")
    return cases


def select_cases(only: str | None) -> list[Case]:
    cases = all_cases()
    if only is None:
        return cases
    selected = [case for case in cases if re.search(only, case.name)]
    if not selected:
        raise ValueError(f"--only {only!r} matches no configuration.")
    return selected


def record_tuned_lookups() -> None:
    """Wrap the tuned-table lookup, so that every lowering records in
    TUNED_LOOKUPS each key it looked up: hit, miss, or an entry the build
    rejected (with the reason). Every lookup and rejection goes through
    these two core.tuning functions; the wrappers replace them for the
    process, so a caller has to reach lookup through the module
    (tuning.lookup), not a name it imported.
    """
    lookup = tuning.lookup
    # pylint: disable-next=protected-access
    warn_untuned_once = tuning._warn_untuned_once
    if getattr(lookup, "is_recorder", False):
        raise RuntimeError("the tuned lookups are recorded already.")
    bound_by_name = [
        name
        for name, module in list(sys.modules.items())
        if name.startswith(PACKAGE_DIR + ".")
        and module is not tuning
        and any(value is lookup for value in vars(module).values())
    ]
    if bound_by_name:
        raise RuntimeError(
            f"{bound_by_name} import core.tuning.lookup by name, so their"
            " lookups would not be recorded; call tuning.lookup instead."
        )

    def _lookup(table: pathlib.Path, key: str) -> list[int | str | bool] | None:
        config = lookup(table, key)
        TUNED_LOOKUPS.setdefault(key, "miss" if config is None else "hit")
        return config

    def _warn_once(
        table: pathlib.Path, device_name: str, key: str, reason: str
    ) -> None:
        if reason:
            TUNED_LOOKUPS[key] = f"rejected: {reason}"
        else:
            TUNED_LOOKUPS.setdefault(key, "miss")
        warn_untuned_once(table, device_name, key, reason)

    _lookup.is_recorder = True
    tuning.lookup = _lookup
    # pylint: disable-next=protected-access
    tuning._warn_untuned_once = _warn_once


def clear_caches() -> None:
    """Drop JAX's trace caches and every functools cache of a flywheel_tpu
    module (kernel builders, tile resolvers, tuned-table loads), so the next
    lowering builds, and looks up, everything afresh."""
    jax.clear_caches()
    for name, module in list(sys.modules.items()):
        if name != PACKAGE_DIR and not name.startswith(PACKAGE_DIR + "."):
            continue
        for value in vars(module).values():
            cache_clear = getattr(value, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()


def is_tpu_custom_call(op: ir.Operation) -> bool:
    return (
        op.name == "stablehlo.custom_call"
        and ir.StringAttr(op.attributes["call_target_name"]).value
        == "tpu_custom_call"
    )


def iter_nested_ops(op: ir.Operation) -> Iterator[ir.Operation]:
    """Every op nested in op, in program order."""
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                child = child.operation
                yield child
                yield from iter_nested_ops(child)


def iter_tpu_custom_calls(module: ir.Module) -> Iterator[ir.Operation]:
    """The tpu_custom_call ops of a lowered program in execution order: @main
    walked op by op, each func.call followed into its callee, so outlining a
    helper into its own function does not reorder the calls. A function
    called twice yields its ops twice."""
    functions = {}
    for op in module.body.operations:
        if op.operation.name == "func.func":
            symbol = ir.StringAttr(op.operation.attributes["sym_name"]).value
            functions[symbol] = op.operation
    if "main" not in functions:
        raise ValueError("the lowered program has no @main function.")

    def _walk(op: ir.Operation) -> Iterator[ir.Operation]:
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    child = child.operation
                    if child.name == "func.call":
                        callee = ir.FlatSymbolRefAttr(
                            child.attributes["callee"]
                        ).value
                        yield from _walk(functions[callee])
                    elif is_tpu_custom_call(child):
                        yield child
                    yield from _walk(child)

    yield from _walk(functions["main"])


def defined_symbols(op: ir.Operation) -> list[str]:
    """sym_name of op and of every op nested in it, in definition order."""
    symbols = []
    if "sym_name" in op.attributes:
        symbols.append(ir.StringAttr(op.attributes["sym_name"]).value)
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                symbols.extend(defined_symbols(child.operation))
    return symbols


def rename_symbols(text: str, symbols: list[str]) -> str:
    """text with every symbol in symbols renamed to @sym<index>, references
    included; raises if source information survives."""
    if len(set(symbols)) != len(symbols):
        raise ValueError(f"a module defines a symbol twice: {symbols}")
    renamed = {symbol: f"sym{index}" for index, symbol in enumerate(symbols)}

    def _rename(match: re.Match[str]) -> str:
        symbol = match.group(2) if match.group(1) is None else match.group(1)
        if symbol in renamed:
            return f"@{renamed[symbol]}"
        else:
            return match.group(0)

    text = SYMBOL_REF.sub(_rename, text)
    leak = SOURCE_LEAK.search(text)
    if leak is not None:
        context_text = text[max(0, leak.start() - 80) : leak.end() + 80]
        raise ValueError(
            "source information survived normalization near:\n" + context_text
        )
    return text


def mosaic_text(serialized: bytes) -> str:
    """The Mosaic module of a tpu_custom_call body as normalized text."""
    context = jax_mlir.make_ir_context()
    mosaic_tpu.register_dialect(context)
    with context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(serialized)
        # Note: the body is stored in Mosaic's versioned serialized form;
        # deserializing gives the ops as the current dialect prints them.
        passmanager.PassManager.parse(
            "builtin.module(mosaic-serde{serialize=false})"
        ).run(module.operation)
        symbols = defined_symbols(module.operation)
        text = module.operation.get_asm(enable_debug_info=False)
    return rename_symbols(text, symbols)


def kernel_call(op: ir.Operation) -> KernelCall:
    attributes = op.attributes
    names = sorted(attributes)
    if "kernel_name" not in names:
        raise ValueError("a tpu_custom_call carries no kernel_name.")
    labels = {
        name: (
            ir.StringAttr(attributes[name]).value
            if name == "kernel_name"
            else str(attributes[name])
        )
        for name in LABEL_ATTRIBUTES
        if name in names
    }
    lines = [
        f"// operand {index}: {value.type}"
        for index, value in enumerate(op.operands)
    ]
    lines += [
        f"// result {index}: {value.type}"
        for index, value in enumerate(op.results)
    ]
    for name in names:
        if name not in LABEL_ATTRIBUTES and name not in SKIPPED_ATTRIBUTES:
            lines.append(f"// attribute {name} = {attributes[name]}")
    backend_config = json.loads(
        ir.StringAttr(attributes["backend_config"]).value
    )
    body = base64.b64decode(backend_config["custom_call_config"].pop("body"))
    lines.append(
        "// backend_config = " + json.dumps(backend_config, sort_keys=True)
    )
    lines.append(mosaic_text(body).rstrip("\n"))
    return KernelCall(labels=labels, section="\n".join(lines))


def split_program(program: str) -> tuple[list[KernelCall], str]:
    """The kernel calls of a lowered program in execution order, and the
    normalized program text with each kernel's backend config replaced by
    'call <index of its first execution>' and its labels dropped.

    Raises unless the calls reached from @main are every tpu_custom_call of
    the module, so a kernel referenced another way (a composite
    decomposition, an uncalled function) cannot drop out silently.
    """
    with jax_mlir.make_ir_context(), ir.Location.unknown():
        module = ir.Module.parse(program)
        num_kernel_ops = sum(
            is_tpu_custom_call(op) for op in iter_nested_ops(module.operation)
        )
        calls = []
        num_reached_ops = 0
        for op in iter_tpu_custom_calls(module):
            config = ir.StringAttr(op.attributes["backend_config"]).value
            marker = CALL_MARKER.fullmatch(config)
            if marker is not None:
                calls.append(calls[int(marker.group(1))])
                continue
            calls.append(kernel_call(op))
            num_reached_ops += 1
            op.attributes["backend_config"] = ir.StringAttr.get(
                f"call {len(calls) - 1}"
            )
            for name in LABEL_ATTRIBUTES:
                if name in op.attributes:
                    del op.attributes[name]
        if num_reached_ops != num_kernel_ops:
            raise ValueError(
                f"the module holds {num_kernel_ops} tpu_custom_call ops, but"
                f" only {num_reached_ops} are reached from @main by func.call."
            )
        symbols = [
            symbol
            for symbol in defined_symbols(module.operation)
            if symbol != "main"
        ]
        text = module.operation.get_asm(enable_debug_info=False)
    return calls, rename_symbols(text, symbols)


def lower_case(case: Case) -> Snapshot:
    clear_caches()
    TUNED_LOOKUPS.clear()
    entry = functools.partial(case.entry, **case.static)
    program = jax.jit(entry).lower(**case.arrays).compiler_ir("stablehlo")
    calls, wrapper = split_program(
        program.operation.get_asm(enable_debug_info=False)
    )
    if not calls:
        raise ValueError(f"{case.name} reaches no tpu_custom_call.")
    return Snapshot(
        calls=calls, wrapper=wrapper, tuned=dict(sorted(TUNED_LOOKUPS.items()))
    )


def render_config(name: str, calls: list[KernelCall]) -> str:
    """The config file: every call's section in order, a repeated call as a
    reference to its first occurrence."""
    lines = [f"// config {name}: {len(calls)} tpu_custom_call(s)"]
    first_call = {}
    for index, call in enumerate(calls):
        if call.section in first_call:
            lines.append(
                f"// ===== call {index}: identical to call"
                f" {first_call[call.section]} ====="
            )
        else:
            first_call[call.section] = index
            lines.append(f"// ===== call {index} =====")
            lines.append(call.section)
    return "\n".join(lines) + "\n"


def describe_environment() -> dict[str, Any]:
    """What the IR depends on besides the code: a change here makes every
    comparison meaningless, so compare refuses it."""
    if jax.default_backend() != "tpu":
        raise RuntimeError(
            "the snapshot lowers for the chip it runs on (tuned configs, VMEM"
            f" size); run it on a TPU host, not {jax.default_backend()}."
        )
    device = jax.devices()[0]
    return dict(
        jax=jax.__version__,
        jaxlib=importlib.metadata.version("jaxlib"),
        libtpu=importlib.metadata.version("libtpu"),
        device_kind=device.device_kind,
        jax_flags={
            flag: str(jax.config.values[flag]) for flag in LOWERING_FLAGS
        },
        env_vars={name: os.environ.get(name) for name in LOWERING_ENV_VARS},
    )


def run_git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def describe_source() -> dict[str, Any]:
    """Where the snapshot came from: the commit, the uncommitted changes of
    flywheel_tpu/, a hash of its files as they are on disk (gitignored ones
    excluded), and a hash of this script."""
    files = run_git(
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        PACKAGE_DIR,
    ).split("\0")
    tree = hashlib.sha256()
    # Note: a file deleted but still in the index is not on disk; the status
    # lists it.
    for relative_path in sorted(
        path for path in files if path and (REPO_ROOT / path).is_file()
    ):
        content = (REPO_ROOT / relative_path).read_bytes()
        tree.update(f"{relative_path}\0{len(content)}\0".encode())
        tree.update(content)
    return dict(
        git_head=run_git("rev-parse", "HEAD").strip(),
        package_status=run_git(
            "status", "--porcelain", "--untracked-files=all", "--", PACKAGE_DIR
        ).splitlines(),
        package_sha256=tree.hexdigest(),
        script_sha256=hashlib.sha256(
            pathlib.Path(__file__).read_bytes()
        ).hexdigest(),
    )


def check_import_root() -> None:
    """Fail unless flywheel_tpu is this checkout's: a stale editable install
    would snapshot another tree."""
    package_root = pathlib.Path(flywheel_tpu.__file__).resolve().parents[1]
    if package_root != REPO_ROOT:
        raise RuntimeError(
            f"flywheel_tpu is imported from {package_root}, not this checkout"
            f" {REPO_ROOT}; run with PYTHONPATH={REPO_ROOT}."
        )


def summarize_tuned(tuned: dict[str, str]) -> str:
    if not tuned:
        return "no tuned lookup"
    num_hits = sum(outcome == "hit" for outcome in tuned.values())
    return f"tuned {num_hits}/{len(tuned)} hit"


def dump(out_dir: pathlib.Path, only: str | None, require_clean: bool) -> None:
    check_import_root()
    source = describe_source()
    if require_clean and source["package_status"]:
        raise RuntimeError(
            f"{PACKAGE_DIR}/ has uncommitted changes, so this snapshot would"
            " name no commit; commit or stash them, or use compare:\n"
            + "\n".join(source["package_status"])
        )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty.")
    cases = select_cases(only)
    environment = describe_environment()
    record_tuned_lookups()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Note: index.json records every lookup outcome, so the per-key warnings
    # are noise.
    warnings.filterwarnings(
        "ignore", message="flywheel_tpu: no usable tuned config"
    )
    configs = {}
    start = time.perf_counter()
    for case in cases:
        case_start = time.perf_counter()
        try:
            snapshot = lower_case(case)
        except Exception as error:
            raise RuntimeError(f"configuration {case.name} failed") from error
        (out_dir / f"{case.name}.mlir").write_text(
            render_config(case.name, snapshot.calls), encoding="utf-8"
        )
        (out_dir / f"{case.name}.wrapper.mlir").write_text(
            snapshot.wrapper, encoding="utf-8"
        )
        configs[case.name] = dict(
            api=case.api,
            calls=[call.labels for call in snapshot.calls],
            tuned=snapshot.tuned,
        )
        kernel_names = ", ".join(
            call.labels["kernel_name"] for call in snapshot.calls
        )
        print(
            f"{case.name}: {kernel_names}; {summarize_tuned(snapshot.tuned)}"
            f" ({time.perf_counter() - case_start:.1f}s)",
            flush=True,
        )
    index = dict(
        format=FORMAT_VERSION,
        environment=environment,
        source=source,
        configs=configs,
    )
    (out_dir / INDEX_FILE).write_text(
        json.dumps(index, indent=1) + "\n", encoding="utf-8"
    )
    print(
        f"dumped {len(cases)} configurations to {out_dir} in"
        f" {time.perf_counter() - start:.1f}s (flywheel_tpu from"
        f" {pathlib.Path(flywheel_tpu.__file__).parent})"
    )


def load_index(snapshot_dir: pathlib.Path) -> dict[str, Any]:
    index = json.loads((snapshot_dir / INDEX_FILE).read_text("utf-8"))
    if index["format"] != FORMAT_VERSION:
        raise ValueError(
            f"{snapshot_dir} has snapshot format {index['format']}, not"
            f" {FORMAT_VERSION}."
        )
    return index


def split_calls(text: str) -> list[str]:
    """Every call's section of a config file, repeats resolved."""
    calls, current = [], None
    for line in text.splitlines()[1:]:
        header = CALL_HEADER.fullmatch(line)
        if header is None:
            if current is None:
                raise ValueError(f"IR line outside a call section: {line!r}")
            current.append(line)
        elif header.group(2) is None:
            current = []
            calls.append(current)
        else:
            calls.append(calls[int(header.group(2))])
            current = None
    return ["\n".join(call) for call in calls]


def anonymized_lines(text: str) -> list[str]:
    return [SSA_VALUE.sub("%_", line) for line in text.splitlines()]


def is_line_permutation(baseline: str, candidate: str) -> bool:
    """Whether the texts hold the same lines up to SSA names and order: what
    hoisting or sinking a traced op gives. Rewired operands give it too, so
    this only labels a difference; it never passes one."""
    return sorted(anonymized_lines(baseline)) == sorted(
        anonymized_lines(candidate)
    )


def describe_changed_calls(
    baseline: str, candidate: str, labels: list[dict[str, str]]
) -> str:
    baseline_calls = split_calls(baseline)
    candidate_calls = split_calls(candidate)
    if len(baseline_calls) != len(candidate_calls):
        return f"{len(baseline_calls)} -> {len(candidate_calls)} calls"
    changed = []
    for index, (baseline_call, candidate_call) in enumerate(
        zip(baseline_calls, candidate_calls)
    ):
        if baseline_call == candidate_call:
            continue
        if is_line_permutation(baseline_call, candidate_call):
            note = "; same lines reordered or rewired"
        else:
            note = ""
        changed.append(f"{index} ({labels[index]['kernel_name']}{note})")
    return "call " + ", ".join(changed)


def diff_head(
    baseline: str, candidate: str, name: str, max_lines: int
) -> list[str]:
    """The head of a unified diff with SSA value names shown as %_, since one
    inserted op renumbers every later value; the raw diff if only names
    differ."""
    baseline_lines = anonymized_lines(baseline)
    candidate_lines = anonymized_lines(candidate)
    if baseline_lines == candidate_lines:
        baseline_lines = baseline.splitlines()
        candidate_lines = candidate.splitlines()
    diff = difflib.unified_diff(
        baseline_lines,
        candidate_lines,
        f"baseline/{name}",
        f"candidate/{name}",
        lineterm="",
    )
    return list(itertools.islice(diff, max_lines))


def print_source(side: str, source: dict[str, Any]) -> None:
    if source["package_status"]:
        status = f"{len(source['package_status'])} uncommitted change(s)"
    else:
        status = "clean"
    print(
        f"{side}: commit {source['git_head'][:12]}, {PACKAGE_DIR}/ {status},"
        f" tree {source['package_sha256'][:12]}, script"
        f" {source['script_sha256'][:12]}"
    )


def compare_snapshots(
    baseline_dir: pathlib.Path,
    candidate_dir: pathlib.Path,
    only: str | None,
    max_diff_lines: int,
) -> bool:
    """Print every difference; True when the snapshots match."""
    baseline, candidate = load_index(baseline_dir), load_index(candidate_dir)
    print_source("baseline ", baseline["source"])
    print_source("candidate", candidate["source"])
    if (
        baseline["source"]["script_sha256"]
        != candidate["source"]["script_sha256"]
    ):
        print(
            "NOTE: this script changed since the baseline; its configurations"
            " or rendering may differ from the baseline's."
        )
    if baseline["environment"] != candidate["environment"]:
        print(
            "ENVIRONMENT differs, so the IR is not comparable:\n  baseline "
            f" {baseline['environment']}\n  candidate"
            f" {candidate['environment']}"
        )
        return False
    baseline_names = [
        name
        for name in baseline["configs"]
        if only is None or re.search(only, name)
    ]
    candidate_names = list(candidate["configs"])
    is_match = True
    for name in sorted(set(baseline_names) ^ set(candidate_names)):
        side = "baseline" if name in baseline["configs"] else "candidate"
        print(f"MISSING {name}: only in the {side}")
        is_match = False
    for name in baseline_names:
        if name not in candidate["configs"]:
            continue
        baseline_config = baseline["configs"][name]
        candidate_config = candidate["configs"][name]
        baseline_text = (baseline_dir / f"{name}.mlir").read_text("utf-8")
        candidate_text = (candidate_dir / f"{name}.mlir").read_text("utf-8")
        if baseline_text != candidate_text:
            is_match = False
            changed_calls = describe_changed_calls(
                baseline_text, candidate_text, baseline_config["calls"]
            )
            print(f"IR DIFFERS {name}: {changed_calls}")
            for line in diff_head(
                baseline_text, candidate_text, f"{name}.mlir", max_diff_lines
            ):
                print(f"    {line}")
        wrapper_file = f"{name}.wrapper.mlir"
        baseline_wrapper = (baseline_dir / wrapper_file).read_text("utf-8")
        candidate_wrapper = (candidate_dir / wrapper_file).read_text("utf-8")
        if baseline_wrapper != candidate_wrapper:
            is_match = False
            if is_line_permutation(baseline_wrapper, candidate_wrapper):
                note = ": same lines reordered or rewired"
            else:
                note = ""
            print(f"WRAPPER DIFFERS {name}{note}")
            for line in diff_head(
                baseline_wrapper,
                candidate_wrapper,
                wrapper_file,
                max_diff_lines,
            ):
                print(f"    {line}")
        if baseline_config["calls"] != candidate_config["calls"]:
            is_match = False
            print(
                f"LABELS DIFFER {name}:\n  baseline "
                f" {baseline_config['calls']}\n  candidate"
                f" {candidate_config['calls']}"
            )
        if baseline_config["tuned"] != candidate_config["tuned"]:
            is_match = False
            keys = sorted(
                set(baseline_config["tuned"]) | set(candidate_config["tuned"])
            )
            print(f"TUNED LOOKUPS DIFFER {name}:")
            for key in keys:
                baseline_outcome = baseline_config["tuned"].get(key, "-")
                candidate_outcome = candidate_config["tuned"].get(key, "-")
                if baseline_outcome != candidate_outcome:
                    print(f"  {key}: {baseline_outcome} -> {candidate_outcome}")
    num_compared = sum(name in candidate["configs"] for name in baseline_names)
    verdict = "PASS" if is_match else "FAIL"
    print(f"{verdict}: compared {num_compared} configurations")
    return is_match


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    dump_parser = subparsers.add_parser(
        "dump",
        help=(
            "lower every configuration and write its snapshot; flywheel_tpu/"
            " must have no uncommitted changes"
        ),
    )
    compare_parser = subparsers.add_parser(
        "compare", help="dump into --out, then diff it against --baseline"
    )
    for subparser in (dump_parser, compare_parser):
        subparser.add_argument(
            "--out",
            type=pathlib.Path,
            required=True,
            help="snapshot directory to write; must be absent or empty",
        )
        subparser.add_argument(
            "--only",
            help="regex: only the configurations whose name it matches",
        )
    compare_parser.add_argument("--baseline", type=pathlib.Path, required=True)
    compare_parser.add_argument(
        "--diff-lines",
        type=int,
        default=40,
        help="unified diff lines printed per differing file",
    )
    args = parser.parse_args()
    if args.command == "dump":
        dump(args.out, args.only, require_clean=True)
    else:
        # Note: read the baseline before lowering, so a bad path fails fast.
        load_index(args.baseline)
        dump(args.out, args.only, require_clean=False)
        if not compare_snapshots(
            args.baseline, args.out, args.only, args.diff_lines
        ):
            sys.exit(1)


if __name__ == "__main__":
    main()
