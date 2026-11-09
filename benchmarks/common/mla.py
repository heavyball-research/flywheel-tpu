"""SHA-pinned loader for tpu-inference's MLA v2 kernel, the baseline of
decode_microbench.py's MLA cases, and the MLA prefill paths the serving
scenario benchmarks share.

--mla-v2-source defaults to the third_party/tpu-inference submodule
(git submodule update --init third_party/tpu-inference), with no patch
applied. Only MLA_V2_SOURCE_SHA256's files are loaded, each verified against
its SHA-256 at MLA_V2_GIT_SHA, the submodule pin: MLA v2's kernel.py,
kv_utils.py, transpose.py and tuned_params.py (the block sizes tpu-inference
serves it with), and RPA v3's util.py, whose get_dtype_packing transpose.py
imports. Their imports outside jax are stubbed, so neither tpu_inference nor
its vLLM dependency is installed: tpu_inference.envs (MLA_XPOSE_N_TILE_SIZE,
read from the environment with envs.py's default), tpu_inference.logger (a
stdlib logger) and sympy (divisors, transpose.py's only use of it).
"""

# Note (david): kvcache_scenarios.py imports this module on its host-only
# paths too, so jax and flywheel_tpu load inside the prefill helpers.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import logging
import math
import os
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.common.rpa import DEFAULT_RPA_SOURCE, RPA_V3_GIT_SHA

if TYPE_CHECKING:
    import jax

# Note (david): RPA v3 and MLA v2 come from the same submodule pin.
DEFAULT_MLA_V2_SOURCE = DEFAULT_RPA_SOURCE
MLA_V2_GIT_SHA = RPA_V3_GIT_SHA
# Paths relative to the checkout root.
MLA_V2_SOURCE_SHA256 = {
    "tpu_inference/kernels/mla/v2/kernel.py": (
        "13d0e2806241956e68e3435c9463722c466441a92e29baee20e6fd31dc5cc5d1"
    ),
    "tpu_inference/kernels/mla/v2/kv_utils.py": (
        "5e606cd47f14124c2e637c7efcc09f0a098b8c4fbe612d855350f6a609b3556f"
    ),
    "tpu_inference/kernels/mla/v2/transpose.py": (
        "1826b3b0c802951ea28e7d7da285f310be793e875d085b5aa58d2bde1bb8132a"
    ),
    "tpu_inference/kernels/mla/v2/tuned_params.py": (
        "ba5254c7b0d741b95ec27903083eb1ce19bf4b7dc64f20a4742184502a1e7943"
    ),
    "tpu_inference/kernels/ragged_paged_attention/v3/util.py": (
        "f9a1c1f189ef0b5d9e9babaa9d221faf01f12bc9bb809fe227d17901867b56c0"
    ),
}
# The packages the loaded files import from; stubbed, so their __init__.py
# never runs.
MLA_V2_PACKAGES = (
    "tpu_inference.kernels.mla.v2",
    "tpu_inference.kernels.ragged_paged_attention.v3",
)
# tpu_inference/envs.py's default.
XPOSE_N_TILE_SIZE_DEFAULT = "160"


def stub_module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def divisors(number: int) -> list[int]:
    """sympy.divisors for a positive int: its divisors in ascending order."""
    small = [d for d in range(1, math.isqrt(number) + 1) if number % d == 0]
    return sorted({*small, *(number // d for d in small)})


def load_mla_v2(source_root: Path):
    """The pinned MLA v2 kernel module (mla_ragged_paged_attention,
    get_kv_cache_shape) and tuned_params module (TuningKey,
    get_tuned_params), imported under stub parent packages and stub envs,
    logger and sympy modules."""
    source_root = Path(source_root).expanduser().resolve()
    for name, expected in MLA_V2_SOURCE_SHA256.items():
        path = source_root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing; --mla-v2-source must be a tpu-inference"
                f" checkout at {MLA_V2_GIT_SHA} (for the default, run git"
                " submodule update --init third_party/tpu-inference)."
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                f"{path} does not match tpu-inference {MLA_V2_GIT_SHA}; got"
                f" sha256 {digest}."
            )

    modules = {
        "tpu_inference.envs": stub_module(
            "tpu_inference.envs",
            MLA_XPOSE_N_TILE_SIZE=int(
                os.environ.get(
                    "MLA_XPOSE_N_TILE_SIZE", XPOSE_N_TILE_SIZE_DEFAULT
                )
            ),
        ),
        "tpu_inference.logger": stub_module(
            "tpu_inference.logger", init_logger=logging.getLogger
        ),
    }
    for package in MLA_V2_PACKAGES:
        parts = package.split(".")
        for depth in range(1, len(parts) + 1):
            name = ".".join(parts[:depth])
            module = types.ModuleType(name)
            module.__path__ = [str(source_root.joinpath(*parts[:depth]))]
            module.__package__ = name
            modules[name] = module
    for name, module in modules.items():
        sys.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)

    # Note (david): only the import binds sympy.divisors, so the stub is
    # installed for it alone and whatever sympy the process had comes back.
    previous = sys.modules.get("sympy")
    sys.modules["sympy"] = stub_module("sympy", divisors=divisors)
    try:
        kernel = importlib.import_module("tpu_inference.kernels.mla.v2.kernel")
    finally:
        if previous is None:
            del sys.modules["sympy"]
        else:
            sys.modules["sympy"] = previous
    tuned_params = importlib.import_module(
        "tpu_inference.kernels.mla.v2.tuned_params"
    )
    return kernel, tuned_params


# Note (david): DeepSeek V3's MLA dims.
LATENT_DIM = 512
ROPE_DIM = 64
NOPE_DIM = 128
SM_SCALE = (NOPE_DIM + ROPE_DIM) ** -0.5
MLA_PREFILL_IMPLS = ("flywheel_mla", "mla_v2")
# Note (david): MLA v2's serving pages, so v2 runs as it is served.
SERVING_PAGE = {"bfloat16": 512, "float8_e4m3fn": 1024}
# Note (david): maps a few sigma onto e4m3's largest value and is no power of
# two, so a scale applied where it rounds shows in the cos_diff.
KV_SCALE = {"bfloat16": None, "float8_e4m3fn": 6 / 448}
# Note (david): the first and last rows of a chunk sit at both ends of its
# causal mask.
CHECK_ROWS = 8
# Note (david): catches a dropped page and stays far above rounding, fp8
# included.
COS_DIFF_BOUND = 1e-3


@dataclasses.dataclass(frozen=True)
class MlaPrefill:
    """One MLA prefill path at a step's static shape: queries, tokens and meta
    take the step's token-major operands to the path's layout, attend(cache,
    *queries, *tokens, *meta) -> (out, cache) writes the step's tokens and
    attends, and token_major takes out back to (T, H, LATENT_DIM)."""

    queries: Callable[..., tuple[jax.Array, ...]]
    tokens: Callable[..., tuple[jax.Array, ...]]
    meta: Callable[..., tuple[jax.Array, ...]]
    attend: Callable[..., tuple[jax.Array, jax.Array]]
    token_major: Callable[[jax.Array], jax.Array]
    info: dict[str, object]


def build_mla_prefill(
    impl: str,
    *,
    heads: int,
    kv_dtype: str,
    cache_shape: tuple[int, ...],
    num_tokens: int,
    batch: int,
    pages_per_seq: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    mla_v2_source: Path,
) -> MlaPrefill:
    """impl's path for steps of num_tokens tokens of batch requests on a cache
    of cache_shape; max_seqlen_q and max_seqlen_k bound flywheel_mla's q_lens
    and kv_lens."""
    import jax.numpy as jnp
    import numpy as np

    kv_scale = KV_SCALE[kv_dtype]
    if impl == "flywheel_mla":
        from flywheel_tpu.softmax_attention.mla_cache_write import (
            mla_cache_write,
        )
        from flywheel_tpu.softmax_attention.mla_prefill import (
            flash_mla_prefill,
        )

        if kv_scale is None:
            traced_kv_scale = None
        else:
            traced_kv_scale = jnp.asarray(kv_scale, jnp.float32)

        def _attend(
            cache, q_latent, q_pe, new_kv_c, new_k_pe, cu_q_lens, kv_lens, table
        ):
            cache = mla_cache_write(
                new_kv_c, new_k_pe, cache, cu_q_lens, kv_lens, table
            )
            out = flash_mla_prefill(
                q_latent,
                q_pe,
                cache,
                cu_q_lens,
                kv_lens,
                table,
                sm_scale=SM_SCALE,
                kv_scale=traced_kv_scale,
                causal=True,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )
            return out, cache

        return MlaPrefill(
            queries=lambda q_latent, q_pe: (q_latent, q_pe),
            tokens=lambda new_kv_c, new_k_pe: (new_kv_c, new_k_pe),
            meta=lambda *meta: tuple(
                jnp.asarray(np.asarray(values, np.int32)) for values in meta
            ),
            attend=_attend,
            token_major=lambda out: out,
            info={},
        )
    elif impl == "mla_v2":
        from flywheel_tpu.softmax_attention.config import next_pow2

        v2_kernel, v2_tuned = load_mla_v2(mla_v2_source)
        # Note (david): the model runner pads the step's tokens and requests to
        # the next power of two, and mla_attention keys the tuned params on the
        # padded q and metadata.
        padded_tokens = next_pow2(num_tokens)
        max_num_seqs = next_pow2(batch)
        _, words, packing, _ = cache_shape
        fields = {
            "max_num_tokens": padded_tokens,
            "actual_num_q_heads": heads,
            "actual_lkv_dim": LATENT_DIM,
            "actual_r_dim": ROPE_DIM,
            "kv_dtype": jnp.dtype(kv_dtype).name,
            "q_dtype": jnp.dtype(jnp.bfloat16).name,
            "page_size_per_kv_packing": words,
            "kv_packing": packing,
            "max_num_seqs": max_num_seqs,
            "pages_per_seq": pages_per_seq,
        }
        decode = v2_tuned.get_tuned_params(
            v2_tuned.TuningKey(case="batched_decode", **fields)
        )
        mixed = v2_tuned.get_tuned_params(
            v2_tuned.TuningKey(case="mixed", **fields)
        )
        # Note (david): mla_attention runs prefill on the mixed params.
        params = {
            "num_kv_pages_per_block": (
                decode.num_kv_pages_per_block,
                mixed.num_kv_pages_per_block,
                mixed.num_kv_pages_per_block,
            ),
            "num_queries_per_block": (
                decode.num_queries_per_block,
                mixed.num_queries_per_block,
                mixed.num_queries_per_block,
            ),
            "decode_batch_size": decode.decode_batch_size,
            "mixed_q_split": mixed.q_split,
        }
        if kv_scale is None:
            scales = {}
        else:
            scales = {"k_scale": kv_scale, "v_scale": kv_scale}
        padding = ((0, padded_tokens - num_tokens), (0, 0))

        def _meta(cu_q_lens, kv_lens, block_table):
            padded_kv_lens = np.zeros(max_num_seqs, np.int32)
            padded_kv_lens[:batch] = kv_lens
            padded_cu_q_lens = np.full(max_num_seqs + 1, num_tokens, np.int32)
            padded_cu_q_lens[: batch + 1] = cu_q_lens
            page_indices = np.zeros((max_num_seqs, pages_per_seq), np.int32)
            page_indices[:batch] = block_table
            return (
                jnp.asarray(padded_kv_lens),
                jnp.asarray(page_indices.reshape(-1)),
                jnp.asarray(padded_cu_q_lens),
                jnp.asarray([0, 0, batch], jnp.int32),
            )

        def _attend(cache, q_latent, q_pe, new_kv_c, new_k_pe, *meta):
            # Note (david): the caller's jit fixes the static arguments.
            return v2_kernel.mla_ragged_paged_attention.__wrapped__(
                q_latent,
                q_pe,
                new_kv_c,
                new_k_pe,
                cache,
                *meta,
                sm_scale=SM_SCALE,
                **params,
                **scales,
            )

        # Note (david): v2's absorb einsum emits head-major q_latent, so the
        # transpose is not v2's time.
        return MlaPrefill(
            queries=lambda q_latent, q_pe: (
                jnp.pad(jnp.transpose(q_latent, (1, 0, 2)), ((0, 0), *padding)),
                jnp.pad(q_pe, (*padding, (0, 0))),
            ),
            tokens=lambda new_kv_c, new_k_pe: (
                jnp.pad(new_kv_c, padding),
                jnp.pad(new_k_pe, padding),
            ),
            meta=_meta,
            attend=_attend,
            token_major=lambda out: jnp.transpose(
                out[:, :num_tokens], (1, 0, 2)
            ),
            info={"v2_params": params},
        )
    else:
        raise ValueError(f"unsupported MLA prefill impl: {impl}")


def draw_normal(key: jax.Array, shape: tuple[int, ...], dtype) -> jax.Array:
    """Unit-normal values drawn on the device, in dtype."""
    import jax
    import jax.numpy as jnp

    return jax.random.normal(key, shape, jnp.bfloat16).astype(dtype)


def mla_cos_diff(
    out: jax.Array,
    q_latent: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    kv_dtype: str,
) -> float:
    """FlashMLA's cosine diff of a request's chunk output out (n, H,
    LATENT_DIM) against an f32 reference on its first and last CHECK_ROWS
    rows; new_kv_c and new_k_pe are the request's keys, the chunk's last."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    highest = jax.lax.Precision.HIGHEST
    num_rows = out.shape[0]
    num_keys = new_kv_c.shape[0]
    rows = np.unique(
        np.concatenate([
            np.arange(min(CHECK_ROWS, num_rows)),
            np.arange(max(0, num_rows - CHECK_ROWS), num_rows),
        ])
    )
    if KV_SCALE[kv_dtype] is None:
        scale = 1.0
    else:
        scale = KV_SCALE[kv_dtype]
    latent = new_kv_c.astype(jnp.float32) * scale
    rope = new_k_pe.astype(jnp.float32) * scale
    scores = (
        jnp.einsum(
            "nhl,kl->nhk",
            q_latent[rows].astype(jnp.float32),
            latent,
            precision=highest,
        )
        + jnp.einsum(
            "nhr,kr->nhk",
            q_pe[rows].astype(jnp.float32),
            rope,
            precision=highest,
        )
    ) * SM_SCALE
    visible = (
        np.arange(num_keys)[None, :] <= (num_keys - num_rows + rows)[:, None]
    )
    scores = jnp.where(jnp.asarray(visible)[:, None, :], scores, -jnp.inf)
    want = np.asarray(
        jnp.einsum(
            "nhk,kl->nhl",
            jax.nn.softmax(scores, axis=-1),
            latent,
            precision=highest,
        ),
        np.float64,
    )
    got = np.asarray(out[rows].astype(jnp.float32), np.float64)
    return 1.0 - 2.0 * float((got * want).sum()) / float(
        (got * got + want * want).sum()
    )
