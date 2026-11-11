"""DeepSeek sparse attention (DSA) layer steps for the serving scenario
benchmarks, and the SHA-pinned loader for vllm-torchtpu's DSA kernels.

A step is one DSA layer's forward: the index-K cache write, the latent cache
write, the indexer's scores, their top-k and the sparse attention.

  flywheel_dsa               dsa_index_cache_write, dsa_cache_write, then
                             dsa_attention
  torchtpu_dsa               vllm-torchtpu's sparse attention indexer op,
                             then its sparse MLA op
  torchtpu_dsa_masked_dense  the same with masked-dense routing admitted, as
                             TPU_MLA_MASKED_DENSE_ENABLED=1 does for glm52

--torchtpu-source is a vllm-torchtpu checkout at TORCHTPU_DSA_GIT_SHA. Its
kernel files are imported under stub parent packages, each verified against
its SHA-256, so neither torch nor vLLM is installed. The torchtpu_* functions
mirror the serving code around the kernels, whose files are verified only.
"""

# Note (david): the scenario scripts import this module on their host-only
# paths too, so jax and flywheel_tpu load inside the functions.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import importlib
import inspect
import logging
import math
import os
import sys
import textwrap
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from benchmarks.common.mla import (
    CHECK_ROWS,
    COS_DIFF_BOUND,
    KV_SCALE,
    LATENT_DIM,
    ROPE_DIM,
    draw_normal,
    stub_module,
)
from benchmarks.common.timing import time_call

if TYPE_CHECKING:
    import jax


@dataclasses.dataclass(frozen=True)
class DsaGeometry:
    """A DSA model's attention shape. qk_head_dim is the query head's nope +
    rope dims before absorption; it sets the softmax scale."""

    heads: int
    index_heads: int
    index_head_dim: int
    top_k: int
    qk_head_dim: int

    @property
    def sm_scale(self) -> float:
        return self.qk_head_dim**-0.5


DSA_GEOMETRIES = {
    # DeepSeek: 128 heads of 128 nope + 64 rope dims.
    "v32": DsaGeometry(
        heads=128,
        index_heads=64,
        index_head_dim=128,
        top_k=2048,
        qk_head_dim=192,
    ),
    # GLM: 64 heads of 192 + 64 dims.
    "glm52": DsaGeometry(
        heads=64,
        index_heads=32,
        index_head_dim=128,
        top_k=2048,
        qk_head_dim=256,
    ),
}
FLYWHEEL_DSA = "flywheel_dsa"
TORCHTPU_DSA = "torchtpu_dsa"
TORCHTPU_DSA_MASKED_DENSE = "torchtpu_dsa_masked_dense"
TORCHTPU_DSA_IMPLS = (TORCHTPU_DSA, TORCHTPU_DSA_MASKED_DENSE)
DSA_IMPLS = (FLYWHEEL_DSA, *TORCHTPU_DSA_IMPLS)
# Note (david): kernels/mla/dispatch.py admits masked-dense routing for this
# profile only.
MASKED_DENSE_GEOMETRY = "glm52"
# Note (david): GLM's serving page, on fp8 caches.
DSA_PAGE = 1024
DSA_KV_DTYPE = "float8_e4m3fn"
DSA_KV_SCALE = KV_SCALE[DSA_KV_DTYPE]
# A token's cache bytes: fp8 nope and rope, and the index key's 128 fp8 bytes,
# its scale byte and 3 bytes of padding.
DSA_CACHE_TOKEN_BYTES = LATENT_DIM + ROPE_DIM + 132
# Note (david): pages of random latents a decode cache is tiled from, and
# the pages of random index keys a decode cache's fill draws at once.
DSA_BASE_PAGES = 8
FILL_PAGES_PER_BLOCK = 256
# Note (david): a selection off by its share of score mass beyond rounding.
INDEX_DEFICIT_BOUND = 1e-3

TORCHTPU_DSA_GIT_SHA = "b08f10fdc6166ff3aa3306629fcaf5b133012cb2"
# Paths relative to src/vllm_torchtpu of the checkout.
TORCHTPU_DSA_SOURCE_SHA256 = {
    "envs.py": (
        "01403153b9598b684f602bcd3533f093a97624ffcaffb43246df57a5fac9650e"
    ),
    "kernels/deepseek_v4/sparsecore_topk.py": (
        "5329def90dda7aa4223794c2fd0f669fe69a24350bdd3a282d2281c2c8d63390"
    ),
    "kernels/deepseek_v4/streamindex_topk/bref_override.py": (
        "bd710ad19a597e96a63aee44ec05bf8a6b318f55b672a026e90470610b315aec"
    ),
    "kernels/deepseek_v4/streamindex_topk/config.py": (
        "9a7435d65169d30ebf949a2d4211996cbf88b403a21fb1c993b46436ddc16972"
    ),
    "kernels/deepseek_v4/streamindex_topk/dcp_sc_compact.py": (
        "35169d67781887c937dc296d99c1235d307f648e391a13fce988c4fb28ec24d6"
    ),
    "kernels/deepseek_v4/streamindex_topk/kv_cache_append.py": (
        "8d471f366f65fea401180fb0c8403b93969e15257de0ac0c216cd98ec5458c30"
    ),
    "kernels/deepseek_v4/streamindex_topk/kv_cache_utils.py": (
        "192064b02ad998c3aef609f209b187b46f947b1b1681c85bd3a502e2783a735c"
    ),
    "kernels/deepseek_v4/streamindex_topk/metadata.py": (
        "c3985c09a323ba60b8559618be67bd23fc64f987f15e7dea7ccbd5571b2fdd2f"
    ),
    "kernels/deepseek_v4/streamindex_topk/streamindex_topk.py": (
        "0afd729c97d71db82f11e72853d575bcad85bceebc1e4aee53501c3ea75960ed"
    ),
    "kernels/mla/kv_cache_utils.py": (
        "03fa5fc23509a5c3735f246570087ff671b55b9f5e91b92ee5e899a5e7ef8b8a"
    ),
    "kernels/mla/scatter.py": (
        "ce8caf5f73a07f511210e77427ce5d84b6ee0033d2e98383d6ca6c49242a4d21"
    ),
    "kernels/mla/scatter_utils.py": (
        "ab23976f97f0a5307e786dd5aa6a001efc6d4bb0307af5f6de690a26bb175176"
    ),
    "kernels/mla/sparse/kernel.py": (
        "93683bab145a5b7e693cfb4e9cab8841cd00a772939579913730529ae2e4a2f8"
    ),
    "kernels/mla/sparse/dsa_gather.py": (
        "20d38f91f69a4e2576d2552f7ae98e627ef6c4740f99c61c7b4137b5862e6773"
    ),
    "kernels/mla/sparse/native_sc_gather.py": (
        "32994e3e43048f861d2e1a3d1a6972d177e7d415731ec4070425605db03a76e3"
    ),
    "kernels/mla/dispatch.py": (
        "0c4c16c3e2997a04ca0445a7016b4b3c0b4281d864735d1c5b0e8275a2661a84"
    ),
    "kernels/mla/masked_dense/cost_model.py": (
        "f1fe5f21b6c61f9b1d4d02ae31528962ad13cbc10836f3178f5dd4f56a5e1c3d"
    ),
    "kernels/mla/masked_dense/kernel.py": (
        "47a6e733083ff55eab65d7dfab47652a2d2240f60a345e84832c127485d00a34"
    ),
    "kernels/mla/masked_dense/csa_mask.py": (
        "d7674f854b6b07fa5d3e94593f3ec59cad7fa29206390ea24ec8b992d4f05ef1"
    ),
    "kernels/sparse_core/core_map_helper.py": (
        "6fea75cb4d7c4f226c9f16e847e3bdfc8e2bc3a29e00dbdfe3205c3d28a3ceeb"
    ),
}
TORCHTPU_SERVING_SHA256 = {
    "layers/adapter/custom_ops/mla_attention_op.py": (
        "3cffef770f2fe5907e8b4b98833677465af52636a487ca4ffbb571bd1f60e825"
    ),
    "layers/core/attention_interface.py": (
        "1c0f690435986dc54f19b26bb3efecf32f5f27868c926817d55eb8f923802e70"
    ),
    "layers/core/quantization.py": (
        "9c5da957013c698ff675caa68925d65d93b796b32b3aa623c7415279e87e49fc"
    ),
}
# The packages the imported files live in; stubbed, so no __init__.py runs.
TORCHTPU_PACKAGES = (
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk",
    "vllm_torchtpu.kernels.mla.sparse",
    "vllm_torchtpu.kernels.mla.masked_dense",
    "vllm_torchtpu.kernels.sparse_core",
)
TORCH_MODULES = (
    "torch",
    "torch_tpu",
    "torch_tpu._internal",
    "torch_tpu._internal.pallas",
)
# mla_attention_op.py's StreamIdx settings.
STREAMIDX_ENABLE_EARLY_EXIT = True
STREAMIDX_KV_PAGES_PER_BLOCK = (2, 2, 2)
STREAMIDX_QUERIES_PER_BLOCK = (1, 128, 128)
STREAMIDX_SMALL_BUCKET_DEFAULT_BKVP = 8
# The indexer's keys are uncompressed.
INDEX_COMPRESSION_RATIO = 1
# The index cache modes whose Pallas writer this module mirrors.
PALLAS_INDEX_CACHE_MODES = ("has", "sal")
# The runner's smallest token bucket and request count.
MIN_TOKEN_BUCKET = 16
MIN_NUM_SEQS = 8
# fp8 values a 32-bit cache word packs.
KV_PACKING = 4
FP8_MAX = 448.0
# quantize_tensor's UE8M0 exponent range; the stored byte is exponent + 127.
E8M0_MIN_EXP = -127
E8M0_MAX_EXP = 127
# act_quant floors a key's max abs here before deriving its scale.
AMAX_FLOOR = 1e-4
F32_MANTISSA_BITS = 23
F32_MANTISSA_MASK = (1 << F32_MANTISSA_BITS) - 1
F32_EXPONENT_MASK = 0xFF
DYNAMIC_SMEM_FLAG_NAME = "--xla_tpu_use_dynamic_smem_negotiation"
DYNAMIC_SMEM_FLAG = f"{DYNAMIC_SMEM_FLAG_NAME}=true"
FORI_REMAINDER_NEEDLE = "elif has_dynamic_remainder:"
FORI_REMAINDER_FIX = "elif has_dynamic_remainder and unroll != 1:"


class DsaQueries(NamedTuple):
    """A call's per-token queries, token-major: the indexer's fp8 queries
    (T, index_heads, index_head_dim) and f32 weights (T, index_heads), and
    the absorbed bf16 q_nope (T, H, LATENT_DIM) and q_pe (T, H, ROPE_DIM)."""

    index_queries: jax.Array
    index_weights: jax.Array
    q_nope: jax.Array
    q_pe: jax.Array


class DsaKeys(NamedTuple):
    """Per-token cache entries, token-major: the bf16 index key (T,
    index_head_dim), the fp8 latent (T, LATENT_DIM) and rope (T, ROPE_DIM)."""

    index_keys: jax.Array
    latent: jax.Array
    rope: jax.Array

    def rows(self, rows) -> DsaKeys:
        return DsaKeys(*(values[rows] for values in self))


class DsaRequests(NamedTuple):
    """A call's ragged batch on the host: request r's tokens are rows
    [cu_q_lens[r], cu_q_lens[r + 1]) and its pages block_table[r]; the first
    num_decode requests decode one token each."""

    cu_q_lens: np.ndarray
    kv_lens: np.ndarray
    block_table: np.ndarray
    num_decode: int

    @property
    def num_tokens(self) -> int:
        return int(self.cu_q_lens[-1])

    def token_requests(self) -> np.ndarray:
        return np.repeat(
            np.arange(len(self.kv_lens)), np.diff(self.cu_q_lens)
        ).astype(np.int32)

    def token_positions(self) -> np.ndarray:
        requests = self.token_requests()
        tokens = np.arange(self.num_tokens)
        return (
            self.kv_lens[requests] - self.cu_q_lens[requests + 1] + tokens
        ).astype(np.int32)


def draw_dsa_queries(
    key: jax.Array, geometry: DsaGeometry, num_tokens: int
) -> DsaQueries:
    import jax
    import jax.numpy as jnp

    keys = jax.random.split(key, 4)
    index_shape = (num_tokens, geometry.index_heads)
    # Note (david): the index weights are scaled by (index_heads *
    # index_head_dim)^-0.5.
    weight_scale = (geometry.index_heads * geometry.index_head_dim) ** -0.5
    return DsaQueries(
        index_queries=draw_normal(
            keys[0],
            (*index_shape, geometry.index_head_dim),
            jnp.float8_e4m3fn,
        ),
        index_weights=draw_normal(keys[1], index_shape, jnp.float32)
        * weight_scale,
        q_nope=draw_normal(
            keys[2], (num_tokens, geometry.heads, LATENT_DIM), jnp.bfloat16
        ),
        q_pe=draw_normal(
            keys[3], (num_tokens, geometry.heads, ROPE_DIM), jnp.bfloat16
        ),
    )


def draw_dsa_keys(
    key: jax.Array, geometry: DsaGeometry, num_tokens: int
) -> DsaKeys:
    import jax
    import jax.numpy as jnp

    keys = jax.random.split(key, 3)
    return DsaKeys(
        index_keys=draw_normal(
            keys[0], (num_tokens, geometry.index_head_dim), jnp.bfloat16
        ),
        latent=draw_normal(
            keys[1], (num_tokens, LATENT_DIM), jnp.float8_e4m3fn
        ),
        rope=draw_normal(keys[2], (num_tokens, ROPE_DIM), jnp.float8_e4m3fn),
    )


@dataclasses.dataclass(frozen=True)
class DsaLayer:
    """One impl's DSA layer at one static step shape, on its own caches.

    queries, keys and meta map a call's token-major operands and requests to
    the impl's layout. step(caches, queries, keys, *meta) -> (outputs, caches)
    is the jitted layer step with its caches donated. selection(...) gives
    request's selected (q_len, top_k) positions, -1 padded."""

    empty_caches: Callable[[int], tuple[jax.Array, ...]]
    write: Callable[..., tuple[jax.Array, ...]]
    queries: Callable[[DsaQueries], DsaQueries]
    keys: Callable[[DsaKeys], DsaKeys]
    meta: Callable[[DsaRequests], tuple[jax.Array, ...]]
    step: Callable[..., tuple[object, tuple[jax.Array, ...]]]
    selection: Callable[..., jax.Array]
    attention_out: Callable[[object], jax.Array]
    info: dict[str, object]


def dsa_args_error(
    impl: str, dsa_geometry: str | None, page_size: int | None
) -> str | None:
    """Why a scenario script's DSA arguments do not fit together, or None."""
    if impl not in DSA_IMPLS:
        if dsa_geometry is not None:
            return f"--impl {impl} takes no --dsa-geometry"
        return None
    elif dsa_geometry is None:
        return f"--impl {impl} needs --dsa-geometry"
    elif page_size not in (None, DSA_PAGE):
        return f"--impl {impl} runs {DSA_PAGE}-token pages only"
    elif (
        impl == TORCHTPU_DSA_MASKED_DENSE
        and dsa_geometry != MASKED_DENSE_GEOMETRY
    ):
        return (
            "masked-dense routing admits --dsa-geometry"
            f" {MASKED_DENSE_GEOMETRY} only"
        )
    return None


def flywheel_meta(requests: DsaRequests) -> tuple[jax.Array, ...]:
    import jax.numpy as jnp

    return tuple(
        jnp.asarray(np.asarray(values, np.int32))
        for values in (
            requests.cu_q_lens,
            requests.kv_lens,
            requests.block_table,
        )
    )


def flywheel_empty_caches(num_pages: int) -> tuple[jax.Array, ...]:
    import jax.numpy as jnp

    from flywheel_tpu.sparse_attention.dsa.cache_write import dsa_cache_shapes
    from flywheel_tpu.sparse_attention.dsa.indexer import index_cache_shape

    nope_shape, rope_shape = dsa_cache_shapes(num_pages, DSA_PAGE)
    return (
        jnp.zeros(index_cache_shape(num_pages, DSA_PAGE), jnp.uint8),
        jnp.zeros(nope_shape, jnp.int32),
        jnp.zeros(rope_shape, jnp.int32),
    )


def flywheel_write(caches, keys, cu_q_lens, kv_lens, block_table):
    from flywheel_tpu.sparse_attention.dsa.cache_write import dsa_cache_write
    from flywheel_tpu.sparse_attention.dsa.index_cache_write import (
        dsa_index_cache_write,
    )

    index_cache, nope_cache, rope_cache = caches
    index_cache = dsa_index_cache_write(
        keys.index_keys, index_cache, cu_q_lens, kv_lens, block_table
    )
    nope_cache, rope_cache = dsa_cache_write(
        keys.latent,
        keys.rope,
        nope_cache,
        rope_cache,
        cu_q_lens,
        kv_lens,
        block_table,
    )
    return index_cache, nope_cache, rope_cache


def flywheel_topk(geometry: DsaGeometry, max_seqlen_k: int) -> Callable:
    """The jitted top_k of the indexer's scores on a written index cache."""
    import jax

    from flywheel_tpu.sparse_attention.dsa.cache_write import token_positions
    from flywheel_tpu.sparse_attention.dsa.indexer import dsa_indexer_scores
    from flywheel_tpu.topk import top_k

    @jax.jit
    def topk(index_cache, queries, cu_q_lens, kv_lens, block_table):
        scores = dsa_indexer_scores(
            queries.index_queries,
            queries.index_weights,
            index_cache,
            cu_q_lens,
            kv_lens,
            block_table,
            max_seqlen_k=max_seqlen_k,
        )
        _, position = token_positions(cu_q_lens, kv_lens, scores.shape[0])
        return top_k(scores, geometry.top_k, row_lengths=position + 1)

    return topk


def flywheel_selection(topk, outputs, caches, queries, requests, request):
    """dsa_attention's own selection of request's tokens, by topk. A decode
    call is scored whole, since its tokens share the indexer's work items; a
    prefill request alone."""
    del outputs  # Unused.
    first, last = requests.cu_q_lens[request : request + 2]
    if requests.num_decode == len(requests.kv_lens):
        selection = topk(caches[0], queries, *flywheel_meta(requests))
        return selection[first:last]
    alone = DsaRequests(
        cu_q_lens=np.array([0, last - first]),
        kv_lens=requests.kv_lens[request : request + 1],
        block_table=requests.block_table[request : request + 1],
        num_decode=0,
    )
    rows = DsaQueries(*(values[first:last] for values in queries))
    return topk(caches[0], rows, *flywheel_meta(alone))


def build_flywheel_layer(
    geometry: DsaGeometry, max_seqlen_k: int, route: str | None
) -> DsaLayer:
    """flywheel's layer step: its two cache writes, then dsa_attention
    through route, or the route it picks on device if None."""
    import jax
    import jax.numpy as jnp

    from flywheel_tpu.sparse_attention.dsa.attention import dsa_attention

    kv_scale = jnp.asarray(DSA_KV_SCALE, jnp.float32)

    def _step(caches, queries, keys, cu_q_lens, kv_lens, block_table, scale):
        caches = flywheel_write(caches, keys, cu_q_lens, kv_lens, block_table)
        out = dsa_attention(
            queries.index_queries,
            queries.index_weights,
            caches[0],
            queries.q_nope,
            queries.q_pe,
            caches[1],
            caches[2],
            cu_q_lens,
            kv_lens,
            block_table,
            max_seqlen_k=max_seqlen_k,
            sm_scale=geometry.sm_scale,
            kv_scale=scale,
            top_k=geometry.top_k,
            route=route,
        )
        return out, caches

    write = jax.jit(flywheel_write, donate_argnums=(0,))
    return DsaLayer(
        empty_caches=flywheel_empty_caches,
        write=lambda caches, keys, requests: write(
            caches, keys, *flywheel_meta(requests)
        ),
        queries=lambda queries: queries,
        keys=lambda keys: keys,
        meta=lambda requests: (*flywheel_meta(requests), kv_scale),
        step=jax.jit(_step, donate_argnums=(0,)),
        selection=functools.partial(
            flywheel_selection, flywheel_topk(geometry, max_seqlen_k)
        ),
        attention_out=lambda out: out,
        info={"max_seqlen_k": max_seqlen_k, "route": route},
    )


def verify_sources(root: Path, expected: dict[str, str]) -> None:
    for name, digest in expected.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing; --torchtpu-source must be a vllm-torchtpu"
                f" checkout at {TORCHTPU_DSA_GIT_SHA}."
            )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(
                f"{path} does not match vllm-torchtpu {TORCHTPU_DSA_GIT_SHA};"
                f" got sha256 {actual}."
            )


@contextlib.contextmanager
def torch_stubs() -> Iterator[None]:
    """Stub torch and torch_tpu._internal.pallas for one import, and put
    back whatever the process had."""
    previous = {name: sys.modules.get(name) for name in TORCH_MODULES}
    stubs = {
        "torch": stub_module("torch", dtype=object),
        "torch_tpu": stub_module("torch_tpu"),
        "torch_tpu._internal": stub_module("torch_tpu._internal"),
        "torch_tpu._internal.pallas": stub_module(
            "torch_tpu._internal.pallas", pallas=stub_module("pallas")
        ),
    }
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = module


def install_torchtpu_packages(root: Path) -> None:
    """Stub parent packages over root and a stdlib vllm_torchtpu.logger."""
    modules = {}
    for package in TORCHTPU_PACKAGES:
        parts = package.split(".")
        for depth in range(1, len(parts) + 1):
            name = ".".join(parts[:depth])
            module = types.ModuleType(name)
            module.__path__ = [str(root.joinpath(*parts[1:depth]))]
            module.__package__ = name
            modules[name] = module
    modules["vllm_torchtpu.logger"] = stub_module(
        "vllm_torchtpu.logger", init_logger=logging.getLogger
    )
    for name, module in modules.items():
        sys.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)


@functools.cache
def load_torchtpu_dsa(source_root: Path) -> types.SimpleNamespace:
    """The pinned vllm-torchtpu modules a DSA layer step calls."""
    root = Path(source_root).expanduser().resolve() / "src" / "vllm_torchtpu"
    verify_sources(
        root, {**TORCHTPU_DSA_SOURCE_SHA256, **TORCHTPU_SERVING_SHA256}
    )
    if "vllm_torchtpu" in sys.modules:
        raise RuntimeError(
            "vllm_torchtpu was imported before the pinned DSA source; refusing"
            " to benchmark an ambiguous baseline."
        )
    install_torchtpu_packages(root)
    with torch_stubs():
        mla_kv = importlib.import_module(
            "vllm_torchtpu.kernels.mla.kv_cache_utils"
        )
    kernels = "vllm_torchtpu.kernels"
    return types.SimpleNamespace(
        envs=importlib.import_module("vllm_torchtpu.envs"),
        mla_kv=mla_kv,
        stream=importlib.import_module(
            f"{kernels}.deepseek_v4.streamindex_topk.streamindex_topk"
        ),
        kv_append=importlib.import_module(
            f"{kernels}.deepseek_v4.streamindex_topk.kv_cache_append"
        ),
        sparse=importlib.import_module(f"{kernels}.mla.sparse.kernel"),
        dispatch=importlib.import_module(f"{kernels}.mla.dispatch"),
    )


def patch_pallas_fori_lowering() -> None:
    """env_override._patch_jax_pallas_fori_lowering: Pallas lowers a
    dynamic-bound fori_loop without its dead remainder loop when unroll is
    1."""
    from jax._src.pallas.mosaic import lowering

    source = textwrap.dedent(
        inspect.getsource(lowering._lower_jaxpr_to_for_loop)
    )
    if FORI_REMAINDER_NEEDLE not in source:
        raise RuntimeError(
            "jax's _lower_jaxpr_to_for_loop no longer holds the line the"
            " torchtpu patch rewrites."
        )
    namespace = {}
    exec(  # pylint: disable=exec-used
        compile(
            source.replace(FORI_REMAINDER_NEEDLE, FORI_REMAINDER_FIX, 1),
            lowering.__file__,
            "exec",
        ),
        lowering.__dict__,
        namespace,
    )
    lowering._lower_jaxpr_to_for_loop = namespace["_lower_jaxpr_to_for_loop"]


def prepare_torchtpu_process() -> None:
    """What vllm-torchtpu's env_override sets at import that reaches these
    kernels; must run before the TPU backend starts."""
    from jax._src import xla_bridge

    if xla_bridge.backends_are_initialized():
        raise RuntimeError(
            "prepare_torchtpu_process must run before the TPU backend starts."
        )
    flags = os.environ.get("LIBTPU_INIT_ARGS", "")
    if DYNAMIC_SMEM_FLAG_NAME not in flags:
        os.environ["LIBTPU_INIT_ARGS"] = f"{DYNAMIC_SMEM_FLAG} {flags}".strip()
    patch_pallas_fori_lowering()


def pack_index_keys(keys: jax.Array) -> jax.Array:
    """VllmTPUSparseAttnIndexer's packed keys: quantize_tensor(k, fp8, axis=-1,
    use_ue8m0=True) and the fp8 bytes followed by the exponent byte."""
    import jax.numpy as jnp
    from jax import lax

    values = keys.astype(jnp.float32)
    scale = jnp.max(jnp.abs(values), axis=-1, keepdims=True) * jnp.float32(
        1.0 / FP8_MAX
    )
    exponent = jnp.ceil(jnp.log2(scale))
    exponent = jnp.clip(
        jnp.where(jnp.exp2(exponent) < scale, exponent + 1, exponent),
        E8M0_MIN_EXP,
        E8M0_MAX_EXP,
    )
    scale = jnp.exp2(exponent)
    inverse = jnp.where(scale == 0, 0.0, 1.0 / scale)
    quantized = jnp.clip(values * inverse, -FP8_MAX, FP8_MAX).astype(
        jnp.float8_e4m3fn
    )
    return jnp.concatenate(
        [
            lax.bitcast_convert_type(quantized, jnp.uint8),
            (exponent - E8M0_MIN_EXP).astype(jnp.uint8),
        ],
        axis=-1,
    )


def streamidx_block_sizes(envs, num_tokens: int):
    """mla_attention_op._streamidx_block_sizes: (num_kv_pages_per_block,
    num_queries_per_block) of a padded token bucket."""
    small_bucket_bq = envs.TPU_STREAMIDX_SMALL_BUCKET_BQ
    if (
        small_bucket_bq is None
        or num_tokens > envs.TPU_STREAMIDX_SMALL_BUCKET_TOKENS
    ):
        return STREAMIDX_KV_PAGES_PER_BLOCK, STREAMIDX_QUERIES_PER_BLOCK
    small_bucket_bkvp = (
        envs.TPU_STREAMIDX_SMALL_BUCKET_BKVP
        or STREAMIDX_SMALL_BUCKET_DEFAULT_BKVP
    )
    return (
        (*STREAMIDX_KV_PAGES_PER_BLOCK[:2], small_bucket_bkvp),
        (*STREAMIDX_QUERIES_PER_BLOCK[:2], small_bucket_bq),
    )


def index_cache_layout(torchtpu):
    """_resolve_indexer_kv_cache_mode's layout at DSA_PAGE, for a mode whose
    Pallas writer this module mirrors."""
    envs = torchtpu.envs
    resolved = envs.VLLM_TPU_INDEXER_KV_CACHE_MODE.strip().lower()
    if resolved not in PALLAS_INDEX_CACHE_MODES:
        raise NotImplementedError(
            f"VLLM_TPU_INDEXER_KV_CACHE_MODE={resolved!r}: only the modes"
            f" {PALLAS_INDEX_CACHE_MODES} are mirrored."
        )
    if envs.is_indexer_kv_cache_mode_explicit():
        mode = envs.VLLM_TPU_INDEXER_KV_CACHE_MODE
    else:
        mode = None
    return torchtpu.stream.indexer_kv_layout(DSA_PAGE, mode=mode)


def sparse_mla_chunk_kwargs(torchtpu) -> dict[str, int]:
    """attention_interface.sparse_mla_chunk_kwargs."""
    chunk = torchtpu.envs.TPU_SPARSE_MLA_CHUNK_TOKENS
    if chunk is None:
        return {}
    else:
        return {"gather_and_attention_chunk_size": chunk}


@dataclasses.dataclass(frozen=True)
class TorchtpuConfig:
    """A torchtpu DSA layer's static settings."""

    geometry: DsaGeometry
    index_layout: object
    nope_layout: object
    rope_layout: object
    is_masked_dense: bool

    def specs(self, torchtpu, num_pages: int):
        mla_kv = torchtpu.mla_kv
        return (
            mla_kv.SparseMLAKVCacheSpec.create(
                mla_kv.KVCacheType.NOPE,
                self.nope_layout,
                num_pages,
                DSA_PAGE,
                LATENT_DIM,
                KV_PACKING,
            ),
            mla_kv.SparseMLAKVCacheSpec.create(
                mla_kv.KVCacheType.ROPE,
                self.rope_layout,
                num_pages,
                DSA_PAGE,
                ROPE_DIM,
                KV_PACKING,
            ),
        )


def torchtpu_config(
    torchtpu, geometry: DsaGeometry, is_masked_dense: bool
) -> TorchtpuConfig:
    """Serving's layouts: TPU_SPARSE_MLA_*_LAYOUT, or the TensorCore layouts
    masked-dense admission requires."""
    layout = torchtpu.mla_kv.KVCacheLayout
    if is_masked_dense:
        nope_layout = rope_layout = layout.TENSORCORE
    else:
        nope_layout = layout(torchtpu.envs.TPU_SPARSE_MLA_NOPE_LAYOUT)
        rope_layout = layout(torchtpu.envs.TPU_SPARSE_MLA_ROPE_LAYOUT)
    return TorchtpuConfig(
        geometry=geometry,
        index_layout=index_cache_layout(torchtpu),
        nope_layout=nope_layout,
        rope_layout=rope_layout,
        is_masked_dense=is_masked_dense,
    )


def torchtpu_write(torchtpu, config, caches, keys, seq_lens, pages, cu_q_lens):
    """Both ops' cache writes, as the indexer and the sparse MLA op do."""
    index_cache, nope_cache, rope_cache = caches
    nope_spec, rope_spec = config.specs(torchtpu, nope_cache.shape[0])
    index_cache = torchtpu.kv_append.coalesced_kv_append(
        index_cache,
        pack_index_keys(keys.index_keys),
        seq_lens,
        pages,
        cu_q_lens,
        kv_layout=config.index_layout,
    )
    nope_cache, rope_cache = torchtpu.mla_kv.update_sparse_mla_kv_cache(
        nope_cache,
        rope_cache,
        keys.latent,
        keys.rope,
        seq_lens,
        pages,
        cu_q_lens,
        nope_spec=nope_spec,
        rope_spec=rope_spec,
    )
    return index_cache, nope_cache, rope_cache


def torchtpu_topk(torchtpu, config, index_cache, queries, *meta):
    """_streamidx_topk_jax's top-k, on the written indexer cache."""
    import jax.numpy as jnp

    seq_lens, pages, cu_q_lens, distribution = meta
    blocks, queries_per_block = streamidx_block_sizes(
        torchtpu.envs, queries.index_queries.shape[0]
    )
    return torchtpu.stream.streamindex_topk(
        queries.index_queries,
        queries.index_weights,
        index_cache,
        seq_lens,
        pages,
        cu_q_lens,
        jnp.stack([distribution[0], distribution[0], distribution[2]]),
        k=config.geometry.top_k,
        compression_ratio=INDEX_COMPRESSION_RATIO,
        num_kv_pages_per_block=blocks,
        num_queries_per_block=queries_per_block,
        enable_early_exit=STREAMIDX_ENABLE_EARLY_EXIT,
        kv_layout=config.index_layout,
        chunk_tokens=torchtpu.envs.TPU_STREAMIDX_CHUNK_TOKENS,
    )


def torchtpu_attend(torchtpu, config, caches, queries, topk, *meta):
    """sparse_mla_attention's attention, on the written latent caches."""
    import jax.numpy as jnp

    _, nope_cache, rope_cache = caches
    sparse = torchtpu.sparse
    q_nope, q_pe = queries.q_nope, queries.q_pe
    cache_layout = config.nope_layout.value
    scales = {"sm_scale": config.geometry.sm_scale, "k_scale": DSA_KV_SCALE}
    if config.is_masked_dense and torchtpu.dispatch.applies(
        q_nope, q_pe, nope_cache, rope_cache, topk, cache_layout
    ):
        return torchtpu.dispatch.attention_from_parts(
            q_nope, q_pe, nope_cache, rope_cache, topk, *meta, **scales
        )
    if torchtpu.envs.TPU_MLA_SPARSE_SPLIT_Q and sparse.supports_split_q(
        q_nope, q_pe
    ):
        q, split_kwargs = q_nope, {"q_pe": q_pe}
    else:
        q, split_kwargs = jnp.concatenate([q_nope, q_pe], axis=-1), {}
    _, pages, cu_q_lens, distribution = meta
    out = sparse.sparse_ragged_paged_attention(
        q,
        nope_cache,
        rope_cache,
        topk,
        pages,
        cu_q_lens,
        distribution,
        cache_layout=cache_layout,
        **scales,
        **sparse_mla_chunk_kwargs(torchtpu),
        **split_kwargs,
    )
    return out[..., :LATENT_DIM]


def torchtpu_meta(
    requests: DsaRequests, padded_seqs: int
) -> tuple[jax.Array, ...]:
    """(seq_lens, page_indices, cu_q_lens, request_distribution) as the
    runner pads them to padded_seqs requests."""
    import jax.numpy as jnp

    batch, pages_per_seq = requests.block_table.shape
    seq_lens = np.zeros(padded_seqs, np.int32)
    seq_lens[:batch] = requests.kv_lens
    cu_q_lens = np.full(padded_seqs + 1, requests.num_tokens, np.int32)
    cu_q_lens[: batch + 1] = requests.cu_q_lens
    pages = np.zeros((padded_seqs, pages_per_seq), np.int32)
    pages[:batch] = requests.block_table
    distribution = [requests.num_decode, requests.num_decode, batch]
    return tuple(
        jnp.asarray(np.asarray(values, np.int32))
        for values in (seq_lens, pages.reshape(-1), cu_q_lens, distribution)
    )


def torchtpu_empty_caches(torchtpu, config, num_pages: int):
    import jax.numpy as jnp

    nope_spec, rope_spec = config.specs(torchtpu, num_pages)
    index_shape = torchtpu.stream.get_kv_cache_shape(
        num_pages, DSA_PAGE, config.geometry.index_head_dim, config.index_layout
    )
    return (
        jnp.zeros(index_shape, jnp.uint8),
        jnp.zeros(nope_spec.shape, nope_spec.jax_dtype),
        jnp.zeros(rope_spec.shape, rope_spec.jax_dtype),
    )


def pad_tokens(values: jax.Array, num_tokens: int) -> jax.Array:
    import jax.numpy as jnp

    padding = ((0, num_tokens - values.shape[0]),) + ((0, 0),) * (
        values.ndim - 1
    )
    return jnp.pad(values, padding)


def torchtpu_selection(outputs, caches, queries, requests, request):
    """The step's own top-k of request's tokens."""
    del caches, queries  # Unused.
    first, last = requests.cu_q_lens[request : request + 2]
    return outputs[1][first:last]


def build_torchtpu_layer(
    geometry: DsaGeometry,
    *,
    num_tokens: int,
    batch: int,
    is_masked_dense: bool,
    source: Path,
) -> DsaLayer:
    """torchtpu's layer step: the indexer op, then the sparse MLA op, on the
    runner's padded tokens and requests."""
    import jax

    from flywheel_tpu.softmax_attention.config import next_pow2

    torchtpu = load_torchtpu_dsa(source)
    config = torchtpu_config(torchtpu, geometry, is_masked_dense)
    padded_tokens = max(MIN_TOKEN_BUCKET, next_pow2(num_tokens))
    padded_seqs = max(MIN_NUM_SEQS, next_pow2(batch))

    def _write(caches, keys, *meta):
        return torchtpu_write(torchtpu, config, caches, keys, *meta)

    def _step(caches, queries, keys, *meta):
        caches = _write(caches, keys, *meta[:3])
        topk = torchtpu_topk(torchtpu, config, caches[0], queries, *meta)
        out = torchtpu_attend(torchtpu, config, caches, queries, topk, *meta)
        return (out, topk), caches

    write = jax.jit(_write, donate_argnums=(0,))
    return DsaLayer(
        empty_caches=functools.partial(torchtpu_empty_caches, torchtpu, config),
        write=lambda caches, keys, requests: write(
            caches, keys, *torchtpu_meta(requests, len(requests.kv_lens))[:3]
        ),
        queries=lambda queries: DsaQueries(
            *(pad_tokens(values, padded_tokens) for values in queries)
        ),
        keys=lambda keys: DsaKeys(
            *(pad_tokens(values, padded_tokens) for values in keys)
        ),
        meta=lambda requests: torchtpu_meta(requests, padded_seqs),
        step=jax.jit(_step, donate_argnums=(0,)),
        selection=torchtpu_selection,
        attention_out=lambda outputs: outputs[0],
        info={
            "padded_tokens": padded_tokens,
            "padded_seqs": padded_seqs,
            "index_layout": config.index_layout.name,
            "nope_layout": config.nope_layout.value,
            "rope_layout": config.rope_layout.value,
            "streamidx_blocks": streamidx_block_sizes(
                torchtpu.envs, padded_tokens
            ),
        },
    )


def build_dsa_layer(
    impl: str,
    geometry: DsaGeometry,
    *,
    num_tokens: int,
    batch: int,
    max_seqlen_k: int,
    torchtpu_source: Path | None,
    route: str | None = None,
) -> DsaLayer:
    """impl's layer for calls of num_tokens tokens of batch requests whose
    kv_lens stay within max_seqlen_k; route pins flywheel's route."""
    if impl == FLYWHEEL_DSA:
        return build_flywheel_layer(geometry, max_seqlen_k, route)
    elif torchtpu_source is None:
        raise ValueError(f"--impl {impl} needs --torchtpu-source.")
    else:
        return build_torchtpu_layer(
            geometry,
            num_tokens=num_tokens,
            batch=batch,
            is_masked_dense=impl == TORCHTPU_DSA_MASKED_DENSE,
            source=torchtpu_source,
        )


def page_index_keys(key: jax.Array, pages: jax.Array, dim: int) -> jax.Array:
    """Random bf16 index keys of physical pages, (len(pages) * DSA_PAGE, dim);
    page p's keys depend on p alone."""
    import jax
    import jax.numpy as jnp

    keys = jax.vmap(
        lambda page: draw_normal(
            jax.random.fold_in(key, page), (DSA_PAGE, dim), jnp.bfloat16
        )
    )(pages)
    return keys.reshape(-1, dim)


def index_cache_pages(keys: jax.Array) -> jax.Array:
    """Whole pages of keys in the index-K cache layout both impls share
    (flywheel's index_cache_shape, torchtpu's SEQ_ALONG_LANE), (pages,
    dim / 4 + 1, 4, DSA_PAGE) uint8: byte i of group g < dim / 4 a token's
    fp8 key dim 4 g + i, the last group its UE8M0 exponent byte and zeros."""
    import jax.numpy as jnp
    from jax import lax

    quantized, exponent = quantized_index_keys(keys)
    num_tokens, dim = quantized.shape
    scale_group = jnp.zeros((num_tokens, KV_PACKING), jnp.uint8)
    rows = jnp.concatenate(
        [
            lax.bitcast_convert_type(quantized, jnp.uint8),
            scale_group.at[:, 0].set(exponent),
        ],
        axis=-1,
    )
    pages = rows.reshape(-1, DSA_PAGE, dim + KV_PACKING).transpose(0, 2, 1)
    return pages.reshape(pages.shape[0], -1, KV_PACKING, DSA_PAGE)


def fill_index_cache(index_cache: jax.Array, key: jax.Array) -> jax.Array:
    """index_cache with every page's page_index_keys, written block by
    block in the layout index_cache_pages makes."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    num_pages, groups, packing, page_size = index_cache.shape
    dim = (groups - 1) * packing
    if (packing, page_size) != (KV_PACKING, DSA_PAGE) or (
        index_cache.dtype != jnp.uint8
    ):
        raise ValueError(
            "The index cache is not in the shared index-K layout; got"
            f" {index_cache.dtype} {index_cache.shape}."
        )
    block = math.gcd(num_pages, FILL_PAGES_PER_BLOCK)

    @functools.partial(jax.jit, donate_argnums=(0,))
    def fill_block(cache, first):
        pages = first + jnp.arange(block, dtype=jnp.int32)
        values = index_cache_pages(page_index_keys(key, pages, dim))
        return lax.dynamic_update_slice(cache, values, (first, 0, 0, 0))

    for first in range(0, num_pages, block):
        index_cache = fill_block(index_cache, jnp.int32(first))
    return index_cache


def filled_caches(
    layer: DsaLayer, base_keys: DsaKeys, num_pages: int, index_key: jax.Array
) -> tuple[jax.Array, ...]:
    """num_pages pages of caches: latents tiled from base_keys' pages, written
    with the impl's own writes, and random index keys per page, so that no two
    positions share an index score."""
    import jax
    import jax.numpy as jnp

    base_pages = base_keys.index_keys.shape[0] // DSA_PAGE
    requests = DsaRequests(
        cu_q_lens=np.arange(base_pages + 1) * DSA_PAGE,
        kv_lens=np.full(base_pages, DSA_PAGE),
        block_table=np.arange(base_pages).reshape(base_pages, 1),
        num_decode=0,
    )
    caches = layer.write(layer.empty_caches(base_pages), base_keys, requests)
    tile = jax.jit(
        lambda cache: jnp.tile(
            cache, (num_pages // base_pages,) + (1,) * (cache.ndim - 1)
        )
    )
    index_cache, *latent_caches = (tile(cache) for cache in caches)
    return jax.block_until_ready(
        (fill_index_cache(index_cache, index_key), *latent_caches)
    )


def edge_rows(first: int, last: int) -> np.ndarray:
    """The first and last CHECK_ROWS token rows of [first, last)."""
    return np.unique(
        np.concatenate([
            np.arange(first, min(first + CHECK_ROWS, last)),
            np.arange(max(first, last - CHECK_ROWS), last),
        ])
    )


@dataclasses.dataclass(frozen=True)
class DsaScenario:
    """A serving scenario's DSA calls on one impl. fresh() makes its caches;
    check_dsa reads rows of the last call's tokens, and logical(request) gives
    that request's keys as the calls leave the cache."""

    layer: DsaLayer
    geometry: DsaGeometry
    fresh: Callable[[], tuple[jax.Array, ...]]
    queries: DsaQueries
    layer_queries: DsaQueries
    calls: list[tuple[DsaKeys, tuple[jax.Array, ...]]]
    requests: list[DsaRequests]
    rows: np.ndarray
    logical: Callable[[int], DsaKeys]

    def step(self, caches):
        outputs = None
        for keys, meta in self.calls:
            outputs, caches = self.layer.step(
                caches, self.layer_queries, keys, *meta
            )
        return outputs, caches


def exponent_scale(exponent: jax.Array) -> jax.Array:
    """f32 power of two of UE8M0 biased exponents."""
    import jax.numpy as jnp
    from jax import lax

    return lax.bitcast_convert_type(
        exponent.astype(jnp.uint32) << F32_MANTISSA_BITS, jnp.float32
    )


def quantized_index_keys(keys: jax.Array) -> tuple[jax.Array, jax.Array]:
    """act_quant with UE8M0 scales of (L, D) keys: their fp8 values and the
    (L,) uint8 biased exponents of their scales."""
    import jax.numpy as jnp
    from jax import lax

    values = keys.astype(jnp.float32)
    ratio = jnp.maximum(jnp.max(jnp.abs(values), axis=-1), AMAX_FLOOR)
    bits = lax.bitcast_convert_type(
        ratio * jnp.float32(1.0 / FP8_MAX), jnp.uint32
    )
    exponent = ((bits >> F32_MANTISSA_BITS) & F32_EXPONENT_MASK) + (
        (bits & F32_MANTISSA_MASK) != 0
    ).astype(jnp.uint32)
    quantized = jnp.clip(
        values / exponent_scale(exponent)[:, None], -FP8_MAX, FP8_MAX
    )
    return quantized.astype(jnp.float8_e4m3fn), exponent.astype(jnp.uint8)


def dequantized_index_keys(keys: jax.Array) -> tuple[jax.Array, jax.Array]:
    """quantized_index_keys' fp8 values as f32, and the (L,) f32 scales."""
    import jax.numpy as jnp

    quantized, exponent = quantized_index_keys(keys)
    return quantized.astype(jnp.float32), exponent_scale(exponent)


def unique_selection(
    selection: jax.Array, length: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """(positions, is_selected): the selection's distinct positions in
    [0, length), sorted, and which entries hold one."""
    import jax.numpy as jnp

    positions = jnp.sort(selection)
    is_first = jnp.concatenate([
        jnp.ones((1,), bool),
        positions[1:] != positions[:-1],
    ])
    is_selected = is_first & (positions >= 0) & (positions < length)
    return jnp.where(is_selected, positions, 0), is_selected


def reference_row(
    queries: DsaQueries,
    keys: DsaKeys,
    length: jax.Array,
    selection: jax.Array,
    *,
    top_k: int,
    sm_scale: float,
) -> tuple[jax.Array, jax.Array]:
    """One token against its request's logical keys in f32: the attention
    over the positions the impl selected, (H, LATENT_DIM), and the share of
    the f32 index scores' top_k mass the selection misses. queries hold the
    token's rows, keys the request's positions, length the positions it
    sees."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    highest = lax.Precision.HIGHEST
    index_keys, key_scale = dequantized_index_keys(keys.index_keys)
    logits = jnp.einsum(
        "hd,ld->hl",
        queries.index_queries.astype(jnp.float32),
        index_keys,
        precision=highest,
    )
    scores = key_scale * jnp.einsum(
        "h,hl->l",
        queries.index_weights,
        jnp.maximum(logits, 0.0),
        precision=highest,
    )
    scores = jnp.where(jnp.arange(scores.shape[0]) < length, scores, -jnp.inf)
    best = lax.top_k(scores, min(top_k, scores.shape[0]))[0]
    best = jnp.where(jnp.isfinite(best), best, 0.0)
    positions, is_selected = unique_selection(selection, length)
    picked = jnp.sum(jnp.where(is_selected, scores[positions], 0.0))
    deficit = (jnp.sum(best) - picked) / jnp.sum(jnp.abs(best))
    latent = keys.latent[positions].astype(jnp.float32) * DSA_KV_SCALE
    rope = keys.rope[positions].astype(jnp.float32) * DSA_KV_SCALE
    attention = (
        jnp.einsum(
            "hl,kl->hk",
            queries.q_nope.astype(jnp.float32),
            latent,
            precision=highest,
        )
        + jnp.einsum(
            "hr,kr->hk",
            queries.q_pe.astype(jnp.float32),
            rope,
            precision=highest,
        )
    ) * sm_scale
    attention = jnp.where(is_selected[None, :], attention, -jnp.inf)
    out = jnp.einsum(
        "hk,kl->hl",
        jax.nn.softmax(attention, axis=-1),
        latent,
        precision=highest,
    )
    return out, deficit


@functools.cache
def jitted_reference_row():
    import jax

    return jax.jit(reference_row, static_argnames=("top_k", "sm_scale"))


def cos_diff(got: np.ndarray, want: np.ndarray) -> float:
    """1 - 2 <got, want> / (|got|^2 + |want|^2): 0 when they match."""
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    return 1.0 - 2.0 * float((got * want).sum()) / float(
        (got * got + want * want).sum()
    )


def check_dsa(
    scenario: DsaScenario, outputs, caches: tuple[jax.Array, ...]
) -> dict[str, float]:
    """The last call's cos_diff on the scenario's rows against the f32
    attention over the impl's own selection, and the worst row's index score
    deficit against the f32 indexer's top-k."""
    layer, requests = scenario.layer, scenario.requests[-1]
    out = layer.attention_out(outputs)
    token_requests = requests.token_requests()
    positions = requests.token_positions()
    reference = jitted_reference_row()
    got, want, deficits = [], [], []
    for request in np.unique(token_requests[scenario.rows]):
        keys = scenario.logical(int(request))
        selection = layer.selection(
            outputs, caches, scenario.queries, requests, int(request)
        )
        first = requests.cu_q_lens[request]
        for row in scenario.rows[token_requests[scenario.rows] == request]:
            row_out, deficit = reference(
                DsaQueries(*(values[row] for values in scenario.queries)),
                keys,
                int(positions[row]) + 1,
                selection[row - first],
                top_k=scenario.geometry.top_k,
                sm_scale=scenario.geometry.sm_scale,
            )
            got.append(np.asarray(out[row].astype(np.float32)))
            want.append(np.asarray(row_out))
            deficits.append(float(deficit))
    return {
        "cos_diff": cos_diff(np.stack(got), np.stack(want)),
        "index_score_deficit": max(deficits),
        "check_rows": len(deficits),
    }


def measure_dsa(scenario: DsaScenario) -> dict[str, object]:
    """Time the scenario, then check one fresh run of it."""
    timing, _ = time_call(scenario.step, scenario.fresh())
    outputs, caches = scenario.step(scenario.fresh())
    checked = check_dsa(scenario, outputs, caches)
    is_ok = (
        checked["cos_diff"] <= COS_DIFF_BOUND
        and checked["index_score_deficit"] <= INDEX_DEFICIT_BOUND
    )
    num_tokens = sum(requests.num_tokens for requests in scenario.requests)
    return {
        **scenario.layer.info,
        **timing,
        **checked,
        "ms_per_call": timing["median_ms"] / len(scenario.calls),
        "tokens_per_s": num_tokens / (timing["median_ms"] / 1e3),
        "status": "ok" if is_ok else "check_failed",
    }


def dsa_work(geometry: DsaGeometry, calls: list[DsaRequests]) -> dict[str, int]:
    """The (token, key) pairs the scenario's calls attend and index, and the
    flops of each."""
    attention_pairs = 0
    indexed_pairs = 0
    for requests in calls:
        visible = requests.token_positions().astype(np.int64) + 1
        attention_pairs += int(np.minimum(visible, geometry.top_k).sum())
        indexed_pairs += int(visible.sum())
    return {
        "attention_pairs": attention_pairs,
        "indexed_pairs": indexed_pairs,
        # Note (david): QK over the latent and the rope, P @ V over the latent.
        "attn_flops": (
            2 * attention_pairs * geometry.heads * (2 * LATENT_DIM + ROPE_DIM)
        ),
        "index_flops": (
            2 * indexed_pairs * geometry.index_heads * geometry.index_head_dim
        ),
    }
