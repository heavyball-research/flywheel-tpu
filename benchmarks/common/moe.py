"""SHA-pinned MoE baselines, and the single-device forwards built from them.

All of them start from the router logits and return the bf16 combined output,
as fused_moe does after a top-k:

- gmm: tpu-inference's GMM_TP backend on one device: XLA routing and token
  gather, tokamax's gmm_v2 twice (gate|up with the fused activation, then the
  down projection), and the SparseCore dense_gather_reduce combine. The
  kernels are tokamax/_src/ops/experimental/gmm_v2/gmm_v2.py and
  tpu_inference/kernels/sparse_core/dense_gather_reduce.py; the glue follows
  fused_moe_func and moe_gmm_local in
  tpu_inference/layers/common/fused_moe_gmm.py, which imports vLLM and so is
  not loaded.
- ragged_dot: XLA's jax.lax.ragged_dot twice between an XLA sort and combine.
- torchtpu: vllm-torchtpu's default path for unquantized TP (non-EP) experts:
  moe_routing.select_experts (softmax, its rowmax router top-k kernel,
  renormalize, weights in the activation dtype) and
  layers/core/fused_moe_gmm.fused_moe_func with USE_MOE_SPARSE_CORE on, as
  its unquantized method calls them, every env knob at its default. The jax
  modules are loaded from a vllm-torchtpu checkout at TORCHTPU_GIT_SHA
  (--torchtpu-source), each verified against its SHA-256; vllm_torchtpu's
  logger is stubbed, so neither torch nor vLLM is needed. The process-wide
  libtpu flag and Pallas lowering patch of its env_override.py are applied
  as its import would.

--tpu-inference-source and --tokamax-source default to the third_party
submodules, at the pins RPA v3 and splash use; every loaded file is verified
against its SHA-256 there.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import logging
import os
import sys
import textwrap
import types
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
from jax._src.pallas.mosaic import lowering

THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"
DEFAULT_TPU_INFERENCE_SOURCE = THIRD_PARTY / "tpu-inference"
DEFAULT_TOKAMAX_SOURCE = THIRD_PARTY / "tokamax"
TPU_INFERENCE_GIT_SHA = "4420cae3615711b3b8f66bac1c802ae3f9fa3439"
TOKAMAX_GIT_SHA = "84e5f36a0c1a6b9c3c2f0e997a09e62038a2bf07"
# Note (david): vllm-torchtpu main on 2026-11-07; the files below, relative
# to src/vllm_torchtpu, are the import closure of fused_moe_gmm and
# router_topk.
TORCHTPU_GIT_SHA = "b08f10fdc6166ff3aa3306629fcaf5b133012cb2"
TORCHTPU_PACKAGE = "vllm_torchtpu"
TORCHTPU_SOURCE_SHA256 = {
    "env_override.py": (
        "b430e488043e5324d2d85fb8293bb7037ba073a0f3aadc3774936e69786f91ce"
    ),
    "envs.py": (
        "01403153b9598b684f602bcd3533f093a97624ffcaffb43246df57a5fac9650e"
    ),
    "kernels/experimental/moe_dense_expert.py": (
        "47b35eab6a637a19017ca68dcf741e0ba9c0882b701849d0ee879b2b81dfa3f2"
    ),
    "kernels/megablox/gmm_v2.py": (
        "ed059c02fb1494cb39a22ea9e74d1c8ebdbe90ed4cf9c0dd86dfaf051f39154b"
    ),
    "kernels/megablox/moe_onehot_unpermute.py": (
        "4220a4cb954db4cd9d02ef927071549edf568148b7fad7bc4ed0615fdc05618e"
    ),
    "kernels/router_topk.py": (
        "56507596616a66b1c1008eccb2781d1b63c06e4527a94a5995ebadf9e09c14e3"
    ),
    "kernels/sparse_core/__init__.py": (
        "43955acccf4b719dbab4dceb4c9437274dc864bf3d15b22c252cd9d010255c7d"
    ),
    "kernels/sparse_core/core_map_helper.py": (
        "6fea75cb4d7c4f226c9f16e847e3bdfc8e2bc3a29e00dbdfe3205c3d28a3ceeb"
    ),
    "kernels/sparse_core/ragged_gather_reduce.py": (
        "a195d0283eec8face789a31d2272874a10c43be3934fcad91ff722d0fcdd342d"
    ),
    "kernels/sparse_core/ragged_gather_reduce_v2/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "kernels/sparse_core/ragged_gather_reduce_v2/config.py": (
        "0b8082c17707cfe414b742d377b318fcdea66447bc1e2f8ae93747d2af206c19"
    ),
    "kernels/sparse_core/ragged_gather_reduce_v2/kernel.py": (
        "9e79f28c1b8bc34b6798d3ac1912614e0cd887584d66573457e4e9b911e40105"
    ),
    "kernels/sparse_core/ragged_gather_reduce_v2/memory_ref.py": (
        "b4f6f0df9eab54a5faeace4edb605f0a65357eac8d65c64baa3a0b35df811d49"
    ),
    "kernels/sparse_core/ragged_gather_reduce_v2/wrapper.py": (
        "8edb4e6c961086b294140b83b218f82f37ca9e1b9b7d803ddcfc0cadde0e0d43"
    ),
    "kernels/sparse_core/ragged_gather_reduce_v3.py": (
        "a9d488dc24390066ae85d6390da5bf30b6d6a582a9a9e91158f1fc37f8f54070"
    ),
    "kernels/sparse_core/ragged_gather_v2.py": (
        "670d8d32718da9b9f4bf23e6b6f1c57015b0a6de5eaf5777f0289a657f78f194"
    ),
    "kernels/sparse_core/support.py": (
        "0d9cf4bc7daa3d74af7a07005e823321e1284618a43abac6722ed5cea640116d"
    ),
    "layers/core/fused_moe_gmm.py": (
        "3b5c039ecbe93f1e932e7e2aa5ab920a84dcb53cf5210116d20d8ea8c7323425"
    ),
}
TPU_INFERENCE_SOURCE_SHA256 = {
    "tpu_inference/kernels/sparse_core/dense_gather_reduce.py": (
        "f4bf9e2fe9a733825cb2c96ce283bfdebfd04f0433186dfce6934a05abb3ec79"
    ),
}
TOKAMAX_SOURCE_SHA256 = {
    "tokamax/_src/ops/experimental/gmm_v2/gmm_v2.py": (
        "ae434daa7e830a65eb3cbdd4e541f618191c42add8a0f1394568baef9c07b06a"
    ),
}

# Note (david): importing vllm_torchtpu runs env_override.py, which changes
# the whole process: it adds this libtpu flag (before the TPU backend starts)
# and patches the Pallas fori_loop lowering to drop the dead remainder loop of
# a dynamic-bound unroll=1 loop. The stubbed import skips both, so they are
# applied here; they then hold for every kernel in the process, as they do in
# a vllm-torchtpu deployment.
TORCHTPU_LIBTPU_FLAG = "--xla_tpu_use_dynamic_smem_negotiation=true"
TORCHTPU_FORI_NEEDLE = "elif has_dynamic_remainder:"
TORCHTPU_FORI_FIX = "elif has_dynamic_remainder and unroll != 1:"
# Note (david): the floor of select_experts' renormalization, so a token
# whose top-k weights all underflow divides by a positive number.
RENORMALIZE_FLOOR = 1e-20


def verify_sources(
    root: Path, pin: str, hashes: dict[str, str], flag: str
) -> Path:
    """root, resolved, once every pinned file under it matches its
    SHA-256."""
    resolved_root = root.expanduser().resolve()
    for name, expected in hashes.items():
        path = resolved_root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing; {flag} must be a checkout at {pin}."
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                f"{path} does not match {pin}; got sha256 {digest}."
            )
    return resolved_root


def stub_packages(root: Path, dotted: str) -> None:
    """Empty parent packages for dotted, rooted at root, so a pinned file
    imports without its real package __init__ (and its vLLM imports)."""
    parts = dotted.split(".")
    for depth in range(1, len(parts) + 1):
        name = ".".join(parts[:depth])
        if name in sys.modules:
            continue
        package = types.ModuleType(name)
        package.__path__ = [str(root.joinpath(*parts[:depth]))]
        package.__package__ = name
        sys.modules[name] = package


class OnceLogger(logging.LoggerAdapter):
    """The logger vllm_torchtpu.logger.init_logger returns: a logger with
    info_once."""

    def __init__(self, name: str):
        super().__init__(logging.getLogger(name), {})
        self.logged_messages = set()

    def info_once(self, msg: str, *args) -> None:
        if msg not in self.logged_messages:
            self.logged_messages.add(msg)
            self.info(msg, *args)


def stub_logger_module(name: str) -> None:
    """A logger module whose init_logger the pinned files import."""
    logger_module = types.ModuleType(name)
    logger_module.init_logger = OnceLogger
    sys.modules[name] = logger_module


def load_baselines(
    tpu_inference_source: Path, tokamax_source: Path
) -> tuple[Callable, Callable]:
    """(gmm_v2, dense_gather_reduce) from the pinned files."""
    for name in ("tpu_inference", "tokamax"):
        if name in sys.modules:
            raise RuntimeError(
                f"{name} was imported before the pinned MoE baselines;"
                " refusing to benchmark an ambiguous baseline."
            )
    tpu_inference_root = verify_sources(
        tpu_inference_source,
        TPU_INFERENCE_GIT_SHA,
        TPU_INFERENCE_SOURCE_SHA256,
        "--tpu-inference-source",
    )
    tokamax_root = verify_sources(
        tokamax_source,
        TOKAMAX_GIT_SHA,
        TOKAMAX_SOURCE_SHA256,
        "--tokamax-source",
    )

    stub_packages(tpu_inference_root, "tpu_inference.kernels.sparse_core")

    # Note (david): gmm_v2 reads one tokamax flag, disable_multi_core_mode,
    # whose absl default is False; the stub fixes it there.
    stub_packages(tokamax_root, "tokamax._src.ops.experimental.gmm_v2")
    config = types.ModuleType("tokamax._src.config")
    config.disable_multi_core_mode = types.SimpleNamespace(value=False)
    sys.modules[config.__name__] = config

    gmm_v2 = importlib.import_module(
        "tokamax._src.ops.experimental.gmm_v2.gmm_v2"
    )
    gather_reduce = importlib.import_module(
        "tpu_inference.kernels.sparse_core.dense_gather_reduce"
    )
    return gmm_v2.gmm_v2, gather_reduce.dense_gather_reduce


def set_torchtpu_libtpu_flags() -> None:
    """env_override.py's libtpu flag; must run before the TPU backend
    starts."""
    args = os.environ.get("LIBTPU_INIT_ARGS", "")
    flag_name = TORCHTPU_LIBTPU_FLAG.split("=", maxsplit=1)[0]
    if not any(
        token.split("=", maxsplit=1)[0] == flag_name for token in args.split()
    ):
        os.environ["LIBTPU_INIT_ARGS"] = (
            f"{TORCHTPU_LIBTPU_FLAG} {args}".strip()
        )


def patch_pallas_fori_lowering() -> None:
    """env_override.py's _patch_jax_pallas_fori_lowering, which must run
    before any Pallas kernel is traced."""
    # Note (david): the patch rewrites a private jax function, which is what
    # the baseline's import does to the whole process.
    # pylint: disable=protected-access
    source = textwrap.dedent(
        inspect.getsource(lowering._lower_jaxpr_to_for_loop)
    )
    if source.count(TORCHTPU_FORI_NEEDLE) != 1:
        raise RuntimeError(
            "the vllm-torchtpu fori_loop lowering patch does not apply to this"
            " jax version; pin the jax that vllm-torchtpu"
            f" {TORCHTPU_GIT_SHA} pins (0.11.2)."
        )
    namespace = {}
    exec(  # pylint: disable=exec-used
        compile(
            source.replace(TORCHTPU_FORI_NEEDLE, TORCHTPU_FORI_FIX),
            lowering.__file__,
            "exec",
        ),
        lowering.__dict__,
        namespace,
    )
    lowering._lower_jaxpr_to_for_loop = namespace["_lower_jaxpr_to_for_loop"]


def load_torchtpu_moe(torchtpu_source: Path) -> tuple[Callable, Callable, int]:
    """(fused_moe_func, router_topk, onehot_permute_threshold) from a
    vllm-torchtpu checkout at TORCHTPU_GIT_SHA, with the process-wide
    runtime patch its import applies (set_torchtpu_libtpu_flags must have run
    before the TPU backend started)."""
    if TORCHTPU_PACKAGE in sys.modules:
        raise RuntimeError(
            "vllm_torchtpu was imported before the pinned MoE baseline;"
            " refusing to benchmark an ambiguous baseline."
        )
    source_root = torchtpu_source.expanduser().resolve() / "src"
    verify_sources(
        source_root / TORCHTPU_PACKAGE,
        f"vllm-torchtpu {TORCHTPU_GIT_SHA}",
        TORCHTPU_SOURCE_SHA256,
        "--torchtpu-source",
    )
    # Note (david): the real package __init__ files import torch_tpu and
    # vLLM; the parents are stubbed instead (kernels/sparse_core/__init__.py
    # is a license header only). envs.py imports only the standard library.
    for dotted in (
        "vllm_torchtpu.layers.core",
        "vllm_torchtpu.kernels.megablox",
        "vllm_torchtpu.kernels.experimental",
        "vllm_torchtpu.kernels.sparse_core",
    ):
        stub_packages(source_root, dotted)
    stub_logger_module("vllm_torchtpu.logger")
    patch_pallas_fori_lowering()
    fused_moe_gmm = importlib.import_module(
        "vllm_torchtpu.layers.core.fused_moe_gmm"
    )
    router = importlib.import_module("vllm_torchtpu.kernels.router_topk")
    return (
        fused_moe_gmm.fused_moe_func,
        router.router_topk,
        fused_moe_gmm.resolve_onehot_permute_threshold(),
    )


def torchtpu_moe(
    x: jax.Array,
    logits: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    *,
    top_k: int,
    fused_moe_func: Callable,
    router_topk: Callable,
    onehot_threshold: int,
) -> jax.Array:
    """vllm-torchtpu's unquantized TP MoE forward: select_experts with
    softmax scoring and renormalize, then fused_moe_func. w1 is the merged
    [E, hidden, 2 * intermediate] gate|up, the layout its
    process_weights_after_loading makes from vLLM's w13 (no half padding at
    intermediate a multiple of 128)."""
    scores = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    topk_weights, topk_ids = router_topk(scores, top_k)
    renormalized_weights = topk_weights / jnp.maximum(
        topk_weights.sum(axis=-1, keepdims=True), RENORMALIZE_FLOOR
    )
    return fused_moe_func(
        x,
        w1,
        w2,
        None,
        None,
        None,
        None,
        renormalized_weights.astype(x.dtype),
        topk_ids.astype(jnp.int32),
        None,
        topk=top_k,
        activation="silu",
        use_ep=False,
        use_sparse_core=True,
        onehot_moe_permute_threshold=onehot_threshold,
        rhs_quant_dtype=None,
        skip_padded_tokens=False,
    )


def gmm_moe(
    x: jax.Array,
    logits: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    *,
    top_k: int,
    gmm_v2: Callable,
    dense_gather_reduce: Callable,
) -> jax.Array:
    """tpu-inference's GMM_TP forward on one device: fused_moe_func with
    use_ep=False, renormalize=True, softmax scoring and no padding, whose
    _process_tokens_locally sorts the tokens and whose moe_gmm_local with
    parallelism tp runs gmm_v2 with group_offset 0 and no zero fill."""
    num_tokens = x.shape[0]
    num_experts = w1.shape[0]
    scores = jax.nn.softmax(logits, axis=-1)
    top_scores, topk_indices = jax.lax.top_k(scores, k=top_k)
    topk_weights = (top_scores / top_scores.sum(axis=-1, keepdims=True)).astype(
        x.dtype
    )

    topk_indices_flat = topk_indices.flatten()
    topk_argsort_indices = jnp.argsort(topk_indices_flat)
    token_indices = jnp.arange(num_tokens, dtype=jnp.int32).repeat(top_k)
    token_indices_sorted = token_indices[topk_argsort_indices]
    group_sizes = jax.nn.one_hot(
        topk_indices_flat, num_experts, dtype=jnp.int32
    ).sum(axis=0)
    topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)
    sorted_x = x[token_indices_sorted]

    group_offset = jnp.array(0, dtype=jnp.int32)
    gmm1 = gmm_v2(
        lhs=sorted_x,
        rhs=w1,
        group_sizes=group_sizes,
        group_offset=group_offset,
        zero_initialize=False,
        fuse_act="silu",
        preferred_element_type=x.dtype,
    )
    gmm1 = gmm1[:, : w2.shape[1]]
    gmm2 = gmm_v2(
        lhs=gmm1,
        rhs=w2,
        group_sizes=group_sizes,
        group_offset=group_offset,
        zero_initialize=False,
    )
    out = dense_gather_reduce(
        gmm2, topk_argsort_revert_indices, topk_weights, top_k
    )
    return out.astype(x.dtype)


def ragged_dot_moe(
    x: jax.Array, logits: jax.Array, w1: jax.Array, w2: jax.Array, *, top_k: int
) -> jax.Array:
    """The MoE in XLA: softmax over the top-k logits, sort by expert,
    jax.lax.ragged_dot twice, unsort and sum the top-k rows in f32."""
    num_tokens, hidden = x.shape
    intermediate = w2.shape[1]
    top_logits, topk_ids = jax.lax.top_k(logits, top_k)
    topk_weights = jax.nn.softmax(top_logits.astype(jnp.float32), axis=-1)
    flat = topk_ids.reshape(-1)
    order = jnp.argsort(flat, stable=True)
    group_sizes = jnp.bincount(flat, length=w1.shape[0]).astype(jnp.int32)
    gate_up = jax.lax.ragged_dot(
        x[order // top_k], w1, group_sizes, preferred_element_type=jnp.float32
    )
    activated = (
        jax.nn.silu(gate_up[:, :intermediate]) * gate_up[:, intermediate:]
    ).astype(x.dtype)
    rows = jax.lax.ragged_dot(
        activated, w2, group_sizes, preferred_element_type=jnp.float32
    )
    scaled_rows = rows * topk_weights.reshape(-1)[order][:, None]
    out = (
        jnp.zeros((num_tokens * top_k, hidden), jnp.float32)
        .at[order]
        .set(scaled_rows)
    )
    return out.reshape(num_tokens, top_k, hidden).sum(axis=1).astype(x.dtype)
