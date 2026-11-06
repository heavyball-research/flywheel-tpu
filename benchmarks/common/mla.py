"""SHA-pinned loader for tpu-inference's MLA v2 kernel, the baseline of
decode_microbench.py's MLA cases.

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

from __future__ import annotations

import hashlib
import importlib
import logging
import math
import os
import sys
import types
from pathlib import Path

from benchmarks.common.rpa import DEFAULT_RPA_SOURCE, RPA_V3_GIT_SHA

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
