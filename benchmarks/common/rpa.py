"""SHA-pinned loader for the tpu-inference RPA v3 kernel, the RPA baseline.

--rpa-source defaults to the third_party/tpu-inference submodule
(git submodule update --init third_party/tpu-inference). Only
tpu_inference/kernels/ragged_paged_attention/v3/{kernel,util}.py are loaded,
and both are verified against their SHA-256 at RPA_V3_GIT_SHA, the submodule
pin. Those two files and the vLLM backend's flash_attn.py are unchanged from
1bd693780c5f, the revision softmax_attention/rpa_tuned_v6e.json was searched
on.

    PYTHONPATH=. uv run --no-sync python benchmarks/common/rpa.py   # host-only self-test
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import types
from pathlib import Path

DEFAULT_RPA_SOURCE = Path(__file__).resolve().parents[2] / "third_party" / (
    "tpu-inference")
RPA_V3_GIT_SHA = "4420cae3615711b3b8f66bac1c802ae3f9fa3439"
RPA_V3_PACKAGE = "tpu_inference.kernels.ragged_paged_attention.v3"
RPA_V3_SOURCE_SHA256 = {
    "kernel.py":
        "bcd21fdaff6b333903a09991d0ad57f3683fef18bf8abef449c0f10d6879609c",
    "util.py":
        "f9a1c1f189ef0b5d9e9babaa9d221faf01f12bc9bb809fe227d17901867b56c0",
}
FIELDS = ("bq_sz", "bkv_sz", "bq_csz", "bkv_csz")


def validate_rpa_source(source_root: Path) -> Path:
    """The pinned v3 package directory inside a tpu-inference checkout."""
    source_root = Path(source_root).expanduser().resolve()
    module_root = source_root.joinpath(*RPA_V3_PACKAGE.split("."))
    missing = [name for name in RPA_V3_SOURCE_SHA256
               if not (module_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{module_root} is missing {missing}; --rpa-source must be a "
            f"tpu-inference checkout at {RPA_V3_GIT_SHA} (for the default, run "
            f"git submodule update --init third_party/tpu-inference).")
    for name, expected in RPA_V3_SOURCE_SHA256.items():
        digest = hashlib.sha256((module_root / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                f"{module_root / name} does not match tpu-inference "
                f"{RPA_V3_GIT_SHA}; got sha256 {digest}.")
    return module_root


def load_rpa_v3(source_root: Path):
    """The pinned RPA v3 kernel module (ragged_paged_attention,
    get_kv_cache_shape), imported under stub parent packages so no
    tpu-inference install (or its vLLM dependency) is needed."""
    module_root = validate_rpa_source(source_root)
    if "tpu_inference" in sys.modules:
        raise RuntimeError(
            "tpu_inference was imported before the pinned RPA v3 source; "
            "refusing to benchmark an ambiguous baseline.")

    parts = RPA_V3_PACKAGE.split(".")
    for depth in range(1, len(parts) + 1):
        name = ".".join(parts[:depth])
        package = types.ModuleType(name)
        path = (module_root if depth == len(parts)
                else module_root.parents[len(parts) - depth - 1])
        package.__path__ = [str(path)]
        package.__package__ = name
        sys.modules[name] = package
    return importlib.import_module(f"{RPA_V3_PACKAGE}.kernel")


def vllm_page_size(max_model_len: int, max_num_seqs: int) -> int:
    """The KV page size vLLM on TPU uses when none is specified.

    PallasAttentionBackend.get_page_size, raised to get_min_page_size
    (tpu_inference/layers/vllm/backends/flash_attn.py at RPA_V3_GIT_SHA): split
    max_model_len into ~16 pages within [16, 256], 16 past 8192 tokens, and
    never so small that the per-request page table overflows SMEM.
    """
    # Note (david): flywheel_tpu imports jax; kept local so a bench that
    # skips its cell never loads it.
    from flywheel_tpu.pallas.block_sizes import next_pow2

    if max_model_len > 8192:
        page_size = 16
    else:
        page_size = min(max(next_pow2(max_model_len) // 16, 16), 256)
    max_pages_per_seq = 1024 * 1024 // 2 // max_num_seqs // 4
    min_page_size = next_pow2(-(-max_model_len // max_pages_per_seq))
    return max(page_size, min_page_size)


def self_test():
    assert vllm_page_size(1024, 8) == 64
    assert vllm_page_size(8192, 16) == 256
    assert vllm_page_size(65536, 1) == 16
    assert vllm_page_size(131072, 128) == 128
    print("self-test ok")


if __name__ == "__main__":
    self_test()
