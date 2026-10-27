"""SHA-pinned loaders for the RPA baselines: tpu-inference's RPA v3 kernel and
vllm-torchtpu's batched RPA.

--rpa-source defaults to the third_party/tpu-inference submodule
(git submodule update --init third_party/tpu-inference). Only
tpu_inference/kernels/ragged_paged_attention/v3/{kernel,util}.py are loaded,
and both are verified against their SHA-256 at RPA_V3_GIT_SHA, the submodule
pin. Those two files and the vLLM backend's flash_attn.py are unchanged from
1bd693780c5f, the revision softmax_attention/rpa_tuned_v6e.json was searched
on.

load_batched_rpa takes a vllm-torchtpu checkout (--torchtpu-source) and loads
src/vllm_torchtpu/kernels/experimental/batched_rpa, every file verified
against its SHA-256 at BATCHED_RPA_GIT_SHA.

    PYTHONPATH=. uv run --no-sync python benchmarks/common/rpa.py   # host-only self-test
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import types
from pathlib import Path

DEFAULT_RPA_SOURCE = (
    Path(__file__).resolve().parents[2] / "third_party" / "tpu-inference"
)
RPA_V3_GIT_SHA = "4420cae3615711b3b8f66bac1c802ae3f9fa3439"
RPA_V3_PACKAGE = "tpu_inference.kernels.ragged_paged_attention.v3"
RPA_V3_SOURCE_SHA256 = {
    "kernel.py": (
        "bcd21fdaff6b333903a09991d0ad57f3683fef18bf8abef449c0f10d6879609c"
    ),
    "util.py": (
        "f9a1c1f189ef0b5d9e9babaa9d221faf01f12bc9bb809fe227d17901867b56c0"
    ),
}
FIELDS = ("bq_sz", "bkv_sz", "bq_csz", "bkv_csz")
# vllm-torchtpu's batched RPA, the kernel its CUSTOM attention backend serves
# with, speculative verify windows in the decode case.
BATCHED_RPA_GIT_SHA = "6f66b7babaec12e98e241320c7f46ff406d3f7de"
BATCHED_RPA_PACKAGE = "vllm_torchtpu.kernels.experimental.batched_rpa"
BATCHED_RPA_SOURCE_SHA256 = {
    "__init__.py": (
        "dae715a4f8e5c127d40a795ddf630d294563ac44234e333077f4aa9cbeae3e05"
    ),
    "bref_override.py": (
        "447be6f37e5b2bf5632a0bd9f8e9724e4b9373c2f6bf452fa9165e8dccb5b380"
    ),
    "configs.py": (
        "dfa7df3e22eda2ae12d93b56a02facef4d7c0c0bfbeff03af4c9dc6253b408fd"
    ),
    "flash_attention.py": (
        "b490517cfc503ccc298d5d511469370730fbe50c82288048337810f54dedb804"
    ),
    "kernel.py": (
        "522246e55347638f05aa060722612294ca54bc63631c78d5aa74a53cc93718e7"
    ),
    "pipeline.py": (
        "ee702226a72f5db5eab4bdbe90fefef1ec43a4cb027105b8923276880a53ef34"
    ),
    "schedule_cp.py": (
        "fab697e2e4af64f58704f8056431140a527c418bd99e43b40c0f2e843e35b577"
    ),
    "schedule.py": (
        "3e77d06c7669ea83869f3fa19cacc98608c891357517b91a736551335c2a429b"
    ),
    "stitch_utils.py": (
        "221b32d923a22dd12dd5d2a268defceb62936d237cc035bd0f3962e988d3e9ce"
    ),
    "utils.py": (
        "1c917940957aef3c04f24e6125b78a31e1229c2221dbf6ad6dd2c519947858d5"
    ),
    "wrapper.py": (
        "67380581df5ac1bdc67e403e36b264c9248fcc7946814653a2c099eafdb782ff"
    ),
}


def validate_rpa_source(source_root: Path) -> Path:
    """The pinned v3 package directory inside a tpu-inference checkout."""
    source_root = Path(source_root).expanduser().resolve()
    module_root = source_root.joinpath(*RPA_V3_PACKAGE.split("."))
    missing = [
        name
        for name in RPA_V3_SOURCE_SHA256
        if not (module_root / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{module_root} is missing {missing}; --rpa-source must be a "
            f"tpu-inference checkout at {RPA_V3_GIT_SHA} (for the default, run "
            "git submodule update --init third_party/tpu-inference)."
        )
    for name, expected in RPA_V3_SOURCE_SHA256.items():
        digest = hashlib.sha256((module_root / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                f"{module_root / name} does not match tpu-inference "
                f"{RPA_V3_GIT_SHA}; got sha256 {digest}."
            )
    return module_root


def load_rpa_v3(source_root: Path):
    """The pinned RPA v3 kernel module (ragged_paged_attention,
    get_kv_cache_shape), imported under stub parent packages so no
    tpu-inference install (or its vLLM dependency) is needed."""
    module_root = validate_rpa_source(source_root)
    if "tpu_inference" in sys.modules:
        raise RuntimeError(
            "tpu_inference was imported before the pinned RPA v3 source; "
            "refusing to benchmark an ambiguous baseline."
        )

    parts = RPA_V3_PACKAGE.split(".")
    for depth in range(1, len(parts) + 1):
        name = ".".join(parts[:depth])
        package = types.ModuleType(name)
        path = (
            module_root
            if depth == len(parts)
            else module_root.parents[len(parts) - depth - 1]
        )
        package.__path__ = [str(path)]
        package.__package__ = name
        sys.modules[name] = package
    return importlib.import_module(f"{RPA_V3_PACKAGE}.kernel")


def load_batched_rpa(source_root: Path):
    """The pinned batched RPA wrapper module (ragged_paged_attention,
    get_kv_cache_shape) from a vllm-torchtpu checkout, loaded like load_rpa_v3:
    the package imports only jax and itself, so stub parents suffice and no
    torch_tpu or vLLM is needed."""
    source_root = Path(source_root).expanduser().resolve()
    parts = BATCHED_RPA_PACKAGE.split(".")
    module_root = source_root.joinpath("src", *parts)
    for name, expected in BATCHED_RPA_SOURCE_SHA256.items():
        path = module_root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing; --torchtpu-source must be a "
                f"vllm-torchtpu checkout at {BATCHED_RPA_GIT_SHA}."
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                f"{path} does not match vllm-torchtpu {BATCHED_RPA_GIT_SHA}; "
                f"got sha256 {digest}."
            )
    if "vllm_torchtpu" in sys.modules:
        raise RuntimeError(
            "vllm_torchtpu was imported before the pinned batched RPA source; "
            "refusing to benchmark an ambiguous baseline."
        )

    for depth in range(1, len(parts) + 1):
        name = ".".join(parts[:depth])
        package = types.ModuleType(name)
        package.__path__ = [str(source_root.joinpath("src", *parts[:depth]))]
        package.__package__ = name
        sys.modules[name] = package
    return importlib.import_module(f"{BATCHED_RPA_PACKAGE}.wrapper")


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


def unjitted_kernel(rpa):
    """ragged_paged_attention without its own jax.jit, which donates q, k and v
    as well as the cache; the benchmarks jit it themselves, donating only the
    cache, so the same q, k and v serve every call."""
    return getattr(
        rpa.ragged_paged_attention, "__wrapped__", rpa.ragged_paged_attention
    )


def request_distribution(*, num_decode: int, num_seqs: int):
    """RPA's (decode_end, prefill_end, mixed_end) as vLLM's TPU runner sends
    it: the first num_decode sequences decode, every other one is mixed."""
    import jax.numpy as jnp

    return jnp.array([num_decode, num_decode, num_seqs], jnp.int32)


def self_test():
    assert vllm_page_size(1024, 8) == 64
    assert vllm_page_size(8192, 16) == 256
    assert vllm_page_size(65536, 1) == 16
    assert vllm_page_size(131072, 128) == 128
    print("self-test ok")


if __name__ == "__main__":
    self_test()
