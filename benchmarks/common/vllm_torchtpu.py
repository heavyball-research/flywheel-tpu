"""SHA-pinned loader for vLLM TorchTPU's kernels: batched RPA, the kernel its
`--attention-backend CUSTOM` serves softmax attention with, and the GDN v3
kernel it serves GDN and KDA linear attention with.

--vllm-torchtpu-source defaults to the third_party/vllm-torchtpu submodule
(git submodule update --init third_party/vllm-torchtpu). Every
src/vllm_torchtpu file either kernel imports is verified against its SHA-256
at VLLM_TORCHTPU_GIT_SHA, the submodule pin, and the kernels are imported
through the real vllm_torchtpu package before JAX starts its backend, so the
import-time JAX patches and libtpu flag defaults vLLM TorchTPU serves with
apply here too. Those files need only JAX; no vLLM or torch install. The
patches are process-wide, so flywheel is timed in a process of its own.

    PYTHONPATH=. uv run --no-sync python benchmarks/common/vllm_torchtpu.py   # host-only self-test
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

DEFAULT_VLLM_TORCHTPU_SOURCE = Path(__file__).resolve().parents[2] / (
    "third_party") / "vllm-torchtpu"
VLLM_TORCHTPU_GIT_SHA = "fe86920a4c851ed80c82ad78ca9ea7cc88add3a6"
BATCHED_RPA_PACKAGE = "vllm_torchtpu.kernels.experimental.batched_rpa"
GDN_V3_PACKAGE = "vllm_torchtpu.kernels.gdn.v3"
# Both kernels' whole vllm_torchtpu import closure, relative to
# src/vllm_torchtpu.
VLLM_TORCHTPU_SOURCE_SHA256 = {
    "__init__.py":
        "a1fc9f8de9942c0e27dfa639e4acfa59b48aa0443189eb33ff0609f1f14cba3d",
    "env_override.py":
        "564eff39afb19621938f51ca67590ca6eaac3d9a1973e08cf537dcece08a3061",
    "envs.py":
        "51fe1c89a983ba03bd4b7b049c8a7676b6afcb155a22fc0f2b0f0db8a81444fa",
    "kernels/__init__.py":
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "kernels/experimental/__init__.py":
        "43955acccf4b719dbab4dceb4c9437274dc864bf3d15b22c252cd9d010255c7d",
    "kernels/experimental/batched_rpa/__init__.py":
        "dae715a4f8e5c127d40a795ddf630d294563ac44234e333077f4aa9cbeae3e05",
    "kernels/experimental/batched_rpa/bref_override.py":
        "447be6f37e5b2bf5632a0bd9f8e9724e4b9373c2f6bf452fa9165e8dccb5b380",
    "kernels/experimental/batched_rpa/configs.py":
        "dfa7df3e22eda2ae12d93b56a02facef4d7c0c0bfbeff03af4c9dc6253b408fd",
    "kernels/experimental/batched_rpa/flash_attention.py":
        "b490517cfc503ccc298d5d511469370730fbe50c82288048337810f54dedb804",
    "kernels/experimental/batched_rpa/kernel.py":
        "522246e55347638f05aa060722612294ca54bc63631c78d5aa74a53cc93718e7",
    "kernels/experimental/batched_rpa/pipeline.py":
        "ee702226a72f5db5eab4bdbe90fefef1ec43a4cb027105b8923276880a53ef34",
    "kernels/experimental/batched_rpa/schedule.py":
        "3e77d06c7669ea83869f3fa19cacc98608c891357517b91a736551335c2a429b",
    "kernels/experimental/batched_rpa/schedule_cp.py":
        "fab697e2e4af64f58704f8056431140a527c418bd99e43b40c0f2e843e35b577",
    "kernels/experimental/batched_rpa/stitch_utils.py":
        "221b32d923a22dd12dd5d2a268defceb62936d237cc035bd0f3962e988d3e9ce",
    "kernels/experimental/batched_rpa/utils.py":
        "1c917940957aef3c04f24e6125b78a31e1229c2221dbf6ad6dd2c519947858d5",
    "kernels/experimental/batched_rpa/wrapper.py":
        "67380581df5ac1bdc67e403e36b264c9248fcc7946814653a2c099eafdb782ff",
    "kernels/gdn/__init__.py":
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "kernels/gdn/v3/__init__.py":
        "43955acccf4b719dbab4dceb4c9437274dc864bf3d15b22c252cd9d010255c7d",
    "kernels/gdn/v3/compute_conv1d.py":
        "f4f4bccb0d553ddf2ee9767e3c83b38ad1e839f6f8e8e74c7d9f70f14e5603c7",
    "kernels/gdn/v3/compute_gdn.py":
        "9d138ad1dca9d22d415321a159614ea52dc303a14bc42eaf58ec57f8819612b3",
    "kernels/gdn/v3/compute_kda.py":
        "e0565c2c9dfd6a5c501893f5af2336fc6676f717637b77a44bcfe272ab87b018",
    "kernels/gdn/v3/config.py":
        "cade8aaf377f549b8f5fdaea4195bf34aac75a04716872c5d5eea52473e075c8",
    "kernels/gdn/v3/memory_ref.py":
        "5491f7811cd09f88a8ed78258dfcbcc39f73630dbf28c0aa44e805004b1807d9",
    "kernels/gdn/v3/metadata.py":
        "c425de2d55d4db23cb1d7171ed9f03e05eabed74df370b0ba39a9264767e4210",
    "kernels/gdn/v3/tiling.py":
        "36bba1bdf572efd19468b48488cbebf3321da84f778cfe82f21136a2200bce33",
    "kernels/gdn/v3/vmem_ldst.py":
        "42e3e57832298993644d789bb0fccdd1eb7ac4232d0ef2d08f246143525326fa",
    "kernels/gdn/v3/wrapper.py":
        "7a134197d3ed3b40d8753a683e3d5bf9208ced8aa6cb6d730144a5f41a21900e",
    "kernels/typed_ldst.py":
        "accdb5db52dbdf541497e5dd88750e8ea92ff21dbb6430a4d7a82c597f874834",
    "logger.py":
        "2505160410a746207d8a406b64ac26e2b8fb6090d58dc1c726ba25ccb5650136",
    "patch_registry.py":
        "21e631bb1f98477c89e567c1bd0e425da81e42be1844213f9d4257b2004e0fa0",
}
# The page size vLLM TorchTPU allocates under --attention-backend CUSTOM when
# --block-size is not given (PallasBatchedRPAAttentionBackend.
# get_preferred_block_size at VLLM_TORCHTPU_GIT_SHA).
CUSTOM_PAGE_SIZE = 256
# JAX device_kind -> pltpu.ChipVersion value, for get_kv_cache_shape, which
# vLLM TorchTPU resolves from the accelerator type instead of a live device.
CHIP_VERSIONS = {"TPU v6 lite": "v6e", "TPU7x": "7x"}


def validate_vllm_torchtpu_source(source_root: Path) -> Path:
    """The src/vllm_torchtpu directory inside a vLLM TorchTPU checkout."""
    source_root = Path(source_root).expanduser().resolve()
    package_root = source_root / "src" / "vllm_torchtpu"
    missing = [name for name in VLLM_TORCHTPU_SOURCE_SHA256
               if not (package_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{package_root} is missing {missing}; --vllm-torchtpu-source "
            f"must be a vLLM TorchTPU checkout at {VLLM_TORCHTPU_GIT_SHA} (for "
            f"the default, run git submodule update --init "
            f"third_party/vllm-torchtpu).")
    for name, expected in VLLM_TORCHTPU_SOURCE_SHA256.items():
        digest = hashlib.sha256((package_root / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                f"{package_root / name} does not match vLLM TorchTPU "
                f"{VLLM_TORCHTPU_GIT_SHA}; got sha256 {digest}.")
    return package_root


def load_pinned_modules(source_root: Path, package: str, names):
    """The pinned modules package.<name> for each of names.

    Call it before JAX starts its backend: importing vllm_torchtpu sets the
    LIBTPU_INIT_ARGS defaults vLLM TorchTPU serves with (among them the
    dynamic SMEM negotiation batched RPA's scheduling metadata needs), and
    libtpu reads them only once.
    """
    from jax._src import xla_bridge

    package_root = validate_vllm_torchtpu_source(source_root)
    if "vllm_torchtpu" in sys.modules:
        raise RuntimeError(
            "vllm_torchtpu was imported before the pinned vLLM TorchTPU "
            "source; refusing to benchmark an ambiguous baseline.")
    if xla_bridge.backends_are_initialized():
        raise RuntimeError(
            "JAX started its backend before vLLM TorchTPU was loaded, so its "
            "LIBTPU_INIT_ARGS defaults would not apply.")
    sys.path.insert(0, str(package_root.parent))
    modules = tuple(importlib.import_module(f"{package}.{name}")
                    for name in names)
    loaded = Path(sys.modules["vllm_torchtpu"].__file__).resolve()
    if loaded.parent != package_root:
        raise RuntimeError(
            f"vllm_torchtpu resolved to {loaded}, not {package_root}.")
    return modules


def load_batched_rpa(source_root: Path):
    """The pinned batched RPA (wrapper, configs) modules."""
    return load_pinned_modules(source_root, BATCHED_RPA_PACKAGE,
                               ("wrapper", "configs"))


def load_gdn_v3(source_root: Path):
    """The pinned GDN v3 (wrapper, config) modules."""
    return load_pinned_modules(source_root, GDN_V3_PACKAGE,
                               ("wrapper", "config"))


def chip_version(device_kind: str):
    """The pltpu.ChipVersion of the chip JAX calls device_kind."""
    from jax.experimental.pallas import tpu as pltpu

    if device_kind not in CHIP_VERSIONS:
        raise ValueError(
            f"no chip version for {device_kind!r}; add it to CHIP_VERSIONS.")
    return pltpu.ChipVersion(CHIP_VERSIONS[device_kind])


def self_test():
    validate_vllm_torchtpu_source(DEFAULT_VLLM_TORCHTPU_SOURCE)
    print("self-test ok")


if __name__ == "__main__":
    self_test()
