"""SHA-pinned loader for tokamax's splash attention kernel, the splash baseline.

--splash-source defaults to the third_party/tokamax submodule
(git submodule update --init third_party/tokamax), pinned at SPLASH_GIT_SHA,
the revision softmax_attention/splash_tuned_v6e.json was searched on. Only
tokamax/_src/ops/experimental/tpu/splash_attention/{splash_attention_kernel,
splash_attention_mask,splash_attention_mask_info,reference}.py are loaded,
and each is verified against its SHA-256 at that pin.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import types
from pathlib import Path

DEFAULT_SPLASH_SOURCE = Path(__file__).resolve().parents[2] / "third_party" / (
    "tokamax")
SPLASH_GIT_SHA = "84e5f36a0c1a6b9c3c2f0e997a09e62038a2bf07"
SPLASH_PACKAGE = "tokamax._src.ops.experimental.tpu.splash_attention"
SPLASH_SOURCE_SHA256 = {
    "splash_attention_kernel.py":
        "aa1b4c9058402f98940c5abaa9dbaee8173284acb782b28f782f18ba6f249603",
    "splash_attention_mask.py":
        "f03e41192c76b9964e84749b973fc144860048b376867386d3f96d448a717a06",
    "splash_attention_mask_info.py":
        "d1466d2115a6fe351a6e0c8b53ac96420a7675913acbe63576a8ccba9e5d29b4",
    "reference.py":
        "c4e856879efec4c54348a578a8765938369272c20194d6ad7467c618f8c4213b",
}

# Note (david): PallasMosaicTpuSplashAttention._get_heuristics_config
# (pallas_mosaic_tpu.py at SPLASH_GIT_SHA), what tokamax runs untuned; the
# counterpart of RPA's get_default_block_sizes formula. layout is the q, k, v
# layouts, h = HEAD_DIM_MINOR and s = SEQ_MINOR; scheduler is
# use_experimental_scheduler; diag_grid 0 leaves qk/sv_diag_skip off.
HEURISTIC = {"block_q": 128, "block_kv": 128, "block_kv_compute": 128,
             "num_stacked_q_heads": 1, "layout": "hhh", "scheduler": 1,
             "diag_grid": 0}
FIELDS = ("block_q", "block_kv", "block_kv_compute", "num_stacked_q_heads",
          "layout", "scheduler", "diag_grid")


def validate_splash_source(source_root: Path) -> Path:
    """The pinned splash_attention package directory inside a tokamax checkout."""
    source_root = Path(source_root).expanduser().resolve()
    module_root = source_root.joinpath(*SPLASH_PACKAGE.split("."))
    missing = [name for name in SPLASH_SOURCE_SHA256
               if not (module_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{module_root} is missing {missing}; --splash-source must be a "
            f"tokamax checkout at {SPLASH_GIT_SHA} (for the default, run "
            f"git submodule update --init third_party/tokamax).")
    for name, expected in SPLASH_SOURCE_SHA256.items():
        digest = hashlib.sha256((module_root / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                f"{module_root / name} does not match tokamax "
                f"{SPLASH_GIT_SHA}; got sha256 {digest}.")
    return module_root


def load_splash(source_root: Path):
    """The pinned (splash_attention_kernel, splash_attention_mask) modules,
    imported under stub parent packages so no tokamax install (or its
    pydantic/jaxtyping/xprof dependencies) is needed."""
    module_root = validate_splash_source(source_root)
    if "tokamax" in sys.modules:
        raise RuntimeError(
            "tokamax was imported before the pinned splash source; refusing "
            "to benchmark an ambiguous baseline.")

    parts = SPLASH_PACKAGE.split(".")
    for depth in range(1, len(parts) + 1):
        name = ".".join(parts[:depth])
        package = types.ModuleType(name)
        path = (module_root if depth == len(parts)
                else module_root.parents[len(parts) - depth - 1])
        package.__path__ = [str(path)]
        package.__package__ = name
        sys.modules[name] = package

    # Note (david): the kernel reaches into base for these four names only,
    # and tokamax's base.py defines each as a plain re-export of reference.py,
    # so a stub base re-exporting them from the pinned reference.py suffices.
    reference = importlib.import_module(f"{SPLASH_PACKAGE}.reference")
    base = types.ModuleType(f"{SPLASH_PACKAGE}.base")
    for name in ("DEFAULT_MASK_VALUE", "SegmentIds", "SplashCustomReturnType",
                 "SplashResidualsType"):
        setattr(base, name, getattr(reference, name))
    sys.modules[base.__name__] = base
    sys.modules[SPLASH_PACKAGE].base = base

    kernel = importlib.import_module(f"{SPLASH_PACKAGE}.splash_attention_kernel")
    mask = importlib.import_module(f"{SPLASH_PACKAGE}.splash_attention_mask")
    return kernel, mask
