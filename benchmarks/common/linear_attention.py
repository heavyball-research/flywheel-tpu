"""Shared paths and throughput reporting for linear-attention benchmarks."""

from __future__ import annotations

import itertools
from pathlib import Path

import jax

BASELINE_KERNEL_DIR = Path(__file__).resolve().parents[1] / "linear_attention" / "baseline_kernel"
PARAM_SCALE = 0.1
KernelState = tuple[jax.Array, ...]
Row = dict[str, int | float | str]
Header = dict[str, dict[str, int | float | str]]


def kda_fwd_flops(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    value_dim: int,
    chunk_size: int,
    nseqs: int = 1,
) -> float:
    """Approximate forward FLOPs of the chunked KDA algorithm.

    Counts the dominant per-chunk terms at 2 FLOPs per MAC: the decay-weighted
    Aqk and L score tensors, the unit-lower-triangular solve over the combined
    [V + K + BT] right-hand side, the two state-recurrence matmuls (w @ h,
    k^T @ v) and the two output matmuls (qg @ h, A @ v). Elementwise gate / exp
    work is ignored, so the derived TFLOP/s is a model-level throughput number,
    not MXU utilization.
    """
    per_chunk = (
        4 * chunk_size * chunk_size * head_dim
        + chunk_size * chunk_size * (value_dim + head_dim + chunk_size)
        + 4 * chunk_size * head_dim * value_dim
        + 2 * chunk_size * head_dim * value_dim
        + 2 * chunk_size * chunk_size * value_dim
    )
    chunks_per_seq = (num_tokens // nseqs + chunk_size - 1) // chunk_size
    return float(nseqs * chunks_per_seq) * num_heads * per_chunk


def report_throughput(
    num_tokens: int, nseqs: int, median_ms: float, flops: float, label: str
) -> dict[str, float]:
    """Prints one sweep point and returns its tokens_per_s and approx_tflops."""
    tokens_per_s = num_tokens / median_ms * 1e3
    approx_tflops = flops / median_ms / 1e9
    print(f"  T={num_tokens:6d} nseqs={nseqs:3d}{label}  {median_ms:8.3f} ms  "
          f"{tokens_per_s / 1e6:6.2f} Mtok/s  "
          f"{approx_tflops:6.2f} TFLOP/s(approx)", flush=True)
    return {"tokens_per_s": tokens_per_s, "approx_tflops": approx_tflops}


def print_speedup_table(
    rows: list[Row], pkgs: list[str], seqlens: list[int], nseqs_list: list[int]
) -> None:
    """Prints median ms per package and each package's speedup over pkgs[0]."""
    base_pkg = pkgs[0]
    median_ms_by_run = {(row["T"], row["nseqs"], row["pkg"]): row["median_ms"] for row in rows}
    print()
    print(f"{'T':>7}{'nseqs':>7}"
          + "".join(f"{pkg + ' ms':>20}" for pkg in pkgs)
          + "".join(f"{pkg + ' x':>20}" for pkg in pkgs[1:]))
    for num_tokens, nseqs in itertools.product(seqlens, nseqs_list):
        if (num_tokens, nseqs, base_pkg) not in median_ms_by_run:
            continue
        median_ms_by_pkg = {pkg: median_ms_by_run[(num_tokens, nseqs, pkg)] for pkg in pkgs}
        print(f"{num_tokens:>7}{nseqs:>7}"
              + "".join(f"{median_ms_by_pkg[pkg]:>20.3f}" for pkg in pkgs)
              + "".join(f"{median_ms_by_pkg[base_pkg] / median_ms_by_pkg[pkg]:>19.3f}x"
                        for pkg in pkgs[1:]))
