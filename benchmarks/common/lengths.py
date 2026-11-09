"""Sequence-length patterns of the varlen benchmarks (FlashAttention's
benchmarks/clc_bench.py): a total of tokens split across a batch by the
pattern's weights, so every pattern at the same (total, batch) holds the same
tokens and differs only in shape.

  uniform   every sequence the same length
  spiky     one sequence takes 32 / (batch + 31) of the total, the rest split
            the remainder evenly
  longtail  lengths fall off linearly, the longest batch times the shortest
"""

from __future__ import annotations

import math

PATTERNS = ("uniform", "spiky", "longtail")


def pattern_weights(pattern: str, batch: int) -> list[float]:
    """Relative sequence lengths of a pattern."""
    if pattern == "uniform":
        return [1.0] * batch
    elif pattern == "spiky":
        return [32.0] + [1.0] * (batch - 1)
    elif pattern == "longtail":
        return [float(batch - index) for index in range(batch)]
    else:
        raise ValueError(f"unsupported pattern: {pattern}")


def split_budget(weights: list[float], total_tokens: int) -> list[int]:
    """Lengths in proportion to weights that sum exactly to total_tokens.

    Floor each share, then hand out (or take back) the remainder one token at a
    time in order of the dropped fraction.
    """
    if total_tokens < len(weights):
        raise ValueError(f"budget {total_tokens} below batch {len(weights)}")
    scaled = [w / sum(weights) * total_tokens for w in weights]
    lengths = [max(1, int(math.floor(value))) for value in scaled]
    delta = total_tokens - sum(lengths)
    order = sorted(
        range(len(weights)),
        key=lambda i: scaled[i] - math.floor(scaled[i]),
        reverse=delta > 0,
    )
    cursor = 0
    while delta != 0:
        index = order[cursor % len(order)]
        if delta > 0:
            lengths[index] += 1
            delta -= 1
        elif lengths[index] > 1:
            lengths[index] -= 1
            delta += 1
        cursor += 1
    return lengths
