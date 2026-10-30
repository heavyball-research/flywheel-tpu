"""Integer helpers for sizing tiles and blocks."""

from collections.abc import Sequence


def next_pow2(value: int) -> int:
    return 1 << max(0, int(value) - 1).bit_length()


def round_up(value: int, alignment: int) -> int:
    """Smallest multiple of alignment that is >= value."""
    return -(-value // alignment) * alignment


def pick_tile(candidates: Sequence[int], block: int) -> int:
    """Largest of the descending candidates dividing block, else block
    itself."""
    return next((tile for tile in candidates if block % tile == 0), block)
