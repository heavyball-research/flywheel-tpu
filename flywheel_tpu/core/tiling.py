"""Integer helpers for sizing tiles and blocks."""


def round_up(value: int, alignment: int) -> int:
    """Smallest multiple of alignment that is >= value."""
    return -(-value // alignment) * alignment
