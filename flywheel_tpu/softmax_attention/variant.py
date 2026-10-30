"""Elementwise math of the softmax attention builds.

A Variant fixes what one build does to its scores and outputs, apart from
their layout (traits.py): the softmax and Q scales, the softcap, the causal or
window mask with its causal offset, learnable sinks, the K and V descales and
whether lse is returned. Every field is static, so a Variant can key a
functools cache or be a jit static; the traced values a build may carry (sink
logits, the K and V descales, a runtime causal offset) stay kernel operands.
Rotary is not part of a Variant: it stages Q and K rather than transforming
scores.
"""

import dataclasses

from .config import LOG2E


def fused_q_scale(softmax_scale: float, softcap: float) -> float:
    # Note (david): the kernel softmax runs in log2 units, so log2(e) rides in
    # the fused Q scale. With softcap, Q takes softmax_scale / softcap instead
    # and the kernel applies log2(e) after its tanh.
    if softcap == 0.0:
        return softmax_scale * LOG2E
    else:
        return softmax_scale / softcap


@dataclasses.dataclass(frozen=True, kw_only=True)
class Variant:
    """Static elementwise math of one attention build.

    softmax_scale multiplies q @ k^T, and softcap > 0 applies
    tanh(scores / softcap) * softcap to the scaled scores. causal and
    window_size are flash_attn's: window_size is the (left, right) reach
    around the diagonal, -1 unbounded, and causal forces the right one to 0.
    The diagonal is shifted by causal_offset, so query i sees key j iff
    i + causal_offset - left <= j <= i + causal_offset + right.
    runtime_causal_offset moves that offset into a device-side int32 scalar
    operand (dense causal only; causal_offset stays 0). has_sinks: the build
    takes (nheads,) sink logits, each joining its head's softmax denominator.
    has_kv_scales: the build takes K and V descales as runtime scalars; K's
    multiplies every score and V's the output. return_lse adds the natural-log
    lse output.
    """

    softmax_scale: float
    softcap: float = 0.0
    causal: bool = False
    window_size: tuple[int, int] = (-1, -1)
    causal_offset: int = 0
    runtime_causal_offset: bool = False
    has_sinks: bool = False
    has_kv_scales: bool = False
    return_lse: bool = False

    def __post_init__(self) -> None:
        if type(self.return_lse) is not bool:
            raise TypeError("return_lse must be a bool.")
        if self.softcap < 0.0:
            raise ValueError(
                f"softcap must be non-negative; got {self.softcap}."
            )

    @property
    def q_scale(self) -> float:
        """The scale applied in-kernel to each staged Q tile (fused_q_scale):
        the softmax runs in exp2, so Q reaches the QK matmul multiplied by
        log2(e), or by 1 / softcap under a softcap."""
        return float(fused_q_scale(self.softmax_scale, self.softcap))

    @property
    def log2_softcap(self) -> float:
        """The softcap in the log2 units of the kernel's scores."""
        # Note (david): Q carries softmax_scale / softcap rather than log2(e),
        # so the log2(e) the exp2 softmax needs is folded into the tanh output
        # scale; 0.0 stays 0.0, so the kernel never traces the tanh.
        return self.softcap * LOG2E

    @property
    def window(self) -> tuple[int | None, int | None]:
        """The mask's (left, right) reach around the shifted diagonal, None
        unbounded; (None, None) masks nothing."""
        left = None if self.window_size[0] == -1 else self.window_size[0]
        # Note (david): window semantics: causal forces the right bound to 0,
        # and -1 means unbounded.
        if self.causal:
            right = 0
        elif self.window_size[1] == -1:
            right = None
        else:
            right = self.window_size[1]
        return left, right

    def with_lse(self, return_lse: bool) -> "Variant":
        """This variant, returning lse or not."""
        return dataclasses.replace(self, return_lse=return_lse)

    def check_backward(self) -> None:
        """Raise NotImplementedError for what the backward kernel lacks."""
        if self.window_size != (-1, -1):
            unsupported = "window_size"
        elif self.softcap != 0.0:
            unsupported = "softcap"
        elif self.has_sinks:
            unsupported = "learnable_sink"
        else:
            return
        raise NotImplementedError(
            f"The TPU backward kernel does not support {unsupported}; that"
            " configuration is forward-only."
        )
