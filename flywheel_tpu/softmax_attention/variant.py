# Copyright 2026 heavyball-research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Elementwise math of the softmax attention builds.

A Variant fixes what one build does to its scores and outputs, apart from
their layout (traits.py): the softmax and Q scales, the softcap, the causal or
window mask with its causal offset, learnable sinks, the K and V descales and
whether lse is returned. Every field is static, so a Variant can key a
functools cache or be a jit static; the traced values a build may carry (sink
logits, the K and V descales, a runtime causal offset) stay kernel operands,
which the kernel hands its hooks as VariantOperands. Rotary is not part of a
Variant: it stages Q and K rather than transforming scores.

The forward building blocks (fwd_blocks.py) call the hooks at fixed points:
transform_q on each staged Q tile (wait_q_row), transform_logits on each QK
fragment (qk), mask_logits where the static mask applies (masked_scores),
fold_sink and lse at a row's end (fold_out_sinks, finish_row_lse), and
output_scale where the output is normalized (finish_row_out). The decode
kernel (kernels/decode.py) calls transform_logits on each score fragment and
fold_sink at a row's end; its caller scales Q, and it keeps its own window
mask, output scale and lse, whose op order differs. The backward
(kernels/bwd.py) reads softmax_scale and the causal mask with its causal
offset, and check_backward rejects every other feature it lacks.

Adding a Variant feature (a new score term such as ALiBi, say) touches:
1. A static field with a default that means "off", checked in __post_init__
   (Variants key caches, so every field is a hashable Python value). A traced
   value also gets a VariantOperands field and a check_operands entry; each
   entry kernel passes it as an SMEM operand (see sink_logits and
   descaled_q_scale).
2. The hooks. transform_logits sees each QK fragment before any mask, but
   only its values: no query or key position and no head. mask_logits runs
   only on fragments the schedule flags as partial (under lax.cond), never on
   full blocks or in the per-seq builds, which take without_mask() and mask
   in their schedule. A position- or head-dependent term therefore needs a
   hook that fwd_blocks.qk hands its fragment's place (its KVBlock's q_tok
   and kv_tok, the row's base_head and g; a per-seq row's diagonal is
   k = q + its ScheduleRow's bottom_right_offset in packed coordinates).
3. The decode kernel, whose score rows interleave the heads of a head group,
   calls only transform_logits and fold_sink, and keeps its own mask, output
   scale and lse.
4. The backward calls no hook: name the feature in check_backward unless
   kernels/bwd.py and bwd_pipeline.py learn it.
5. dispatch: matches_tuned_table keeps a non-default field off the tuned
   blocks until it is measured and named in TUNED_TABLE_FIELDS. The decode
   table (dispatch.resolve_kvcache_config) keys on the left window bound and
   return_lse only, so a decode feature that changes the best tiles needs its
   own key there.
6. Entry checks that limit where a feature runs, like check_paged_window and
   make_flash_attn_mha's runtime_causal_offset check.
"""

import dataclasses
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from flywheel_tpu.softmax_attention.config import DEFAULT_MASK_VALUE, LOG2E
from flywheel_tpu.softmax_attention.softmax import fold_sink

# The Variant fields a tuned block entry may vary in (matches_tuned_table).
TUNED_TABLE_FIELDS = ("softmax_scale", "causal", "has_sinks", "return_lse")


def fused_q_scale(softmax_scale: float, softcap: float) -> float:
    # Note (david): the kernel softmax runs in log2 units, so log2(e) rides in
    # the fused Q scale. With softcap, Q takes softmax_scale / softcap instead
    # and the kernel applies log2(e) after its tanh.
    if softcap == 0.0:
        return softmax_scale * LOG2E
    else:
        return softmax_scale / softcap


class VariantOperands(NamedTuple):
    """The traced values a Variant's hooks take in one kernel run, each set
    exactly when its Variant flag is.

    causal_offset: runtime_causal_offset's int32 scalar, read from SMEM.
    q_scale / v_scale: has_kv_scales' f32 scalars, the Q scale times K's
      descale (Variant.descaled_q_scale) and V's descale.
    sinks_ref: has_sinks' SMEM (nheads,) float32 sink logits in log2 units.
    """

    causal_offset: jax.Array | None = None
    q_scale: jax.Array | None = None
    v_scale: jax.Array | None = None
    sinks_ref: jax.Array | None = None


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
        # A Variant keys functools caches and jit statics, so every field must
        # be a hashable Python value; make_variant converts the user's.
        for name in ("softmax_scale", "softcap"):
            if type(getattr(self, name)) is not float:
                raise TypeError(
                    f"{name} must be a Python float; got"
                    f" {type(getattr(self, name)).__name__}."
                )
        for name in (
            "causal",
            "runtime_causal_offset",
            "has_sinks",
            "has_kv_scales",
            "return_lse",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")
        if type(self.window_size) is not tuple or len(self.window_size) != 2:
            raise TypeError(
                "window_size must be a (left, right) tuple; got"
                f" {self.window_size!r}."
            )
        if type(self.causal_offset) is not int:
            raise TypeError(
                "causal_offset must be a Python int (a traced offset is"
                " runtime_causal_offset's operand); got"
                f" {type(self.causal_offset).__name__}."
            )
        if self.runtime_causal_offset and self.causal_offset != 0:
            raise ValueError(
                "a runtime causal offset replaces the static one, so"
                f" causal_offset must stay 0; got {self.causal_offset}."
            )
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
    def matches_tuned_table(self) -> bool:
        """Whether dispatch may use a tuned block entry for this variant.

        The tuned entries key on causal and return_lse, scale-free blocks
        serve every softmax_scale, and sink builds share the sink-free entries;
        every other field was never measured, so it must hold its default. A
        new field thus keeps its builds on the analytic blocks until it is
        measured and named in TUNED_TABLE_FIELDS.
        """
        return all(
            getattr(self, field.name) == field.default
            for field in dataclasses.fields(self)
            if field.name not in TUNED_TABLE_FIELDS
        )

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

    def with_causal_offset(self, causal_offset: int) -> "Variant":
        """This variant with its mask's diagonal shifted by causal_offset."""
        return dataclasses.replace(self, causal_offset=causal_offset)

    def without_mask(self) -> "Variant":
        """This variant with no causal or window mask, for a build whose
        schedule applies the mask instead (the per-seq ones)."""
        return dataclasses.replace(self, causal=False, window_size=(-1, -1))

    def check_paged_window(self) -> None:
        """Raise unless the mask's right bound is 0 (causal) or unbounded: a
        paged KV cache's schedule bounds a row's keys on the right only at the
        causal diagonal."""
        if self.window[1] not in (None, 0):
            raise NotImplementedError(
                "a paged KV cache takes a window right bound of -1 (unbounded)"
                f" or 0 (causal) only; got window_size={self.window_size}."
            )

    def check_sinks_operand(self, sinks: jax.Array | None) -> None:
        """Raise unless sinks is given exactly when has_sinks."""
        if (sinks is not None) != self.has_sinks:
            raise ValueError(
                "sinks must be passed exactly when the Variant has_sinks; got"
                f" has_sinks={self.has_sinks} and"
                f" sinks={None if sinks is None else sinks.shape}."
            )

    def check_operands(self, operands: VariantOperands) -> None:
        """Assert that operands holds exactly the traced values the flags
        name."""
        assert (
            operands.causal_offset is not None
        ) == self.runtime_causal_offset, operands
        assert (operands.q_scale is not None) == self.has_kv_scales, operands
        assert (operands.v_scale is not None) == self.has_kv_scales, operands
        assert (operands.sinks_ref is not None) == self.has_sinks, operands

    def descaled_q_scale(self, k_descale: jax.Array) -> jax.Array:
        """The q_scale operand of a has_kv_scales build: K's descale joins the
        Q scale, since it multiplies every score."""
        return self.q_scale * k_descale

    def mask_offset(self, operands: VariantOperands) -> int | jax.Array:
        """The causal offset the mask shifts query positions by: the static
        causal_offset, or the runtime scalar."""
        if self.runtime_causal_offset:
            return operands.causal_offset
        else:
            return self.causal_offset

    def _operand_q_scale(self, operands: VariantOperands) -> float | jax.Array:
        if operands.q_scale is None:
            return self.q_scale
        else:
            return operands.q_scale

    def scales_q(self, operands: VariantOperands) -> bool:
        """Whether transform_q changes a Q tile: a static 1.0 leaves it
        untouched."""
        q_scale = self._operand_q_scale(operands)
        return isinstance(q_scale, jax.Array) or q_scale != 1.0

    def transform_q(self, q: jax.Array, operands: VariantOperands) -> jax.Array:
        """A staged Q tile scaled by the Q scale (K's descale included), in
        f32 and cast back to q's dtype."""
        q_scale = self._operand_q_scale(operands)
        return (q.astype(jnp.float32) * q_scale).astype(q.dtype)

    def transform_logits(self, logits: jax.Array) -> jax.Array:
        """The softcap on a QK fragment's log2-unit logits."""
        if self.log2_softcap:
            return jnp.tanh(logits) * self.log2_softcap
        else:
            return logits

    def mask_logits(
        self,
        scores: jax.Array,
        q_first: int | jax.Array,
        k_first: int | jax.Array,
        tile_shape: tuple[int, int],
    ) -> jax.Array:
        """scores with the causal / window mask applied to a token tile whose
        rows are queries from q_first, already shifted by the causal offset,
        and whose columns are keys from k_first."""
        left, right = self.window
        q_ids = q_first + lax.broadcasted_iota(jnp.int32, tile_shape, 0)
        k_ids = k_first + lax.broadcasted_iota(jnp.int32, tile_shape, 1)
        if left is None:
            left_masked = scores
        else:
            left_masked = jnp.where(
                q_ids - left <= k_ids, scores, DEFAULT_MASK_VALUE
            )
        if right is None:
            return left_masked
        else:
            return jnp.where(
                k_ids <= q_ids + right, left_masked, DEFAULT_MASK_VALUE
            )

    def sink_logits(self, sinks: jax.Array) -> jax.Array:
        """The sink logits as sinks_ref holds them: float32, in the log2 units
        of the kernel's scores."""
        return sinks.astype(jnp.float32) * LOG2E

    def fold_sink(
        self, row_max: jax.Array, row_sum: jax.Array, sink: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """softmax.fold_sink: a finished row's (anchor, row_sum, out_scale)
        with its sink logit joined."""
        return fold_sink(row_max, row_sum, sink)

    def lse(
        self,
        row_max: jax.Array,
        row_sum: jax.Array,
        sink: jax.Array | None = None,
    ) -> jax.Array:
        """The natural-log lse of a finished row's static-anchor state, its
        sink folded in first when given."""
        if sink is not None:
            row_max, row_sum, _ = self.fold_sink(row_max, row_sum, sink)
        # Note (david): the static anchor keeps sum_j exp2(logit_j) ==
        # exp2(row_max) * row_sum, so lse is row_max + log2(row_sum) in log2
        # units; scale by ln(2) for the natural-log lse that is returned.
        return (row_max + jnp.log2(row_sum)) * math.log(2.0)

    def output_scale(
        self, row_sum: jax.Array, operands: VariantOperands
    ) -> jax.Array:
        """The factor that normalizes the output accumulator's rows: 1 /
        row_sum, times V's descale under has_kv_scales."""
        # Note (david): a quantized cache's V descale is one per-tensor factor,
        # folded into the softmax normalization.
        if operands.v_scale is None:
            v_scale = 1.0
        else:
            v_scale = operands.v_scale
        return v_scale / row_sum

    def check_backward(self) -> None:
        """Raise NotImplementedError for what the backward kernel lacks: it
        takes softmax_scale, causal and a static causal_offset, and nothing
        else of a Variant (return_lse only shapes the forward)."""
        if self.window_size != (-1, -1):
            unsupported = "window_size"
        elif self.softcap != 0.0:
            unsupported = "softcap"
        elif self.has_sinks:
            unsupported = "learnable_sink"
        elif self.runtime_causal_offset:
            unsupported = "a runtime causal_offset"
        elif self.has_kv_scales:
            unsupported = "k_descale / v_descale"
        else:
            return
        raise NotImplementedError(
            f"The TPU backward kernel does not support {unsupported}; that"
            " configuration is forward-only."
        )
