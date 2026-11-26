# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The packed-sequence rows of the H3 transformer forward (RFC #420 ``packed_index_copy``,
``packed_index_select``).

Diffusers' ``MiniMaxH3Transformer3DModel.forward`` scatters the text, video and audio rows into the
packed sequence inline (``new_zeros`` and three ``index_copy`` calls) and selects the video and
audio rows after the output heads (``index_select``). Neither is a submodule, so the
module-forward rebinding of ``h3_full_model`` cannot reach them. The candidate can bind the
input scatter to ``packed_index_copy`` (#545) and the two output gathers to the shared
``index_select`` (#663). Each binding is optional so a missing scatter kernel still allows
the gather to run, and vice versa.

``packed_index_copy`` is pure data movement, one writer per output row and no arithmetic, so the
forward and every gradient are bitwise the provider's (``tests/models/minimax_h3/
test_h3_packed_forward.py``). The copy is guarded: ``bind`` accepts only the provider forwards in
``KNOWN_FORWARDS`` (by the sha256 of their source) and refuses any other, which then keeps running
unchanged. The two known ones differ only in how they read a module's dtype.
"""

from __future__ import annotations

import hashlib
import inspect
from contextlib import contextmanager
from functools import partial
from types import MethodType

import torch

MODALITY_NUM = 3  # MINIMAX_H3_MODALITY_NUM
# sha256 of ``inspect.getsource`` of the provider forward -> how it reads a module's dtype.
KNOWN_FORWARDS = {
    # diffusers@f53d552 (the pinned provider): ``module.weight.dtype``
    "f0d5dd4b76be39ae0e10b19e26d2521cbbf66d2cb0dfb36f01ce75f0c3b1d26e": "weight",
    # diffusers 0.43.0 (the verl-omni run environment): ``get_parameter_dtype(module)``
    "ae200f2868ad388f1f40fccb7529f39ec3bc00c999725babef1f6d73d05361d4": "parameter",
}
ROWS = {"packed_index_copy": "row", "packed_index_select": "row"}


class UnknownForwardError(RuntimeError):
    """The model's forward is not one of ``KNOWN_FORWARDS``; it is left unchanged."""


def forward_source_sha256(model_cls) -> str:
    """sha256 of the forward's source; ``UnknownForwardError`` if the source cannot be read."""

    try:
        source = inspect.getsource(inspect.unwrap(model_cls.forward))
    except (OSError, TypeError) as exc:  # not file-backed (generated, compiled, a C function)
        raise UnknownForwardError(
            f"cannot read {model_cls.__name__}.forward source: {exc}"
        ) from exc
    return hashlib.sha256(source.encode()).hexdigest()


def _dtype_reader(variant: str):
    if variant == "weight":
        return lambda module: module.weight.dtype
    from diffusers.models.modeling_utils import get_parameter_dtype

    return get_parameter_dtype


def _packed_forward(model_cls, pack, variant, *, select=None):
    """Build the provider forward with optional input scatter and output selection operators."""

    module = inspect.getmodule(inspect.unwrap(model_cls.forward))
    output_cls = module.MiniMaxH3TransformerOutput
    dtype_of = _dtype_reader(variant)
    # The H3 packer assigns each video/audio token its own sequence position.
    # Both output projections select along S in [B, S, D_out], with distinct indices.
    select_rows = (
        partial(torch.index_select, dim=1)
        if select is None
        else partial(select, dim=1, unique_indices=True)
    )

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        attention_kwargs=None,
        return_dict=True,
    ):
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}."
            )
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors matching "
                f"`position_ids`, got {list(token_tags.shape)} and {list(timestep_indices.shape)} "
                f"for seq_len={sequence_length}."
            )
        rotary_emb = self.rope(position_ids)
        video_embeds = self.proj_in(hidden_states.to(dtype_of(self.proj_in)))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(dtype_of(self.audio_proj_in)))
        text_embeds = self.context_embedder(
            encoder_hidden_states.to(dtype_of(self.context_embedder))
        )
        text_embeds = self.token_refiner(text_embeds)
        if pack is None:
            # Keep the provider's write order and casts when only output selection is bound.
            hidden_states = text_embeds.new_zeros(
                (text_embeds.shape[0], sequence_length, text_embeds.shape[-1])
            )
            hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
            hidden_states = hidden_states.index_copy(
                1, video_indices, video_embeds.to(text_embeds.dtype)
            )
            hidden_states = hidden_states.index_copy(
                1, audio_indices, audio_embeds.to(text_embeds.dtype)
            )
        else:
            # The CUDA scatter requires contiguous views. Selection accepts the original strides.
            hidden_states = pack(
                text_embeds.contiguous(),
                video_embeds.contiguous(),
                audio_embeds.contiguous(),
                text_indices.contiguous(),
                video_indices.contiguous(),
                audio_indices.contiguous(),
                sequence_length,
            )
        temb = self.time_proj(timestep)
        temb = self.time_embedder(
            temb.to(
                self.time_embedder.linear_1.weight.dtype
                if variant == "weight"
                else dtype_of(self.time_embedder)
            )
        )
        adaln_indices = timestep_indices * MODALITY_NUM + token_tags
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, temb, adaln_indices, rotary_emb
                )
            else:
                hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb)
        hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(
            dtype_of(self.proj_out)
        )
        video_output = select_rows(self.proj_out(hidden_states), index=video_indices)
        audio_output = select_rows(self.audio_proj_out(hidden_states), index=audio_indices)
        if not return_dict:
            return (video_output, audio_output)
        return output_cls(sample=video_output, audio_sample=audio_output)

    from diffusers.utils import apply_lora_scale

    return apply_lora_scale("attention_kwargs")(forward)


@contextmanager
def bind(model, pack, *, select=None):
    """Run the known H3 forward with the supplied packed-sequence operators.

    Raises ``UnknownForwardError`` (before changing anything) if the model's forward is not a
    known provider forward, or if the instance already has its own ``forward`` (a caller's wrapper,
    e.g. a record / replay hook, which this would bypass). Restores the forward on exit.
    """

    if "forward" in model.__dict__:
        raise UnknownForwardError("the model's forward is already wrapped on the instance")
    digest = forward_source_sha256(type(model))
    variant = KNOWN_FORWARDS.get(digest)
    if variant is None:
        raise UnknownForwardError(
            f"{type(model).__name__}.forward source {digest[:12]} is not a known provider forward"
        )
    model.forward = MethodType(_packed_forward(type(model), pack, variant, select=select), model)
    try:
        yield
    finally:
        model.__dict__.pop("forward", None)
