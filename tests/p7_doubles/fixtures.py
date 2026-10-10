# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic synthetic M10 fixtures for T05 (no real data, no RNG).

Every tensor value comes from an integer formula and is exactly representable
in BF16, so fixtures are bit-identical across machines and torch versions.
``fixture_digest`` covers raw bytes and layout metadata; the expected digests
are pinned in ``fixtures/manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import torch

DOUBLE_OF = "T01 synthetic M10 fixtures and checksums"
FIXTURE_SCHEMA_VERSION = "test-double.t05_m10_fixture.v0"


@dataclass(frozen=True)
class M10Fixture:
    """Global (canonical-order) M10 inputs plus the declared distributed layout."""

    name: str
    hidden: torch.Tensor  # BF16 [N, H], canonical global token order
    lm_head_weight: torch.Tensor  # BF16 [V_padded, H], full padded vocabulary
    target_ids: torch.Tensor  # INT64 [N]
    active_mask: torch.Tensor  # BOOL [N]
    real_vocab: int
    padded_vocab: int
    sample_lengths: tuple[int, ...]
    tp_world: int = 1
    cp_world: int = 1
    vocab_shards: tuple[tuple[int, int], ...] = ()  # declared [start, end) per TP rank
    present_tp_ranks: tuple[int, ...] = ()
    canonical_tp: Optional[int] = None
    notes: str = ""
    schema_version: str = FIXTURE_SCHEMA_VERSION

    @property
    def num_tokens(self) -> int:
        return int(self.hidden.size(0))

    def cp_local_rows(self, cp_rank: int) -> tuple[int, ...]:
        """Canonical row index for each local row of ``cp_rank`` (Megatron zigzag)."""

        rows: list[int] = []
        offset = 0
        for length in self.sample_lengths:
            if self.cp_world == 1:
                rows.extend(range(offset, offset + length))
            else:
                width = length // (2 * self.cp_world)
                first = cp_rank
                second = 2 * self.cp_world - 1 - cp_rank
                rows.extend(range(offset + first * width, offset + (first + 1) * width))
                rows.extend(range(offset + second * width, offset + (second + 1) * width))
            offset += length
        return tuple(rows)

    def row_identity(self) -> tuple[tuple[int, int], ...]:
        """(sample_index, position) of every canonical row."""

        identity: list[tuple[int, int]] = []
        for sample, length in enumerate(self.sample_lengths):
            identity.extend((sample, position) for position in range(length))
        return tuple(identity)


def _exact_bf16(values: torch.Tensor) -> torch.Tensor:
    converted = values.to(torch.bfloat16)
    if not torch.equal(converted.float(), values.float()):
        raise AssertionError("fixture value is not exactly representable in bf16")
    return converted


def _hidden(tokens: int, width: int) -> torch.Tensor:
    i = torch.arange(tokens).reshape(-1, 1)
    j = torch.arange(width).reshape(1, -1)
    return _exact_bf16((((i * 31 + j * 17) % 97) - 48).float() / 64.0)


def _weight(vocab: int, width: int) -> torch.Tensor:
    v = torch.arange(vocab).reshape(-1, 1)
    j = torch.arange(width).reshape(1, -1)
    return _exact_bf16((((v * 13 + j * 7) % 89) - 44).float() / 128.0)


def _targets(tokens: int, real_vocab: int) -> torch.Tensor:
    return (torch.arange(tokens, dtype=torch.int64) * 7 + 3) % real_vocab


def _even_shards(padded_vocab: int, world: int) -> tuple[tuple[int, int], ...]:
    width = padded_vocab // world
    return tuple((rank * width, (rank + 1) * width) for rank in range(world))


def _base(
    name: str,
    *,
    sample_lengths: tuple[int, ...],
    width: int = 16,
    real_vocab: int = 64,
    padded_vocab: int = 64,
    tp_world: int = 1,
    cp_world: int = 1,
    notes: str = "",
) -> M10Fixture:
    tokens = sum(sample_lengths)
    active = torch.tensor([(index % 5) != 4 for index in range(tokens)], dtype=torch.bool)
    return M10Fixture(
        name=name,
        hidden=_hidden(tokens, width),
        lm_head_weight=_weight(padded_vocab, width),
        target_ids=_targets(tokens, real_vocab),
        active_mask=active,
        real_vocab=real_vocab,
        padded_vocab=padded_vocab,
        sample_lengths=sample_lengths,
        tp_world=tp_world,
        cp_world=cp_world,
        vocab_shards=_even_shards(padded_vocab, tp_world),
        present_tp_ranks=tuple(range(tp_world)),
        notes=notes,
    )


def make_fixture(name: str, **kwargs: Any) -> M10Fixture:
    """Public builder for larger ad-hoc fixtures (same integer formulas; not pinned)."""

    return _base(name, **kwargs)


def _flip_one_weight_bit(weight: torch.Tensor) -> torch.Tensor:
    flipped = weight.clone()
    raw = flipped.view(torch.int16)
    raw[3, 5] ^= 1
    return flipped


def build_fixtures() -> dict[str, M10Fixture]:
    normal = _base("normal", sample_lengths=(8,), notes="TP1/CP1, real==padded vocab")
    padded = _base(
        "padded_vocab_tp2",
        sample_lengths=(8,),
        real_vocab=60,
        padded_vocab=64,
        tp_world=2,
        notes="tail [60,64) is padding; two equal TP shards",
    )
    zigzag = _base(
        "packed_varlen_zigzag_cp2",
        sample_lengths=(8, 4),
        cp_world=2,
        notes="two packed samples, Megatron zigzag CP=2",
    )
    canonical = replace(
        _base("canonical_tp4_on_tp1", sample_lengths=(8,), notes="physical TP1, canonical TP4"),
        canonical_tp=4,
    )
    wrong_shard = replace(
        _base("wrong_shard_tp2", sample_lengths=(8,), tp_world=2, notes="rank 1 declares [30,64)"),
        vocab_shards=((0, 32), (30, 64)),
    )
    drift = replace(
        _base("weight_drift", sample_lengths=(8,), notes="one LM-head bit flipped vs normal"),
        lm_head_weight=_flip_one_weight_bit(normal.lm_head_weight),
    )
    missing_rank = replace(
        _base("missing_rank_tp2", sample_lengths=(8,), tp_world=2, notes="rank 1 record absent"),
        present_tp_ranks=(0,),
    )
    zero_active = replace(
        _base("zero_active", sample_lengths=(8,), notes="no active token"),
        active_mask=torch.zeros(8, dtype=torch.bool),
    )
    fixtures = (normal, padded, zigzag, canonical, wrong_shard, drift, missing_rank, zero_active)
    return {fixture.name: fixture for fixture in fixtures}


def _tensor_bytes(value: torch.Tensor) -> bytes:
    flat = value.detach().contiguous().cpu().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


def fixture_digest(fixture: M10Fixture) -> str:
    meta = {
        "name": fixture.name,
        "schema_version": fixture.schema_version,
        "real_vocab": fixture.real_vocab,
        "padded_vocab": fixture.padded_vocab,
        "sample_lengths": list(fixture.sample_lengths),
        "tp_world": fixture.tp_world,
        "cp_world": fixture.cp_world,
        "vocab_shards": [list(pair) for pair in fixture.vocab_shards],
        "present_tp_ranks": list(fixture.present_tp_ranks),
        "canonical_tp": fixture.canonical_tp,
    }
    digest = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8"))
    for name in ("hidden", "lm_head_weight", "target_ids", "active_mask"):
        tensor = getattr(fixture, name)
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode("utf-8"))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def local_logits(fixture: M10Fixture, *, rows: tuple[int, ...], tp_rank: int) -> torch.Tensor:
    """Reference local logits (fp32 matmul, BF16 output) for CPU-only doubles."""

    start, end = fixture.vocab_shards[tp_rank]
    hidden = fixture.hidden[list(rows)].float()
    weight = fixture.lm_head_weight[start:end].float()
    return (hidden @ weight.t()).to(torch.bfloat16)


def vime_request(
    fixture: M10Fixture,
    *,
    cp_rank: int = 0,
    tp_rank: int = 0,
    temperature: Any = 1.0,
    keep_mask: Optional[torch.Tensor] = None,
    metadata_overrides: Optional[Mapping[str, Any]] = None,
    chunk_size: int = 1024,
) -> SimpleNamespace:
    """Structural double of VIME ``LinearLogpRequest`` (VIME tree f1e20b7)."""

    rows = fixture.cp_local_rows(cp_rank)
    start, end = fixture.vocab_shards[tp_rank]
    hidden = fixture.hidden[list(rows)].clone()
    metadata: dict[str, Any] = {
        "tp_rank": tp_rank,
        "tp_world_size": fixture.tp_world,
        "real_vocab_size": fixture.real_vocab,
        "padded_vocab_size": fixture.padded_vocab,
    }
    metadata.update(metadata_overrides or {})
    layout = "single" if fixture.cp_world == 1 else "zigzag"
    return SimpleNamespace(
        logits=local_logits(fixture, rows=rows, tp_rank=tp_rank),
        target_ids=fixture.target_ids[list(rows)].clone(),
        tensor_parallel_group=None,
        token_layout=SimpleNamespace(world_size=fixture.cp_world, rank=cp_rank, layout=layout),
        with_entropy=False,
        with_entropy_grad=False,
        chunk_size=chunk_size,
        log_prob_keep_mask=keep_mask,
        context=SimpleNamespace(
            hidden=hidden,
            projection=SimpleNamespace(weight=fixture.lm_head_weight[start:end].clone(), bias=None),
            vocab_partition=SimpleNamespace(
                local_start=start,
                local_size=end - start,
                real_size=fixture.real_vocab,
                padded_size=fixture.padded_vocab,
            ),
        ),
        temperature=temperature,
        metadata=metadata,
    )


__all__ = [
    "DOUBLE_OF",
    "FIXTURE_SCHEMA_VERSION",
    "M10Fixture",
    "build_fixtures",
    "fixture_digest",
    "local_logits",
    "make_fixture",
    "vime_request",
]
