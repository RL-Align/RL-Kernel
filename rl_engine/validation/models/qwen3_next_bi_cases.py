# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Batch-invariance cells for the Qwen3-Next full model (RFC #428 ``batch_invariance_ci``).

Every cell runs one *target* sequence through the native rollout, optionally
next to *companion* sequences, and compares the target bitwise - response
tokens, BF16 logits, FP32 logprobs, the embedding and every layer, sub-node
and recurrent/KV-state digest - with a *reference*: the same target, alone,
with the same prefill chunk plan. A cell passes only if nothing about the
target changed. Families:

* ``concurrency`` - N = 1, 2, 4, 8, 16, 32, 64 requests in one engine, the
  target first or last, all arriving together;
* ``interference`` - companion content, lengths and arrival (during the
  target's prefill, during its decode, after it) varied around one target;
* ``chunk`` - the target alone across prefill chunk boundaries (prompt
  lengths at and around 64, 128, 528 and 1024 tokens, chunk 64 or 128, first
  chunk 1, 63 or 65), compared with the replay engine's full-sequence
  recompute of the same tokens: the train/rollout boundary itself;
* ``packing`` - the replay engine packs the target with companions into one
  THD sequence batch (offsets and padding varied) and the target must equal
  its recompute alone.

Prompts repeat the four pinned L2 texts to any length, so a cell is fully
determined by its parameters. :func:`cells` is pinned by
``qwen3_next_bi_manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_engine.validation.models.qwen3_next_l2 import json_digest

MANIFEST = Path(__file__).with_name("qwen3_next_bi_manifest.json")
MAX_MODEL_LEN = 1040
RESPONSE_LENGTH = 16
CONCURRENCY = (1, 2, 4, 8, 16, 32, 64)
CHUNK_PROMPTS = (63, 64, 65, 127, 128, 129, 527, 528, 529, 1023, 1024)
CHUNK_SIZES = (64, 128)
FIRST_CHUNKS = (1, 63, 65)


@dataclass(frozen=True)
class Sequence:
    """One request: which pinned text it repeats, how long its prompt is, when it arrives."""

    text: int
    prompt_length: int
    max_tokens: int = RESPONSE_LENGTH
    arrival_step: int = 0
    offset: int = 0


@dataclass(frozen=True)
class Cell:
    cell_id: str
    family: str
    target: Sequence
    first_chunk: int
    chunk_size: int
    companions: tuple[Sequence, ...] = ()
    target_last: bool = False
    engine: str = "rollout"

    def chunk_plan(self):
        """The target's exact prefill chunks: first chunk, then fixed chunks, then the rest."""
        remaining, plan = self.target.prompt_length, []
        size = min(self.first_chunk, remaining)
        while remaining:
            plan.append(size)
            remaining -= size
            size = min(self.chunk_size, remaining)
        return plan

    def reference(self):
        """The same target, alone, with the same chunk plan."""
        return Cell(
            cell_id=f"ref-{self.target.text}-{self.target.prompt_length}"
            f"-{self.target.offset}-{self.first_chunk}-{self.chunk_size}",
            family="reference",
            target=self.target,
            first_chunk=self.first_chunk,
            chunk_size=self.chunk_size,
            engine=self.engine,
        )

    def to_dict(self):
        return asdict(self)


def _companions(count, *, seed, lengths=(37, 256, 512, 1000), arrival=0, tokens=RESPONSE_LENGTH):
    return tuple(
        Sequence(
            text=(seed + index) % 4,
            prompt_length=lengths[(seed + index) % len(lengths)],
            max_tokens=tokens,
            arrival_step=arrival,
            offset=7 * (index + 1),
        )
        for index in range(count)
    )


def cells():
    out = []
    target = Sequence(text=0, prompt_length=256)
    for n in CONCURRENCY:
        for last in (False, True) if n > 1 else (False,):
            out.append(
                Cell(
                    cell_id=f"concurrency-n{n:02d}-{'last' if last else 'first'}",
                    family="concurrency",
                    target=target,
                    first_chunk=64,
                    chunk_size=64,
                    companions=_companions(n - 1, seed=n),
                    target_last=last,
                )
            )
    interference = {
        "same-text": _companions(3, seed=0, lengths=(256,)),
        "short-companions": _companions(7, seed=1, lengths=(1, 2, 3, 5)),
        "long-companions": _companions(3, seed=2, lengths=(1000, 1024)),
        "mixed-lengths": _companions(15, seed=3, lengths=(8, 63, 129, 777, 1024)),
        "arrive-in-prefill": _companions(4, seed=4, arrival=2),
        "arrive-in-decode": _companions(4, seed=5, arrival=6),
        # The 256-token target prefills in four steps and decodes in fifteen: step 18
        # is its last decode.
        "arrive-at-last-decode": _companions(4, seed=6, arrival=18),
        "long-generations": _companions(4, seed=7, tokens=24),
        "single-token-generations": _companions(8, seed=8, tokens=1),
    }
    for name, companions in interference.items():
        out.append(
            Cell(
                cell_id=f"interference-{name}",
                family="interference",
                target=target,
                first_chunk=64,
                chunk_size=64,
                companions=companions,
            )
        )
    out.append(
        Cell(
            cell_id="interference-target-last-arrival",
            family="interference",
            target=Sequence(text=0, prompt_length=256, arrival_step=3),
            first_chunk=64,
            chunk_size=64,
            companions=_companions(4, seed=9),
        )
    )
    for prompt in CHUNK_PROMPTS:
        for chunk in CHUNK_SIZES:
            for first in FIRST_CHUNKS:
                out.append(
                    Cell(
                        cell_id=f"chunk-p{prompt:04d}-c{chunk}-f{first:02d}",
                        family="chunk",
                        target=Sequence(text=prompt % 4, prompt_length=prompt),
                        first_chunk=first,
                        chunk_size=chunk,
                    )
                )
    packing = {
        "two-sequences": _companions(1, seed=10),
        "four-sequences": _companions(3, seed=11),
        "eight-mixed": _companions(7, seed=12, lengths=(8, 63, 129, 777)),
        "one-token-neighbours": _companions(3, seed=13, lengths=(1,)),
    }
    for name, companions in packing.items():
        for last in (False, True):
            out.append(
                Cell(
                    cell_id=f"packing-{name}-{'last' if last else 'first'}",
                    family="packing",
                    target=target,
                    first_chunk=target.prompt_length,
                    chunk_size=target.prompt_length,
                    companions=companions,
                    target_last=last,
                    engine="replay",
                )
            )
    for cell in out:
        validate(cell)
    if len({cell.cell_id for cell in out}) != len(out):
        raise ValueError("Batch-invariance cell IDs must be unique")
    return out


def quick_cells():
    """A CI-sized subset: one cell per family, plus the largest concurrency."""
    keep = {
        "concurrency-n04-last",
        "concurrency-n64-first",
        "interference-arrive-in-decode",
        "chunk-p0129-c64-f63",
        "packing-four-sequences-last",
    }
    return [cell for cell in cells() if cell.cell_id in keep]


def validate(cell):
    sequences = (cell.target, *cell.companions)
    if any(
        s.prompt_length < 1
        or s.max_tokens < 1
        or s.prompt_length + s.max_tokens > MAX_MODEL_LEN
        or s.text not in range(4)
        or s.arrival_step < 0
        for s in sequences
    ):
        raise ValueError(f"{cell.cell_id}: a sequence does not fit the pinned rollout profile")
    if cell.target.max_tokens != RESPONSE_LENGTH:
        raise ValueError(f"{cell.cell_id}: the target always generates sixteen tokens")
    if len(sequences) > 64:
        raise ValueError(f"{cell.cell_id}: at most 64 concurrent requests are certified")
    if cell.first_chunk < 1 or cell.chunk_size < 1:
        raise ValueError(f"{cell.cell_id}: chunks must be positive")
    if cell.engine not in ("rollout", "replay"):
        raise ValueError(f"{cell.cell_id}: unknown engine {cell.engine}")


def prompt_tokens(encoded_texts, sequence):
    """The pinned text ``sequence.text`` repeated, rotated by ``offset``, cut to length."""
    row = encoded_texts[sequence.text]
    start = sequence.offset % len(row)
    rotated = row[start:] + row[:start]
    return (rotated * ((sequence.prompt_length + len(rotated) - 1) // len(rotated)))[
        : sequence.prompt_length
    ]


def manifest_payload():
    entries = [cell.to_dict() for cell in cells()]
    return {
        "schema_version": 1,
        "description": "Qwen3-Next batch-invariance cells; regenerate with "
        "python -m rl_engine.validation.models.qwen3_next_bi_cases --write",
        "cells": entries,
        "cells_sha256": hashlib.sha256(
            json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def load_manifest():
    payload = json.loads(MANIFEST.read_text())
    if payload != json.loads(json.dumps(manifest_payload())):
        raise ValueError("qwen3_next_bi_manifest.json differs from the generated cells")
    return payload


if __name__ == "__main__":
    import sys

    text = json.dumps(manifest_payload(), indent=2) + "\n"
    if "--write" in sys.argv:
        MANIFEST.write_text(text)
    elif not MANIFEST.is_file() or MANIFEST.read_text() != text:
        raise SystemExit("qwen3_next_bi_manifest.json is stale; rerun with --write")


# --------------------------------------------------------------------------- #
# Worker requests
# --------------------------------------------------------------------------- #


def _common(case_id, role, identity, prompt, responses, trace_layers):
    length = len(prompt)
    return {
        "case_id": case_id,
        "role": role,
        "identity": identity,
        "sample_ids": [0],
        "prompt_tokens": [prompt],
        "response_tokens": None if responses is None else [responses],
        "response_length": RESPONSE_LENGTH,
        "prediction_positions": list(range(length - 1, length + RESPONSE_LENGTH - 1)),
        "consumed_response_tokens": RESPONSE_LENGTH - 1,
        "consumed_input_tokens": None if responses is None else [prompt + responses[:-1]],
        "state_position": length + RESPONSE_LENGTH - 2,
        "logits_dtype": "bfloat16",
        "logprobs_dtype": "float32",
        "vocab_size": 151936,
        "trace_layers": list(trace_layers or []),
        "required_layer_digests": list(range(48)),
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": 1234,
            "ignore_eos": True,
        },
    }


def rollout_request(cell, encoded_texts, identity, *, trace_layers=None):
    """Version 2 rollout request: the generating target plus untraced companions."""
    if cell.engine != "rollout":
        raise ValueError(f"{cell.cell_id} is not a rollout cell")
    plan = cell.chunk_plan()
    request = {
        "schema_version": 2,
        **_common(
            cell.cell_id,
            "rollout",
            identity,
            prompt_tokens(encoded_texts, cell.target),
            None,
            trace_layers,
        ),
        "rollout_prefill_chunks": plan,
        "prefill_chunks": plan,
        "execution": {
            "rollout_mode": "autoregressive",
            "prefill_chunks": plan,
            "response_mode": "generate",
        },
        "companions": [prompt_tokens(encoded_texts, s) for s in cell.companions],
        "companion_max_tokens": [s.max_tokens for s in cell.companions],
        "arrival_steps": {
            "target": cell.target.arrival_step,
            "companions": [s.arrival_step for s in cell.companions],
        },
        "chunk_size": cell.chunk_size,
        "target_last": cell.target_last,
    }
    request["request_sha256"] = json_digest(request)
    return request


def replay_request(case_id, prompt, responses, identity, *, packing=None, trace_layers=None):
    """Version 1 recompute request for one target; ``packing`` adds untraced neighbours."""
    if len(responses) != RESPONSE_LENGTH:
        raise ValueError("Replay needs the target's sixteen response tokens")
    request = {
        "schema_version": 1,
        **_common(case_id, "recompute", identity, prompt, responses, trace_layers),
        "rollout_prefill_chunks": [len(prompt)],
        "prefill_chunks": None,
        "execution": {
            "recompute_mode": "full_from_tokens",
            "input_sequence_length": len(prompt) + RESPONSE_LENGTH - 1,
        },
    }
    if packing is not None:
        request["packing"] = packing
    request["request_sha256"] = json_digest(request)
    return request


def packing_layout(cell, encoded_texts, target_tokens):
    """The replay packing of one cell: neighbours are prompt plus continuation tokens."""
    neighbours = [
        prompt_tokens(
            encoded_texts,
            Sequence(s.text, s.prompt_length + s.max_tokens - 1, offset=s.offset),
        )
        for s in cell.companions
    ]
    sequences = (neighbours + [target_tokens]) if cell.target_last else [target_tokens, *neighbours]
    return {"sequences": sequences, "target_index": len(neighbours) if cell.target_last else 0}
