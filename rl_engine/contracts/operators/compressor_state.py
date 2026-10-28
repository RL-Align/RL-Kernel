# SPDX-License-Identifier: Apache-2.0
"""Versioned C4/C128 state contract; distinct from the eight-wave HCA profile."""

from dataclasses import asdict, dataclass
import hashlib
import json

from rl_engine.contracts.operators.kv_compressor_c4 import C4Error, require

CompressorStateError = C4Error


@dataclass(frozen=True)
class CompressorStateContract:
    ratio: int = 4
    max_requests: int = 4
    max_tokens_per_step: int = 256
    max_sequence_tokens: int = 4096
    max_arena_bytes: int = 128 * 1024 * 1024
    world_size: int = 1
    graph: bool = False

    def __post_init__(self):
        for name in (
            "ratio",
            "max_requests",
            "max_tokens_per_step",
            "max_sequence_tokens",
            "max_arena_bytes",
            "world_size",
        ):
            require(
                type(getattr(self, name)) is int and getattr(self, name) > 0,
                f"{name} must be a positive integer",
            )
        require(self.ratio in (4, 128), "only main-KV C4/C128 are admitted")
        require(type(self.graph) is bool, "graph must be a bool")
        require(self.max_sequence_tokens >= self.ratio, "capacity must fit one complete group")
        require(
            self.max_sequence_tokens < 2**31 and self.max_tokens_per_step < 2**31,
            "provider int32 capacity exceeded",
        )
        require(self.resident_bytes < 2**31, "provider buffer addressing envelope exceeded")
        require(self.resident_bytes <= self.max_arena_bytes, "arena byte budget exceeded")

    @property
    def width(self):
        return 1024 if self.ratio == 4 else 512

    @property
    def state_rows(self):
        return 8 if self.ratio == 4 else 128

    @property
    def plan_rows(self):
        return (self.max_tokens_per_step + self.ratio - 1) // self.ratio

    @property
    def pages_per_request(self):
        return (self.max_sequence_tokens // self.ratio + 63) // 64

    @property
    def resident_bytes(self):
        # Packed state + persistent contiguous reader copies, cache, staging,
        # positions/write metadata, plan, tables, and model/position constants.
        return (
            self.max_requests * self.state_rows * self.width * 4 * 4
            + self.max_requests * self.pages_per_request * 64 * 512 * 2
            + self.max_tokens_per_step * (self.width * 4 + 16)
            + self.plan_rows * 16
            + self.max_requests * (self.pages_per_request + 1) * 4
            + self.ratio * self.width * 4
            + 512 * 4
            + self.max_sequence_tokens * 64 * 2
        )

    @property
    def arithmetic_id(self):
        # This explicit single-wave profile must not be mixed with task 18's
        # eight-wave C128 HCA recipe merely because shapes agree.
        return f"aiter.main_c{self.ratio}.online_fp32.wave64.vllm_partial_f32.v1"

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
