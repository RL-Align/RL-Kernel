# SPDX-License-Identifier: Apache-2.0
"""INK-02: explicit 48-visit tensor trace, with byte-exact digests."""

import hashlib
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class Visit:
    logical: int
    physical: int
    pass_index: int | None


def visits() -> tuple[Visit, ...]:
    schedule = [(i, None) for i in range(6)]
    schedule += [(i, p) for p in range(3) for i in range(6, 18)]
    schedule += [(i, None) for i in range(18, 24)]
    return tuple(Visit(j, i, p) for j, (i, p) in enumerate(schedule))


def tensor_digest(tensor: torch.Tensor) -> dict:
    """Hash raw bytes, distinguishing BF16 rounding, signed zero and NaN payloads."""
    if tensor.layout != torch.strided:
        raise ValueError("trace requires a dense strided tensor")
    value = tensor.detach().cpu().contiguous().reshape(-1)
    raw = value.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


class VisitTrace:
    """Debug-only synchronous trace; no performance claim or backend inference."""

    def __init__(self, metadata: dict):
        self.metadata = dict(metadata)
        self._records: dict[tuple, dict] = {}
        self._ordinal: dict[tuple, int] = {}
        self._logprob: list[int] | None = None

    def record(
        self,
        logical: int,
        role: str,
        tensor: torch.Tensor,
        *,
        direction: str = "forward",
        backend: str,
        kernel: str,
    ) -> None:
        if type(logical) is not int or not 0 <= logical < 48:
            raise ValueError("logical visit outside fixed schedule")
        if direction not in ("forward", "backward") or not all((role, backend, kernel)):
            raise ValueError("trace requires direction, role, backend and kernel identity")
        key = (logical, direction, role)
        if key in self._records:
            raise ValueError(f"duplicate trace boundary {key}")
        group = (logical, direction)
        ordinal = self._ordinal.get(group, 0)
        self._ordinal[group] = ordinal + 1
        self._records[key] = {
            "ordinal": ordinal,
            **asdict(visits()[logical]),
            "direction": direction,
            "role": role,
            "backend": backend,
            "kernel": kernel,
            **tensor_digest(tensor),
        }

    def record_logprob(self, logprob: torch.Tensor) -> None:
        """Store FP32 bit patterns for the selected-token vector, not full logits."""
        if self._logprob is not None:
            raise ValueError("selected logprob already recorded")
        if logprob.ndim != 1 or logprob.dtype != torch.float32 or not logprob.numel():
            raise ValueError("selected logprob must be a nonempty FP32 vector")
        if not bool(torch.isfinite(logprob).all()):
            raise ValueError("non-finite selected logprob")
        self._logprob = logprob.detach().cpu().contiguous().view(torch.int32).tolist()

    def report(self) -> dict:
        return {
            "schema": "inkling2-visit-trace-v1",
            "metadata": dict(self.metadata),
            "records": sorted(
                (record.copy() for record in self._records.values()),
                key=lambda r: (r["logical"], r["direction"], r["ordinal"]),
            ),
            "selected_logprob_bits": None if self._logprob is None else list(self._logprob),
        }
