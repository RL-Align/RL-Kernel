# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P2c: read-only per-step gradient write evidence for Megatron DDP training.

Step window: from the end of this step's ``zero_grad_buffer`` to the end of
``finish_grad_sync`` (all accumulated micro-batches and the gradient sync),
before the optimizer step and the next zeroing.

A write is decided from write markers, never from values: gradients can cancel
across micro-batches, so a zero ``main_grad`` or an unchanged digest does not
prove "no write". Markers:

* a post-accumulate hook per trainable parameter that only counts calls
  (it fires for the autograd path and for the dummy gradient Megatron returns
  on the fused-wgrad path);
* Megatron's ``grad_added_to_main_grad`` flag, which the fused wgrad path sets
  and ``zero_grad_buffer`` resets (except under Transformer-Engine CUDA graphs,
  where a flag still set after zeroing makes the write source unknown).

Bucket digests are optional extra evidence and never decide a single parameter.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import torch

from rl_engine.alignment.cross_config.adapters.megatron_training import T05ContractError

GRADIENT_EVIDENCE_SCHEMA_VERSION = "rlkernel.p7.t05.gradient_step_evidence.v1-draft"
_FLAG = "grad_added_to_main_grad"


@dataclass(frozen=True)
class ParameterGradientRecord:
    name: str
    requires_grad: bool
    managed_by_ddp: bool
    buffer_index: Optional[int]
    bucket_id: Optional[int]
    buffer_range: Optional[tuple[int, int]]
    accumulate_hook_calls: int
    flag_at_begin: Optional[bool]
    flag_at_end: Optional[bool]
    grad_present_at_end: bool
    write_path: str  # fused_main_grad | autograd_main_grad | autograd_grad | none | unknown
    wrote: Optional[bool]
    basis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requires_grad": self.requires_grad,
            "managed_by_ddp": self.managed_by_ddp,
            "buffer_index": self.buffer_index,
            "bucket_id": self.bucket_id,
            "buffer_range": None if self.buffer_range is None else list(self.buffer_range),
            "accumulate_hook_calls": self.accumulate_hook_calls,
            "flag_at_begin": self.flag_at_begin,
            "flag_at_end": self.flag_at_end,
            "grad_present_at_end": self.grad_present_at_end,
            "write_path": self.write_path,
            "wrote": self.wrote,
            "basis": self.basis,
        }


@dataclass(frozen=True)
class GradientStepEvidence:
    step_index: int
    records: tuple[ParameterGradientRecord, ...]
    bucket_digests: Optional[tuple[tuple[int, int, str], ...]] = None
    schema_version: str = GRADIENT_EVIDENCE_SCHEMA_VERSION

    def record(self, name: str) -> ParameterGradientRecord:
        for item in self.records:
            if item.name == name:
                return item
        raise KeyError(name)

    @property
    def emitted(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.records if item.wrote is True)

    @property
    def unknown(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.records if item.wrote is None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "step_index": self.step_index,
            "records": [item.to_dict() for item in self.records],
            "emitted": list(self.emitted),
            "unknown": list(self.unknown),
            "bucket_digests": (
                None
                if self.bucket_digests is None
                else [list(item) for item in self.bucket_digests]
            ),
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _flag(param: torch.nn.Parameter) -> Optional[bool]:
    value = getattr(param, _FLAG, None)
    return None if value is None else bool(value)


def _decide(
    *, requires_grad: bool, managed: bool, calls: int, begin: Optional[bool], end: Optional[bool]
) -> tuple[str, Optional[bool], str]:
    if not requires_grad:
        return "none", False, "requires_grad=False"
    if managed:
        if begin:
            return "unknown", None, f"{_FLAG} was still set after zero_grad_buffer"
        if calls > 0 and end:
            return "fused_main_grad", True, f"{calls} accumulate hook call(s) and {_FLAG}=True"
        if calls > 0:
            return "autograd_main_grad", True, f"{calls} accumulate hook call(s), {_FLAG}=False"
        if end:
            return "unknown", None, f"{_FLAG}=True without an accumulate hook call"
        return "none", False, "no accumulate hook call and no fused write flag"
    if calls > 0:
        return "autograd_grad", True, f"{calls} accumulate hook call(s), not DDP-managed"
    return "none", False, "no accumulate hook call, not DDP-managed"


class GradientStepObserver:
    """Wrap one Megatron DDP instance read-only; one evidence record per step window."""

    def __init__(self, ddp: Any, *, digest_buckets: bool = False) -> None:
        for name in ("module", "zero_grad_buffer", "finish_grad_sync"):
            if not hasattr(ddp, name):
                raise T05ContractError(
                    "UNSUPPORTED_CAPABILITY", f"ddp.{name}", "not a Megatron DDP-like module"
                )
        self._ddp = ddp
        self._digest = digest_buckets
        self._handles: list[Any] = []
        self._patched: dict[str, Any] = {}
        self._calls: dict[str, int] = {}
        self._begin_flags: dict[str, Optional[bool]] = {}
        self._step = 0
        self._zero_seen = False
        self._sync_seen = False
        self._installed = False

    @property
    def installed(self) -> bool:
        return self._installed

    def _counter(self, name: str) -> Callable[[torch.Tensor], None]:
        def count(_param: torch.Tensor) -> None:
            self._calls[name] = self._calls.get(name, 0) + 1

        return count

    def _wrap(self, method_name: str, after: Callable[[], None]) -> None:
        original = getattr(self._ddp, method_name)
        had_instance_attr = method_name in vars(self._ddp)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            after()
            return result

        self._patched[method_name] = (original, had_instance_attr)
        setattr(self._ddp, method_name, wrapped)

    def install(self) -> None:
        if self._installed:
            return
        for name, param in self._ddp.module.named_parameters():
            if param.requires_grad:
                self._handles.append(param.register_post_accumulate_grad_hook(self._counter(name)))
        self._wrap("zero_grad_buffer", self._after_zero)
        self._wrap("finish_grad_sync", self._after_sync)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for method_name, (original, had_instance_attr) in self._patched.items():
            if had_instance_attr:
                setattr(self._ddp, method_name, original)
            else:
                delattr(self._ddp, method_name)
        self._patched.clear()
        self._installed = False

    def _after_zero(self) -> None:
        self._step += 1
        self._calls = {}
        self._begin_flags = {
            name: _flag(param) for name, param in self._ddp.module.named_parameters()
        }
        self._zero_seen = True
        self._sync_seen = False

    def _after_sync(self) -> None:
        if self._zero_seen:
            self._sync_seen = True

    def _buffer_map(self) -> dict[int, tuple[int, int, tuple[int, int]]]:
        mapping: dict[int, tuple[int, int, tuple[int, int]]] = {}
        buffers = list(getattr(self._ddp, "buffers", []) or []) + list(
            getattr(self._ddp, "expert_parallel_buffers", []) or []
        )
        for index, buffer in enumerate(buffers):
            for param, (start, end, bucket_id) in getattr(buffer, "param_index_map", {}).items():
                mapping[id(param)] = (index, int(bucket_id), (int(start), int(end)))
        return mapping

    def _bucket_digests(self) -> tuple[tuple[int, int, str], ...]:
        digests: list[tuple[int, int, str]] = []
        buffers = list(getattr(self._ddp, "buffers", []) or []) + list(
            getattr(self._ddp, "expert_parallel_buffers", []) or []
        )
        for index, buffer in enumerate(buffers):
            data = buffer.grad_data.detach()
            for bucket_id, (start, end) in enumerate(getattr(buffer, "bucket_indices", [])):
                chunk = data[start:end].cpu().reshape(-1)
                dense = torch.empty(chunk.shape, dtype=chunk.dtype).copy_(chunk)
                digest = hashlib.sha256(dense.view(torch.uint8).numpy().tobytes()).hexdigest()
                digests.append((index, bucket_id, digest))
        return tuple(digests)

    def end_step(self) -> GradientStepEvidence:
        """Close the window after gradient sync and before the optimizer step."""

        if not self._zero_seen:
            raise T05ContractError(
                "MISSING_PROVENANCE", "gradient_window", "zero_grad_buffer not observed this step"
            )
        if not self._sync_seen:
            raise T05ContractError(
                "MISSING_PROVENANCE", "gradient_window", "finish_grad_sync not observed this step"
            )
        mapping = self._buffer_map()
        records = []
        for name, param in self._ddp.module.named_parameters():
            located = mapping.get(id(param))
            managed = located is not None or hasattr(param, "main_grad")
            calls = self._calls.get(name, 0)
            begin = self._begin_flags.get(name)
            end = _flag(param)
            path, wrote, basis = _decide(
                requires_grad=param.requires_grad,
                managed=managed,
                calls=calls,
                begin=begin,
                end=end,
            )
            records.append(
                ParameterGradientRecord(
                    name=name,
                    requires_grad=bool(param.requires_grad),
                    managed_by_ddp=managed,
                    buffer_index=None if located is None else located[0],
                    bucket_id=None if located is None else located[1],
                    buffer_range=None if located is None else located[2],
                    accumulate_hook_calls=calls,
                    flag_at_begin=begin,
                    flag_at_end=end,
                    grad_present_at_end=param.grad is not None,
                    write_path=path,
                    wrote=wrote,
                    basis=basis,
                )
            )
        evidence = GradientStepEvidence(
            step_index=self._step,
            records=tuple(records),
            bucket_digests=self._bucket_digests() if self._digest else None,
        )
        self._zero_seen = False
        self._sync_seen = False
        return evidence


def check_base_grad_absent(evidence: GradientStepEvidence, base_param_names: Iterable[str]) -> None:
    """Every frozen base parameter must have no gradient write in this step."""

    for name in base_param_names:
        try:
            record = evidence.record(name)
        except KeyError as exc:
            raise T05ContractError(
                "MISSING_PROVENANCE", f"gradient.{name}", "parameter not observed"
            ) from exc
        if record.wrote is None:
            raise T05ContractError("MISSING_PROVENANCE", f"gradient.{name}", record.basis)
        if record.wrote:
            raise T05ContractError(
                "IDENTITY_DRIFT",
                f"gradient.{name}",
                f"base parameter received a gradient: {record.basis}",
            )


def check_emitted_against_producer(
    evidence: GradientStepEvidence, producer_emitted: Optional[Iterable[str]]
) -> None:
    """Cross-check observed writes with the producer's declared emitted-gradient set."""

    if producer_emitted is None:
        raise T05ContractError(
            "UPSTREAM_EVIDENCE_MISSING", "gradient.producer_emitted", "producer gradient set absent"
        )
    if evidence.unknown:
        raise T05ContractError(
            "MISSING_PROVENANCE",
            "gradient.unknown",
            f"write source unknown for {list(evidence.unknown)}",
        )
    declared, observed = set(producer_emitted), set(evidence.emitted)
    if declared != observed:
        raise T05ContractError(
            "UPSTREAM_CONTRACT_MISMATCH",
            "gradient.emitted",
            f"declared-only {sorted(declared - observed)}, "
            f"observed-only {sorted(observed - declared)}",
        )


__all__ = [
    "GRADIENT_EVIDENCE_SCHEMA_VERSION",
    "GradientStepEvidence",
    "GradientStepObserver",
    "ParameterGradientRecord",
    "check_base_grad_absent",
    "check_emitted_against_producer",
]
