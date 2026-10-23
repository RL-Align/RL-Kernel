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

With ``observe_sync=True`` the observer also records, inside the step window,
every gradient-sync collective Megatron issues on a DDP bucket (the
reduce-scatter used with the distributed optimizer, or the all-reduce
otherwise): which bucket, which group, which shard this rank received. The
collective functions are wrapped by pass-through Python callables only, so no
CUDA kernel is added. :func:`observe_optimizer_ownership` reads which
parameter ranges this rank's distributed optimizer owns.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import torch

from rl_engine.alignment.cross_config.adapters.megatron_training import T05ContractError

GRADIENT_EVIDENCE_SCHEMA_VERSION = "rlkernel.p7.t05.gradient_step_evidence.v1-draft"
_FLAG = "grad_added_to_main_grad"
_MEGATRON_BUFFER_MODULE = "megatron.core.distributed.param_and_grad_buffer"


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
    sync: Optional[dict[str, Any]] = None
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
        payload = {
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
        if self.sync is not None:
            payload["sync"] = self.sync
        return payload

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _ddp_buffers(ddp: Any) -> list[Any]:
    return list(getattr(ddp, "buffers", []) or []) + list(
        getattr(ddp, "expert_parallel_buffers", []) or []
    )


def _group_ranks(group: Any) -> list[int]:
    import torch.distributed as dist

    if group is None:
        return list(range(dist.get_world_size()))
    return [int(rank) for rank in dist.get_process_group_ranks(group)]


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

    def __init__(
        self, ddp: Any, *, digest_buckets: bool = False, observe_sync: bool = False
    ) -> None:
        for name in ("module", "zero_grad_buffer", "finish_grad_sync"):
            if not hasattr(ddp, name):
                raise T05ContractError(
                    "UNSUPPORTED_CAPABILITY", f"ddp.{name}", "not a Megatron DDP-like module"
                )
        self._ddp = ddp
        self._digest = digest_buckets
        self._observe_sync = observe_sync
        self._collectives: list[tuple[Any, str, Any]] = []
        self._sync_calls: list[dict[str, Any]] = []
        self._unmatched_calls = 0
        self._bucket_ptrs: dict[tuple[int, int], tuple[int, int, int]] = {}
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
        if self._observe_sync:
            self._wrap_collectives()
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
        for owner, attribute, original in reversed(self._collectives):
            setattr(owner, attribute, original)
        self._collectives.clear()
        self._installed = False

    def _wrap_collectives(self) -> None:
        """Pass-through wrappers that note grad-sync calls on DDP buckets in the window."""

        import torch.distributed as dist

        targets: list[tuple[Any, str, str]] = [(dist, "all_reduce", "all_reduce")]
        buffer_module = sys.modules.get(_MEGATRON_BUFFER_MODULE)
        if buffer_module is not None and hasattr(buffer_module, "dist_reduce_scatter_func"):
            targets.append((buffer_module, "dist_reduce_scatter_func", "reduce_scatter"))
        for owner, attribute, kind in targets:
            original = getattr(owner, attribute)
            self._collectives.append((owner, attribute, original))
            setattr(owner, attribute, self._spy(original, kind))

    def _spy(self, original: Callable[..., Any], kind: str) -> Callable[..., Any]:
        def spy(*args: Any, **kwargs: Any) -> Any:
            if self._zero_seen and not self._sync_seen:
                self._note_collective(kind, args, kwargs)
            return original(*args, **kwargs)

        return spy

    def _note_collective(self, kind: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        import torch.distributed as dist

        if kind == "reduce_scatter":
            output = args[0] if args else kwargs.get("output")
            inputs = args[1] if len(args) > 1 else kwargs.get("input")
            op = args[2] if len(args) > 2 else kwargs.get("op", dist.ReduceOp.SUM)
            group = args[3] if len(args) > 3 else kwargs.get("group")
        else:
            inputs = output = args[0] if args else kwargs.get("tensor")
            op = args[1] if len(args) > 1 else kwargs.get("op", dist.ReduceOp.SUM)
            group = args[2] if len(args) > 2 else kwargs.get("group")
        if not isinstance(inputs, torch.Tensor) or not isinstance(output, torch.Tensor):
            self._unmatched_calls += 1
            return
        located = self._bucket_ptrs.get((inputs.data_ptr(), inputs.numel()))
        if located is None:
            self._unmatched_calls += 1
            return
        buffer_index, bucket_id, _ = located
        self._sync_calls.append(
            {
                "kind": kind,
                "buffer_index": buffer_index,
                "bucket_id": bucket_id,
                "group_ranks": _group_ranks(group),
                "op": str(op),
                "input_numel": int(inputs.numel()),
                "output_offset": (output.data_ptr() - inputs.data_ptr()) // inputs.element_size(),
                "output_numel": int(output.numel()),
            }
        )

    def _bucket_layout(self) -> list[dict[str, Any]]:
        names = {id(param): name for name, param in self._ddp.module.named_parameters()}
        layout = []
        for buffer_index, buffer in enumerate(_ddp_buffers(self._ddp)):
            index_map = getattr(buffer, "param_index_map", {})
            for bucket in getattr(buffer, "buckets", []) or []:
                params = sorted(
                    (names.get(id(param), "?"), int(index_map[param][0]), int(index_map[param][1]))
                    for param in bucket.params_list
                )
                layout.append(
                    {
                        "buffer_index": buffer_index,
                        "bucket_id": int(bucket.bucket_id),
                        "offset": int(bucket.offset),
                        "numel": int(bucket.grad_data.numel()),
                        "numel_unpadded": int(bucket.numel_unpadded),
                        "gradient_scaling_factor": float(bucket.gradient_scaling_factor),
                        "params": [list(item) for item in params],
                    }
                )
        return layout

    def _after_zero(self) -> None:
        self._step += 1
        self._calls = {}
        self._sync_calls = []
        self._unmatched_calls = 0
        if self._observe_sync:
            self._bucket_ptrs = {
                (bucket.grad_data.data_ptr(), bucket.grad_data.numel()): (
                    buffer_index,
                    int(bucket.bucket_id),
                    int(bucket.offset),
                )
                for buffer_index, buffer in enumerate(_ddp_buffers(self._ddp))
                for bucket in getattr(buffer, "buckets", []) or []
            }
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
        for index, buffer in enumerate(_ddp_buffers(self._ddp)):
            for param, (start, end, bucket_id) in getattr(buffer, "param_index_map", {}).items():
                mapping[id(param)] = (index, int(bucket_id), (int(start), int(end)))
        return mapping

    def _bucket_digests(self) -> tuple[tuple[int, int, str], ...]:
        digests: list[tuple[int, int, str]] = []
        for index, buffer in enumerate(_ddp_buffers(self._ddp)):
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
            sync=self._sync_evidence() if self._observe_sync else None,
        )
        self._zero_seen = False
        self._sync_seen = False
        return evidence

    def _sync_evidence(self) -> dict[str, Any]:
        ddp_config = getattr(self._ddp, "ddp_config", None)
        configured = getattr(ddp_config, "use_distributed_optimizer", None)
        average = getattr(ddp_config, "average_in_collective", None)
        return {
            "source": "observed",
            "use_distributed_optimizer": {
                "source": "configured",
                "value": None if configured is None else bool(configured),
            },
            "average_in_collective": {
                "source": "configured",
                "value": None if average is None else bool(average),
            },
            "buckets": self._bucket_layout(),
            "calls": list(self._sync_calls),
            "unmatched_calls": self._unmatched_calls,
        }


def observe_optimizer_ownership(optimizer: Any, ddp: Any) -> dict[str, Any]:
    """Parameter ranges this rank's Megatron distributed optimizer owns, in DDP buffer space."""

    names = {id(param): name for name, param in ddp.module.named_parameters()}
    buffers = _ddp_buffers(ddp)
    members = list(getattr(optimizer, "chained_optimizers", None) or [optimizer])
    ranges: list[dict[str, Any]] = []
    kinds = []
    for member in members:
        gbuf_ranges = getattr(member, "gbuf_ranges", None)
        if gbuf_ranges is None:
            continue
        kinds.append(type(member).__name__)
        for gbuf_index, buffer in enumerate(member.buffers):
            matches = [index for index, item in enumerate(buffers) if item is buffer]
            if len(matches) != 1:
                raise T05ContractError(
                    "IDENTITY_DRIFT",
                    "optimizer.buffers",
                    f"optimizer buffer {gbuf_index} is not one of the DDP buffers",
                )
            for bucket_ranges in gbuf_ranges[gbuf_index].values():
                for bucket_id, bucket_range in enumerate(bucket_ranges):
                    for param, ranges_of in bucket_range["param_map"].items():
                        world, local = ranges_of["gbuf_world"], ranges_of["param"]
                        ranges.append(
                            {
                                "name": names.get(id(param), "?"),
                                "buffer_index": matches[0],
                                "bucket_id": bucket_id,
                                "world_start": int(world.start),
                                "world_end": int(world.end),
                                "param_start": int(local.start),
                                "param_end": int(local.end),
                            }
                        )
    if not kinds:
        raise T05ContractError(
            "UNSUPPORTED_CAPABILITY", "optimizer", "no Megatron distributed optimizer found"
        )
    ranges.sort(key=lambda item: (item["buffer_index"], item["world_start"]))
    return {"source": "observed", "optimizers": kinds, "ranges": ranges}


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
    "observe_optimizer_ownership",
]
