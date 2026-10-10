# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Private scoring contracts and child-process supervision."""

from __future__ import annotations

import hashlib
import inspect
import json
import multiprocessing as mp
import os
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Protocol, Sequence

import torch

from rl_engine.validation.cross_config._json import strict_json_loads
from rl_engine.validation.cross_config.schema import CanonicalScoringBatch, ScorerSpec, ScoreSide


class PairedRunnerError(RuntimeError):
    """Base error for a paired scoring attempt."""


class OperatorExecutionError(PairedRunnerError):
    """Raised when exact operator evidence cannot authorize execution."""


class ChildScoringError(PairedRunnerError):
    """Raised when a scoring child exits without a valid result."""


class ScoringTimeoutError(PairedRunnerError):
    """Raised after all scoring children are stopped at the deadline."""


class RankCompletenessError(PairedRunnerError):
    """Raised when rank results are missing, duplicated, or inconsistent."""


class ScorerIdentityError(PairedRunnerError):
    """Raised when paired scorer model state is not logically identical."""


@dataclass(frozen=True)
class RankScore:
    """One rank's full canonical selected-logprob observation."""

    rank: int
    world_size: int
    selected_logprobs: torch.Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("rank must be >= 0")
        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")
        if self.rank >= self.world_size:
            raise ValueError("rank must be less than world_size")
        if not isinstance(self.selected_logprobs, torch.Tensor):
            raise TypeError("selected_logprobs must be a torch.Tensor")
        object.__setattr__(
            self,
            "selected_logprobs",
            self.selected_logprobs.detach().to(device="cpu").clone(),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


class PairedScorer(Protocol):
    """Small injection boundary used by the paired-runner control plane."""

    spec: ScorerSpec

    def score(
        self,
        batch: CanonicalScoringBatch,
        *,
        batch_size: int,
        operator: Any,
    ) -> torch.Tensor | RankScore | Sequence[RankScore]: ...


class ChildSupervisor:
    """Own the lifecycle of the two isolated scoring children."""

    def __init__(self, start_method: Optional[str] = None):
        available = mp.get_all_start_methods()
        resolved = start_method or ("fork" if "fork" in available else "spawn")
        if resolved not in available:
            raise ValueError(f"multiprocessing start method is unavailable: {resolved}")
        self.start_method = resolved
        self._active_processes: list[mp.Process] = []

    @property
    def active_child_pids(self) -> tuple[int, ...]:
        return tuple(
            process.pid
            for process in self._active_processes
            if process.pid is not None and process.is_alive()
        )

    def run(
        self,
        attempt_dir: Path,
        batch: CanonicalScoringBatch,
        *,
        batch_size: int,
        scorers: Mapping[str, PairedScorer],
        specs: Mapping[str, ScorerSpec],
        instances: Mapping[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Mapping[str, Any]]:
        context: Any = mp.get_context(self.start_method)
        processes: dict[str, mp.Process] = {}
        with tempfile.TemporaryDirectory(prefix=".paired-runner-", dir=attempt_dir) as tmp:
            temporary_dir = Path(tmp)
            try:
                for target in ("rollout", "training"):
                    process = context.Process(
                        target=_score_child,
                        name=f"cross-config-{target}",
                        args=(
                            temporary_dir / f"{target}.pt",
                            temporary_dir / f"{target}.error.json",
                            scorers[target],
                            specs[target],
                            batch,
                            batch_size,
                            instances[target],
                        ),
                    )
                    process.start()
                    processes[target] = process
                self._active_processes = list(processes.values())
                self._wait(
                    processes,
                    temporary_dir,
                    timeout_seconds=timeout_seconds,
                )
                return {
                    target: _load_child_result(temporary_dir / f"{target}.pt")
                    for target in ("rollout", "training")
                }
            finally:
                _stop_processes(tuple(processes.values()))
                self._active_processes = []

    @staticmethod
    def _wait(
        processes: Mapping[str, mp.Process],
        temporary_dir: Path,
        *,
        timeout_seconds: float,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        unfinished = set(processes)
        while unfinished:
            for target in tuple(unfinished):
                process = processes[target]
                process.join(timeout=0.01)
                if process.is_alive():
                    continue
                unfinished.remove(target)
                if process.exitcode != 0:
                    detail = _child_error_detail(temporary_dir / f"{target}.error.json")
                    raise ChildScoringError(
                        f"{target} scoring child failed with exit code "
                        f"{process.exitcode}: {detail}"
                    )
            if unfinished and time.monotonic() >= deadline:
                labels = ", ".join(sorted(unfinished))
                raise ScoringTimeoutError(
                    f"paired scoring exceeded {timeout_seconds:.3f}s; "
                    f"stopped children: {labels}"
                )


def _score_child(
    result_path: Path,
    error_path: Path,
    scorer: PairedScorer,
    spec: ScorerSpec,
    batch: CanonicalScoringBatch,
    batch_size: int,
    operator: Any,
) -> None:
    try:
        with _read_only_scoring_guard(scorer, verify_state=True) as evidence:
            with torch.no_grad():
                output = scorer.score(batch, batch_size=batch_size, operator=operator)
        ranks = _coerce_rank_scores(output, spec.world_size)
        payload = {
            "schema_version": 1,
            "guard_evidence": evidence,
            "ranks": [
                {
                    "rank": rank.rank,
                    "world_size": rank.world_size,
                    "selected_logprobs": rank.selected_logprobs,
                    "metadata": json_safe(rank.metadata),
                }
                for rank in ranks
            ],
        }
        temporary = result_path.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, result_path)
    except BaseException as exc:
        error = {
            "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        error_path.write_text(json.dumps(error, sort_keys=True), encoding="utf-8")
        raise SystemExit(1) from None


@contextmanager
def _read_only_scoring_guard(
    scorer: PairedScorer,
    *,
    verify_state: bool,
) -> Iterator[dict[str, Any]]:
    model = scorer_model(scorer)
    if verify_state and getattr(scorer, "optimizer", None) is not None:
        raise ValueError("scorer must not own an active optimizer")
    if model is None:
        yield {
            "model_state_verified": False,
            "model_eval": False,
            "no_grad": True,
            "optimizer_step": False,
        }
        return

    modes = tuple((module, module.training) for module in model.modules())
    snapshot = _module_tensor_snapshot(model) if verify_state else None
    model.eval()
    evidence = {
        "model_state_verified": verify_state,
        "model_eval": True,
        "no_grad": True,
        "optimizer_step": False,
        "model_modes_restored": False,
        "model_state_unchanged": False if verify_state else None,
    }
    try:
        yield evidence
    finally:
        for module, was_training in modes:
            module.training = was_training
        evidence["model_modes_restored"] = True
        if snapshot is not None:
            mutations = _module_state_mutations(model, snapshot)
            if mutations:
                raise RuntimeError(
                    "read-only scorer mutated model parameters/buffers: " + ", ".join(mutations)
                )
            evidence["model_state_unchanged"] = True


def _module_tensor_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    values = {
        f"parameter:{name}": tensor.detach().to(device="cpu").clone()
        for name, tensor in model.named_parameters()
    }
    values.update(
        {
            f"buffer:{name}": tensor.detach().to(device="cpu").clone()
            for name, tensor in model.named_buffers()
        }
    )
    return values


def _module_state_mutations(
    model: torch.nn.Module,
    before: Mapping[str, torch.Tensor],
) -> list[str]:
    after = _module_tensor_snapshot(model)
    mutations: list[str] = []
    for name in sorted(set(before) | set(after)):
        left = before.get(name)
        right = after.get(name)
        if left is None or right is None:
            mutations.append(name)
            continue
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            mutations.append(name)
    return mutations


def scorer_model(scorer: PairedScorer) -> Optional[torch.nn.Module]:
    if isinstance(scorer, torch.nn.Module):
        return scorer
    candidate = getattr(scorer, "model", None)
    return candidate if isinstance(candidate, torch.nn.Module) else None


def paired_model_state_fingerprints(
    rollout_scorer: PairedScorer,
    training_scorer: PairedScorer,
) -> dict[str, Optional[str]]:
    fingerprints = {
        "rollout": _scorer_model_state_fingerprint(rollout_scorer),
        "training": _scorer_model_state_fingerprint(training_scorer),
    }
    if fingerprints["rollout"] is None or fingerprints["training"] is None:
        raise ScorerIdentityError(
            "rollout and training model state fingerprints must both be observable"
        )
    if fingerprints["rollout"] != fingerprints["training"]:
        raise ScorerIdentityError(
            "rollout and training model state fingerprints differ before scoring"
        )
    return fingerprints


def _scorer_model_state_fingerprint(scorer: PairedScorer) -> Optional[str]:
    declared = getattr(scorer, "model_state_fingerprint", None)
    if declared is not None and (not isinstance(declared, str) or not declared):
        raise ScorerIdentityError("scorer model_state_fingerprint must be a non-empty string")
    model = scorer_model(scorer)
    if model is None:
        return declared
    observed_model = _module_state_fingerprint(model)
    if declared is not None and declared != observed_model:
        raise ScorerIdentityError(
            "declared scorer model_state_fingerprint does not match observed model state"
        )
    return observed_model


def scorer_implementation_fingerprint(scorer: PairedScorer) -> str:
    declared = getattr(scorer, "implementation_fingerprint", None)
    if declared is not None and (not isinstance(declared, str) or not declared):
        raise ScorerIdentityError("scorer implementation_fingerprint must be a non-empty string")
    scorer_type = f"{type(scorer).__module__}.{type(scorer).__qualname__}"
    score_source = _source_text(getattr(type(scorer), "score", None))
    return canonical_fingerprint(
        {
            "declared_implementation": declared,
            "scorer_type": scorer_type,
            "score_source_fingerprint": hashlib.sha256(score_source.encode("utf-8")).hexdigest(),
        }
    )


def _module_state_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    digest.update(f"{type(model).__module__}.{type(model).__qualname__}".encode("utf-8"))
    digest.update(_source_text(type(model)).encode("utf-8"))
    tensors = tuple(
        (f"parameter:{name}", tensor) for name, tensor in model.named_parameters()
    ) + tuple((f"buffer:{name}", tensor) for name, tensor in model.named_buffers())
    for name, tensor in tensors:
        snapshot = tensor.detach().to(device="cpu")
        if snapshot.is_sparse:
            snapshot = snapshot.to_dense()
        snapshot = snapshot.contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(snapshot.dtype).encode("utf-8"))
        digest.update(str(tuple(snapshot.shape)).encode("utf-8"))
        digest.update(snapshot.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _source_text(value: Any) -> str:
    try:
        return inspect.getsource(value)
    except (OSError, TypeError):
        return repr(value)


def _coerce_rank_scores(
    output: torch.Tensor | RankScore | Sequence[RankScore],
    expected_world_size: int,
) -> tuple[RankScore, ...]:
    if isinstance(output, torch.Tensor):
        if expected_world_size != 1:
            raise RankCompletenessError(
                "a bare tensor result is valid only for a world_size=1 scorer"
            )
        return (RankScore(rank=0, world_size=1, selected_logprobs=output),)
    if isinstance(output, RankScore):
        return (output,)
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
        raise TypeError("scorer must return a tensor, RankScore, or sequence of RankScore")
    values = tuple(output)
    if not all(isinstance(value, RankScore) for value in values):
        raise TypeError("every scorer sequence item must be a RankScore")
    return values


def _load_child_result(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ChildScoringError(f"scoring child produced no result artifact: {path.name}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ChildScoringError(f"failed to load scoring child result {path.name}: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ChildScoringError(f"malformed scoring child result: {path.name}")
    return payload


def validate_rank_outputs(
    payload: Mapping[str, Any],
    spec: ScorerSpec,
    *,
    expected_shape: torch.Size,
    target: str,
) -> dict[int, RankScore]:
    raw_ranks = payload.get("ranks")
    if not isinstance(raw_ranks, Sequence):
        raise RankCompletenessError(f"{target} child result has no rank sequence")
    ranks: dict[int, RankScore] = {}
    duplicates: list[int] = []
    for raw in raw_ranks:
        if not isinstance(raw, Mapping):
            raise RankCompletenessError(f"{target} rank result must be a mapping")
        rank_score = RankScore(
            rank=int(raw["rank"]),
            world_size=int(raw["world_size"]),
            selected_logprobs=raw["selected_logprobs"],
            metadata=raw.get("metadata", {}),
        )
        if rank_score.rank in ranks:
            duplicates.append(rank_score.rank)
        ranks[rank_score.rank] = rank_score
    if duplicates:
        raise RankCompletenessError(f"{target} returned duplicate ranks: {sorted(set(duplicates))}")
    expected = set(range(spec.world_size))
    actual = set(ranks)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RankCompletenessError(
            f"{target} rank set is incomplete; missing={missing}, unexpected={unexpected}"
        )
    for rank_index, value in ranks.items():
        if value.world_size != spec.world_size:
            raise RankCompletenessError(
                f"{target} rank {rank_index} reported world_size={value.world_size}, "
                f"expected {spec.world_size}"
            )
        if value.selected_logprobs.shape != expected_shape:
            raise RankCompletenessError(
                f"{target} rank {rank_index} selected_logprobs shape "
                f"{tuple(value.selected_logprobs.shape)} does not match "
                f"canonical shape {tuple(expected_shape)}"
            )
        expected_dtype = torch_dtype(spec.dtype)
        if (
            not value.selected_logprobs.is_floating_point()
            or value.selected_logprobs.dtype != expected_dtype
        ):
            raise RankCompletenessError(
                f"{target} rank {rank_index} selected_logprobs dtype "
                f"{value.selected_logprobs.dtype} does not match scorer dtype {expected_dtype}"
            )
    rank_zero = ranks[0].selected_logprobs
    for rank_index, value in ranks.items():
        if rank_index == 0:
            continue
        if value.selected_logprobs.dtype != rank_zero.dtype or not torch.equal(
            value.selected_logprobs,
            rank_zero,
        ):
            raise RankCompletenessError(
                f"{target} rank {rank_index} selected_logprobs diverge from rank 0"
            )
    return ranks


def scorer_spec(scorer: PairedScorer, expected_side: ScoreSide) -> ScorerSpec:
    spec = getattr(scorer, "spec", None)
    if not isinstance(spec, ScorerSpec):
        raise TypeError("paired scorer must expose a ScorerSpec as .spec")
    if spec.side is not expected_side:
        raise ValueError(f"scorer side {spec.side.value!r} does not match {expected_side.value!r}")
    model = scorer_model(scorer)
    if model is not None:
        _require_module_on_device(model, device_type(spec.device))
        _require_module_float_dtype(model, torch_dtype(spec.dtype))
    return spec


def validate_scorer_identity(
    specs: Mapping[str, ScorerSpec],
    batch: CanonicalScoringBatch,
) -> None:
    identity = batch.identity
    expected = {
        "checkpoint_id": identity.checkpoint_id,
        "model_version": identity.model_version,
        "pre_update_state": identity.pre_update_state,
    }
    for target in ("rollout", "training"):
        observed = specs[target].construction_options
        mismatches = [key for key, value in expected.items() if observed.get(key) != value]
        if mismatches:
            raise ScorerIdentityError(
                f"{target} scorer construction identity differs from canonical identity: "
                + ", ".join(mismatches)
            )


def _child_error_detail(path: Path) -> str:
    try:
        value = _read_json_object(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return "child did not publish structured error evidence"
    return f"{value.get('type', 'error')}: {value.get('message', '')}"


def _read_json_object(path: Path) -> dict[str, Any]:
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _stop_processes(processes: Iterator[mp.Process] | Sequence[mp.Process]) -> None:
    values = tuple(processes)
    for process in values:
        if process.is_alive():
            process.terminate()
    for process in values:
        if process.pid is not None:
            process.join(timeout=1.0)
    for process in values:
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=1.0)


def _require_module_on_device(model: torch.nn.Module, expected: str) -> None:
    for name, tensor in tuple(model.named_parameters()) + tuple(model.named_buffers()):
        if tensor.device.type != expected:
            raise ValueError(
                f"scorer model tensor {name!r} is on {tensor.device}; expected {expected}"
            )


def _require_module_float_dtype(model: torch.nn.Module, expected: torch.dtype) -> None:
    mismatches = [
        f"{name}={tensor.dtype}"
        for name, tensor in tuple(model.named_parameters()) + tuple(model.named_buffers())
        if tensor.is_floating_point() and tensor.dtype != expected
    ]
    if mismatches:
        raise ValueError(
            f"scorer floating model state must use {expected}: " + ", ".join(mismatches)
        )


def device_type(value: str) -> str:
    try:
        return torch.device(value).type
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid scorer device: {value!r}") from exc


def normalized_dtype(value: str) -> str:
    dtype = torch_dtype(value)
    return str(dtype).removeprefix("torch.")


def torch_dtype(value: str) -> torch.dtype:
    normalized = str(value).strip().lower().replace("torch.", "")
    dtypes = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return dtypes[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported stateless scorer dtype: {value!r}") from exc


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset, tuple, list)):
        items = [json_safe(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
        return items
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def canonical_fingerprint(value: Any) -> str:
    serialized = json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
