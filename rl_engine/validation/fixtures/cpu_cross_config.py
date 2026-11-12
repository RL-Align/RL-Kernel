# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU-only adapters for cross-configuration smoke execution.

This module is deliberately outside the production framework package. It gives
the CLI and tests a deterministic execution target without implying CUDA,
distributed, vLLM, or training-runtime support.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import torch

from rl_engine.runtime.executor import (
    StatelessForwardConfig,
    StatelessForwardExecutor,
    StatelessForwardInputs,
)
from rl_engine.runtime.operators import (
    OperatorBridge,
    OperatorOverride,
    selected_logprobs_with_operator,
)
from rl_engine.runtime.registry import kernel_registry
from rl_engine.runtime.semantic_registry import (
    OperatorRequirements,
    OperatorResolutionPolicy,
    SemanticOperatorCatalog,
)
from rl_engine.runtime.semantic_registry import (
    implementation_fingerprint as fingerprint_implementation,
)
from rl_engine.validation.cross_config.artifacts import ArtifactStore
from rl_engine.validation.cross_config.config import ExperimentConfig, OperatorSelection
from rl_engine.validation.cross_config.execution_plan import build_execution_plan
from rl_engine.validation.cross_config.runner import PairedRunner, PairedRunResult, RankScore
from rl_engine.validation.cross_config.runtime import (
    AdapterMaterialization,
    KnobApplication,
    RuntimeBinding,
    RuntimeTools,
)
from rl_engine.validation.cross_config.schema import (
    CanonicalScoringBatch,
    ExperimentCase,
    KnobDescriptor,
    MaterializationStatus,
    ScorerSpec,
    ScoreSide,
)

CPU_SCORER_IMPLEMENTATION_FINGERPRINT = "cross_config.cpu_stateless_scorer.v1"


class SyntheticCpuCausalLM(torch.nn.Module):
    """Deterministic parameter-free model for the named CPU smoke scenario."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_axis: torch.Tensor
        self.register_buffer(
            "vocab_axis",
            torch.arange(vocab_size, dtype=torch.float32),
            persistent=False,
        )
        self.config = SimpleNamespace(use_cache=False, _attn_implementation="eager")
        self.generation_config = SimpleNamespace(use_cache=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
    ) -> Any:
        del attention_mask
        if use_cache not in {None, False}:
            raise ValueError("CPU smoke scoring forbids KV-cache generation")
        if input_ids.device.type != "cpu":
            raise ValueError("the synthetic smoke model accepts CPU tensors only")
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device="cpu").expand_as(input_ids)
        centers = torch.remainder(input_ids + position_ids + 1, self.vocab_axis.numel()).float()
        logits = -torch.abs(self.vocab_axis.view(1, 1, -1) - centers.unsqueeze(-1)) * 0.125
        return SimpleNamespace(logits=logits, past_key_values=None)


class CpuStatelessScorer:
    """Read-only teacher-forcing adapter over ``StatelessForwardExecutor``."""

    optimizer = None
    implementation_fingerprint = CPU_SCORER_IMPLEMENTATION_FINGERPRINT

    def __init__(
        self,
        model: torch.nn.Module,
        spec: ScorerSpec,
        config: Optional[StatelessForwardConfig] = None,
    ):
        if spec.world_size != 1:
            raise ValueError("CpuStatelessScorer supports only world_size=1")
        if _device_type(spec.device) != "cpu":
            raise ValueError("CpuStatelessScorer is explicitly CPU-only")
        resolved_config = config or StatelessForwardConfig(
            mode="reference",
            attention_backend="eager",
            output_dtype=_torch_dtype(spec.dtype),
        )
        if resolved_config.mode not in {"reference", "both"}:
            raise ValueError("CpuStatelessScorer requires reference scoring mode")
        expected_dtype = _torch_dtype(spec.dtype)
        if resolved_config.output_dtype is not expected_dtype:
            raise ValueError("stateless output_dtype must match the scorer dtype")
        _require_module_on_cpu(model)
        _require_module_float_dtype(model, expected_dtype)
        self.model = model
        self.spec = spec
        self.config = resolved_config

    def score(
        self,
        batch: CanonicalScoringBatch,
        *,
        batch_size: int,
        operator: Any,
    ) -> tuple[RankScore, ...]:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")

        def selected_logprob_fn(
            logits: torch.Tensor,
            token_ids: torch.Tensor,
            *,
            mask: Optional[torch.Tensor] = None,
            temperature: float = 1.0,
            output_dtype: torch.dtype = torch.float32,
        ) -> torch.Tensor:
            return selected_logprobs_with_operator(
                operator,
                logits,
                token_ids,
                active_mask=mask,
                temperature=temperature,
                output_dtype=output_dtype,
            )

        executor = StatelessForwardExecutor(
            self.model,
            self.config,
            selected_logprob_fn=selected_logprob_fn,
        )
        chunks: list[torch.Tensor] = []
        observed_ranges: list[tuple[int, int]] = []
        for start in range(0, batch.input_ids.shape[0], batch_size):
            stop = min(start + batch_size, batch.input_ids.shape[0])
            inputs = StatelessForwardInputs(
                input_ids=batch.input_ids[start:stop],
                attention_mask=batch.attention_mask[start:stop],
                completion_mask=batch.active_mask[start:stop],
                labels=batch.selected_token_ids[start:stop],
                position_ids=(
                    None if batch.position_ids is None else batch.position_ids[start:stop]
                ),
            )
            result = executor.score(inputs)
            if result.reference_logps is None:  # pragma: no cover - guarded by config mode
                raise RuntimeError("stateless scorer returned no selected logprobs")
            chunks.append(result.reference_logps.detach().to(device="cpu"))
            observed_ranges.append((start, stop))
        selected = torch.cat(chunks, dim=0)
        return (
            RankScore(
                rank=0,
                world_size=1,
                selected_logprobs=selected,
                metadata={
                    "device": "cpu",
                    "teacher_forcing": True,
                    "use_cache": False,
                    "optimizer_step": False,
                    "batch_ranges": observed_ranges,
                },
            ),
        )


class CpuSmokeMaterializer:
    """Materialize the exact single-process CPU surface used by smoke tests."""

    runtime_kind = "cpu_smoke"

    @property
    def implementation_fingerprint(self) -> str:
        """Seal the adapter's concrete class and materialization entry point."""

        return fingerprint_implementation(
            type(self),
            instance=self,
            entrypoints=("materialize",),
        )

    def __init__(
        self,
        *,
        requested_operator_backends: Optional[Mapping[str, str]] = None,
        actual_operator_backends: Optional[Mapping[str, str]] = None,
    ):
        self.requested_operator_backends = dict(requested_operator_backends or {})
        self.actual_operator_backends = dict(actual_operator_backends or {})

    def materialize(
        self,
        normalized: Mapping[str, Any],
        descriptors: Mapping[str, KnobDescriptor],
    ) -> AdapterMaterialization:
        flat = _flatten(normalized)
        applications = tuple(
            self._application(path, value, descriptors[path]) for path, value in flat.items()
        )
        batch_size = int(flat["batch.size"])
        requested_logp = str(flat["logp.backend"])
        operator_backends = self.requested_operator_backends or {
            "rollout": requested_logp,
            "training": requested_logp,
        }
        return AdapterMaterialization(
            applications=applications,
            binding=RuntimeBinding(
                batch_size=batch_size,
                side_configs={
                    "rollout": {
                        "device": "cpu",
                        "dtype": "float32",
                        "enable_prefix_caching": False,
                        "enforce_eager": True,
                    },
                    "training": {
                        "device": "cpu",
                        "dtype": "float32",
                        "attention_backend": "eager",
                        "sharding": "unsharded",
                    },
                },
                topology={
                    "rollout": {
                        "world_size": 1,
                        "tensor_parallel_size": 1,
                        "context_parallel_size": 1,
                    },
                    "training": {"world_size": 1, "sharding": "unsharded"},
                },
                scorer={
                    "mode": "reference",
                    "use_cache": False,
                    "attention_backend": "eager",
                    "output_dtype": "float32",
                },
                operator_backends=operator_backends,
                runtime_kind=self.runtime_kind,
            ),
        )

    def _application(
        self,
        path: str,
        requested: Any,
        descriptor: KnobDescriptor,
    ) -> KnobApplication:
        fixed_values = {
            "rollout.tensor_parallel_size": 1,
            "rollout.context_parallel_size": 1,
            "rollout.dtype": "float32",
            "rollout.enable_prefix_caching": False,
            "rollout.enforce_eager": True,
            "training.attention_backend": "eager",
            "training.compute_dtype": "float32",
            "training.sharding": "unsharded",
        }
        if path == "batch.size":
            return _application(
                descriptor,
                requested,
                requested,
                requested,
                MaterializationStatus.APPLIED,
                "canonical batch is partitioned at scorer invocation",
            )
        if path == "logp.backend":
            requested_backends = self.requested_operator_backends or {
                "rollout": requested,
                "training": requested,
            }
            if requested_backends.get("rollout") != requested:
                return _application(
                    descriptor,
                    requested,
                    requested_backends,
                    None,
                    MaterializationStatus.ERROR,
                    "rollout operator conflicts with public logp.backend",
                )
            actual_backends = {
                "rollout": self.actual_operator_backends.get("rollout"),
                "training": self.actual_operator_backends.get("training"),
            }
            if None in actual_backends.values():
                return _application(
                    descriptor,
                    requested,
                    requested_backends,
                    None,
                    MaterializationStatus.UNOBSERVABLE,
                    "operator resolution trace has not been supplied",
                )
            status = (
                MaterializationStatus.APPLIED
                if actual_backends == requested_backends
                else MaterializationStatus.FALLBACK
            )
            return _application(
                descriptor,
                requested,
                requested_backends,
                actual_backends,
                status,
                "concrete CPU backends were read from exact resolution traces",
            )

        actual = fixed_values[path]
        status = (
            MaterializationStatus.APPLIED
            if requested == actual
            else MaterializationStatus.UNSUPPORTED
        )
        reason = (
            "read back from the single-process CPU scorer"
            if status is MaterializationStatus.APPLIED
            else f"CPU smoke supports only {path}={actual!r}"
        )
        return _application(descriptor, requested, requested, actual, status, reason)


def run_cpu_experiment(
    config: ExperimentConfig,
    *,
    output_root: str | Path,
    allow_smoke_operators: bool = False,
    timeout_seconds: float = 30.0,
    resume: bool = True,
) -> dict[str, Any]:
    """Run every planned case through the explicit CPU smoke adapter."""

    scenario_device = str(config.definition.scenario.get("device", "")).strip().lower()
    if scenario_device != "cpu":
        raise ValueError("the CPU runtime requires scenario.device='cpu'")
    plan = build_execution_plan(config)

    store = ArtifactStore(output_root)
    experiment_dir = store.initialize_experiment(
        config.definition.experiment_id,
        experiment=plan.experiment,
        plan=plan.rows(),
    )
    batch = canonical_cpu_batch(config)
    runs = [
        run_cpu_case(
            store,
            entry.case,
            batch,
            entry.operators,
            allow_smoke_operators=allow_smoke_operators,
            strict=config.definition.strict_fallback,
            timeout_seconds=timeout_seconds,
            resume=resume,
        )
        for entry in plan.entries
    ]
    cases = [
        {
            "case_id": run.case_id,
            "attempt_id": run.attempt_id,
            "status": str(run.summary["status"]),
            "rollout_backend": run.summary["rollout_backend"],
            "training_backend": run.summary["training_backend"],
            "mismatch_count": run.summary.get("mismatch_count"),
            "worst_token_index": run.summary.get("worst_token_index"),
            "resumed": run.resumed,
            "attempt_dir": str(run.attempt_dir),
        }
        for run in runs
    ]
    return {
        "schema_version": "cross_config.cli_summary.v1",
        "status": "pass" if all(item["status"] == "pass" for item in cases) else "fail",
        "experiment_id": config.definition.experiment_id,
        "scenario_id": config.definition.scenario_id,
        "runtime": "cpu-smoke",
        "artifact_dir": str(experiment_dir),
        "cases": cases,
    }


def run_cpu_case(
    store: ArtifactStore,
    case: ExperimentCase,
    batch: CanonicalScoringBatch,
    selection: OperatorSelection,
    *,
    allow_smoke_operators: bool,
    strict: bool,
    timeout_seconds: float,
    resume: bool,
) -> PairedRunResult:
    """Execute one already-bound CPU case with case-local operator state."""

    catalog = SemanticOperatorCatalog(kernel_registry.semantic.backend_descriptors())
    if allow_smoke_operators:
        from rl_engine.validation.fixtures.smoke_ops import register_smoke_operators

        register_smoke_operators(catalog, allow_smoke_operators=True)
    bridge = OperatorBridge(
        catalog,
        policy=OperatorResolutionPolicy(
            strict=strict,
            allow_test_backends=allow_smoke_operators,
        ),
    )
    override = OperatorOverride(
        semantic_op="selected_logprob",
        rollout_backend=selection.rollout_backend,
        training_backend=selection.training_backend,
    )
    topologies: dict[str, Mapping[str, Any]] = {
        "rollout": {
            "world_size": 1,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
        },
        "training": {"world_size": 1, "sharding": "unsharded"},
    }
    rollout_dtype = str(case.requested["rollout"]["dtype"])
    training_dtype = str(case.requested["training"]["compute_dtype"])
    requirements = {
        "rollout": OperatorRequirements(
            device="cpu",
            dtype=rollout_dtype,
            topology=topologies["rollout"],
            alignment_properties={"deterministic": True},
        ),
        "training": OperatorRequirements(
            device="cpu",
            dtype=training_dtype,
            topology=topologies["training"],
            alignment_properties={"deterministic": True},
        ),
    }
    resolved = bridge.resolve_override(override, requirements=requirements, strict=strict)
    options = {
        target: _factory_options(
            selection.backend_for(target),
            selection.options_for(target),
            allow_smoke_operators=allow_smoke_operators,
        )
        for target in ("rollout", "training")
    }
    instances = {
        target: bridge.instantiate(
            resolved,
            target=target,  # type: ignore[arg-type]
            factory_kwargs=options[target],
        )
        for target in ("rollout", "training")
    }
    provenance = {
        target: bridge.instance_provenance(
            resolved,
            target=target,  # type: ignore[arg-type]
            instance=instances[target],
        )
        for target in ("rollout", "training")
    }
    actual_backends = {target: provenance[target].backend_id for target in provenance}
    materialization = RuntimeTools().materialize(
        case,
        CpuSmokeMaterializer(
            requested_operator_backends={
                "rollout": selection.rollout_backend,
                "training": selection.training_backend,
            },
            actual_operator_backends=actual_backends,
        ),
    )
    RuntimeTools.require_executable(materialization, strict=strict)

    minimum_token_id = min(
        int(batch.input_ids.min().item()),
        int(batch.selected_token_ids.min().item()),
    )
    if minimum_token_id < 0:
        raise ValueError("CPU smoke token IDs must be non-negative")
    vocab_size = (
        max(
            int(batch.input_ids.max().item()),
            int(batch.selected_token_ids.max().item()),
        )
        + 17
    )
    scorers = {
        "rollout": CpuStatelessScorer(
            SyntheticCpuCausalLM(vocab_size),
            _scorer_spec(
                ScoreSide.ROLLOUT,
                rollout_dtype,
                selection.rollout_backend,
                topologies["rollout"],
                case,
            ),
        ),
        "training": CpuStatelessScorer(
            SyntheticCpuCausalLM(vocab_size),
            _scorer_spec(
                ScoreSide.TRAINING,
                training_dtype,
                selection.training_backend,
                topologies["training"],
                case,
            ),
        ),
    }
    return PairedRunner(store, timeout_seconds=timeout_seconds).run(
        case,
        materialization,
        batch,
        scorers["rollout"],
        scorers["training"],
        resolved,
        instances,
        provenance,
        operator_factory_options=options,
        strict=strict,
        timeout_seconds=timeout_seconds,
        resume=resume,
    )


def canonical_cpu_batch(config: ExperimentConfig) -> CanonicalScoringBatch:
    """Build the immutable CPU tensors frozen by an experiment identity."""

    identity = config.definition.identity
    position_ids = (
        torch.tensor(identity.position_ids, dtype=torch.long, device="cpu")
        if identity.position_ids
        else None
    )
    return CanonicalScoringBatch(
        identity=identity,
        input_ids=torch.tensor(identity.token_ids, dtype=torch.long, device="cpu"),
        selected_token_ids=torch.tensor(
            identity.selected_token_ids,
            dtype=torch.long,
            device="cpu",
        ),
        active_mask=torch.tensor(identity.active_mask, dtype=torch.bool, device="cpu"),
        attention_mask=torch.tensor(
            identity.attention_mask,
            dtype=torch.bool,
            device="cpu",
        ),
        position_ids=position_ids,
        metadata={"source": "named_json", "device": "cpu"},
    )


def _factory_options(
    backend_id: str,
    configured: Mapping[str, Any],
    *,
    allow_smoke_operators: bool,
) -> dict[str, Any]:
    options = dict(configured)
    if backend_id == "smoke_only.logp_offset":
        if not allow_smoke_operators:
            raise PermissionError("smoke offset requires explicit test authorization")
        options["allow_smoke_operators"] = True
    return options


def _scorer_spec(
    side: ScoreSide,
    dtype: str,
    backend_id: str,
    topology: Mapping[str, Any],
    case: ExperimentCase,
) -> ScorerSpec:
    identity = case.identity
    return ScorerSpec(
        side=side,
        backend_id="cpu_stateless_teacher_forcing",
        dtype=dtype,
        device="cpu",
        world_size=1,
        topology=topology,
        construction_options={
            "checkpoint_id": identity.checkpoint_id,
            "model_version": identity.model_version,
            "pre_update_state": identity.pre_update_state,
            "teacher_forcing": True,
            "use_cache": False,
        },
        operator_overrides={"selected_logprob": backend_id},
    )


def _application(
    descriptor: KnobDescriptor,
    requested: Any,
    materialized: Any,
    actual: Any,
    status: MaterializationStatus,
    reason: str,
) -> KnobApplication:
    return KnobApplication(
        path=descriptor.path,
        requested=requested,
        materialized=materialized,
        actual=actual,
        lifecycle=descriptor.lifecycle,
        status=status,
        critical=descriptor.critical,
        evidence={"reason": reason},
    )


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, Mapping):
            result.update(_flatten(child, path))
        else:
            result[path] = child
    return result


def _require_module_on_cpu(model: torch.nn.Module) -> None:
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("CPU smoke models must remain on CPU")


def _require_module_float_dtype(model: torch.nn.Module, expected: torch.dtype) -> None:
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    mismatched = sorted(
        {
            str(tensor.dtype).replace("torch.", "")
            for tensor in tensors
            if tensor.is_floating_point() and tensor.dtype is not expected
        }
    )
    if mismatched:
        raise ValueError(
            f"CPU smoke model floating dtype must be {expected}; observed {mismatched}"
        )


def _device_type(value: str) -> str:
    return value.split(":", 1)[0].strip().lower()


def _torch_dtype(value: str) -> torch.dtype:
    normalized = value.strip().lower().replace("torch.", "")
    aliases = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}
    normalized = aliases.get(normalized, normalized)
    try:
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported scorer dtype {value!r}") from exc


__all__ = [
    "CpuSmokeMaterializer",
    "CpuStatelessScorer",
    "SyntheticCpuCausalLM",
    "canonical_cpu_batch",
    "run_cpu_case",
    "run_cpu_experiment",
]
