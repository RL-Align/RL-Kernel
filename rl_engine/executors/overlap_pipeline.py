# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Optional, Protocol, Sequence, cast

import torch

from rl_engine.executors.bridge import WeightPublisher, WeightUpdateManifest, make_weight_bridge
from rl_engine.executors.rollout import RolloutExecutor
from rl_engine.executors.training_contract import (
    RolloutBatchMixin,
    RolloutStageResult,
    TorchRLTrainingConfig,
    TrainingStageResult,
    extract_rollout_token_groups,
)
from rl_engine.testing import (
    compute_policy_ratio,
    compute_reference_kl,
    masked_mean,
    selected_logprobs_reference,
)


@dataclass(frozen=True)
class PipelineConfig:
    """Local overlap scheduler configuration."""

    max_prefetch: int = 1
    stop_on_error: bool = True
    rollout_workers: int = 1
    training_workers: int = 1
    initial_weight_version: int = 0
    weight_version_policy: str = "published"

    def __post_init__(self) -> None:
        if self.max_prefetch < 1:
            raise ValueError("max_prefetch must be >= 1")
        if self.rollout_workers < 1:
            raise ValueError("rollout_workers must be >= 1")
        if self.training_workers < 1:
            raise ValueError("training_workers must be >= 1")
        if self.weight_version_policy not in {"published", "spec"}:
            raise ValueError("weight_version_policy must be 'published' or 'spec'")


@dataclass(frozen=True)
class IterationSpec:
    """One scheduled rollout/training iteration."""

    iteration: int
    weight_version: Optional[int] = None
    prompts: Sequence[Any] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WeightHandoffRecord:
    """One published and installed weight manifest handoff."""

    iteration: int
    weight_version: int
    update_id: str
    transport: str
    tensor_count: int
    total_nbytes: int
    published_at: float
    installed_at: Optional[float] = None


@dataclass(frozen=True)
class PipelineTimelineSummary:
    """Serializable summary of one pipeline run."""

    started_at: float
    finished_at: float
    elapsed_seconds: float
    sequential_estimate_seconds: float
    overlap_seconds: float
    overlap_ratio: float
    max_queue_depth: int
    rollout_results: list[Mapping[str, Any]]
    training_results: list[Mapping[str, Any]]
    weight_handoffs: list[Mapping[str, Any]]
    final_published_weight_version: int


class RolloutWorker(Protocol):
    def rollout(self, spec: IterationSpec) -> RolloutStageResult: ...


class TrainingWorker(Protocol):
    def train(self, rollout: RolloutStageResult) -> TrainingStageResult: ...


class PipelineExecutionError(RuntimeError):
    """Raised when a rollout, training, or weight handoff stage fails."""

    def __init__(self, stage: str, iteration: int, cause: BaseException):
        super().__init__(f"{stage} failed for iteration {iteration}: {cause}")
        self.stage = stage
        self.iteration = iteration
        self.cause = cause


class ManifestWeightHandoff:
    """
    Publish training weights and install them on rollout workers at safe boundaries.

    Publication happens only after a complete training step. Installation is
    deferred until no rollout future is active, so a rollout engine is not
    hot-updated while it may still be generating with the previous version.
    """

    def __init__(self, *, release_on_shutdown: bool = True):
        self.release_on_shutdown = bool(release_on_shutdown)
        self.records: list[WeightHandoffRecord] = []
        self._pending: list[tuple[WeightUpdateManifest, WeightHandoffRecord]] = []
        self._installed_update_ids: list[str] = []

    def publish(
        self,
        training_worker: TrainingWorker,
        result: TrainingStageResult,
    ) -> Optional[WeightHandoffRecord]:
        if result.published_weight_version is None:
            return None
        self._release_pending(training_worker)
        publish_weights = getattr(training_worker, "publish_weights", None)
        if not callable(publish_weights):
            raise RuntimeError(
                "manifest weight handoff requires the training worker to expose "
                "publish_weights(weight_version=..., metadata=...)"
            )

        manifest = publish_weights(
            weight_version=result.published_weight_version,
            metadata={
                "issue": "18",
                "pipeline_iteration": result.iteration,
                "consumed_weight_version": result.consumed_weight_version,
            },
        )
        record = WeightHandoffRecord(
            iteration=result.iteration,
            weight_version=manifest.weight_version,
            update_id=manifest.update_id,
            transport=manifest.transport,
            tensor_count=manifest.tensor_count,
            total_nbytes=manifest.total_nbytes,
            published_at=time.perf_counter(),
        )
        self._pending.append((manifest, record))
        return record

    def pending_iteration(self) -> int:
        if not self._pending:
            return -1
        return int(self._pending[-1][1].iteration)

    def install_latest(
        self,
        rollout_worker: RolloutWorker,
        training_worker: Optional[TrainingWorker] = None,
    ) -> Optional[WeightHandoffRecord]:
        del training_worker
        if not self._pending:
            return None

        install = _resolve_manifest_install(rollout_worker)
        latest_manifest, latest_record = self._pending[-1]
        installed_at = time.perf_counter()
        install(latest_manifest)
        installed_record = replace(latest_record, installed_at=installed_at)
        self.records.append(installed_record)
        self._installed_update_ids.append(latest_manifest.update_id)
        self._pending.clear()
        return installed_record

    def release_all(
        self,
        training_worker: TrainingWorker,
        rollout_worker: RolloutWorker,
    ) -> None:
        if not self.release_on_shutdown:
            return

        release_rollout = getattr(rollout_worker, "release_weight_manifest", None)
        if callable(release_rollout):
            for update_id in reversed(self._installed_update_ids):
                release_rollout(update_id)

        release_training = getattr(training_worker, "release_weights", None)
        if callable(release_training):
            update_ids = [record.update_id for record in self.records]
            update_ids.extend(manifest.update_id for manifest, _record in self._pending)
            for update_id in reversed(update_ids):
                release_training(update_id)

        self._pending.clear()
        self._installed_update_ids.clear()

    def _release_pending(self, training_worker: TrainingWorker) -> None:
        if not self._pending:
            return
        release_training = getattr(training_worker, "release_weights", None)
        if callable(release_training):
            for manifest, _record in self._pending:
                release_training(manifest.update_id)
        self._pending.clear()


class OverlapPipeline:
    """Coordinate rollout and training stages with bounded prefetch overlap."""

    def __init__(
        self,
        rollout_worker: RolloutWorker,
        training_worker: TrainingWorker,
        config: Optional[PipelineConfig] = None,
        *,
        weight_handoff: Optional[ManifestWeightHandoff] = None,
    ):
        self.rollout_worker = rollout_worker
        self.training_worker = training_worker
        self.config = config or PipelineConfig()
        self.weight_handoff = weight_handoff
        self.rollout_results: list[RolloutStageResult] = []
        self.training_results: list[TrainingStageResult] = []
        self.weight_handoffs: list[WeightHandoffRecord] = []
        self.max_queue_depth = 0
        self.started_at = 0.0
        self.finished_at = 0.0
        self.final_published_weight_version = self.config.initial_weight_version

    def run(self, iterations: Sequence[IterationSpec]) -> list[TrainingStageResult]:
        specs = list(iterations)
        if not specs:
            return []

        self.rollout_results = []
        self.training_results = []
        self.weight_handoffs = []
        self.max_queue_depth = 0
        self.started_at = time.perf_counter()
        self.finished_at = self.started_at
        self.final_published_weight_version = self.config.initial_weight_version

        rollout_futures: dict[Future[RolloutStageResult], int] = {}
        training_futures: dict[Future[TrainingStageResult], int] = {}
        buffered_rollouts: dict[int, RolloutStageResult] = {}
        submitted_specs: dict[int, IterationSpec] = {}
        next_rollout_index = 0
        next_training_index = 0
        current_published_weight_version = self.config.initial_weight_version
        current_rollout_weight_version = self.config.initial_weight_version

        def rollout_backlog() -> int:
            return len(rollout_futures) + len(buffered_rollouts)

        def rollout_spec_for(index: int) -> IterationSpec:
            base_spec = specs[index]
            if self.config.weight_version_policy == "spec":
                if base_spec.weight_version is None:
                    raise ValueError("IterationSpec.weight_version is required for spec policy")
                return base_spec
            visible_version = (
                current_rollout_weight_version
                if self.weight_handoff is not None
                else current_published_weight_version
            )
            return replace(base_spec, weight_version=visible_version)

        def install_pending_weights_if_idle() -> None:
            nonlocal current_rollout_weight_version
            if self.weight_handoff is None or rollout_futures:
                return
            try:
                installed = self.weight_handoff.install_latest(
                    self.rollout_worker,
                    self.training_worker,
                )
            except BaseException as exc:
                raise PipelineExecutionError(
                    "weight_handoff",
                    self.weight_handoff.pending_iteration(),
                    exc,
                ) from exc
            if installed is None:
                return
            current_rollout_weight_version = installed.weight_version
            self.weight_handoffs.append(installed)

        def submit_rollouts(executor: ThreadPoolExecutor) -> None:
            nonlocal next_rollout_index
            while next_rollout_index < len(specs) and rollout_backlog() < self.config.max_prefetch:
                index = next_rollout_index
                rollout_spec = rollout_spec_for(index)
                submitted_specs[index] = rollout_spec
                future = executor.submit(self.rollout_worker.rollout, rollout_spec)
                rollout_futures[future] = index
                next_rollout_index += 1
            self.max_queue_depth = max(self.max_queue_depth, rollout_backlog())

        def submit_training(executor: ThreadPoolExecutor) -> None:
            nonlocal next_training_index
            while (
                next_training_index < len(specs)
                and next_training_index in buffered_rollouts
                and len(training_futures) < self.config.training_workers
            ):
                rollout = buffered_rollouts.pop(next_training_index)
                future = executor.submit(self.training_worker.train, rollout)
                training_futures[future] = next_training_index
                next_training_index += 1
            self.max_queue_depth = max(self.max_queue_depth, rollout_backlog())

        try:
            with (
                ThreadPoolExecutor(
                    max_workers=self.config.rollout_workers,
                    thread_name_prefix="rl-kernel-rollout",
                ) as rollout_executor,
                ThreadPoolExecutor(
                    max_workers=self.config.training_workers,
                    thread_name_prefix="rl-kernel-training",
                ) as training_executor,
            ):
                submit_rollouts(rollout_executor)

                while len(self.training_results) < len(specs):
                    install_pending_weights_if_idle()
                    submit_training(training_executor)
                    submit_rollouts(rollout_executor)

                    active_futures: set[Future[Any]] = set(rollout_futures) | set(training_futures)
                    if not active_futures:
                        break

                    done, _ = wait(active_futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        if future in rollout_futures:
                            index = rollout_futures.pop(future)
                            rollout_spec = submitted_specs[index]
                            try:
                                result = future.result()
                            except BaseException as exc:
                                self._cancel_pending(rollout_futures, training_futures)
                                raise PipelineExecutionError(
                                    "rollout", rollout_spec.iteration, exc
                                ) from exc
                            self._validate_rollout_result(rollout_spec, result)
                            buffered_rollouts[index] = result
                            self.rollout_results.append(result)
                            self.max_queue_depth = max(self.max_queue_depth, rollout_backlog())
                        elif future in training_futures:
                            index = training_futures.pop(future)
                            rollout_spec = submitted_specs[index]
                            try:
                                result = future.result()
                            except BaseException as exc:
                                self._cancel_pending(rollout_futures, training_futures)
                                raise PipelineExecutionError(
                                    "training", rollout_spec.iteration, exc
                                ) from exc
                            try:
                                self._validate_training_result(rollout_spec, result)
                                if result.published_weight_version is not None:
                                    if self.weight_handoff is not None:
                                        self.weight_handoff.publish(self.training_worker, result)
                                    current_published_weight_version = max(
                                        current_published_weight_version,
                                        result.published_weight_version,
                                    )
                                self.training_results.append(result)
                            except BaseException as exc:
                                self._cancel_pending(rollout_futures, training_futures)
                                raise PipelineExecutionError(
                                    "weight_handoff", rollout_spec.iteration, exc
                                ) from exc

                return list(self.training_results)
        finally:
            try:
                install_pending_weights_if_idle()
            finally:
                if self.weight_handoff is not None:
                    self.weight_handoff.release_all(self.training_worker, self.rollout_worker)
            self.final_published_weight_version = current_published_weight_version
            self.finished_at = time.perf_counter()

    def timeline_summary(self) -> PipelineTimelineSummary:
        finished_at = self.finished_at or time.perf_counter()
        rollout_rows: list[Mapping[str, Any]] = [
            _stage_to_dict(result) for result in self.rollout_results
        ]
        training_rows: list[Mapping[str, Any]] = [
            _stage_to_dict(result) for result in self.training_results
        ]
        sequential = sum(result.duration_seconds for result in self.rollout_results) + sum(
            result.duration_seconds for result in self.training_results
        )
        overlap = compute_stage_overlap_seconds(self.rollout_results, self.training_results)
        rollout_total = sum(result.duration_seconds for result in self.rollout_results)
        denominator = max(rollout_total, 1e-12)
        return PipelineTimelineSummary(
            started_at=self.started_at,
            finished_at=finished_at,
            elapsed_seconds=finished_at - self.started_at,
            sequential_estimate_seconds=sequential,
            overlap_seconds=overlap,
            overlap_ratio=overlap / denominator,
            max_queue_depth=self.max_queue_depth,
            rollout_results=rollout_rows,
            training_results=training_rows,
            weight_handoffs=[asdict(record) for record in self.weight_handoffs],
            final_published_weight_version=self.final_published_weight_version,
        )

    @staticmethod
    def _cancel_pending(
        rollout_futures: Mapping[Future[RolloutStageResult], int],
        training_futures: Mapping[Future[TrainingStageResult], int],
    ) -> None:
        for future in list(rollout_futures) + list(training_futures):
            future.cancel()

    @staticmethod
    def _validate_rollout_result(spec: IterationSpec, result: RolloutStageResult) -> None:
        if result.iteration != spec.iteration:
            raise ValueError(
                f"rollout result iteration {result.iteration} does not match spec "
                f"{spec.iteration}"
            )
        if result.weight_version != spec.weight_version:
            raise ValueError(
                f"rollout result weight_version {result.weight_version} does not match spec "
                f"{spec.weight_version}"
            )

    @staticmethod
    def _validate_training_result(spec: IterationSpec, result: TrainingStageResult) -> None:
        if result.iteration != spec.iteration:
            raise ValueError(
                f"training result iteration {result.iteration} does not match spec "
                f"{spec.iteration}"
            )


class RolloutExecutorWorker:
    """Production-facing rollout adapter over RolloutExecutor.generate_candidates."""

    def __init__(
        self,
        executor: RolloutExecutor,
        *,
        num_generations: Optional[int] = None,
        sampling_params: Optional[Mapping[str, Any]] = None,
    ):
        self.executor = executor
        self.num_generations = num_generations
        self.sampling_params = dict(sampling_params or {})

    def rollout(self, spec: IterationSpec) -> RolloutStageResult:
        if spec.weight_version is None:
            raise ValueError("IterationSpec.weight_version must be resolved before rollout")
        started_at = time.perf_counter()
        payload = self.executor.generate_candidates(
            spec.prompts,
            num_generations=self.num_generations,
            sampling_params=self.sampling_params,
        )
        finished_at = time.perf_counter()
        return RolloutStageResult(
            iteration=spec.iteration,
            weight_version=int(cast(int, spec.weight_version)),
            payload=payload,
            started_at=started_at,
            finished_at=finished_at,
            metrics={
                "num_prompts": len(spec.prompts),
                "backend": payload.get("backend") if isinstance(payload, Mapping) else None,
            },
        )

    def install_weight_manifest(self, manifest: WeightUpdateManifest) -> Mapping[str, torch.Tensor]:
        return self.executor.update_weights(manifest)

    def release_weight_manifest(self, update_id: str) -> None:
        release_update = getattr(self.executor, "release_weight_update", None)
        if callable(release_update):
            release_update(update_id)
        elif getattr(self.executor, "active_weight_update_id", None) == update_id:
            self.executor.release_weights()


class TorchRLTrainingWorker(RolloutBatchMixin):
    """DeepSpeed-style local training adapter using a real PyTorch optimizer step."""

    config: TorchRLTrainingConfig

    def __init__(
        self,
        config: Optional[TorchRLTrainingConfig] = None,
        *,
        weight_bridge: Optional[WeightPublisher] = None,
        weight_transport: str = "local-clone",
    ):
        self.config = config or TorchRLTrainingConfig()
        self.weight_bridge = weight_bridge or make_weight_bridge(
            weight_transport,
            source_worker="torch-training",
            source_rank=0,
        )
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but torch.cuda.is_available() is false")

        torch.manual_seed(self.config.seed)
        self.model = torch.nn.Sequential(
            torch.nn.Embedding(self.config.vocab_size, self.config.hidden_dim),
            torch.nn.Linear(self.config.hidden_dim, self.config.vocab_size),
        ).to(device=self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self._latest_published_weight_version = -1

    def train(self, rollout: RolloutStageResult) -> TrainingStageResult:
        started_at = time.perf_counter()
        batch, payload_metrics = self._batch_from_rollout_or_synthetic(rollout)

        logits = self.model(batch.token_ids.long())
        current_logps = selected_logprobs_reference(
            logits,
            batch.token_ids,
            mask=batch.completion_mask,
            output_dtype=torch.float32,
        )
        old_logps = current_logps.detach() - 0.01
        ref_logps = current_logps.detach() - 0.02
        ratio = compute_policy_ratio(current_logps, old_logps, batch.completion_mask)
        unclipped = ratio * batch.advantages.float()
        clipped = torch.clamp(ratio, 0.8, 1.2) * batch.advantages.float()
        policy_loss = -torch.minimum(unclipped, clipped)
        kl = compute_reference_kl(current_logps, ref_logps, batch.completion_mask)
        loss = masked_mean(policy_loss + 0.01 * kl, batch.completion_mask)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        finished_at = time.perf_counter()
        published = self._next_published_weight_version(rollout.weight_version)
        return TrainingStageResult(
            iteration=rollout.iteration,
            consumed_weight_version=rollout.weight_version,
            published_weight_version=published,
            metrics={
                "loss": float(loss.detach().cpu().item()),
                "active_tokens": int(batch.completion_mask.sum().item()),
                "payload_type": type(rollout.payload).__name__,
                "training_backend": "torch",
                **payload_metrics,
            },
            started_at=started_at,
            finished_at=finished_at,
        )

    def publish_weights(
        self,
        *,
        weight_version: int,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> WeightUpdateManifest:
        return self.weight_bridge.publish(
            self.model,
            weight_version=weight_version,
            metadata=metadata,
        )

    def release_weights(self, update_id: str) -> None:
        self.weight_bridge.release(update_id)

    def _next_published_weight_version(self, consumed_weight_version: int) -> int:
        published = max(
            self._latest_published_weight_version + 1,
            int(consumed_weight_version) + 1,
        )
        self._latest_published_weight_version = published
        return published


def compute_stage_overlap_seconds(
    rollouts: Sequence[RolloutStageResult],
    trainings: Sequence[TrainingStageResult],
) -> float:
    """Compute pairwise rollout/training interval overlap in seconds."""

    overlap = 0.0
    for rollout in rollouts:
        for training in trainings:
            if rollout.iteration == training.iteration:
                continue
            start = max(rollout.started_at, training.started_at)
            end = min(rollout.finished_at, training.finished_at)
            overlap += max(0.0, end - start)
    return overlap


def timeline_summary_to_dict(summary: PipelineTimelineSummary) -> dict[str, Any]:
    return asdict(summary)


def _resolve_manifest_install(rollout_worker: RolloutWorker):
    install = getattr(rollout_worker, "install_weight_manifest", None)
    if callable(install):
        return install
    update_weights = getattr(rollout_worker, "update_weights", None)
    if callable(update_weights):
        return update_weights
    executor = getattr(rollout_worker, "executor", None)
    update_weights = getattr(executor, "update_weights", None)
    if callable(update_weights):
        return update_weights
    raise RuntimeError(
        "manifest weight handoff requires the rollout worker to expose "
        "install_weight_manifest(manifest) or update_weights(manifest)"
    )


def _stage_to_dict(stage: RolloutStageResult | TrainingStageResult) -> dict[str, Any]:
    row = asdict(stage)
    row.pop("payload", None)
    row["duration_seconds"] = stage.duration_seconds
    return row


__all__ = [
    "IterationSpec",
    "ManifestWeightHandoff",
    "OverlapPipeline",
    "PipelineConfig",
    "PipelineExecutionError",
    "PipelineTimelineSummary",
    "RolloutExecutorWorker",
    "RolloutStageResult",
    "RolloutWorker",
    "TorchRLTrainingConfig",
    "TorchRLTrainingWorker",
    "TrainingStageResult",
    "TrainingWorker",
    "WeightHandoffRecord",
    "compute_stage_overlap_seconds",
    "extract_rollout_token_groups",
    "timeline_summary_to_dict",
]
