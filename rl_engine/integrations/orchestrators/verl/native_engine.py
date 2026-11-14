# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- install the explicit import ABI adapters before upstream imports.
"""Opt-in real verl TrainingWorker engine with pristine upstream dependencies.

Register backend ``rlk_megatron`` before constructing TrainingWorker. Native
MCore constructs and schedules the model; RLK owns packed loading/operator
bindings, the raw-score boundary and versioned export/update only.
"""

from contextlib import ExitStack
from dataclasses import dataclass

import torch

from .native_imports import install_native_import_adapters

install_native_import_adapters()

from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
from megatron.core.utils import get_model_config
from verl.workers.config import McoreEngineConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

from rl_engine.integrations.engines.train.megatron.native_mhc import (
    normalize_native_causal_mask,
    selected_raw_logprobs,
)
from rl_engine.integrations.orchestrators.verl.mhc import bind_mhc_worker


@dataclass
class RLKNativeMcoreEngineConfig(McoreEngineConfig):
    strategy: str = "rlk_megatron"
    rlk_weight_version: int = 0
    rlk_mhc_backend: str = "rocm"

    def __post_init__(self):
        if self.strategy != "rlk_megatron":
            raise ValueError("use the explicit rlk_megatron engine registration")
        object.__setattr__(self, "strategy", "megatron")
        try:
            super().__post_init__()
        finally:
            object.__setattr__(self, "strategy", "rlk_megatron")


@EngineRegistry.register("language_model", "rlk_megatron")
class RLKNativeMegatronEngine(MegatronEngineWithLMHead):
    def _build_megatron_module(self):
        from rl_engine.integrations.engines.train.megatron.checkpoint import load_native_checkpoint

        cfg = self.engine_config
        if any(
            getattr(cfg, key) != 1
            for key in (
                "tensor_model_parallel_size",
                "pipeline_model_parallel_size",
                "expert_model_parallel_size",
                "context_parallel_size",
            )
        ):
            raise ValueError("native packed DSV4 engine initially requires TP=PP=EP=CP=1")
        if self.peft_cls is not None or cfg.use_megatron_fsdp or self.is_value_model:
            raise ValueError("native packed DSV4 engine supports the ordinary actor/DDP path")
        self._rlk_models = []
        self.rlk_weight_version = cfg.rlk_weight_version

        def load_and_bind(models):
            for model in models:
                model._rlk_fused_qnorm = True
                model._rlk_original_qkv_codec = True
                load_native_checkpoint(model, self.model_config.local_path)
                if cfg.rlk_mhc_backend == "rocm_triton":
                    from rl_engine.integrations.engines.train.megatron.native_mhc import (
                        original_mhc_parameters,
                    )

                    self._rlk_mhc_parameters = original_mhc_parameters(model)
                worker = bind_mhc_worker(
                    model,
                    "training",
                    backend=cfg.rlk_mhc_backend,
                    mode="split",
                    version=self.rlk_weight_version,
                    observer=lambda *r: print(
                        "RLK_ACTOR_EXECUTION", r[:-1], tuple(r[-1].shape), flush=True
                    ),
                )
                original = model.forward

                def forward(*args, _forward=original, _worker=worker, **kwargs):
                    if "attention_mask" in kwargs:
                        kwargs["attention_mask"] = normalize_native_causal_mask(
                            kwargs["attention_mask"], kwargs["input_ids"].shape
                        )
                    with ExitStack() as scopes:
                        if _worker.backend == "rocm_triton":
                            from rl_engine.integrations.engines.train.megatron.logical_keys import (
                                logical_keys_for_input,
                            )

                            keys = logical_keys_for_input(
                                kwargs["input_ids"], self.rlk_logical_cases
                            )
                            scopes.enter_context(_worker.logical_rows(keys))
                        scopes.enter_context(_worker.request(self.rlk_weight_version))
                        return _forward(*args, **kwargs)

                model.forward = forward
                self._rlk_models.append(model)
            return models

        self.provider.register_pre_wrap_hook(load_and_bind)
        # Use the native config ABI directly: Bridge 0.7's derived finalize calls
        # an upstream __post_init__ that was removed in this pinned native MCore.
        ddp = None
        if not cfg.forward_only:
            opts = dict(self._resolve_override_ddp_config())
            opts.setdefault("use_distributed_optimizer", cfg.use_distributed_optimizer)
            opts.setdefault("grad_reduce_in_fp32", True)
            if opts.get("overlap_grad_reduce") or opts.get("overlap_param_gather"):
                raise ValueError("initial DSV4 engine does not support overlapped DDP")
            ddp = DistributedDataParallelConfig(**opts)
            ddp.finalize()
        models = self.provider.provide_distributed_model(
            wrap_with_ddp=not cfg.forward_only,
            ddp_config=ddp,
            fp16=self.provider.fp16,
            bf16=self.provider.bf16,
            use_megatron_fsdp=False,
        )
        if len(models) != 1 or len(self._rlk_models) != 1:
            raise ValueError("initial native DSV4 engine requires one local model")
        self.tf_config = get_model_config(models[0])
        if cfg.rlk_mhc_backend == "rocm_triton":
            from rl_engine.integrations.engines.train.megatron.native_mhc import (
                original_mhc_parameters,
            )

            current = original_mhc_parameters(self._rlk_models[0])
            if set(current) != set(self._rlk_mhc_parameters) or any(
                current[name] is not self._rlk_mhc_parameters[name] for name in current
            ):
                raise RuntimeError("native Float16/DDP wrapping replaced an mHC Parameter")
            print("RLK_AMD662_FP32_PARAMETER_IDENTITY", sorted(current), flush=True)
        return models

    def _build_optimizer(self):
        from verl.utils.megatron.optimizer import get_megatron_optimizer, init_megatron_optim_config

        conf = init_megatron_optim_config(
            self.optimizer_config,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            fp16=self.param_dtype == torch.float16,
            bf16=self.param_dtype == torch.bfloat16,
        )
        optimizer = get_megatron_optimizer(model=self.module, config=conf)
        if self.engine_config.rlk_mhc_backend == "rocm_triton":
            registered = {id(p) for group in optimizer.param_groups for p in group["params"]}
            missing = [
                name for name, p in self._rlk_mhc_parameters.items() if id(p) not in registered
            ]
            if missing:
                raise RuntimeError(f"original mHC Parameters absent from optimizer: {missing}")
            print(
                "RLK_AMD662_OPTIMIZER_PARAMETER_IDENTITY", len(self._rlk_mhc_parameters), flush=True
            )
        for chunk in self.module:
            config = get_model_config(chunk)
            config.grad_scale_func = optimizer.scale_loss
            config.finalize_model_grads_func = finalize_model_grads
        return optimizer

    def forward_backward_batch(self, data, loss_function=None, forward_only=False):
        if self.engine_config.rlk_mhc_backend != "rocm_triton" or forward_only:
            return super().forward_backward_batch(data, loss_function, forward_only=forward_only)
        from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

        with canonical_backward_session() as session:
            result = super().forward_backward_batch(data, loss_function, forward_only=False)
            session.validate_complete()
            print(
                "RLK_AMD662_CANONICAL_BACKWARD_COMPLETED",
                {name: len(entries) for name, entries in session.uses.items()},
                session.received,
                flush=True,
            )
        return result

    def _lm_head_logits_processor(self, logits, label, temperature, **kwargs):
        model = self._rlk_models[0]
        with model._rlk_p1.request(self.rlk_weight_version):
            raw = selected_raw_logprobs(
                model, logits, label, weight_version=self.rlk_weight_version
            )
        # Parent owns its transformed log_probs/entropy API. Keep the original
        # custom-autograd logits untouched and expose full-distribution raw scores.
        result = super()._lm_head_logits_processor(logits.clone(), label, temperature, **kwargs)
        result["raw_log_probs"] = raw
        return result

    def optimizer_step(self):
        version = self.rlk_weight_version + 1
        with ExitStack() as stack:
            for model in self._rlk_models:
                stack.enter_context(model._rlk_p1.updating(version, torch.cuda.synchronize))
            norm = super().optimizer_step()
        self.rlk_weight_version = version
        return norm

    def get_per_tensor_param(self, base_sync_done=False, **kwargs):
        from rl_engine.integrations.engines.train.megatron.checkpoint import (
            export_native_checkpoint_parameters,
        )

        return export_native_checkpoint_parameters(self._rlk_models[0]), None
