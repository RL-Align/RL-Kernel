# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Observe native VIME updates before its train_one_step clears gradients.

This module does not implement an optimizer or loss. The native training entry
points remain responsible for both; this observer rejects missing evidence.
"""

import hashlib
import math
import random
import struct
from contextlib import contextmanager

import numpy as np
import torch

from rl_engine.validation.models.qwen3_next.qwen3_next_rl_fixture import (
    require_policy_configuration,
)


def tensor_digest(value, *, chunk_bytes=64 * 1024 * 1024):
    """Hash original tensor bytes with bounded host memory, including dtype/shape."""
    if value.layout != torch.strided or value.device.type == "meta":
        raise ValueError("Only materialized strided tensors have a state identity")
    digest = hashlib.sha256()
    digest.update(str((str(value.dtype), tuple(value.shape))).encode())
    flat = value.detach().contiguous().reshape(-1)
    chunk_size = max(1, chunk_bytes // value.element_size())
    for start in range(0, flat.numel(), chunk_size):
        part = flat[start : start + chunk_size].cpu()
        if (part.is_floating_point() or part.is_complex()) and not torch.isfinite(part).all():
            raise ValueError("Nonfinite tensor in training state")
        digest.update(part.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_digest(state):
    """Canonical recursive identity for optimizer/scheduler/RNG state, without pickle."""
    digest = hashlib.sha256()

    def add(raw):
        digest.update(struct.pack("!Q", len(raw)))
        digest.update(raw)

    def visit(value):
        add(type(value).__name__.encode())
        if isinstance(value, torch.Tensor):
            add(tensor_digest(value).encode())
        elif isinstance(value, np.ndarray):
            add(str((value.dtype.str, value.shape)).encode())
            if value.dtype.hasobject:
                raise TypeError("Object arrays cannot identify recovery state")
            add(value.tobytes(order="C"))
        elif isinstance(value, dict):
            if any(type(key) not in (str, int) for key in value):
                raise TypeError("Recovery state keys must be strings or integers")
            for key in sorted(value, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif value is None or type(value) in (bool, int, str, bytes):
            add(repr(value).encode())
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("Nonfinite scalar in training state")
            add(struct.pack("!d", value))
        else:
            raise TypeError(f"Unsupported recovery state type: {type(value)!r}")

    visit(state)
    return digest.hexdigest()


def capture_rng_state(*, tracker=None):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [value.clone() for value in torch.cuda.get_rng_state_all()],
        "megatron": (
            None
            if tracker is None
            else {name: value.clone() for name, value in tracker.get_states().items()}
        ),
    }


def restore_rng_state(state, *, tracker=None):
    if len(state["torch_cuda"]) != torch.cuda.device_count():
        raise ValueError("RNG restore requires the original visible GPU topology")
    if (state["megatron"] is None) != (tracker is None):
        raise ValueError("Megatron RNG tracker must be restored explicitly")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    if tracker is not None:
        tracker.set_states({name: value.clone() for name, value in state["megatron"].items()})


def named_trainable_parameters(model):
    result = {}
    seen = set()
    for index, chunk in enumerate(model):
        for name, parameter in chunk.named_parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                result[f"chunk{index}.{name}"] = parameter
                seen.add(id(parameter))
    if not result:
        raise ValueError("Native training model has no trainable parameters")
    return result


def _any_rank_nonzero(flags):
    """Combine per-rank nonzero flags across the tensor-parallel ranks (max)."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return flags
    reduced = torch.tensor(flags, dtype=torch.int32, device=torch.cuda.current_device())
    dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    return [bool(value) for value in reduced.tolist()]


def gradient_evidence(parameters, *, required_groups, reduce_flags=_any_rank_nonzero):
    """Unselected experts may be zero; every declared active path must be nonzero.

    "Nonzero" is judged across the tensor-parallel group, not per rank: a
    vocab-parallel shard such as the embedding of a rank whose vocabulary slice
    contains none of the batch's tokens legitimately receives an all-zero
    gradient while the full parameter does train. Every rank evaluates the same
    parameter list in the same order, so the flag exchange is a fixed-order
    collective that all ranks enter together. The per-rank value is kept as
    ``nonzero_local``.
    """
    if not required_groups or any(not prefixes for prefixes in required_groups.values()):
        raise ValueError("Predeclare the active parameter groups before training")
    matched = {group: [] for group in required_groups}
    evidence = {}
    names = []
    local_flags = []
    for name, parameter in parameters.items():
        gradient = getattr(parameter, "main_grad", None)
        if gradient is None:
            gradient = parameter.grad
        if gradient is None:
            evidence[name] = {"present": False, "nonzero": False, "nonzero_local": False}
            continue
        if gradient.shape != parameter.shape or not torch.isfinite(gradient).all():
            raise ValueError(f"Invalid/nonfinite gradient for {name}")
        nonzero_local = bool(torch.count_nonzero(gradient))
        evidence[name] = {
            "present": True,
            "nonzero": nonzero_local,
            "nonzero_local": nonzero_local,
            "dtype": str(gradient.dtype),
        }
        names.append(name)
        local_flags.append(nonzero_local)
    group_flags = list(reduce_flags(local_flags)) if names else []
    if len(group_flags) != len(names):
        raise ValueError("Nonzero-gradient flags did not come back one per parameter")
    for name, nonzero in zip(names, group_flags, strict=True):
        evidence[name]["nonzero"] = bool(nonzero)
        for group, prefixes in required_groups.items():
            if nonzero and any(prefix in name for prefix in prefixes):
                matched[group].append(name)
    missing = [group for group, names_ in matched.items() if not names_]
    if missing:
        raise ValueError(f"Active paths have no finite nonzero gradient: {missing}")
    return {"parameters": evidence, "active_groups": matched}


@contextmanager
def observe_native_optimizer_step(optimizer, model, *, required_groups, evidence):
    """Wrap only step(), preserving the native loss, backward, clipping and update."""
    original = optimizer.step
    had_own = "step" in vars(optimizer)
    previous = vars(optimizer).get("step")
    parameters = named_trainable_parameters(model)
    calls = 0

    def observed(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls != 1:
            raise ValueError("One train_one_step call must perform exactly one optimizer step")
        evidence["gradients"] = gradient_evidence(parameters, required_groups=required_groups)
        before = {name: tensor_digest(value) for name, value in parameters.items()}
        result = original(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 3 or not bool(result[0]):
            raise ValueError("Native Megatron optimizer did not report a successful update")
        norm = float(result[1])
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Native optimizer must report a finite nonzero gradient norm")
        after = {name: tensor_digest(value) for name, value in parameters.items()}
        changed = [name for name in before if before[name] != after[name]]
        if not changed:
            raise ValueError("Optimizer reported success without changing any model parameter")
        evidence.update(
            {
                "grad_norm": norm,
                "parameter_hashes_before": before,
                "parameter_hashes_after": after,
                "changed_parameters": changed,
            }
        )
        return result

    optimizer.step = observed
    try:
        yield
        if calls != 1:
            raise ValueError("Native training did not call the optimizer exactly once")
    finally:
        if had_own:
            optimizer.step = previous
        else:
            del optimizer.step


def native_policy_update(args, model, optimizer, scheduler, fixture, *, step, required_groups):
    """Execute the actual installed VIME train_one_step with its clipped policy loss."""
    require_policy_configuration(args)
    fixture.validate()
    if step not in (1, 2, 3):
        raise ValueError("The fixed acceptance sequence uses updates 1, 2 and recovery control 3")
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.utils import get_model_config
    from vime.backends.megatron_utils import loss as native_loss
    from vime.backends.megatron_utils.data import DataIterator
    from vime.backends.megatron_utils.model import train_one_step

    if train_one_step.__module__ != "vime.backends.megatron_utils.model":
        raise ValueError("Native VIME training entry point was replaced")
    if native_loss.policy_loss_function.__module__ != "vime.backends.megatron_utils.loss":
        raise ValueError("Native VIME policy loss was replaced")
    for chunk in model:
        chunk.train()
        config = get_model_config(chunk)
        if config.hidden_dropout != 0 or config.attention_dropout != 0:
            raise ValueError("The fixed eager profile requires zero dropout")
        config.grad_scale_func = optimizer.scale_loss
        config.finalize_model_grads_func = finalize_model_grads
        config.timers = None
    batch = fixture.batch([0, 1, 2, 3], device=torch.cuda.current_device())
    batch["tokens"] = batch.pop("unconcat_tokens")
    iterator = DataIterator(batch, [[0], [1], [2], [3]])
    original_policy_loss = native_loss.policy_loss_function
    policy_calls = 0

    def observed_loss(*loss_args, **loss_kwargs):
        nonlocal policy_calls
        policy_calls += 1
        return original_policy_loss(*loss_args, **loss_kwargs)

    evidence = {"step": step, "fixture": fixture.content_sha256, "sample_position": step * 4}
    native_loss.policy_loss_function = observed_loss
    try:
        with observe_native_optimizer_step(
            optimizer, model, required_groups=required_groups, evidence=evidence
        ):
            metrics, norm = train_one_step(
                args,
                rollout_id=0,
                step_id=step - 1,
                data_iterator=[iterator],
                model=model,
                optimizer=optimizer,
                opt_param_scheduler=scheduler,
                num_microbatches=4,
                step_global_batch_size=4,
            )
    finally:
        native_loss.policy_loss_function = original_policy_loss
    if policy_calls != 4 or iterator.offset != 4:
        raise ValueError("Native policy loss must consume exactly the four fixed samples")
    if not math.isfinite(float(norm)):
        raise ValueError("Native training returned a nonfinite gradient norm")
    evidence.update({"policy_loss_calls": policy_calls, "metrics": metrics})
    fixture.validate()
    return evidence
