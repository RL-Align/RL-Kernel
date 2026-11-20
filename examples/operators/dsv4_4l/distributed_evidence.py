# SPDX-License-Identifier: Apache-2.0
"""Instance-local observations of native PP boundaries and actual DDP reduction."""

import hashlib
import inspect
import math
from contextlib import contextmanager
from types import MethodType

import torch
import torch.distributed as dist


def tensor_record(value):
    return dict(
        shape=list(value.shape),
        dtype=str(value.dtype),
        sha256=hashlib.sha256(
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        finite=bool(torch.isfinite(value).all()),
        nonzero=bool(value.ne(0).any()),
    )


class PipelineBoundaryEvidence:
    def __init__(self, model, dp_rank):
        self.model = model
        self.dp_rank = dp_rank
        self.records = []
        self.phase = None
        self.counter = 0
        original = model.set_input_tensor

        def receive(_model, value):
            tensors = value if isinstance(value, list) else [value]
            if not model.pre_process and tensors[0] is not None:
                index = self.counter
                self.counter += 1
                self.record("forward_recv", index, tensors[0])
                if tensors[0].requires_grad:
                    tensors[0].register_hook(
                        lambda grad, i=index: self.record("backward_send", i, grad)
                    )
            return original(value)

        model.set_input_tensor = MethodType(receive, model)

        def sent(_model, _args, value):
            if not model.post_process:
                index = self.counter
                self.counter += 1
                self.record("forward_send", index, value)
                if value.requires_grad:
                    value.register_hook(lambda grad, i=index: self.record("backward_recv", i, grad))

        self.handle = model.register_forward_hook(sent)

    def begin(self, phase):
        self.phase = phase
        self.counter = 0
        self.records = []

    def record(self, direction, microbatch, value):
        self.records.append(
            dict(
                phase=self.phase,
                direction=direction,
                microbatch=microbatch,
                dp_rank=self.dp_rank,
                **tensor_record(value),
            )
        )
        # A tensor hook must preserve the actual native gradient object.
        return value


@contextmanager
def ddp_step_evidence(engine, model):
    """Observe upstream finish_grad_sync and optimizer inputs without replacing math."""
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    records = dict(local_gradients={}, reduced_gradients={}, native_ddp_finish_calls=[])
    handles = []
    restores = []
    for name, parameter in parameters.items():

        def gradient(value, name=name):
            records["local_gradients"][name] = tensor_record(value)
            return value

        handles.append(parameter.register_hook(gradient))
    for chunk in engine.module:
        original = chunk.finish_grad_sync

        def finish(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            records["native_ddp_finish_calls"].append(
                dict(
                    source=inspect.getfile(_original.__func__),
                    actual_callable=_original.__qualname__,
                )
            )
            return result

        restores.append((chunk, "finish_grad_sync", original))
        chunk.finish_grad_sync = finish
    original_step = engine.optimizer_step

    def step():
        for name, parameter in parameters.items():
            gradient = getattr(parameter, "main_grad", None)
            if gradient is None:
                gradient = parameter.grad
            if gradient is not None:
                records["reduced_gradients"][name] = tensor_record(gradient)
        replicas = [None] * engine.get_data_parallel_size()
        dist.all_gather_object(
            replicas,
            dict(local=records["local_gradients"], reduced=records["reduced_gradients"]),
            group=engine.get_data_parallel_group(),
        )
        if any(r["reduced"] != replicas[0]["reduced"] for r in replicas):
            raise RuntimeError("actual reduced DDP gradients differ across replicas")
        if not records["native_ddp_finish_calls"]:
            raise RuntimeError("native DDP finish_grad_sync was not executed")
        if not records["reduced_gradients"] or not all(
            v["finite"] for v in records["reduced_gradients"].values()
        ):
            raise RuntimeError("actual optimizer gradients are empty or nonfinite")
        records["different_local_gradients"] = any(
            r["local"] != replicas[0]["local"] for r in replicas
        )
        if engine.get_data_parallel_size() > 1 and not records["different_local_gradients"]:
            raise RuntimeError("DP2 test did not exercise distinct local backward gradients")
        records["dp_replicas"] = len(replicas)
        from rl_engine.integrations.engines.train.megatron.native_mhc import original_mhc_parameters

        names = {id(p): name for name, p in parameters.items()}
        expected = [names[id(p)] for p in original_mhc_parameters(model).values()]
        missing = [name for name in expected if name not in records["local_gradients"]]
        records["missing_mhc_backward_parameters"] = missing
        records["missing_backward_parameters"] = [
            name for name in parameters if name not in records["local_gradients"]
        ]
        if missing:
            raise RuntimeError(f"original mHC Parameters missing actual backward: {missing}")
        return original_step()

    engine.optimizer_step = step
    try:
        yield records
    finally:
        engine.optimizer_step = original_step
        for obj, name, original in restores:
            setattr(obj, name, original)
        for handle in handles:
            handle.remove()


def distributed_token_loss(model_output, data, *, dp_size):
    """Use the real engine's globally reduced valid-token denominator for DDP."""
    from verl.utils import tensordict_utils as tu

    total = tu.get_non_tensor_data(data, key="batch_num_tokens", default=None)
    if (
        total is None
        or not math.isfinite(float(total))
        or total <= 0
        or type(dp_size) is not int
        or dp_size <= 0
    ):
        raise ValueError("native DDP loss requires a positive actual global valid-token count")
    raw, mask = model_output["raw_log_probs"].values(), data["loss_mask"].values()
    if raw.dtype != torch.float32 or raw.shape != mask.shape:
        raise ValueError("native DDP loss requires aligned original FP32 raw scores")
    return -(raw * mask).sum() * dp_size / total, {}
