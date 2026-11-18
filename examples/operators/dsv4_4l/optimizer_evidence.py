# SPDX-License-Identifier: Apache-2.0
"""Observe actual optimizer state without serializing it or changing its arithmetic."""

import hashlib
import json
import os


def optimizer_snapshot(engine):
    import torch

    from examples.operators.dsv4_4l.distributed_evidence import tensor_record

    optimizer = engine.optimizer
    tensors, scalars, objects = {}, {}, []

    def visit(value, path):
        if isinstance(value, torch.Tensor):
            tensors[path] = dict(tensor_record(value), data_ptr=value.data_ptr())
            if value.numel() == 1 and path.rsplit("/", 1)[-1] == "step":
                scalars[path] = value.item()
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, path + "/" + str(key))
        elif isinstance(value, (tuple, list)):
            for key, item in enumerate(value):
                visit(item, path + "/" + str(key))
        elif value is None or isinstance(value, (int, float, bool, str)):
            scalars[path] = value
        else:
            raise TypeError(f"unrecorded optimizer state at {path}: {type(value)}")

    def identity(opt):
        objects.append(dict(id=id(opt), type=type(opt).__qualname__))
        if hasattr(opt, "chained_optimizers"):
            for child in opt.chained_optimizers:
                identity(child)
        elif hasattr(opt, "optimizer"):
            identity(opt.optimizer)

    identity(optimizer)
    visit(optimizer.state_dict(), "state")
    if any(not v["finite"] for v in tensors.values()):
        raise RuntimeError("actual optimizer contains nonfinite state")
    steps = {key: value for key, value in scalars.items() if key.rsplit("/", 1)[-1] == "step"}
    payload = dict(tensors=tensors, scalars=scalars)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return dict(
        pid=os.getpid(),
        engine_id=id(engine),
        optimizer_id=id(optimizer),
        optimizer_objects=objects,
        state_digest=digest,
        step_counters=steps,
        state_tensors=tensors,
        state_scalars=scalars,
    )


def verify_optimizer_transition(before, after, *, version, previous=None):
    for key in ["pid", "engine_id", "optimizer_id", "optimizer_objects"]:
        if before[key] != after[key] or (previous is not None and before[key] != previous[key]):
            raise ValueError(f"actor/optimizer identity changed: {key}")
    if previous is not None and before["state_digest"] != previous["state_digest"]:
        raise ValueError("optimizer state was reset or changed between consecutive steps")
    if after["state_digest"] == before["state_digest"] or not after["state_tensors"]:
        raise ValueError("actual optimizer state did not change")
    if not after["step_counters"] or any(
        float(v) != version for v in after["step_counters"].values()
    ):
        raise ValueError("actual optimizer counters do not match consecutive update version")
    if version > 1 and (
        not before["state_tensors"]
        or not before["step_counters"]
        or any(float(v) != version - 1 for v in before["step_counters"].values())
    ):
        raise ValueError("actual previous optimizer moments/counters were lost")
    return True
