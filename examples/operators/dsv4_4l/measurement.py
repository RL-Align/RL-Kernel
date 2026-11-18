# SPDX-License-Identifier: Apache-2.0
"""Opt-in observation timers; no arithmetic, sampling or optimizer replacements."""

import functools
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


def workload(cases):
    return dict(
        samples=len(cases),
        padded_input_tokens=len(cases)
        * max(
            (len(c["prompt_token_ids"]) + len(c["response_token_ids"]) for c in cases), default=0
        ),
        prompt_tokens=sum(len(c["prompt_token_ids"]) for c in cases),
        response_tokens=sum(len(c["response_token_ids"]) for c in cases),
        input_tokens=sum(len(c["prompt_token_ids"]) + len(c["response_token_ids"]) for c in cases),
    )


def emit(phase, **fields):
    if os.environ.get("DSV4_MEASURE") != "1":
        return
    root = Path("/outputs")
    record = dict(phase=phase, pid=os.getpid(), rank=os.environ.get("RANK"), **fields)
    with (root / f"timings-{os.getpid()}.jsonl").open("a") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")


def memory_snapshot(phase, **fields):
    if os.environ.get("DSV4_MEASURE") != "1":
        return
    import torch

    if torch.cuda.is_initialized():
        emit(
            phase,
            logical_gpu=torch.cuda.current_device(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            **fields,
        )


@contextmanager
def measure(phase, *, cuda_fence=False, **fields):
    if os.environ.get("DSV4_MEASURE") != "1":
        yield
        return
    if cuda_fence:
        import torch

        torch.cuda.synchronize()
    start = time.perf_counter()
    success = False
    try:
        yield
        if cuda_fence:
            torch.cuda.synchronize()
        success = True
    finally:
        emit(
            phase,
            seconds=time.perf_counter() - start,
            success=success,
            completion_boundary=(
                "GPU synchronize" if cuda_fence else "completed API/IPC acknowledgement"
            ),
            diagnostic_overhead_included=True,
            **fields,
        )
        if cuda_fence:
            memory_snapshot(phase + "_memory", **fields)


def instrument_actor(engine):
    if os.environ.get("DSV4_MEASURE") != "1":
        return
    original = engine.optimizer_step

    @functools.wraps(original)
    def step(*args, **kwargs):
        with measure("optimizer_step", cuda_fence=True, version=engine.rlk_weight_version + 1):
            return original(*args, **kwargs)

    engine.optimizer_step = step


def instrument_server(server):
    if os.environ.get("DSV4_MEASURE") != "1":
        return
    for name in ("launch_server", "generate", "collective_rpc"):
        original = getattr(server, name)

        @functools.wraps(original)
        async def wrapped(*args, _original=original, _name=name, **kwargs):
            phase = {
                "launch_server": "rollout_init_load_compile_warmup",
                "generate": "rollout_generate",
                "collective_rpc": "rollout_control",
            }[_name]
            fields = {}
            if _name == "collective_rpc":
                fields["method"] = args[0]
                if args[0] == "load_actor_update":
                    phase = "weight_sync"
                    fields["path"] = kwargs.get("args", args[1:] if len(args) > 1 else ())
            if _name == "generate":
                fields.update(request_id=kwargs.get("request_id"), prompt_tokens=len(args[0]))
            with measure(phase, **fields):
                result = await _original(*args, **kwargs)
            if _name == "generate":
                value = result.model_dump()
                emit(
                    "generation_workload",
                    **fields,
                    response_tokens=len(value["token_ids"]),
                )
            return result

        setattr(server, name, wrapped)
