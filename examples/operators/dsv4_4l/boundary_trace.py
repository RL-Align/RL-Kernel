# SPDX-License-Identifier: Apache-2.0
"""Observe real native-MCore/vLLM module boundaries without replacing arithmetic."""

import json
import dataclasses
import os
from pathlib import Path


def metadata_snapshot(value, depth=0):
    if depth > 5:
        return repr(type(value))
    if isinstance(value, dict):
        return {k: metadata_snapshot(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [metadata_snapshot(v, depth + 1) for v in value]
    if dataclasses.is_dataclass(value) and type(value).__module__.startswith("vllm"):
        return {
            field.name: metadata_snapshot(getattr(value, field.name), depth + 1)
            for field in dataclasses.fields(value)
        }
    import torch

    if value is None or isinstance(value, (torch.Tensor, str, bool, int, float)):
        return value
    # Methods can receive module/worker handles. Serializing those also walks
    # their local trace hooks and fails on lambdas; they are not tensor state.
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def capture(model, root, role):
    from .trace_store import save_trace

    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    events = []
    handles = []
    context = {"layer": None}
    restores = []
    worker = model._rlk_p1 if role == "actor" else model._rlk_trace_worker
    original_observer = worker.observer

    def dump(name, value):
        file = save_trace(metadata_snapshot(value), root / f"{len(events):05d}.pt")
        events.append(
            dict(
                name=name,
                layer=context["layer"],
                file=file,
                pid=os.getpid(),
                weight_version=worker.version,
            )
        )
        (root / "manifest.json").write_text(json.dumps(events, indent=2))

    def observer(*args):
        dump("rlk." + args[-2], args[-1])
        if original_observer:
            original_observer(*args)

    worker.observer = observer
    handles.append(
        model.register_forward_pre_hook(
            lambda m, a, k: dump("model.input", dict(args=a, kwargs=k)), with_kwargs=True
        )
    )
    branches = []
    if role == "actor":
        if model.pre_process:
            branches.append(("embedding", model.embedding))
        for block in model.decoder.layers:
            index = (block.layer_number - 1) // 2
            handles.append(
                block.register_forward_pre_hook(lambda m, a, i=index: context.update(layer=i))
            )
            attention = (block.layer_number - 1) % 2 == 0
            branch = block.inner_layer.self_attention if attention else block.inner_layer.mlp
            branches.append((f"layer.{index}." + ("attn" if attention else "ffn"), branch))
    else:
        if hasattr(model.embed_tokens, "weight"):
            branches.append(("embedding", model.embed_tokens))
        if hasattr(model.norm, "weight"):
            branches.append(("final_norm", model.norm))
        for i in range(model.start_layer, model.end_layer):
            block = model.layers[i]
            handles.append(
                block.register_forward_pre_hook(lambda m, a, i=i: context.update(layer=i))
            )
            branches.extend(((f"layer.{i}.attn", block.attn), (f"layer.{i}.ffn", block.ffn)))
    for label, branch in branches:
        if not label.endswith(".attn"):
            continue
        for method in (
            "_bpre_attn_gemm",
            "_fused_wqa_wkv_gemm",
            "_run_parallel_input_projections",
            "_wq_b_proj",
            "_fused_qnorm_rope_kv_insert",
            "_rlk_prefill_sparse",
            "_o_proj",
        ):
            if not hasattr(branch, method):
                continue
            original = getattr(branch, method)

            def observed(*args, _original=original, _label=label + ".method." + method, **kwargs):
                # An out= destination contains uninitialized bytes before execution.
                inputs = {k: v for k, v in kwargs.items() if k != "output"}
                dump(_label + ".input", dict(args=args, kwargs=inputs))
                result = _original(*args, **kwargs)
                dump(_label + ".output", kwargs.get("output") if result is None else result)
                return result

            restores.append((branch, method, original))
            setattr(branch, method, observed)
    for label, branch in branches:
        for child_name, module in branch.named_modules():
            name = label + ("." + child_name if child_name else "")
            handles.append(
                module.register_forward_pre_hook(lambda m, a, n=name: dump(n + ".input", a))
            )
            handles.append(
                module.register_forward_hook(lambda m, a, o, n=name: dump(n + ".output", o))
            )

    def stop():
        for obj, name, original in reversed(restores):
            setattr(obj, name, original)
        for handle in handles:
            handle.remove()
        worker.observer = original_observer

    return stop
