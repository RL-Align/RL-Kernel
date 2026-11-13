"""Capture original tensors from the real native actor; no arithmetic replacements."""

import json
import types
import torch
from pathlib import Path


def install(model, root):
    root = Path(root)
    root.mkdir(exist_ok=True)
    events = []
    context = {"layer": None}
    enabled = [True]
    handles = []
    restores = []

    def replace(obj, name, value):
        old = getattr(obj, name)
        restores.append((obj, name, old))
        setattr(obj, name, value)

    def clone(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, tuple):
            return tuple(clone(v) for v in value)
        if isinstance(value, list):
            return [clone(v) for v in value]
        return (
            value
            if value is None or isinstance(value, (str, bool, int, float))
            else repr(type(value))
        )

    def dump(name, value):
        if not enabled[0]:
            return
        index = len(events)
        file = f"{index:04d}.pt"
        torch.save(clone(value), root / file)
        events.append(dict(index=index, layer=context["layer"], name=name, file=file))
        (root / "manifest.json").write_text(json.dumps(events, indent=2))

    observer = model._rlk_p1.observer

    def observe(*args):
        dump("rlk." + args[-2], args[-1])
        if observer:
            observer(*args)

    replace(model._rlk_p1, "observer", observe)

    def after_embedding(m, args, result):
        dump("embedding.output", result)

    handles.append(model.embedding.register_forward_hook(after_embedding))
    for index, layer in enumerate(model.decoder.layers):

        def entering(m, args, index=index):
            context["layer"] = index // 2

        handles.append(layer.register_forward_pre_hook(entering))
        branch = layer.inner_layer.self_attention if index % 2 == 0 else layer.inner_layer.mlp
        prefix = f"layer.{index // 2}." + ("attn" if index % 2 == 0 else "ffn")
        if index % 2 == 1:
            original_gating = branch.router.gating

            def gating(*args, _original=original_gating, _prefix=prefix, **kwargs):
                result = _original(*args, **kwargs)
                dump(_prefix + ".router.gating.output", result)
                return result

            replace(branch.router, "gating", gating)
            if hasattr(branch.experts, "backend"):
                base = branch.experts.backend.base
                old_quantize = base.quantize

                def quantize(x, _original=old_quantize, _prefix=prefix):
                    result = _original(x)
                    dump(_prefix + ".packed.quantize", (x, result.codes, result.scales))
                    return result

                replace(base, "quantize", quantize)
                old_projection = base.forward

                def projection(*args, _original=old_projection, _prefix=prefix, **kwargs):
                    result = _original(*args, **kwargs)
                    dump(_prefix + ".packed.projection.output", result)
                    return result

                replace(base, "forward", projection)
        if index % 2 == 0:
            original_qkv = branch.get_query_key_value_tensors

            def query_key_value(*args, _original=original_qkv, _prefix=prefix, **kwargs):
                result = _original(*args, **kwargs)
                dump(_prefix + ".method.get_query_key_value_tensors", result)
                return result

            replace(branch, "get_query_key_value_tensors", query_key_value)
            core = branch.core_attention
            original_forward = core.forward.__func__
            sparse = original_forward.__globals__["unfused_compressed_sparse_attn"]

            def trace_sparse(*args, _sparse=sparse, _prefix=prefix, **kwargs):
                dump(_prefix + ".sparse.input", args)
                out = _sparse(*args, **kwargs)
                dump(_prefix + ".sparse.output", out)
                return out

            cloned = types.FunctionType(
                original_forward.__code__,
                dict(original_forward.__globals__, unfused_compressed_sparse_attn=trace_sparse),
                original_forward.__name__,
                original_forward.__defaults__,
                original_forward.__closure__,
            )
            cloned.__kwdefaults__ = original_forward.__kwdefaults__
            replace(core, "forward", types.MethodType(cloned, core))
        for name, module in branch.named_modules():
            label = prefix + ("." + name if name else "")

            def entering_module(m, args, label=label):
                dump(label + ".input", args)
                provider = getattr(m, "_qnorm_provider", None)
                if provider is not None:
                    dump(label + ".normalized_quantized_input", provider.quantized_input(args[0]))

            handles.append(module.register_forward_pre_hook(entering_module))
            handles.append(
                module.register_forward_hook(
                    lambda m, args, out, label=label: dump(label + ".output", out)
                )
            )

    def stop():
        enabled[0] = False
        for handle in handles:
            handle.remove()
        for obj, name, original in reversed(restores):
            setattr(obj, name, original)

    return stop
