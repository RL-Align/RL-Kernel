# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Lossless, framework-primitive parameter contracts for automatic discovery.

No model architecture or module-path conventions are used here. In particular,
an arbitrary matrix is NEVER transposed just because that makes it match.
"""

from __future__ import annotations

from collections import Counter
import json

import torch

from .core import fingerprint


def _inherits(module, package: str, name: str) -> bool:
    return any(
        cls.__module__.startswith(package) and cls.__name__ == name for cls in type(module).__mro__
    )


def value_key(role: str, value: torch.Tensor) -> str:
    return json.dumps(
        [role, list(value.shape), str(value.dtype), fingerprint(value)], separators=(",", ":")
    )


def in_scope(path: str, scope: str) -> bool:
    return not scope or path == scope or path.startswith(scope + ".")


def scope_inventory(inventory: dict, scope: str | None = None) -> Counter:
    return Counter(
        inventory["parameters"][index]["key"] for index in scope_parameters(inventory, scope)
    )


def scope_parameters(inventory: dict, scope: str | None = None) -> set[int]:
    return {
        index
        for path, owner in inventory["owners"].items()
        if scope is None or in_scope(path, scope)
        for atom in owner["atoms"]
        for index in atom["parameters"]
    }


def align_primitive_views(left: dict, right: dict) -> None:
    """Choose a declared whole projection when its peer is already whole.

    A GPT-style packed Conv1D and a fused native QKV projection can represent
    the same complete matrix. This is exact PARAMETER correspondence, not
    fitting activations or inferring arbitrary packing from shapes.
    """
    singles = [
        {
            tuple(owner["atoms"][0]["keys"])
            for owner in inventory["owners"].values()
            if len(owner["atoms"]) == 1 and owner["atoms"][0]["kind"] == "affine"
        }
        for inventory in (left, right)
    ]
    for inventory, peers in ((left, singles[1]), (right, singles[0])):
        for owner in inventory["owners"].values():
            whole = owner.get("whole_atom")
            if whole is not None and tuple(whole["keys"]) in peers:
                owner["atoms"] = [whole]
                owner["selected_view"] = "verified_whole_projection"


def parameter_inventory(model: torch.nn.Module) -> dict:
    """Inspect actual parameters; retain physical identity AND lossless views.

    TP/CP > 1 is deliberately not inferred. The discovery entry point rejects it
    and directs callers to the existing explicitly mapped distributed API.
    """
    records, fragments, owners, physical = [], {}, {}, {}
    for path, module in model.named_modules():
        direct = dict(module.named_parameters(recurse=False))
        if not direct:
            continue
        for name, parameter in direct.items():
            full_name = f"{path}.{name}" if path else name
            if not parameter.is_floating_point() or parameter.dtype in (
                getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e5m2", None),
            ):
                raise ValueError(f"{full_name}: quantized/non-floating parameters need a mapping")
            physical[full_name] = fingerprint(parameter)

        linear = isinstance(module, torch.nn.Linear)
        conv1d = _inherits(module, "transformers.pytorch_utils", "Conv1D")
        vllm_linear = _inherits(module, "vllm.model_executor.layers.linear", "LinearBase")
        vocab = _inherits(
            module, "vllm.model_executor.layers.vocab_parallel_embedding", "VocabParallelEmbedding"
        )
        if vllm_linear:
            method = getattr(module, "quant_method", None)
            if type(method).__name__ != "UnquantizedLinearMethod":
                raise ValueError(f"{path}: automatic discovery requires unquantized linear weights")
        affine = linear or conv1d or vllm_linear
        weight = direct.get("weight")
        if affine and (weight is None or weight.ndim != 2):
            raise ValueError(f"{path}: invalid affine parameter contract")
        transpose = conv1d
        if vllm_linear:
            if (getattr(weight, "output_dim", None), getattr(weight, "input_dim", None)) != (0, 1):
                raise ValueError(f"{path}: unsupported native linear storage axes")
        canonical = weight.T if transpose else weight
        width = canonical.shape[0] if affine else None
        sizes = [width] if affine else []
        if vllm_linear and (
            _inherits(module, "vllm.model_executor.layers.linear", "MergedColumnParallelLinear")
            or _inherits(module, "vllm.model_executor.layers.linear", "QKVParallelLinear")
        ):
            sizes = list(getattr(module, "output_sizes", []))
            if not sizes or sum(sizes) != width:
                raise ValueError(f"{path}: fused projection partition metadata is not complete")
        atom_records = []

        def add(name, view, transform, direct=direct, path=path):
            parameter = direct[name]
            # Aliases of tied parameters count once; different physical slices
            # of a fused parameter remain distinct logical parameters.
            identity = (id(parameter), tuple(transform))
            key = value_key(name, view)
            if identity not in fragments:
                fragments[identity] = len(records)
                records.append(
                    {
                        "key": key,
                        "owners": [],
                        "names": [],
                        "transform": list(transform),
                        "physical_shape": list(parameter.shape),
                    }
                )
            index = fragments[identity]
            record = records[index]
            if record["key"] != key:
                raise ValueError(f"{path}: conflicting parameter alias interpretations")
            record["owners"].append(path)
            record["names"].append(f"{path}.{name}" if path else name)
            return index, key

        consumed = set()
        if affine:
            start = 0
            for size in sizes:
                keys, indices = [], []
                for name in ("weight", "bias"):
                    if name not in direct:
                        continue
                    value = canonical if name == "weight" else direct[name]
                    if value.shape[0] != width:
                        raise ValueError(f"{path}: affine bias/output shape mismatch")
                    transform = (
                        "transpose" if transpose and name == "weight" else "identity",
                        start,
                        start + size,
                    )
                    index, key = add(name, value[start : start + size], transform)
                    keys.append(key)
                    indices.append(index)
                    consumed.add(name)
                atom_records.append(
                    {
                        "kind": "affine",
                        "keys": sorted(keys),
                        "parameters": indices,
                        "in_features": canonical.shape[1],
                        "out_features": size,
                        "output_slice": [start, start + size],
                        "input_visible": True,
                        "output_visible": not (
                            bool(getattr(module, "skip_bias_add", False)) and "bias" in direct
                        ),
                    }
                )
                start += size
        for name, value in direct.items():
            if name in consumed:
                continue
            transform = ("identity", 0, value.shape[0] if value.ndim else 1)
            if vocab and name == "weight":
                logical_size = int(module.num_embeddings)
                if value.ndim != 2 or not 0 < logical_size <= value.shape[0]:
                    raise ValueError(f"{path}: invalid vocabulary-padding metadata")
                value = value[:logical_size]
                transform = ("identity", 0, logical_size)
            index, key = add(name, value, transform)
            atom_records.append({"kind": "parameter", "keys": [key], "parameters": [index]})
        owner = {
            "atoms": atom_records,
            "type": f"{type(module).__module__}.{type(module).__name__}",
        }
        if affine and len(sizes) > 1:
            keys, indices = [], []
            for name in ("weight", "bias"):
                if name not in direct:
                    continue
                value = canonical if name == "weight" else direct[name]
                index, key = add(
                    name,
                    value,
                    ("transpose" if transpose and name == "weight" else "identity", 0, width),
                )
                indices.append(index)
                keys.append(key)
            owner["whole_atom"] = {
                **atom_records[0],
                "keys": sorted(keys),
                "parameters": indices,
                "out_features": width,
                "output_slice": [0, width],
            }
        owners[path] = owner
    return {"parameters": records, "owners": owners, "physical": physical}
