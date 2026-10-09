# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Bounded native DistributedOptimizer parameter-state I/O for fixed TP4/DP1.

The installed native DP-zero saver packs entire gradient buckets. With CPU Adam
offload and no communication overlap, a bucket can contain most of the model.
This adapter uses the native parameter getters/setters and verifies their exact
DP1 range coverage, while copying at most 64 MiB into each serialized block.
It never implements an optimizer, changes dtype, or changes native state layout.
"""

import json
from pathlib import Path

import torch
from qwen3_next_contract import atomic_json

from rl_engine.integrations.qwen3_next_training import named_trainable_parameters, state_digest

PARAMETER_KEYS = ("param", "exp_avg", "exp_avg_sq")


def _hdo_param_generator(cpu_optimizer):
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import _param_generator

    return _param_generator(cpu_optimizer)


def verify_hdo_optimizer_master_aliases(model, leaves):
    """CPU Adam must update the same FP32 master tensors the checkpoint getter exposes.

    The distributed optimizer keys its HybridDeviceOptimizer by the DP-sharded
    parameter (``optimizer.param_groups[group][order]`` found through
    ``model_param_group_index_map``), not by the model parameter, and the
    checkpoint getter returns that shard's ``state["master_param"]``. The FP32
    copy CPU Adam steps (``param_to_fp32_param[shard]``) is a separate tensor
    that must hold the same values, and the CPU optimizer's state entry must be
    the very dict the HDO keeps for the shard.
    """
    try:
        from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
    except ModuleNotFoundError:
        return

    for record in native_parameter_layout(model, leaves).values():
        leaf = record["leaf"]
        hdo = leaf.optimizer
        if not isinstance(hdo, HybridDeviceOptimizer):
            continue
        orig = record["parameter"]
        group_index, group_order = leaf.model_param_group_index_map[orig]
        shard = hdo.param_groups[group_index]["params"][group_order]
        master = leaf._get_main_param_and_optimizer_states(orig)["param"]
        hdo_state = hdo.state.get(shard)
        if not isinstance(hdo_state, dict) or hdo_state.get("master_param") is not master:
            raise ValueError("HDO state master_param is not the checkpoint-visible master")
        if shard not in hdo.param_to_fp32_param:
            raise ValueError("Native HDO must keep an FP32 copy for each sharded parameter")
        fp32_inner = hdo.param_to_fp32_param[shard]
        if hdo.param_to_inner_param.get(shard) is not fp32_inner:
            raise ValueError("HDO inner optimizer parameter is not the FP32 master copy")
        if fp32_inner.dtype != torch.float32 or not torch.equal(master, fp32_inner):
            raise ValueError("HDO FP32 master copy diverged from checkpoint-visible master weights")
        linked = False
        for cpu_optimizer in hdo.cpu_optimizers:
            for param in _hdo_param_generator(cpu_optimizer):
                if param is not fp32_inner:
                    continue
                if cpu_optimizer.state.get(param) is not hdo_state:
                    raise ValueError("CPU Adam state is not aliased to the HDO parameter state")
                linked = True
        if not linked:
            raise ValueError("CPU Adam does not own the FP32 master for a model parameter")


BLOCK_BYTES = 64 * 1024 * 1024


def native_parameter_layout(model, leaves):
    names = {id(value): name for name, value in named_trainable_parameters(model).items()}
    records = {}
    for leaf_index, leaf in enumerate(leaves):
        covered = set()
        for buffer_index, dtype_maps in enumerate(leaf.gbuf_ranges):
            for dtype, buckets in dtype_maps.items():
                for bucket_index, bucket in enumerate(buckets):
                    intervals = []
                    capacity = leaf.buffers[buffer_index].buckets[bucket_index].grad_data.numel()
                    for parameter, ranges in bucket["param_map"].items():
                        name = names.get(id(parameter))
                        if name is None or name in records or parameter in covered:
                            raise ValueError(
                                "Native optimizer ranges must partition model parameters"
                            )
                        local, relative = ranges["gbuf_local"], ranges["param"]
                        if (
                            relative.start != 0
                            or relative.end != parameter.numel()
                            or local.end - local.start != parameter.numel()
                            or not 0 <= local.start < local.end <= capacity
                        ):
                            raise ValueError("Native DP1 ranges must cover each complete parameter")
                        if parameter not in leaf.model_param_group_index_map:
                            raise ValueError(
                                "Native bucket parameter is absent from optimizer groups"
                            )
                        intervals.append((local.start, local.end))
                        covered.add(parameter)
                        records[name] = {
                            "parameter": parameter,
                            "leaf": leaf,
                            "layout": {
                                "leaf": leaf_index,
                                "buffer": buffer_index,
                                "dtype": [str(item) for item in dtype],
                                "bucket": bucket_index,
                                "bucket_numel": capacity,
                                "group_order": list(leaf.model_param_group_index_map[parameter]),
                                "param_range": [relative.start, relative.end],
                                "local_range": [local.start, local.end],
                            },
                        }
                    intervals.sort()
                    if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
                        raise ValueError("Native optimizer bucket ranges overlap")
        if covered != leaf.model_param_group_index_map.keys():
            raise ValueError("Native bucket ranges do not cover all optimizer parameters")
    if set(records) != set(names.values()):
        raise ValueError("Native optimizer state does not cover all model parameters")
    return dict(sorted(records.items()))


def native_parameter_states(record):
    parameter, leaf = record["parameter"], record["leaf"]
    states = leaf._get_main_param_and_optimizer_states(parameter)
    if not set(PARAMETER_KEYS).issubset(states) or set(states) - set(PARAMETER_KEYS) - {"step"}:
        raise ValueError("Native master weights or Adam moments are missing or unsupported")
    for key in PARAMETER_KEYS:
        value = states[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or value.numel() != parameter.numel()
            or not value.is_contiguous()
        ):
            raise ValueError(f"Invalid fully offloaded native {key}")
    if "step" in states:
        step = states["step"]
        if isinstance(step, torch.Tensor):
            if step.device.type != "cpu" or step.numel() != 1:
                raise ValueError("Native Adam step must be a CPU scalar")
        elif type(step) not in (int, float):
            raise ValueError("Native Adam step must be a numeric scalar")
    return states


def parameter_state_records(model, leaves):
    records, byte_count = {}, 0
    for name, record in native_parameter_layout(model, leaves).items():
        states = native_parameter_states(record)
        records[name] = {key: state_digest(value) for key, value in states.items()}
        byte_count += sum(
            value.numel() * value.element_size()
            for value in states.values()
            if isinstance(value, torch.Tensor)
        )
    return records, byte_count


def save_bounded_parameter_state(model, leaves, rank_dir, save_tensor_file):
    rank_dir = Path(rank_dir)
    directory = rank_dir / "optimizer-state"
    directory.mkdir()
    index = {
        "schema_version": 1,
        "scope": "native_dp1_parameter_state",
        "block_bytes": BLOCK_BYTES,
        "parameters": {},
    }
    extras = {}
    width = BLOCK_BYTES // (len(PARAMETER_KEYS) * 4)
    for parameter_index, (name, record) in enumerate(
        native_parameter_layout(model, leaves).items()
    ):
        states = native_parameter_states(record)
        entry = {
            "layout": record["layout"],
            "shapes": {key: list(states[key].shape) for key in PARAMETER_KEYS},
            "blocks": [],
        }
        count = record["parameter"].numel()
        for block_index, start in enumerate(range(0, count, width)):
            end = min(start + width, count)
            filename = f"optimizer-state/p{parameter_index:05d}-b{block_index:05d}.pt"
            # Clone bounded views: torch.save otherwise serializes their entire
            # underlying native storage, which can itself be a whole buffer.
            block = {key: states[key].reshape(-1)[start:end].clone() for key in PARAMETER_KEYS}
            save_tensor_file(rank_dir / filename, block)
            del block
            entry["blocks"].append({"path": filename, "start": start, "end": end})
        extras[name] = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in states.items()
            if key not in PARAMETER_KEYS
        }
        index["parameters"][name] = entry
    save_tensor_file(rank_dir / "optimizer-extra.pt", extras)
    atomic_json(rank_dir / "optimizer-index.json", index)
    return index


def parameter_state_files(rank_dir, index=None):
    rank_dir = Path(rank_dir)
    if index is None:
        index = json.loads((rank_dir / "optimizer-index.json").read_text())
    if (index.get("schema_version"), index.get("scope"), index.get("block_bytes")) != (
        1,
        "native_dp1_parameter_state",
        BLOCK_BYTES,
    ):
        raise ValueError("Unsupported native bounded optimizer checkpoint")
    files = ["optimizer-header.pt", "optimizer-index.json", "optimizer-extra.pt"]
    for entry in index["parameters"].values():
        for block in entry["blocks"]:
            path = Path(block["path"])
            if len(path.parts) != 2 or path.parts[0] != "optimizer-state" or path.suffix != ".pt":
                raise ValueError("Optimizer block path is outside its rank state directory")
            files.append(str(path))
    if len(files) != len(set(files)):
        raise ValueError("Native optimizer blocks cannot alias another parameter's state")
    return files


def load_bounded_parameter_state(model, leaves, rank_dir):
    rank_dir = Path(rank_dir)
    index = json.loads((rank_dir / "optimizer-index.json").read_text())
    parameter_state_files(rank_dir, index)
    current = native_parameter_layout(model, leaves)
    if current.keys() != index["parameters"].keys():
        raise ValueError("Saved native optimizer parameter inventory differs")
    extras = torch.load(rank_dir / "optimizer-extra.pt", map_location="cpu", weights_only=True)
    if extras.keys() != current.keys():
        raise ValueError("Saved native optimizer extra-state inventory differs")
    width = BLOCK_BYTES // (len(PARAMETER_KEYS) * 4)
    touched_leaves = []
    with torch.no_grad():
        for name, record in current.items():
            entry = index["parameters"][name]
            states = native_parameter_states(record)
            shapes = {key: list(states[key].shape) for key in PARAMETER_KEYS}
            if entry["layout"] != record["layout"] or entry["shapes"] != shapes:
                raise ValueError("Saved native optimizer layout or state shapes differ")
            position = 0
            for block in entry["blocks"]:
                start, end = block["start"], block["end"]
                if (
                    type(start) is not int
                    or type(end) is not int
                    or start != position
                    or not start < end <= min(start + width, record["parameter"].numel())
                ):
                    raise ValueError(
                        "Native optimizer blocks must cover each parameter exactly once"
                    )
                payload = torch.load(
                    rank_dir / block["path"], map_location="cpu", weights_only=True
                )
                if payload.keys() != set(PARAMETER_KEYS):
                    raise ValueError("Native optimizer block omits masters or moments")
                for key in PARAMETER_KEYS:
                    value = payload[key]
                    if (
                        value.dtype != torch.float32
                        or value.shape != (end - start,)
                        or not torch.isfinite(value).all()
                    ):
                        raise ValueError("Native optimizer block shape/dtype/finite check failed")
                    states[key].reshape(-1)[start:end].copy_(value)
                del payload
                position = end
            if position != record["parameter"].numel():
                raise ValueError("Native optimizer blocks do not cover the complete parameter")
            if extras[name].keys() != states.keys() - set(PARAMETER_KEYS):
                raise ValueError("Native optimizer extra-state schema differs")
            for key, value in extras[name].items():
                if isinstance(states[key], torch.Tensor):
                    if not isinstance(value, torch.Tensor) or value.dtype != states[key].dtype:
                        raise ValueError("Native Adam scalar dtype differs")
                    states[key].copy_(value)
                else:
                    if type(value) is not type(states[key]):
                        raise ValueError("Native Adam scalar type differs")
                    states[key] = value
            record["leaf"]._set_main_param_and_optimizer_states(record["parameter"], states)
            if record["leaf"] not in touched_leaves:
                touched_leaves.append(record["leaf"])
    try:
        from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
    except ModuleNotFoundError:
        return

    for leaf in touched_leaves:
        hdo = leaf.optimizer
        if isinstance(hdo, HybridDeviceOptimizer):
            hdo._sync_hdo_state_to_sub_optimizers()
