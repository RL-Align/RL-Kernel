# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def load_sample(source: Path, sample: int, step: int) -> dict[str, Any]:
    if sample < 0 or step < 0:
        raise ValueError("--sample and --step must be nonnegative")
    if source.is_dir():
        candidates = list(source.glob(f"**/rollout_data/{step}.pt"))
        if not candidates:
            candidates = list(source.glob(f"train-data/{step}.rank0.pt"))
        if not candidates:
            candidate = source / "frozen-replay.json"
            candidates = [candidate] if candidate.is_file() else []
        if len(candidates) != 1:
            raise ValueError(
                "Source must identify one run with a saved rollout; pass its "
                "rollout .pt file explicitly"
            )
        source = candidates[0]
    value = (
        json.loads(source.read_text(encoding="utf-8"))
        if source.suffix == ".json"
        else torch.load(source, map_location="cpu", weights_only=True)
    )
    inferred_step = step
    if source.parent.name in {"rollout_data", "train-data", "train_data"}:
        prefix = source.name.split(".")[0]
        if prefix.isdecimal():
            inferred_step = int(prefix)
    source_step = value.get("source_step", value.get("rollout_id", inferred_step))
    if "rollout_data" in value:
        value = value["rollout_data"]
    if "samples" in value:
        value = value["samples"][sample]
    elif "tokens" in value and "response_lengths" in value:
        value = {
            "tokens": value["tokens"][sample],
            "response_length": value["response_lengths"][sample],
            "loss_mask": value.get("loss_masks", [None] * (sample + 1))[sample],
        }
    raw_tokens = torch.as_tensor(value["tokens"])
    if raw_tokens.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise ValueError("token IDs must be integers")
    tokens = raw_tokens.long().reshape(-1).tolist()
    prompt = value.get("prompt_length")
    if prompt is None:
        prompt = len(tokens) - int(value["response_length"])
    mask = value.get("mask", value.get("loss_mask"))
    if mask is None:
        mask = [1] * (len(tokens) - prompt)
    mask = torch.as_tensor(mask).reshape(-1).tolist()
    if not 0 < prompt < len(tokens) or len(mask) != len(tokens) - prompt:
        raise ValueError("source token/prompt/mask lengths do not agree")
    if any(t < 0 for t in tokens) or any(m not in (0, 1) for m in mask):
        raise ValueError("source must contain nonnegative token IDs and a binary active mask")
    return {
        "schema_version": "rlkernel.frozen_replay.v1",
        "tokens": tokens,
        "prompt_length": prompt,
        "mask": mask,
        "source": str(source.resolve()),
        "source_sample": sample,
        "source_step": int(source_step if source_step is not None else step),
        **({"sampling": value["sampling"]} if "sampling" in value else {}),
    }


def source_configuration(source: Path) -> dict[str, Any]:
    """Prefer the failing run's parameters; explicit CLI options still win."""
    root = source if source.is_dir() else source.parent
    for candidate in [root, *list(root.parents)[:4]]:
        manifest = candidate / "manifest.json"
        frozen = candidate / "frozen-inputs.before.json"
        if manifest.is_file():
            value = json.loads(manifest.read_text(encoding="utf-8"))
            topology = value.get("topology", {})
            options = {
                name: topology[key]
                for name, key in (
                    ("tp_size", "tp"),
                    ("cp_size", "cp"),
                    ("rollout_tp_size", "rollout_tp"),
                    ("rollout_cp_size", "rollout_cp"),
                )
                if key in topology
            }
            options.update(
                {
                    f"rollout_{key}": value["sampling"][key]
                    for key in ("temperature", "top_p", "top_k")
                    if key in value.get("sampling", {})
                }
            )
            command = value.get("train_command", [])
            legacy_flags = {
                "--rollout-temperature": ("rollout_temperature", float),
                "--rollout-top-p": ("rollout_top_p", float),
                "--rollout-top-k": ("rollout_top_k", int),
                "--vllm-prefill-context-parallel-size": ("rollout_cp_size", int),
            }
            for flag, (key, cast) in legacy_flags.items():
                if key not in options and flag in command:
                    options[key] = cast(command[command.index(flag) + 1])
            if "rollout_tp_size" not in options and "rollout_gpus_per_engine" in topology:
                cp = options.get("rollout_cp_size", 1)
                options["rollout_tp_size"] = topology["rollout_gpus_per_engine"] // cp
                options.setdefault("rollout_cp_size", 1)
            return {"options": options, "model_root": value.get("paths", {}).get("model_root")}
        if frozen.is_file():
            value = json.loads(frozen.read_text(encoding="utf-8"))
            parameters = value.get("parameters", {})
            options = {}
            for section, prefix in (("training", ""), ("rollout", "rollout_")):
                for name, key in (
                    ("tp_size", "tensor_parallel_size"),
                    ("cp_size", "context_parallel_size"),
                ):
                    if key in parameters.get(section, {}):
                        options[prefix + name] = parameters[section][key]
            options.update(
                {
                    f"rollout_{key}": parameters["rollout"][key]
                    for key in ("temperature", "top_p", "top_k")
                    if key in parameters.get("rollout", {})
                }
            )
            return {"options": options}
    return {}


def source_baseline(source: Path, explicit: str | None = None) -> tuple[str, str]:
    """Read executed/requested source cases; never assume a successful repair."""
    from .session import matrix_groups

    def checked(value, origin):
        groups = matrix_groups(value)
        if len(groups) != 1:
            raise ValueError("baseline must be one Mxxx group")
        return groups[0], origin

    if explicit:
        return checked(explicit, "explicit --baseline")
    root = source if source.is_dir() else source.parent
    candidates = [source] if source.is_file() and source.suffix == ".json" else []
    for parent in [root, *list(root.parents)[:4]]:
        candidates.extend(
            parent / name
            for name in (
                "matrix-report.json",
                "manifest.json",
                "single-arm-summary.json",
                "launch-plan.json",
                "arms/r-r/launch.json",
                "arms/p-p/launch.json",
            )
        )
    for path in candidates:
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("baseline"):
            return checked(value["baseline"], str(path))
        env = value.get("runtime_env", {}).get("env_vars", value.get("environment", {}))
        if env.get("RL_KERNEL_ALIGNMENT_GROUP"):
            return checked(env["RL_KERNEL_ALIGNMENT_GROUP"], str(path))
        cases = [env.get(f"RL_KERNEL_{name}_CASE") for name in ("ATTENTION", "FFN", "LOGP")]
        if all(case in {"P/P", "R/R"} for case in cases):
            return "M" + "".join("1" if case == "R/R" else "0" for case in cases), str(path)
        arm = value.get("arm", {})
        group = value.get("group", arm.get("group") if isinstance(arm, dict) else arm)
        if group and group.startswith("M"):
            return checked(group, str(path))
        mode = value.get("mode")
        if mode in {"native", "consistency"}:
            return ("M000" if mode == "native" else "M111"), str(path)
    return "M000", "source has no module configuration; explicit native baseline"
