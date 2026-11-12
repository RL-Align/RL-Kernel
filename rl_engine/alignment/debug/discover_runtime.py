# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Turnkey HF/vLLM automatic discovery; no model-specific mappings or kernels."""

from __future__ import annotations

import gc
import json
import os
from functools import partial
from pathlib import Path

import torch

from .discovery import DiscoverCapture
from .entry import analyze, print_report


def _edit_leaf(value, path):
    if not path:
        result = value.clone()
        result.reshape(-1)[0] += 0.125
        return result
    key, *rest = path
    if isinstance(value, dict):
        return {k: _edit_leaf(v, rest) if k == key else v for k, v in value.items()}
    items = [_edit_leaf(v, rest) if i == key else v for i, v in enumerate(value)]
    return tuple(items) if isinstance(value, tuple) else items


def _install(model, root, tokens, stack, injection):
    tokens = torch.tensor(tokens, dtype=torch.long)
    observer = DiscoverCapture(
        Path(root),
        "rollout",
        model,
        tokens=tokens,
        stack=stack,
        context={"phase": "prefill", "cache": "empty", "mask": "causal"},
    )
    handle = None
    if injection is not None:
        path = json.loads(injection["field"])

        def change(module, args, kwargs):
            values = _edit_leaf({"args": args, "kwargs": kwargs}, path)
            return values["args"], values["kwargs"]

        handle = model.get_submodule(injection["module"]).register_forward_pre_hook(
            change,
            with_kwargs=True,
        )
    try:
        observer.__enter__()
    except Exception:
        if handle:
            handle.remove()
        raise
    model._automatic_discovery = observer, handle
    return True


def _close(model, error=None):
    observer, handle = model._automatic_discovery
    failure = RuntimeError(error) if error else None
    observer.__exit__(RuntimeError if failure else None, failure, None)
    if handle:
        handle.remove()
    del model._automatic_discovery
    return True


def run(
    model_path: Path,
    tokens: list[list[int]],
    output: Path,
    *,
    dtype: str = "bfloat16",
    training_stack: str | None = None,
    rollout_stack: str | None = None,
    fault_control: bool = False,
):
    """Native prefill replay for any architecture BOTH frameworks can construct.

    The model path and tokens are inputs, not a registered RL Kernel model name.
    A fault control, if requested, uses the generated plan to choose its target.
    """
    if (
        not tokens
        or not tokens[0]
        or any(len(row) != len(tokens[0]) for row in tokens)
        or any(type(token) is not int or token < 0 for row in tokens for token in row)
    ):
        raise ValueError("tokens must be a rectangular nonempty batch of nonnegative integer IDs")
    model_path, output = Path(model_path).resolve(), Path(output).resolve()
    if not model_path.is_dir() or not list(model_path.glob("*.safetensors")):
        raise ValueError("provide a local HF checkpoint with safetensors weights")
    if not torch.cuda.is_available():
        raise ValueError("native HF/vLLM discovery requires a CUDA or ROCm GPU")
    output.mkdir(parents=True, exist_ok=False)
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from transformers import AutoModelForCausalLM
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    arms = ["native", "injected"] if fault_control else ["native"]
    frozen = torch.tensor(tokens, dtype=torch.long)
    positions = torch.arange(frozen.shape[1]).expand_as(frozen)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=getattr(torch, dtype),
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
        )
        .cuda()
        .eval()
    )
    for arm in arms:
        with (
            torch.inference_mode(),
            DiscoverCapture(
                output / arm,
                "training",
                model,
                tokens=frozen,
                positions=positions,
                stack=training_stack,
                context={"phase": "prefill", "cache": "empty", "mask": "causal"},
            ),
        ):
            model(
                input_ids=frozen.cuda(),
                position_ids=positions.cuda(),
                attention_mask=torch.ones_like(frozen, device="cuda"),
                use_cache=False,
            )
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    engine = LLM(
        model=str(model_path),
        dtype=dtype,
        trust_remote_code=False,
        skip_tokenizer_init=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        compilation_config=0,
        max_model_len=frozen.shape[1] + 1,
        max_num_seqs=len(tokens),
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        kv_cache_memory_bytes=256 * 1024 * 1024,
        gpu_memory_utilization=0.8,
    )
    reports, injection = {}, None
    try:
        for arm in arms:
            engine.apply_model(
                partial(
                    _install,
                    root=str(output / arm),
                    tokens=tokens,
                    stack=rollout_stack,
                    injection=injection,
                )
            )
            try:
                engine.generate(
                    [{"prompt_token_ids": prompt} for prompt in tokens],
                    SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, detokenize=False),
                    use_tqdm=False,
                )
            except Exception as exc:
                engine.apply_model(partial(_close, error=str(exc)))
                raise
            engine.apply_model(_close)
            report = analyze(output / arm)
            reports[arm] = report
            print_report(report, arm)
            if report["status"] == "not_comparable":
                break
            if fault_control and arm == "native":
                first = report["automatic_mapping"]["blocks"][0]
                if "block_input" not in first:
                    raise ValueError("no verified first-block input exists for the fault control")
                injection = {"module": first["rollout"], "field": first["block_input"]["rollout"]}
            if arm == "injected":
                point = report.get("first_divergence") or {}
                if point.get("boundary") != "block.0.input":
                    raise ValueError(
                        "generated-plan fault did not localize to the first block input"
                    )
    finally:
        engine.llm_engine.engine_core.shutdown()
        cleanup_dist_env_and_memory()
    summary = {
        "model": str(model_path),
        "scope": "native HF/vLLM TP1 eager prefill; automatic block/primitive mapping",
        "results": {
            arm: {
                "status": report["status"],
                "mapping": report["automatic_mapping"]["status"],
                "blocks": len(report["automatic_mapping"].get("blocks", [])),
                "events": len(report["boundaries"]),
                "first_divergence": report["first_divergence"],
                "errors": report["errors"],
            }
            for arm, report in reports.items()
        },
    }
    (output / "discovery-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return reports["native"]


def run_cli(args):
    report = run(
        args.model,
        json.loads(args.tokens.read_text("utf-8")),
        args.output,
        dtype=args.dtype,
        training_stack=args.training_stack,
        rollout_stack=args.rollout_stack,
    )
    return {"equal": 0, "diverged": 1, "not_comparable": 2}[report["status"]]
