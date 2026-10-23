#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Exercise actual Vime training and vLLM WS2 sampler providers on shared logits.

Run locally or with torchrun (one rank per GPU by default). No model checkpoint is
needed: the real vLLM Sampler.forward samples from frozen caller-provided logits.
The return-frame observer reads actual provider LSE tensors without replacing any
provider, registry, collective, or numerical kernel. No performance is measured.

    python benchmarks/fused_logp_provider_evidence.py --out /tmp/providers-tp1.json
    torchrun --standalone --nproc-per-node=2 benchmarks/fused_logp_provider_evidence.py \
        --out /tmp/providers-tp2.json --flash-attn-src /path/to/flash-attention
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Choose the actual native sampler without an optional FlashInfer installation.
# The selected implementation and environment are recorded in the report.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from rl_engine.contracts.operators.logprob import (  # noqa: E402
    LogprobContract,
    LogprobDType,
    LogprobRole,
    MaskSpec,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.integrations.engines.rollout.vllm.operators import VllmLogpOperator  # noqa: E402
from rl_engine.integrations.orchestrators.vime.providers.logp import (  # noqa: E402
    SelectedLogprobProviderUnavailable,
)
from rl_engine.integrations.orchestrators.vime.providers.logp import (  # noqa: E402
    provider as vime_provider,
)
from rl_engine.reference.logprob.vocab_parallel_logp import (  # noqa: E402
    BACKEND_ID,
    DEFAULT_NUM_VOCAB_TILES,
)
from rl_engine.runtime.registry import kernel_registry  # noqa: E402


def _comparison(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Compare raw element bytes, including signed zero and NaN payloads."""
    if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
        raise ValueError("comparison requires equal shapes and dtypes")
    candidate, reference = candidate.detach().contiguous(), reference.detach().contiguous()
    different = (
        (candidate.view(torch.uint8) != reference.view(torch.uint8))
        .reshape(candidate.numel(), candidate.element_size())
        .any(dim=-1)
    )
    rows = candidate.shape[0] if candidate.ndim else 1
    finite = torch.isfinite(candidate) & torch.isfinite(reference)
    finite_error = (candidate.double() - reference.double()).abs()[finite]
    return {
        "elements_checked": candidate.numel(),
        "rows_checked": rows,
        "elements_differing": different.sum().item(),
        "rows_differing": different.reshape(rows, -1).any(dim=-1).sum().item() if rows else 0,
        "max_abs": finite_error.max().item() if finite_error.numel() else 0.0,
        "nonfinite_elements": (~finite).sum().item(),
    }


def _readback(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().contiguous().cpu()
    raw = bytes(value.view(torch.uint8).reshape(-1).tolist())
    values = value.tolist()
    if value.is_floating_point():
        # Keep JSON valid even when a candidate produces NaN on a padded shard.
        # Raw bytes remain the authoritative exact representation.
        def json_number(v):
            if isinstance(v, list):
                return [json_number(item) for item in v]
            return v if math.isfinite(v) else str(v)

        values = json_number(values)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "values": values,
        "raw_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_bytes_hex": raw.hex(),
    }


def _fingerprint(value: torch.Tensor) -> dict[str, Any]:
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "byte_count": value.numel() * value.element_size(),
        "raw_bytes_sha256": hashlib.sha256(memoryview(raw)).hexdigest(),
    }


def _observed_call(function: Callable, invoke: Callable) -> tuple[Any, dict[str, Any]]:
    """Read the actual returning function frame; delegate any existing profiler."""
    previous = sys.getprofile()
    observed: dict[str, Any] = {}

    def observe(frame, event, arg):
        if previous is not None:
            previous(frame, event, arg)
        if event == "return" and frame.f_code is function.__code__:
            for name in ("_lse", "lse", "selected", "selected_logp", "contract"):
                if name in frame.f_locals:
                    observed[name] = frame.f_locals[name]
            dispatch = frame.f_locals.get("dispatch")
            if dispatch is not None:
                observed["dispatch_provenance"] = dict(dispatch.provenance)

    sys.setprofile(observe)
    try:
        result = invoke()
    finally:
        sys.setprofile(previous)
    return result, observed


def _sampler_metadata(rows: int, device: torch.device):
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata

    return SamplingMetadata(
        temperature=torch.ones(rows, device=device),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=0,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(rows, device=device),
        presence_penalties=torch.zeros(rows, device=device),
        repetition_penalties=torch.ones(rows, device=device),
        output_token_ids=[[] for _ in range(rows)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def _vllm_forward(logits: torch.Tensor) -> dict[str, Any]:
    from vllm.v1.sample.sampler import Sampler

    sampler = Sampler(logprobs_mode="raw_logprobs")
    operator = VllmLogpOperator(Sampler.forward)
    owned = logits.detach().clone()
    before = owned.clone()
    with torch.no_grad():
        result, observed = _observed_call(
            VllmLogpOperator.__call__,
            lambda: operator(sampler, owned, _sampler_metadata(logits.shape[0], logits.device)),
        )
    if "_lse" not in observed or "selected" not in observed:
        raise RuntimeError("actual vLLM WS2 provider did not expose observed logp/LSE locals")
    labels = result.sampled_token_ids.reshape(-1).long()
    ids = result.logprobs_tensors.logprob_token_ids
    columns = (ids == labels[:, None]).long().argmax(dim=-1)
    selected = result.logprobs_tensors.logprobs.gather(1, columns[:, None]).reshape(-1)
    return {
        "logp": selected.detach(),
        "lse": observed["_lse"].detach(),
        "labels": labels.detach(),
        "observed_logp": observed["selected"].detach(),
        "provenance": {
            **dict(operator.provenance),
            "sampler_class": f"{type(sampler).__module__}.{type(sampler).__qualname__}",
            "native_callback": "vllm.v1.sample.sampler.Sampler.forward",
            "sampler_logprobs_mode": sampler.logprobs_mode,
            "sampler_all_greedy": True,
            "sampler_temperature": 1.0,
            "scoring_tp_world_size": 1,
            "contract": dataclasses.asdict(observed["contract"]),
            "lse_source": "actual VllmLogpOperator.__call__ return-frame local _lse",
            "source_input_mutation": _comparison(owned, before),
        },
    }


def _vime_request(local_logits, labels, real_vocab, padded_vocab, tp_group, tp_rank, tp_world):
    return SimpleNamespace(
        logits=local_logits,
        target_ids=labels,
        tensor_parallel_group=tp_group,
        context_parallel=SimpleNamespace(world_size=1, rank=0, layout="single"),
        with_entropy=False,
        with_entropy_grad=False,
        log_prob_keep_mask=None,
        metadata={
            "real_vocab_size": real_vocab,
            "padded_vocab_size": padded_vocab,
            "tp_rank": tp_rank,
            "tp_world_size": tp_world,
            "num_vocab_tiles": DEFAULT_NUM_VOCAB_TILES,
        },
    )


def _vime_forward(request, *, training: bool) -> dict[str, Any]:
    manager = torch.enable_grad if training else torch.no_grad
    with manager():
        result, observed = _observed_call(vime_provider, lambda: vime_provider(request))
    if "_lse" not in observed:
        raise RuntimeError("actual Vime provider did not expose observed LSE local")
    return {
        "logp": result.selected_logprobs.reshape(-1).detach(),
        "lse": observed["_lse"].detach(),
        "provenance": {
            **dict(result.provenance),
            "backend_id": result.backend_id,
            "contract_id": result.contract_id,
            "contract": dataclasses.asdict(observed["contract"]),
            "grad_enabled_at_boundary": training,
            "lse_source": "actual Vime provider return-frame local _lse",
        },
    }


def _load_flash(source: Path):
    path = source / "flash_attn/ops/triton/cross_entropy.py"
    spec = importlib.util.spec_from_file_location("provider_evidence_flash_ce", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load flash-attn CE from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = source / "SOURCE.json"
    provenance = json.loads(manifest.read_text()) if manifest.exists() else {}
    provenance.update({"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return module, provenance


def _flash_forward(module, logits, labels, *, tp_group=None, training=True):
    manager = torch.enable_grad if training else torch.no_grad
    x = logits.detach().requires_grad_(training)
    with manager():
        result, observed = _observed_call(
            module.CrossEntropyLoss.forward,
            lambda: module.cross_entropy_loss(x, labels, process_group=tp_group),
        )
    if "lse" not in observed:
        raise RuntimeError("actual flash-attn CE forward did not expose observed LSE")
    return {"logp": -result[0].detach(), "lse": observed["lse"].detach()}


def _pair_comparisons(left, right, prefix):
    return {f"{prefix}_{key}": _comparison(left[key], right[key]) for key in ("logp", "lse")}


def _pack_provider(value):
    return {
        "provenance": value["provenance"],
        "readback": {key: _readback(value[key]) for key in ("logp", "lse")},
    }


def _explicit_contract(logits, active, real_vocab, tp_rank, tp_world, role):
    width = logits.shape[1]
    return LogprobContract(
        role=role,
        dtype={torch.bfloat16: LogprobDType.BF16, torch.float32: LogprobDType.FP32}[logits.dtype],
        mask=MaskSpec(num_tokens=logits.shape[0], active_mask=tuple(active)),
        sharding=ShardingSpec(
            tp_rank=tp_rank,
            tp_world_size=tp_world,
            vocab_shard_bounds=tuple((r * width, (r + 1) * width) for r in range(tp_world)),
            real_vocab_size=real_vocab,
            padded_vocab_size=width * tp_world,
        ),
        reduction=ReductionSpec(),
    )


def run_case(
    *,
    rows: int,
    vocab: int,
    seed: int,
    dtype=torch.bfloat16,
    real_vocab: int | None = None,
    temperature: float = 1.0,
    vocab_mask: bool = False,
    active_mask: bool = False,
    flash_attn_src: Path | None = None,
    tp_group=None,
    tp_rank=0,
    tp_world_size=1,
    layouts: bool = True,
) -> dict[str, Any]:
    """Call real integration providers; mask-only contracts have separate scope."""
    real_vocab = vocab if real_vocab is None else real_vocab
    if not 0 < real_vocab <= vocab or vocab % DEFAULT_NUM_VOCAB_TILES or vocab % tp_world_size:
        raise ValueError("real vocab, padded vocab, TP degree and fixed tiles are incompatible")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    generator = torch.Generator().manual_seed(seed)
    frozen = torch.randn(rows, vocab, generator=generator).mul_(2).to(torch.bfloat16)
    # Share the caller's transformed execution values with every provider.
    full = (frozen.float() / temperature).to(dtype).cuda()
    full[:, real_vocab:] = -float("inf")
    if vocab_mask:
        support = torch.arange(vocab, device=full.device) % 7 != 1
        full.masked_fill_(~support[None, :], -float("inf"))
    width = vocab // tp_world_size
    local_slice = slice(tp_rank * width, (tp_rank + 1) * width)
    module, flash_source = _load_flash(Path(flash_attn_src)) if flash_attn_src else (None, None)

    def execute(batch):
        rollout = _vllm_forward(batch)
        shard = batch[:, local_slice].detach().contiguous().requires_grad_(True)
        request = _vime_request(
            shard, rollout["labels"], real_vocab, vocab, tp_group, tp_rank, tp_world_size
        )
        train = _vime_forward(request, training=True)
        values = {"vllm_rollout": rollout, "vime_train": train}
        if module is not None:
            values["flash_train"] = _flash_forward(
                module, shard, rollout["labels"], tp_group=tp_group
            )
            values["flash_rollout"] = _flash_forward(
                module, batch, rollout["labels"], training=False
            )
        return values, request

    values, request = execute(full)
    train, rollout = values["vime_train"], values["vllm_rollout"]
    for value in (train, rollout):
        if value["provenance"]["actual_backend"] != BACKEND_ID or value["provenance"]["fallback"]:
            raise RuntimeError("actual integration provider silently changed WS2 backend")
    no_grad = _vime_forward(request, training=False)
    comparisons = {
        **_pair_comparisons(train, rollout, "vime_vs_vllm"),
        **_pair_comparisons(train, no_grad, "vime_train_vs_no_grad"),
        "vllm_selected_vs_observed_logp": _comparison(rollout["logp"], rollout["observed_logp"]),
    }
    result = {
        "configuration": {
            "rows": rows,
            "padded_vocab": vocab,
            "real_vocab": real_vocab,
            "seed": seed,
            "dtype": str(dtype),
            "temperature": temperature,
            "temperature_application": "shared caller FP32 divide, then one execution-dtype cast",
            "vocab_mask": (
                "shared caller -inf support (column index % 7 == 1)" if vocab_mask else None
            ),
            "padding_application": (
                "shared -inf padding; Vime explicit real/padded contract; "
                "vLLM full-width contract"
            ),
            "training_tp": tp_world_size,
            "training_tp_rank": tp_rank,
            "vllm_scoring_tp": 1,
            "cp_world_size": 1,
        },
        "providers": {key: _pack_provider(values[key]) for key in ("vime_train", "vllm_rollout")},
        "input_fingerprints": {
            "full_scoring_logits": _fingerprint(full),
            "training_logits_shard": _fingerprint(request.logits),
        },
        "sampled_token_ids": _readback(rollout["labels"]),
        "comparisons": comparisons,
        "layouts": {},
        "flash_attn": {"status": "not_requested"},
    }
    if module is not None:
        flash_no_grad = _flash_forward(
            module, request.logits, rollout["labels"], tp_group=tp_group, training=False
        )
        flash_full_train = _flash_forward(module, full, rollout["labels"])
        result["flash_attn"] = {
            "status": "executed",
            "source": flash_source,
            "rollout_composition": (
                "experimental CE scorer on frozen full logits and token IDs produced by the "
                "actual vLLM Sampler.forward; no flash-attn rollout provider is installed"
            ),
            "training_tp": tp_world_size,
            "rollout_tp": 1,
            "lse_source": "actual CrossEntropyLoss.forward return-frame local lse",
            "same_provider_train_vs_no_grad": _pair_comparisons(
                values["flash_train"], flash_no_grad, "flash"
            ),
            "coordinated_train_vs_rollout": _pair_comparisons(
                values["flash_train"], values["flash_rollout"], "flash"
            ),
            "replicated_tp1_train_vs_rollout_control": _pair_comparisons(
                flash_full_train, values["flash_rollout"], "flash"
            ),
            "changed_implementation_vs_ws2": _pair_comparisons(
                values["flash_train"], train, "flash_vs_ws2"
            ),
            "readback": {
                key: {kind: _readback(values[key][kind]) for kind in ("logp", "lse")}
                for key in ("flash_train", "flash_rollout")
            },
        }
    # The adapters themselves have no active-token argument. Exercise the real
    # registry contract separately and state that boundary explicitly.
    active = [not active_mask or r % 4 != 1 for r in range(rows)]
    active_tensor = torch.tensor(active, device=full.device)
    targets = torch.where(active_tensor, rollout["labels"], -100)
    direct = {}
    for role in (LogprobRole.TRAIN, LogprobRole.INFER):
        contract = _explicit_contract(
            request.logits, active, real_vocab, tp_rank, tp_world_size, role
        )
        dispatch = kernel_registry.get_logprob_op(contract, requested_backend=BACKEND_ID)
        with torch.enable_grad() if role is LogprobRole.TRAIN else torch.no_grad():
            lp, lse = dispatch.op(
                request.logits,
                targets,
                contract=contract,
                tp_group=tp_group,
                num_vocab_tiles=DEFAULT_NUM_VOCAB_TILES,
            )
        direct[role.value] = {"logp": lp.detach(), "lse": lse.detach()}
    result["explicit_active_mask"] = {
        "scope": (
            "actual WS2 registry operators; integration adapters themselves declare all-active"
        ),
        "active_rows": active,
        "ignored_targets": targets.cpu().tolist(),
        "comparisons": {
            **_pair_comparisons(direct["train"], direct["infer"], "train_vs_infer"),
            "active_or_zero_logp": _comparison(
                direct["train"]["logp"], torch.where(active_tensor, rollout["logp"], 0.0)
            ),
            "all_rows_lse": _comparison(direct["train"]["lse"], rollout["lse"]),
        },
        "readback": {key: _readback(direct["train"][key]) for key in ("logp", "lse")},
    }
    result["provider_active_compaction"] = {}
    if active_mask:
        compact, _ = execute(full[active_tensor])
        result["provider_active_compaction"] = {
            name: _pair_comparisons(
                compact[name],
                {
                    kind: value[active_tensor]
                    for kind, value in values[name].items()
                    if kind in ("logp", "lse")
                },
                "compact_vs_active_full",
            )
            for name in values
        }
    rejected = _vime_request(
        request.logits, request.target_ids, real_vocab, vocab, tp_group, tp_rank, tp_world_size
    )
    rejected.log_prob_keep_mask = torch.ones_like(request.logits, dtype=torch.bool)
    try:
        vime_provider(rejected)
    except SelectedLogprobProviderUnavailable as exc:
        rejection = {"status": "rejected", "type": type(exc).__name__, "message": str(exc)}
    else:
        rejection = {"status": "unexpectedly_accepted"}
    result["unsupported_requests"] = {"vime_top_p_replay": rejection}
    if layouts:
        layouts_to_run = [
            ("reverse_permutation", [torch.arange(rows - 1, -1, -1, device=full.device)])
        ]
        layouts_to_run += [
            (f"chunks_{size}", list(torch.arange(rows, device=full.device).split(size)))
            for size in (1, 7)
        ]
        for name, indices in layouts_to_run:
            assembled = {
                key: {kind: torch.empty_like(values[key][kind]) for kind in ("logp", "lse")}
                for key in values
            }
            for index in indices:
                part, _ = execute(full[index])
                for key in assembled:
                    for kind in ("logp", "lse"):
                        assembled[key][kind][index] = part[key][kind]
            result["layouts"][name] = {
                key: _pair_comparisons(assembled[key], values[key], "layout_vs_full")
                for key in assembled
            }
            result["layouts"][name]["integration_train_vs_rollout"] = _pair_comparisons(
                assembled["vime_train"], assembled["vllm_rollout"], "vime_vs_vllm"
            )
            if module is not None:
                result["layouts"][name]["coordinated_flash_train_vs_rollout"] = _pair_comparisons(
                    assembled["flash_train"], assembled["flash_rollout"], "flash"
                )
    return result


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _summarize_comparisons(entries):
    entries = list(entries)
    result = {
        "comparisons_checked": len(entries),
        "elements_checked": sum(m["elements_checked"] for m in entries),
        "rows_checked": sum(m["rows_checked"] for m in entries),
        "elements_differing": sum(m["elements_differing"] for m in entries),
        "rows_differing": sum(m["rows_differing"] for m in entries),
        "nonfinite_elements": sum(m["nonfinite_elements"] for m in entries),
        "max_abs": max((m["max_abs"] for m in entries), default=0.0),
    }
    result["all_bitwise_equal"] = result["elements_differing"] == 0
    result["all_finite"] = result["nonfinite_elements"] == 0
    result["passes"] = bool(entries) and result["all_bitwise_equal"] and result["all_finite"]
    return result


def _summary(cases):
    baseline, self_modes, tp1_controls, candidate_layouts, coordinated, changed = (
        [] for _ in range(6)
    )
    for case in cases:
        baseline.extend(case["comparisons"].values())
        baseline.extend(case["explicit_active_mask"]["comparisons"].values())
        for name, metrics in case["provider_active_compaction"].items():
            if name in ("vime_train", "vllm_rollout"):
                baseline.extend(metrics.values())
            elif name in ("flash_train", "flash_rollout"):
                candidate_layouts.extend(metrics.values())
        for layout in case["layouts"].values():
            for name in ("vime_train", "vllm_rollout", "integration_train_vs_rollout"):
                baseline.extend(layout[name].values())
            for name in ("flash_train", "flash_rollout"):
                if name in layout:
                    candidate_layouts.extend(layout[name].values())
            coordinated.extend(layout.get("coordinated_flash_train_vs_rollout", {}).values())
        flash = case["flash_attn"]
        if flash["status"] == "executed":
            self_modes.extend(flash["same_provider_train_vs_no_grad"].values())
            tp1_controls.extend(flash["replicated_tp1_train_vs_rollout_control"].values())
            coordinated.extend(flash["coordinated_train_vs_rollout"].values())
            changed.extend(flash["changed_implementation_vs_ws2"].values())
    candidate = {
        "status": "executed" if self_modes else "not_requested",
        "same_provider_train_vs_no_grad": _summarize_comparisons(self_modes),
        "replicated_tp1_train_vs_rollout_control": _summarize_comparisons(tp1_controls),
        "fixed_topology_layout_invariance": _summarize_comparisons(candidate_layouts),
        "coordinated_train_vs_rollout": _summarize_comparisons(coordinated),
        "changed_implementation_vs_ws2_diagnostic": _summarize_comparisons(changed),
    }
    candidate["coordinated_replacement_passes_sampled_contract"] = (
        bool(self_modes)
        and all(
            candidate[name]["passes"]
            for name in (
                "same_provider_train_vs_no_grad",
                "replicated_tp1_train_vs_rollout_control",
                "coordinated_train_vs_rollout",
            )
        )
        and (not candidate_layouts or candidate["fixed_topology_layout_invariance"]["passes"])
    )
    return {
        "case_count": len(cases),
        "existing_ws2_provider_baseline": _summarize_comparisons(baseline),
        "flash_attn_candidate": candidate,
        "top_p_replay_rejection_observed": all(
            c["unsupported_requests"]["vime_top_p_replay"]["status"] == "rejected" for c in cases
        ),
        "scope": (
            "sampled provider/scorer contracts; "
            "CE-vs-WS2 drift is diagnostic, not a replacement gate"
        ),
    }


def _module_source(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot locate actual module source: {module_name}")
    path = Path(spec.origin)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--flash-attn-src", type=Path)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--rows", type=int, nargs="+", default=(17, 33))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--seeds", type=int, nargs="+", help="override --seed with a seed list")
    parser.add_argument("--temperatures", type=float, nargs="+", default=(1.0, 0.7))
    parser.add_argument(
        "--dtypes", choices=("bfloat16", "float32"), nargs="+", default=("bfloat16", "float32")
    )
    parser.add_argument("--padding", type=int, default=7)
    parser.add_argument("--no-layouts", action="store_true")
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--allow-shared-gpus", action="store_true")
    args = parser.parse_args()
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    devices = torch.cuda.device_count()
    if not devices:
        raise SystemExit("requires CUDA and an importable genuine vLLM Sampler")
    if local_rank >= devices and not args.allow_shared_gpus:
        raise SystemExit(
            "rank count exceeds physical GPUs; "
            "pass --allow-shared-gpus to label rank sharing explicitly"
        )
    if world > devices and args.distributed_backend == "nccl":
        raise SystemExit(
            "NCCL duplicate-device ranks are unsupported; use a valid transport for GPU sharing"
        )
    device = local_rank % devices
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(
            backend=args.distributed_backend,
            device_id=torch.device("cuda", device) if args.distributed_backend == "nccl" else None,
        )
    properties = torch.cuda.get_device_properties(device)
    topology = {
        "rank": rank,
        "local_rank": local_rank,
        "device_index": device,
        "device_name": properties.name,
        "device_uuid": str(getattr(properties, "uuid", "unknown")),
        "visible_gpu_count": devices,
        "world_size": world,
        "transport": dist.get_backend() if world > 1 else "none",
        "physical_gpu_topology_claim": (
            "one rank per GPU"
            if world <= devices
            else "multiple ranks per GPU; not four physical GPUs"
        ),
    }
    report = {
        "kind": "actual_vime_vllm_ws2_provider_evidence",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "vllm": _version("vllm"),
            "triton": _version("triton"),
            "VLLM_USE_FLASHINFER_SAMPLER": os.getenv("VLLM_USE_FLASHINFER_SAMPLER"),
            "installed_vllm_sources": {
                name: _module_source(name)
                for name in ("vllm.v1.sample.sampler", "vllm.v1.sample.metadata")
            },
        },
        "topology": topology,
        "scope": (
            "real integration providers on frozen logits; actual Sampler.forward; "
            "no full model or serving engine"
        ),
        "cases": [],
    }
    try:
        for seed in args.seeds or [args.seed]:
            for rows in args.rows:
                for dtype in args.dtypes:
                    for temperature in args.temperatures:
                        for masked in (False, True):
                            case = run_case(
                                rows=rows,
                                vocab=args.vocab,
                                seed=seed,
                                dtype=getattr(torch, dtype),
                                real_vocab=args.vocab - args.padding if masked else args.vocab,
                                temperature=temperature,
                                vocab_mask=masked,
                                active_mask=masked,
                                flash_attn_src=args.flash_attn_src,
                                tp_group=dist.group.WORLD if world > 1 else None,
                                tp_rank=rank,
                                tp_world_size=world,
                                layouts=not args.no_layouts,
                            )
                            report["cases"].append(case)
                            report["summary"] = _summary(report["cases"])
                            destination = (
                                args.out
                                if world == 1
                                else args.out.with_name(
                                    f"{args.out.stem}.rank{rank}{args.out.suffix}"
                                )
                            )
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            destination.write_text(
                                json.dumps(report, indent=2, allow_nan=False) + "\n"
                            )
                            print(
                                f"rank {rank}: seed={seed} rows={rows} dtype={dtype} "
                                f"temperature={temperature} masked={masked} complete",
                                flush=True,
                            )
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, report)
            if rank == 0:
                args.out.write_text(
                    json.dumps(
                        {
                            "kind": report["kind"],
                            "ranks": gathered,
                            "summary": _summary(
                                [case for item in gathered for case in item["cases"]]
                            ),
                        },
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
