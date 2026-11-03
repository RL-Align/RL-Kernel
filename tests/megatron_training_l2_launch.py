# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 WS1 L2 launch variants and canonical grouping on one SM90 GPU.

Runs in a fresh single-rank process (``torchrun --nproc_per_node=1``) so the
first strict call measured below is the first one in the process. Every case
compares the selected-token FP32 raw bytes of the same rows and records the
entry point that produced them; different entry points are recorded, never
merged into one verdict.

Cases (plan section 5.5, L2 launch and L2 canonical grouping):

* ``first_call``: the process's first strict call vs three warm repeats;
* ``chunked``: one whole call vs contiguous row chunks and single-row calls,
  on the canonical TP entry (size-1 TP group, as VIME passes it) and on the
  ungrouped single-GPU entry;
* ``reuse_recompute``: a tiny Megatron GPT scored by VIME's provider through
  the reused LM-head logits vs recomputation from the captured hidden rows;
* ``canonical``: canonical virtual TP 1/2/4 at physical TP1; canonical vocab
  padding (padded 512 with a canonical vocab of 520 vs a 520-row LM head with
  the same boundary); invalid canonical settings must be refused;
* ``cuda_graph``: capture the canonical TP entry forward in a CUDA graph and
  replay it; capture failure is recorded (the training side has no CUDA graph
  for this route without Transformer Engine). Runs last.

    torchrun --standalone --nproc_per_node=1 tests/megatron_training_l2_launch.py \
        --out runs/<id>/l2_launch.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from p7_doubles import t02_sampling  # noqa: E402
from p7_doubles.fixtures import make_fixture  # noqa: E402

from rl_engine.alignment.cross_config.adapters import megatron_training_gate as gate  # noqa: E402
from rl_engine.alignment.cross_config.adapters.megatron_training import (  # noqa: E402
    PreUpdateWeightLock,
    ShardDigest,
)
from rl_engine.alignment.cross_config.adapters.megatron_training_provider import (  # noqa: E402
    M10ProviderInput,
    RecordedTrainingProvider,
)

SAMPLE_LENGTHS = (64, 32)
REAL_VOCAB, PADDED_VOCAB, WIDTH = 500, 512, 128
ROWS = sum(SAMPLE_LENGTHS)
CANONICAL_ENV = "RL_KERNEL_STRICT_CANONICAL_TP"
CANONICAL_VOCAB_ENV = "RL_KERNEL_STRICT_CANONICAL_VOCAB_SIZE"
FIXTURE = make_fixture(
    "l2_launch",
    sample_lengths=SAMPLE_LENGTHS,
    width=WIDTH,
    real_vocab=REAL_VOCAB,
    padded_vocab=PADDED_VOCAB,
)


@contextlib.contextmanager
def _env(**values: str | None):
    saved = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _bits(values: torch.Tensor) -> list[int]:
    return values.detach().float().contiguous().reshape(-1).cpu().view(torch.int32).tolist()


def _diff(reference: list[int], other: list[int]) -> int:
    if len(reference) != len(other):
        return max(len(reference), len(other))
    return sum(1 for a, b in zip(reference, other, strict=True) if a != b)


def _inputs(weight_rows: int = PADDED_VOCAB):
    hidden = FIXTURE.hidden.cuda()
    weight = FIXTURE.lm_head_weight.cuda()
    if weight_rows > weight.size(0):
        extra = torch.zeros(weight_rows - weight.size(0), weight.size(1), dtype=weight.dtype)
        weight = torch.cat([weight, extra.cuda()])
    return hidden, weight, FIXTURE.target_ids.cuda()


def _wrapper_call(group: Any, *, rows=slice(None), weight_rows: int = PADDED_VOCAB):
    from rl_engine.integrations.linear_logp import LinearLogpWrapper

    hidden, weight, targets = _inputs(weight_rows)
    wrapper = LinearLogpWrapper()
    with torch.no_grad():
        out = wrapper(
            hidden[rows].contiguous(),
            weight,
            targets[rows].contiguous(),
            tp_group=group,
            vocab_start_index=0,
            global_vocab_size=weight.size(0),
            real_vocab_size=REAL_VOCAB,
        )
    return out, dict(wrapper.provenance)


def _m10(group: Any) -> M10ProviderInput:
    weight = FIXTURE.lm_head_weight.cuda()
    shard = ShardDigest.from_tensor(weight, tp_rank=0, vocab_start=0, vocab_end=PADDED_VOCAB)
    return M10ProviderInput(
        final_decoder_rows=FIXTURE.hidden.cuda(),
        lm_head_weight_local=weight,
        target_token_ids=FIXTURE.target_ids.cuda(),
        active_token_mask=FIXTURE.active_mask.clone(),
        vocab_start=0,
        real_vocab_size=REAL_VOCAB,
        padded_vocab_size=PADDED_VOCAB,
        token_layout_world_size=1,
        token_layout_rank=0,
        token_layout="single",
        row_layout=gate.RowLayout(("s0", "s1"), SAMPLE_LENGTHS),
        checkpoint_id="ckpt-0",
        weight_id="w-0",
        weight_lock=PreUpdateWeightLock("ckpt-0", "w-0", (shard,)),
        lm_head_lora=False,
        sampling_record=t02_sampling.record(t02_sampling.IDENTITY_REQUEST),
        tensor_parallel_group=group,
        metadata={"tp_rank": 0, "tp_world_size": 1},
    )


def case_first_call(group: Any) -> dict[str, Any]:
    first = RecordedTrainingProvider().score(_m10(group))
    warm = [RecordedTrainingProvider().score(_m10(group)) for _ in range(3)]
    return {
        "entrypoint": first.entrypoint,
        "canonical_configured": first.to_dict().get("canonical_configured"),
        "warm_entrypoints": sorted({item.entrypoint for item in warm}),
        "active_tokens": int(first.to_dict()["active_count"]),
        "warm_identical": [item.active_raw_bytes == first.active_raw_bytes for item in warm],
        "ok": all(item.active_raw_bytes == first.active_raw_bytes for item in warm)
        and all(item.entrypoint == first.entrypoint for item in warm),
    }


def case_chunked(group: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    ok = True
    for label, call_group in (("canonical_tp_entry", group), ("ungrouped_entry", None)):
        whole, provenance = _wrapper_call(call_group)
        whole_bits = _bits(whole)
        variants: dict[str, int] = {}
        entries = {provenance.get("strict_entrypoint")}
        for name, bounds in (
            ("chunks_17_33_46", (0, 17, 50, ROWS)),
            ("chunks_of_32", (0, 32, 64, ROWS)),
            ("single_rows", tuple(range(ROWS + 1))),
        ):
            pieces = []
            for start, end in zip(bounds[:-1], bounds[1:], strict=True):
                out, prov = _wrapper_call(call_group, rows=slice(start, end))
                pieces.extend(_bits(out))
                entries.add(prov.get("strict_entrypoint"))
            variants[name] = _diff(whole_bits, pieces)
        ok = ok and all(count == 0 for count in variants.values()) and len(entries) == 1
        result[label] = {
            "entrypoints": sorted(str(item) for item in entries),
            "rows": ROWS,
            "rows_differing_from_whole": variants,
        }
    group_bits = _bits(_wrapper_call(group)[0])
    plain_bits = _bits(_wrapper_call(None)[0])
    result["record_only_entry_difference"] = _diff(group_bits, plain_bits)
    result["ok"] = ok
    return result


def case_canonical(group: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"by_canonical_tp": {}}
    bits: dict[str, list[int]] = {}
    for canonical in (None, "1", "2", "4", "8"):
        with _env(**{CANONICAL_ENV: canonical, CANONICAL_VOCAB_ENV: None}):
            out, provenance = _wrapper_call(group)
        key = canonical or "unset"
        bits[key] = _bits(out)
        result["by_canonical_tp"][key] = {"entrypoint": provenance.get("strict_entrypoint")}
    for key in bits:
        result["by_canonical_tp"][key]["rows_differing_from_unset"] = _diff(
            bits["unset"], bits[key]
        )
    with _env(**{CANONICAL_ENV: "4", CANONICAL_VOCAB_ENV: str(PADDED_VOCAB)}):
        explicit = _bits(_wrapper_call(group)[0])
    with _env(**{CANONICAL_ENV: "4", CANONICAL_VOCAB_ENV: "520"}):
        padded = _bits(_wrapper_call(group)[0])
        wide_head = _bits(_wrapper_call(group, weight_rows=520)[0])
    result["same_boundary"] = {
        "canonical_vocab_explicit_512_vs_default": _diff(bits["4"], explicit),
        "padded_512_to_520_vs_520_row_head": _diff(padded, wide_head),
        "padded_520_vs_unpadded_512_record_only": _diff(bits["4"], padded),
    }
    refused: dict[str, str] = {}
    for label, settings in (
        ("canonical_tp_3_does_not_divide_512", {CANONICAL_ENV: "3", CANONICAL_VOCAB_ENV: None}),
        ("canonical_tp_0", {CANONICAL_ENV: "0", CANONICAL_VOCAB_ENV: None}),
        ("canonical_vocab_below_padded", {CANONICAL_ENV: "4", CANONICAL_VOCAB_ENV: "500"}),
        ("canonical_vocab_not_divisible", {CANONICAL_ENV: "4", CANONICAL_VOCAB_ENV: "514"}),
    ):
        with _env(**settings):
            try:
                _wrapper_call(group)
                torch.cuda.synchronize()
                refused[label] = "ACCEPTED (expected refusal)"
            except (ValueError, RuntimeError, ZeroDivisionError) as error:
                refused[label] = f"{type(error).__name__}: {error}"
    result["invalid_settings"] = refused
    result["ok"] = (
        result["same_boundary"]["canonical_vocab_explicit_512_vs_default"] == 0
        and result["same_boundary"]["padded_512_to_520_vs_520_row_head"] == 0
        and bits["unset"] == bits["1"]
        and not any(value.startswith("ACCEPTED") for value in refused.values())
    )
    return result


def case_reuse_recompute() -> dict[str, Any]:
    from megatron.core import parallel_state as mpu
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.module import Float16Module
    from megatron.core.transformer.transformer_config import TransformerConfig
    from vime.backends.megatron_utils.linear_logp_provider import LinearLogpRequest, TokenLayout
    from vime.backends.megatron_utils.model import (
        _install_linear_logp_capture,
        _take_linear_logp_context,
    )

    from rl_engine.integrations.vime.linear_logp_provider import provider

    seq = 32
    mpu.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1234)
    try:
        config = TransformerConfig(
            num_layers=2,
            hidden_size=WIDTH,
            num_attention_heads=4,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        torch.manual_seed(0)
        model = GPTModel(
            config,
            get_gpt_layer_local_spec(),
            vocab_size=PADDED_VOCAB,
            max_sequence_length=seq,
            parallel_output=True,
            share_embeddings_and_output_weights=False,
        ).cuda()
        model = Float16Module(config, model)
        _install_linear_logp_capture(
            [model], argparse.Namespace(padded_vocab_size=PADDED_VOCAB, vocab_size=REAL_VOCAB)
        )
        tokens = ((torch.arange(seq, device="cuda") * 7 + 3) % REAL_VOCAB).reshape(1, seq)
        positions = torch.arange(seq, device="cuda").reshape(1, seq)
        targets = torch.zeros(seq, dtype=torch.long, device="cuda")
        targets[: seq - 1] = tokens[0, 1:]
        outputs = {}
        with torch.no_grad():
            logits = model(tokens, positions, None).squeeze(0).contiguous()
            context = _take_linear_logp_context(model)
            for label, reuse in (("reuse", True), ("recompute", False)):
                # As VIME loss.py builds it: with complete sampling support in strict mode
                # (RL_KERNEL_COMPLETE_SAMPLING_SUPPORT=1) the FP32 promotion made by
                # Float16Module is undone and the logits are not temperature-scaled.
                request_logits = logits.to(torch.bfloat16).contiguous() if reuse else logits
                request = LinearLogpRequest(
                    logits=request_logits,
                    target_ids=targets,
                    tensor_parallel_group=mpu.get_tensor_model_parallel_group(),
                    token_layout=TokenLayout(world_size=1, rank=0, layout="single"),
                    with_entropy=False,
                    with_entropy_grad=False,
                    chunk_size=1024,
                    log_prob_keep_mask=None,
                    context=context,
                    temperature=1.0,
                    metadata={
                        "logits_are_temperature_scaled": not reuse,
                        "complete_sampling_support": reuse,
                        "real_vocab_size": REAL_VOCAB,
                        "padded_vocab_size": PADDED_VOCAB,
                        "tp_rank": 0,
                        "tp_world_size": 1,
                    },
                )
                scored = provider(request)
                execution = scored.provenance.get("execution", {})
                outputs[label] = {
                    "bits": _bits(scored.logp.reshape(-1)[: seq - 1]),
                    "lm_head_result_reused": execution.get("lm_head_result_reused"),
                    "entrypoint": scored.provenance.get("strict_entrypoint"),
                    "logits_dtype": str(request_logits.dtype),
                }
        reuse, recompute = outputs["reuse"], outputs["recompute"]
        return {
            "rows": seq - 1,
            "reuse": {k: v for k, v in reuse.items() if k != "bits"},
            "recompute": {k: v for k, v in recompute.items() if k != "bits"},
            "rows_differing": _diff(reuse["bits"], recompute["bits"]),
            "paths_recorded_distinctly": reuse["lm_head_result_reused"] is True
            and recompute["lm_head_result_reused"] is False,
            "ok": reuse["lm_head_result_reused"] is True
            and recompute["lm_head_result_reused"] is False,
        }
    finally:
        mpu.destroy_model_parallel()


def case_cuda_graph(group: Any) -> dict[str, Any]:
    from rl_engine.integrations.linear_logp import LinearLogpWrapper

    hidden, weight, targets = _inputs()
    wrapper = LinearLogpWrapper()

    def forward() -> torch.Tensor:
        return wrapper(
            hidden,
            weight,
            targets,
            tp_group=group,
            vocab_start_index=0,
            global_vocab_size=PADDED_VOCAB,
            real_vocab_size=REAL_VOCAB,
        )

    with torch.no_grad():
        eager = _bits(forward())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                static_out = forward()
        except Exception as error:  # capture refusal is the recorded outcome
            torch.cuda.synchronize()
            return {
                "captured": False,
                "error": f"{type(error).__name__}: {str(error)[:400]}",
                "ok": True,
                "note": "record only: training side has no CUDA graph for this route",
            }
        graph.replay()
        torch.cuda.synchronize()
        replay = _bits(static_out)
    return {
        "captured": True,
        "entrypoint": wrapper.provenance.get("strict_entrypoint"),
        "rows_differing_from_eager": _diff(eager, replay),
        "ok": _diff(eager, replay) == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="l2_launch.json")
    args = parser.parse_args()
    os.environ.update(
        VIME_RL_KERNEL_STRICT="1",
        RL_KERNEL_LOGP_CASE="R/R",
        RL_KERNEL_ATTENTION_CASE="P/P",
        RL_KERNEL_FFN_CASE="P/P",
    )
    os.environ.pop(CANONICAL_ENV, None)
    os.environ.pop(CANONICAL_VOCAB_ENV, None)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    if dist.get_world_size() != 1:
        raise SystemExit("single-GPU script: run with --nproc_per_node=1")
    group = dist.new_group([0])

    from rl_engine.integrations.ablation import integration_plan_from_environment
    from rl_engine.integrations.megatron_runtime import install_megatron_integration

    cases: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("first_call", lambda: case_first_call(group)),  # must stay first
        ("chunked", lambda: case_chunked(group)),
        ("canonical", lambda: case_canonical(group)),
        (
            "reuse_recompute",
            lambda: (
                install_megatron_integration(integration_plan_from_environment()),
                case_reuse_recompute(),
            )[1],
        ),
        ("cuda_graph", lambda: case_cuda_graph(group)),  # last: a failed capture may taint state
    ]
    results: dict[str, Any] = {}
    status = 0
    for name, run in cases:
        try:
            results[name] = run()
        except Exception as error:
            traceback.print_exc()
            results[name] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        ok = bool(results[name].get("ok"))
        status |= 0 if ok else 1
        summary = {k: v for k, v in results[name].items() if k not in ("by_canonical_tp",)}
        print(f"{name}: ok={ok} {json.dumps(summary, default=str)[:900]}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1, sort_keys=True, default=str))
    dist.destroy_process_group()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
