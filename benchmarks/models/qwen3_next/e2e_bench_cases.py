# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Workloads, synthetic inputs, arm environments and summaries of the end-to-end speed test.

Pure Python plus Torch on the CPU, so the report logic is testable without GPUs.
"""

import statistics

import torch

VOCAB = 151936
PHASES = ("rollout", "replay", "train")
ROLLOUT_ARMS = ("strict", "native_graph", "native_eager", "native_bi")
ACTOR_ARMS = ("strict", "native")
STRICT_ARM = "strict"

# Shapes are (prompt tokens, batch, response tokens).
WORKLOADS = {
    # The L2 matrix shapes the certified profile covers.
    "certified": {
        "max_model_len": 1040,
        "kv_cache_bytes": 4 << 30,
        "rollout": [(p, b, 16) for p in (64, 256, 1024) for b in (1, 4)],
        "replay": [(p, b, 16) for p in (64, 256, 1024) for b in (1, 4)],
        "train": (64, 4, 16),
    },
    # Longer generation. Strict runs use the certified providers outside the L2 shapes.
    "long": {
        "max_model_len": 2048,
        "kv_cache_bytes": 8 << 30,
        "rollout": [(512, b, 512) for b in (1, 16, 64)],
        "replay": [(512, b, 512) for b in (1, 4, 16)],
        "train": (512, 4, 512),
    },
}


def arms(phase):
    return ROLLOUT_ARMS if phase == "rollout" else ACTOR_ARMS


def shape_id(shape):
    prompt, batch, response = shape
    return f"p{prompt}-b{batch}-r{response}"


def synthetic_row(length, index):
    """Token IDs for row ``index`` of a ``length``-token prompt; equal for every batch size."""
    generator = torch.Generator().manual_seed(1_000_003 * length + index)
    return torch.randint(1000, 150000, (length,), generator=generator).tolist()


def prompts(shape):
    prompt, batch, _ = shape
    return [synthetic_row(prompt, index) for index in range(batch)]


def sequences(shape):
    """Prompt plus synthetic response tokens, for replay and training."""
    prompt, batch, response = shape
    return [
        synthetic_row(prompt, index) + synthetic_row(response, 7919 + index)
        for index in range(batch)
    ]


def response_positions(shape, rows=None):
    """Packed rows whose logits predict the response tokens, and those targets.

    ``rows`` defaults to every sequence of ``shape``; pass a data-parallel rank's share.
    """
    prompt = shape[0]
    positions, targets, start = [], [], 0
    for sequence in sequences(shape) if rows is None else rows:
        positions.extend(range(start + prompt - 1, start + len(sequence) - 1))
        targets.extend(sequence[prompt:])
        start += len(sequence)
    return positions, targets


# Actor layouts. The strict arm is certified at TP4 with the l2 run's EP (expert tensor
# parallel 4 // EP); its training is certified at EP1 only. Stock VIME cannot run TP4 (two
# KV heads; Megatron's attention output gate breaks when TP exceeds them), so it runs its
# recipe's TP2 with sequence parallel and expert tensor parallel 1, experts over all four
# ranks: data parallel 2. The strict profile is certified without sequence parallel.
STOCK_VIME_LAYOUT = {"tp": 2, "sp": True, "ep": 4, "etp": 1, "dp": 2}
STRICT_TRAINING_NEEDS_EP1 = (
    "the certified strict training profile is EP1; run train on an EP1 l2 run"
)


def actor_layout(run_ep, arm):
    """Parallel layout of a replay/train arm on the run's four GPUs."""
    if arm == STRICT_ARM:
        return {"tp": 4, "sp": False, "ep": run_ep, "etp": 4 // run_ep, "dp": 1}
    return dict(STOCK_VIME_LAYOUT)


def packed_lengths(lengths, multiple):
    """Packed sequence lengths plus one pad sequence up to ``multiple`` tokens.

    Sequence parallel splits the packed tokens over the TP ranks, so their count must divide
    by TP; VIME's ``get_batch`` pads the same way (pad token 0, one extra packed sequence).
    """
    pad = -sum(lengths) % multiple
    return [*lengths, pad] if pad else list(lengths)


def actor_unsupported(run_ep, phase, arm):
    """Why this replay/train arm cannot run on the run's topology, or None."""
    if phase == "train" and arm == STRICT_ARM and run_ep != 1:
        return STRICT_TRAINING_NEEDS_EP1
    return None


def data_parallel_share(items, rank, size):
    """A data-parallel rank's share of a batch, and whether it is only a filler copy.

    Items go round-robin. A rank left without any (a batch smaller than ``size``) gets a
    copy of the first item: every rank must still run the forward, because the MoE
    all-to-all spans the data-parallel ranks. Its result is discarded; the time is the
    slowest rank's, so the filler cannot make a step look faster.
    """
    share = list(items)[rank::size]
    return (share, False) if share else ([items[0]], True)


def arm_environment(base, arm):
    """Strict arms keep the certified stage environment; native arms run stock settings.

    ``base`` is ``qwen3_next_run.stage_environment`` of the certified l2 stage. A native arm
    drops the certificate and the strict runtime knobs, so vLLM and Triton use their defaults.
    """
    env = dict(base)
    if arm == STRICT_ARM:
        if env.get("VLLM_BATCH_INVARIANT") != "1" or not env.get("QWEN_SHARED_PROFILE_RESULT"):
            raise ValueError("The strict arm needs the certified l2 stage environment")
        return env
    for name in ("QWEN_SHARED_PROFILE_RESULT", "TRITON_F32_DEFAULT", "VLLM_USE_V2_MODEL_RUNNER"):
        env.pop(name, None)
    env["VLLM_BATCH_INVARIANT"] = "1" if arm == "native_bi" else "0"
    return env


def arm_order(phase, round_index):
    """Alternate the arm order between rounds so drift does not favour one arm."""
    order = list(arms(phase))
    return order if round_index % 2 == 0 else order[::-1]


def rollout_metrics(shape, prefill_seconds, total_seconds):
    prompt, batch, response = shape
    prefill = statistics.median(prefill_seconds)
    total = statistics.median(total_seconds)
    decode = total - prefill
    return {
        "prefill_s": prefill,
        "total_s": total,
        "prefill_tok_per_s": batch * prompt / prefill,
        "decode_tok_per_s": batch * (response - 1) / decode if decode > 0 else None,
        "output_tok_per_s": batch * response / total,
        "prefill_values_s": list(prefill_seconds),
        "total_values_s": list(total_seconds),
    }


def timing(values):
    return {
        "median_s": statistics.median(values),
        "min_s": min(values),
        "max_s": max(values),
        "values_s": list(values),
    }


def merge_rounds(records):
    """Pool one arm/workload's per-round shape records.

    Timing lists are concatenated. Boolean checks are kept per round. ``tokens_sha256`` (the
    greedy output of a rollout shape) is collected per round, so ``same_tokens_across_rounds``
    tells whether separate processes generated the same tokens.
    """
    pooled = {}
    for record in records:
        for shape, entry in record.items():
            target = pooled.setdefault(shape, {"checks": {}})
            for key, value in entry.items():
                if key.endswith("values_s"):
                    target.setdefault(key, []).extend(value)
                elif key == "tokens_sha256":
                    target.setdefault("tokens_sha256", []).append(value)
                elif isinstance(value, bool) or value is None:
                    target["checks"].setdefault(key, []).append(value)
    for target in pooled.values():
        digests = target.get("tokens_sha256")
        if digests is not None:
            target["checks"]["same_tokens_across_rounds"] = [len(set(digests)) == 1]
    return pooled


def summarize(phase, shape, values):
    if phase == "rollout":
        return rollout_metrics(shape, values["prefill_values_s"], values["total_values_s"])
    if phase == "train":
        return {
            "step": timing(values["step_values_s"]),
            "optimizer": timing(values["optimizer_values_s"]),
            "forward_backward_median_s": statistics.median(values["step_values_s"])
            - statistics.median(values["optimizer_values_s"]),
        }
    return {"forward_logprob": timing(values["values_s"])}


def headline(phase, summary):
    """The single number per shape the comparison table uses (seconds, lower is better)."""
    if phase == "rollout":
        return summary["total_s"]
    if phase == "train":
        return summary["step"]["median_s"]
    return summary["forward_logprob"]["median_s"]


def ratios(phase, results):
    """strict / native headline ratio for every native arm and shape both measured."""
    strict = results.get(STRICT_ARM, {})
    out = {}
    for arm, workloads in results.items():
        if arm == STRICT_ARM:
            continue
        for workload, shapes in workloads.items():
            for shape, summary in shapes.items():
                reference = strict.get(workload, {}).get(shape)
                if reference is None or "error" in summary or "error" in reference:
                    continue
                out.setdefault(arm, {}).setdefault(workload, {})[shape] = headline(
                    phase, reference
                ) / headline(phase, summary)
    return out
