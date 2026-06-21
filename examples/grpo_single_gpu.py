# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Minimal single-device GRPO training example for RL-Kernel.

This script intentionally uses a tiny synthetic batch and a toy policy so it can
run on one GPU without external model downloads, rollout services, or launchers.
It also supports ``--device cpu`` for smoke tests.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.kernels.registry import resolve_logp_op_type  # noqa: E402
from rl_engine.testing import (  # noqa: E402
    active_token_count,
    compute_policy_ratio,
    compute_reference_kl,
    make_synthetic_rl_kernel_batch,
    masked_mean,
    selected_logprobs_reference,
    summarize_kernel_drift,
)


@dataclass(frozen=True)
class StepMetrics:
    step: int
    loss: float
    policy_loss: float
    kl: float
    active_tokens: int
    logp_backend: str
    train_logp_source: str
    kernel_max_abs_error: float


class TinyPolicy(torch.nn.Module):
    """Small trainable policy that predicts completion tokens."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.proj = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embedding(token_ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--completion-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--valid-density", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--logp-backend",
        default="auto",
        help=(
            "Selected-token logp backend. Use 'deterministic' or "
            "'batch_invariant' for the CUDA deterministic path."
        ),
    )
    parser.add_argument(
        "--require-batch-invariant-logp",
        action="store_true",
        help="Require a batch-invariant deterministic logp backend.",
    )
    parser.add_argument(
        "--require-fused-logp",
        action="store_true",
        help="Fail if CUDA dispatch falls back instead of using RL-Kernel's fused logp backend.",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(requested)


def resolve_logp_op(
    device: torch.device,
    logp_backend: str,
    require_batch_invariant_logp: bool,
) -> Any:
    op_type = resolve_logp_op_type(
        logp_backend,
        require_batch_invariant=require_batch_invariant_logp,
    )
    if device.type == "cpu":
        from rl_engine.kernels.ops.pytorch.loss.logp import NativeLogpOp

        return NativeLogpOp()

    from rl_engine.kernels.registry import kernel_registry

    return kernel_registry.get_op(op_type)


def is_fused_logp_backend(logp_op: Any) -> bool:
    return bool(getattr(logp_op, "is_fused_logp", False))


def make_group_advantages(
    batch_size: int,
    completion_len: int,
    samples_per_prompt: int,
    completion_mask: torch.Tensor,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size % samples_per_prompt != 0:
        raise ValueError("batch_size must be divisible by samples_per_prompt")
    if samples_per_prompt < 2:
        raise ValueError("samples_per_prompt must be at least 2 for group normalization")

    token_rewards = (token_ids.float() % 7.0) / 6.0
    masked_rewards = token_rewards.masked_fill(~completion_mask, 0.0)
    denom = completion_mask.sum(dim=1).clamp_min(1).float()
    rewards = masked_rewards.sum(dim=1) / denom

    grouped = rewards.view(-1, samples_per_prompt)
    group_mean = grouped.mean(dim=1, keepdim=True)
    group_std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    sample_advantages = ((grouped - group_mean) / group_std).reshape(batch_size)
    advantages = sample_advantages[:, None].expand(batch_size, completion_len).clone()
    advantages = advantages.masked_fill(~completion_mask, 0.0)
    return rewards, advantages


def selected_logps_with_op(
    logp_op: Any,
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if hasattr(logp_op, "apply_fp32"):
        selected = logp_op.apply_fp32(logits, token_ids)
    else:
        selected = logp_op(logits, token_ids).float()
    return selected.masked_fill(~mask, 0.0)


def grpo_loss(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    clip_eps: float,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ratio = compute_policy_ratio(current_logps, old_logps, completion_mask)
    unclipped = ratio * advantages.float()
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages.float()
    policy_loss_terms = -torch.minimum(unclipped, clipped)
    kl_terms = compute_reference_kl(current_logps, ref_logps, completion_mask)
    policy_loss = masked_mean(policy_loss_terms, completion_mask)
    kl = masked_mean(kl_terms, completion_mask)
    return policy_loss + beta * kl, policy_loss, kl


def run_training(args: argparse.Namespace) -> list[StepMetrics]:
    if args.steps <= 0:
        raise ValueError("--steps must be greater than zero")

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    logp_op = resolve_logp_op(
        device,
        args.logp_backend,
        args.require_batch_invariant_logp,
    )
    backend_name = logp_op.__class__.__name__
    if args.require_fused_logp and not is_fused_logp_backend(logp_op):
        raise RuntimeError(
            "--require-fused-logp was set, but kernel dispatch selected "
            f"{backend_name}. Build the CUDA extension with `pip install -e .` "
            "before running the strict fused-logp path."
        )

    batch = make_synthetic_rl_kernel_batch(
        num_prompts=args.num_prompts,
        samples_per_prompt=args.samples_per_prompt,
        prompt_len=args.prompt_len,
        completion_len=args.completion_len,
        vocab_size=args.vocab_size,
        valid_density=args.valid_density,
        dtype=torch.float32,
        device=device,
        seed=args.seed,
    )
    rewards, advantages = make_group_advantages(
        batch.batch_size,
        batch.completion_len,
        args.samples_per_prompt,
        batch.completion_mask,
        batch.token_ids,
    )

    policy = TinyPolicy(args.vocab_size, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    with torch.no_grad():
        initial_logits = policy(batch.token_ids)
        old_logps = selected_logprobs_reference(
            initial_logits,
            batch.token_ids,
            mask=batch.completion_mask,
            output_dtype=torch.float32,
        )
        ref_logps = old_logps.detach().clone()

    metrics: list[StepMetrics] = []
    print(
        "starting grpo_single_gpu "
        f"device={device.type} backend={backend_name} "
        f"batch={batch.batch_size}x{batch.completion_len} "
        f"active_tokens={int(active_token_count(batch.completion_mask).item())}"
    )
    print(
        "reward_stats "
        f"mean={rewards.mean().item():.6f} min={rewards.min().item():.6f} "
        f"max={rewards.max().item():.6f}"
    )

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = policy(batch.token_ids)
        reference_logps = selected_logprobs_reference(
            logits,
            batch.token_ids,
            mask=batch.completion_mask,
            output_dtype=torch.float32,
        )
        kernel_logps = selected_logps_with_op(
            logp_op,
            logits,
            batch.token_ids,
            batch.completion_mask,
        )
        drift = summarize_kernel_drift(
            kernel_logps.detach(),
            reference_logps.detach(),
            batch.completion_mask,
        )

        if kernel_logps.requires_grad:
            train_logps = kernel_logps
            train_source = backend_name
        else:
            train_logps = reference_logps
            train_source = "autograd_reference"

        loss, policy_loss, kl = grpo_loss(
            train_logps,
            old_logps,
            ref_logps,
            advantages,
            batch.completion_mask,
            args.clip_eps,
            args.beta,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

        loss.backward()
        optimizer.step()

        step_metrics = StepMetrics(
            step=step,
            loss=float(loss.detach().cpu().item()),
            policy_loss=float(policy_loss.detach().cpu().item()),
            kl=float(kl.detach().cpu().item()),
            active_tokens=int(active_token_count(batch.completion_mask).item()),
            logp_backend=backend_name,
            train_logp_source=train_source,
            kernel_max_abs_error=float(drift["max_abs_error"]),
        )
        metrics.append(step_metrics)
        print(
            f"step={step_metrics.step} loss={step_metrics.loss:.6f} "
            f"policy_loss={step_metrics.policy_loss:.6f} kl={step_metrics.kl:.6f} "
            f"train_logp_source={step_metrics.train_logp_source} "
            f"kernel_max_abs_error={step_metrics.kernel_max_abs_error:.6e}"
        )

    print(
        f"completed grpo_single_gpu steps={args.steps} "
        f"device={device.type} backend={backend_name}"
    )
    return metrics


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
