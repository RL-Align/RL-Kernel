# SPDX-License-Identifier: Apache-2.0
"""Independent HF PPO math and optimizer control; never imports candidate PPO losses."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from examples.post_training.tp_ppo.fixed_contract import (
    CONFIG,
    OPTIMIZER,
    identity,
    load_packet,
    load_targets,
    save_report,
    sha,
    verify_checkpoint,
    write_tensor,
)


class HFModel(torch.nn.Module):
    def __init__(self, checkpoint, role, dtype):
        """Load an HF actor (causal LM) or critic (backbone plus seeded scalar head) on CUDA.

        Raises ``ValueError`` if the checkpoint has missing, unexpected or mismatched tensors
        (the critic's dropped ``lm_head.weight`` excepted) or a parameter is not ``dtype``.
        """
        super().__init__()
        from transformers import AutoModel, AutoModelForCausalLM

        cls = AutoModelForCausalLM if role == "actor" else AutoModel
        self.backbone, info = cls.from_pretrained(
            checkpoint,
            dtype=dtype,
            local_files_only=True,
            attn_implementation="eager",
            output_loading_info=True,
        )
        # AutoModel deliberately drops the pretrained LM projection for scalar scoring.
        unexpected = set(info.get("unexpected_keys", []))
        if role == "critic":
            unexpected.discard("lm_head.weight")
        if unexpected or any(
            info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")
        ):
            raise ValueError(f"incomplete HF {role} checkpoint: {info}")
        self.role = role
        if role == "critic":
            self.score = torch.nn.Linear(self.backbone.config.hidden_size, 1, dtype=dtype)
            with torch.no_grad():
                self.score.weight.copy_(
                    torch.randn(
                        self.score.weight.shape, generator=torch.Generator().manual_seed(144)
                    )
                    * 0.01
                )
                self.score.bias.zero_()
        self.backbone.gradient_checkpointing_enable()
        self.cuda().eval()
        if {p.dtype for p in self.parameters()} != {dtype}:
            raise ValueError("HF parameter dtype differs")

    def forward(self, packet):
        """Return FP32 next-token log-probs ``[B, T-1]`` (actor) or values ``[B, T]`` (critic).

        Critic values keep the model dtype (BF16 in the BF16 control).
        """
        out = self.backbone(
            **{k: packet[k] for k in ("input_ids", "attention_mask", "position_ids")},
            use_cache=False,
        )
        if self.role == "actor":
            return (
                out.logits[:, :-1]
                .float()
                .log_softmax(-1)
                .gather(-1, packet["input_ids"][:, 1:, None])
                .squeeze(-1)
            )
        return self.score(out.last_hidden_state).squeeze(-1)

    def canonical(self):
        """Return parameters by canonical name (critic: ``model.`` prefix plus ``score.*``)."""
        result = {
            (("model." if self.role == "critic" else "") + n): p
            for n, p in self.backbone.named_parameters()
        }
        if self.role == "critic":
            result.update({"score.weight": self.score.weight, "score.bias": self.score.bias})
        return result


def independent_targets(packet, values, reference):
    """Recompute PPO targets with explicit per-token GAE loops from critic ``values``.

    The reward is the value at each row's last attended position, added at the last action
    token; step rewards are ``-kl_coefficient * (behavior - reference)``. Bootstrap is zero
    for terminated rows. Returns old/bootstrap values, rewards, reference log-probs,
    normalized advantages and returns (from the unnormalized advantages); per-token entries
    are zero outside the action mask. Raises ``ValueError`` if a row has no action token.
    """
    mask = packet["action_mask"]
    if not mask.any(1).all():
        raise ValueError("every rollout row needs at least one action token")
    positions = torch.arange(values.shape[1], device=values.device).expand_as(values)
    last_state = positions.masked_fill(~packet["attention_mask"], -1).max(1).values
    rows = torch.arange(values.shape[0], device=values.device)
    rewards = values[rows, last_state]
    last = positions[:, :-1].masked_fill(~mask, -1).max(1).values
    bootstrap = values[rows, last + 1].masked_fill(packet["terminated"], 0)
    old = values[:, :-1].float().masked_fill(~mask, 0)
    reference = reference.float().masked_fill(~mask, 0)
    step_rewards = -CONFIG.kl_coefficient * (packet["behavior_logps"] - reference)
    step_rewards = step_rewards.masked_fill(~mask, 0)
    step_rewards[rows, last] += rewards
    adv = torch.zeros_like(old)
    for row in range(len(rows)):
        carry = torch.zeros((), device=values.device)
        next_value = bootstrap[row]
        for t in reversed(torch.where(mask[row])[0].tolist()):
            carry = (
                step_rewards[row, t]
                + CONFIG.gamma * next_value
                - old[row, t]
                + CONFIG.gamma * CONFIG.gae_lambda * carry
            )
            adv[row, t] = carry
            next_value = old[row, t]
    returns = adv + old
    valid = adv[mask]
    normalized = ((adv - valid.mean()) / (valid.std(unbiased=False) + 1e-8)).masked_fill(~mask, 0)
    return dict(
        old_values=old,
        bootstrap_values=bootstrap,
        rewards=rewards,
        reference_logps=reference,
        advantages=normalized,
        returns=returns,
    )


def independent_loss(role, scores, packet, targets):
    """Return the mean clipped PPO policy loss (actor) or clipped value loss (critic).

    Both means are taken over the action-mask tokens of ``packet``.
    """
    mask = packet["action_mask"]
    if role == "actor":
        ratio = torch.exp(scores[mask].float() - packet["behavior_logps"][mask])
        a = targets["advantages"][mask]
        return -torch.minimum(
            ratio * a, ratio.clamp(1 - CONFIG.policy_clip, 1 + CONFIG.policy_clip) * a
        ).mean()
    v = scores[:, :-1][mask].float()
    old, ret = targets["old_values"][mask], targets["returns"][mask]
    clipped = old + (v - old).clamp(-CONFIG.value_clip, CONFIG.value_clip)
    return 0.5 * torch.maximum((v - ret).square(), (clipped - ret).square()).mean()


def main():
    """Freeze HF FP32 PPO targets (``--score-only``) or run the HF reference PPO updates.

    ``--score-only`` always uses FP32 actor and critic models, ignoring role and precision.
    Training runs ``CONFIG.epochs`` AdamW steps, each accumulating per-row microbatch
    gradients into FP32 parameters (FP32 masters for BF16) and clipping to ``max_grad_norm``,
    and writes scores, loss, gradients, norms and parameter updates as evidence tensors.
    Once the provenance checks before the main ``try`` pass, ``report.json`` is written to
    ``--output`` on success and on failure.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--targets", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--role", choices=["actor", "critic"], default="actor")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--score-only", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    report = identity()
    report.update(
        role=a.role, precision=a.precision, rollout_manifest_sha256=sha(a.rollout / "manifest.json")
    )
    started = time.perf_counter()
    try:
        verify_checkpoint(a.checkpoint)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        packet = {k: v.cuda() for k, v in load_packet(a.rollout).items()}
        if a.score_only:
            with torch.no_grad():
                model = HFModel(a.checkpoint, "actor", torch.float32)
                reference = model(packet).detach()
                del model
                gc.collect()
                torch.cuda.empty_cache()
                model = HFModel(a.checkpoint, "critic", torch.float32)
                values = model(packet).detach()
                targets = independent_targets(packet, values, reference)
            torch.save({k: v.cpu() for k, v in targets.items()}, a.output / "targets.pt")
            (a.output / "targets.json").write_text(
                json.dumps(
                    dict(
                        sha256=sha(a.output / "targets.pt"),
                        rollout_manifest_sha256=report["rollout_manifest_sha256"],
                    ),
                    indent=2,
                )
                + "\n"
            )
            report["status"] = "frozen_targets_complete"
        else:
            if a.targets is None:
                raise ValueError("training requires frozen HF FP32 targets")
            targets = {k: v.cuda() for k, v in load_targets(a.targets).items()}
            report["targets_manifest_sha256"] = sha(a.targets / "targets.json")
            dtype = torch.float32 if a.precision == "fp32" else torch.bfloat16
            model = HFModel(a.checkpoint, a.role, dtype)
            named = model.canonical()
            # Independent FP32 master optimizer for both precision controls.
            masters = (
                {n: torch.nn.Parameter(p.detach().float().clone()) for n, p in named.items()}
                if dtype == torch.bfloat16
                else named
            )
            optimizer = torch.optim.AdamW(masters.values(), **OPTIMIZER, foreach=False, fused=False)
            artifacts = {}
            for update in range(CONFIG.epochs):
                before = {n: p.detach().float().cpu().clone() for n, p in named.items()}
                optimizer.zero_grad(set_to_none=True)
                for p0 in named.values():
                    p0.grad = None
                total_loss = 0.0
                all_scores = []
                for row in range(len(packet["input_ids"])):
                    micro = {
                        k: (
                            v[row : row + 1]
                            if k not in ("actor_version", "critic_version", "sampling")
                            else v
                        )
                        for k, v in packet.items()
                    }
                    t = {k: v[row : row + 1] for k, v in targets.items()}
                    scores = model(micro)
                    all_scores.append(scores.detach().cpu())
                    fraction = micro["action_mask"].sum() / packet["action_mask"].sum()
                    loss = independent_loss(a.role, scores, micro, t) * fraction
                    total_loss += float(loss.detach())
                    loss.backward()
                    if dtype == torch.bfloat16:
                        for n, p0 in named.items():
                            if p0.grad is None:
                                raise ValueError(f"missing HF gradient: {n}")
                            g = p0.grad.float()
                            if masters[n].grad is None:
                                masters[n].grad = g.clone()
                            else:
                                masters[n].grad.add_(g)
                            p0.grad = None
                    del loss, scores
                mask = (
                    packet["action_mask"].cpu()
                    if a.role == "actor"
                    else packet["attention_mask"].cpu()
                )
                artifacts[f"{update}/scores"] = write_tensor(
                    a.output, f"{update}/scores", torch.cat(all_scores)[mask]
                )
                artifacts[f"{update}/loss"] = write_tensor(
                    a.output, f"{update}/loss", torch.tensor(total_loss, dtype=torch.float64)
                )
                norm = torch.zeros((), device="cuda", dtype=torch.float64)
                for n, p0 in masters.items():
                    if p0.grad is None or not torch.isfinite(p0.grad).all():
                        raise ValueError(f"invalid HF gradient: {n}")
                    norm += p0.grad.double().square().sum()
                    artifacts[f"{update}/gradient/{n}"] = write_tensor(
                        a.output, f"{update}/gradient/{n}", p0.grad
                    )
                norm = norm.sqrt()
                artifacts[f"{update}/grad_norm"] = write_tensor(
                    a.output, f"{update}/grad_norm", norm
                )
                scale = (CONFIG.max_grad_norm / (norm + 1e-6)).clamp(max=1)
                for p0 in masters.values():
                    p0.grad.mul_(scale)
                artifacts[f"{update}/clipped_norm"] = write_tensor(
                    a.output,
                    f"{update}/clipped_norm",
                    sum(p0.grad.double().square().sum() for p0 in masters.values()).sqrt(),
                )
                optimizer.step()
                if dtype == torch.bfloat16:
                    with torch.no_grad():
                        for n, p0 in named.items():
                            p0.copy_(masters[n])
                for n, p0 in named.items():
                    artifacts[f"{update}/update/{n}"] = write_tensor(
                        a.output, f"{update}/update/{n}", p0.detach().float().cpu() - before[n]
                    )
                del before
            report.update(
                status="fixed_reference_complete",
                artifacts=artifacts,
                effective_action_tokens=int(packet["action_mask"].sum()),
            )
        report.update(
            seconds=time.perf_counter() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_report(a.output, report)


if __name__ == "__main__":
    main()
