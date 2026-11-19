# SPDX-License-Identifier: Apache-2.0
"""Independent HF PPO math and optimizer control; never imports candidate PPO losses."""
from __future__ import annotations

import argparse
import gc
import json
import time
import types
from pathlib import Path

import torch
import torch.utils.checkpoint

from tools.validation.models.qwen3_ppo.fixed_contract import (
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

CONTROL_CHUNKS = 3
IMPLEMENTATION_CONTROL = dict(
    attention="sdpa",
    linear=(
        f"tokens, output features and input features each split into {CONTROL_CHUNKS} chunks; "
        "input-feature partials summed in reverse order in FP32"
    ),
    log_softmax=f"logsumexp merged over {CONTROL_CHUNKS} vocabulary chunks",
)


def reordered_linear(x, weight, bias):
    """Return ``F.linear(x, weight, bias)`` computed with a different reduction order.

    Tokens, output features and input features are each split into ``CONTROL_CHUNKS``
    pieces. Input-feature partial products are computed in FP32 and summed in reverse order,
    so the forward sum, the input-gradient sum (over output-feature chunks) and the
    weight-gradient sum (over token chunks) all differ in order from one GEMM. The result is
    cast back to ``x.dtype``, as a BF16 GEMM with FP32 accumulation would be.
    """
    tokens = x.reshape(-1, x.shape[-1])
    rows = []
    for t in tokens.tensor_split(CONTROL_CHUNKS, 0):
        t_chunks = t.float().tensor_split(CONTROL_CHUNKS, 1)
        columns = []
        for w in weight.tensor_split(CONTROL_CHUNKS, 0):
            w_chunks = w.float().tensor_split(CONTROL_CHUNKS, 1)
            acc = None
            for k in reversed(range(CONTROL_CHUNKS)):
                part = t_chunks[k] @ w_chunks[k].T
                acc = part if acc is None else acc + part
            columns.append(acc)
        rows.append(torch.cat(columns, 1))
    y = torch.cat(rows, 0)
    if bias is not None:
        y = y + bias.float()
    return y.to(x.dtype).reshape(*x.shape[:-1], weight.shape[0])


def recomputed_reordered_linear(x, weight, bias):
    """Return :func:`reordered_linear` without keeping its FP32 chunk copies for backward.

    Autograd keeps only the inputs and recomputes the same graph in backward, so gradients
    are bitwise those of :func:`reordered_linear`. Keeping the FP32 copies of every BF16
    weight (about 33 GB for Qwen3-8B) would not fit next to the optimizer state.
    """
    return torch.utils.checkpoint.checkpoint(reordered_linear, x, weight, bias, use_reentrant=False)


def chunked_log_softmax_gather(logits, ids):
    """Return ``logits.log_softmax(-1).gather(-1, ids)`` with a chunk-merged logsumexp."""
    lse = torch.stack(
        [c.logsumexp(-1) for c in logits.tensor_split(CONTROL_CHUNKS, -1)], -1
    ).logsumexp(-1)
    return logits.gather(-1, ids).squeeze(-1) - lse


def use_reordered_linear(module):
    """Make every ``nn.Linear`` under ``module`` run :func:`recomputed_reordered_linear`."""
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            m.forward = types.MethodType(
                lambda self, x: recomputed_reordered_linear(x, self.weight, self.bias), m
            )


class HFModel(torch.nn.Module):
    def __init__(self, checkpoint, role, dtype, implementation_control=False):
        """Load an HF actor (causal LM) or critic (backbone plus seeded scalar head) on CUDA.

        With ``implementation_control``, the model uses ``sdpa`` attention instead of
        ``eager``, :func:`reordered_linear` for every linear layer and a chunked log-softmax:
        the same math in a different reduction order, used to measure implementation spread.
        Raises ``ValueError`` if the checkpoint has missing, unexpected or mismatched tensors
        (the critic's dropped ``lm_head.weight`` excepted) or a parameter is not ``dtype``.
        """
        super().__init__()
        from transformers import AutoModel, AutoModelForCausalLM

        cls = AutoModelForCausalLM if role == "actor" else AutoModel
        self.implementation_control = implementation_control
        self.backbone, info = cls.from_pretrained(
            checkpoint,
            dtype=dtype,
            local_files_only=True,
            attn_implementation="sdpa" if implementation_control else "eager",
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
        if implementation_control:
            use_reordered_linear(self)
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
        if self.role == "actor" and self.implementation_control:
            return chunked_log_softmax_gather(
                out.logits[:, :-1].float(), packet["input_ids"][:, 1:, None]
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


class ReferenceAdamW:
    """AdamW over FP32 parameters, with FP32 masters for BF16 models.

    The update is ``torch.optim.AdamW(..., foreach=False, fused=False)`` applied on the GPU one
    parameter at a time, the per-tensor path a single resident optimizer takes, so where the
    state lives does not change any value. With ``state_device="cpu"`` the moments and the
    BF16 masters stay in host memory between steps and only one parameter's state is on the
    GPU at a time: the GPU then holds the model and its FP32 gradients, which lets Qwen3-8B
    train on an 80-94 GB GPU. ``state_device="cuda"`` keeps everything resident.
    """

    def __init__(self, named, dtype, state_device):
        """Track ``named`` model parameters; BF16 models get FP32 masters on ``state_device``."""
        self.named = named
        self.state_device = torch.device(state_device)
        self.masters = (
            {n: p.detach().float().to(self.state_device, copy=True) for n, p in named.items()}
            if dtype == torch.bfloat16
            else None
        )
        self.master_grads = {}
        self.state = {}

    def zero_grad(self):
        """Drop the model gradients and the accumulated FP32 master gradients."""
        self.master_grads.clear()
        for p in self.named.values():
            p.grad = None

    def accumulate(self):
        """Add BF16 model gradients into FP32 master gradients after each backward.

        FP32 models accumulate in ``.grad`` directly. Raises ``ValueError`` if a BF16
        parameter has no gradient.
        """
        if self.masters is None:
            return
        for n, p in self.named.items():
            if p.grad is None:
                raise ValueError(f"missing HF gradient: {n}")
            g = p.grad.float()
            if n in self.master_grads:
                self.master_grads[n].add_(g)
            else:
                self.master_grads[n] = g.clone()
            p.grad = None

    def gradients(self):
        """Return the FP32 gradient the optimizer will apply, by canonical name."""
        if self.masters is None:
            return {n: p.grad for n, p in self.named.items()}
        return {n: self.master_grads.get(n) for n in self.named}

    def step(self):
        """Apply one AdamW step to every parameter, then copy BF16 masters into the model."""
        for n, p in self.named.items():
            if self.masters is None:
                target = p
            else:
                target = torch.nn.Parameter(self.masters[n].cuda())
                target.grad = self.master_grads[n]
            optimizer = torch.optim.AdamW([target], **OPTIMIZER, foreach=False, fused=False)
            if n in self.state:
                optimizer.state[target] = {
                    k: v.cuda() if k != "step" else v for k, v in self.state[n].items()
                }
            optimizer.step()
            self.state[n] = {
                k: v.to(self.state_device) if k != "step" else v
                for k, v in optimizer.state[target].items()
            }
            if self.masters is not None:
                with torch.no_grad():
                    self.masters[n].copy_(target)
                    p.copy_(target)
                target.grad = None
            del optimizer, target


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
    ``--implementation-control`` trains the same update with :data:`IMPLEMENTATION_CONTROL`
    (a different reduction order) and records it in the report.
    Training runs ``CONFIG.epochs`` AdamW steps, each accumulating per-row microbatch
    gradients into FP32 parameters (FP32 masters for BF16) and clipping to ``max_grad_norm``,
    and writes scores, loss, gradients, norms and parameter updates as evidence tensors.
    ``--optimizer-state`` (default ``cpu``) says where :class:`ReferenceAdamW` keeps the
    moments and BF16 masters between steps; the values are the same either way.
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
    p.add_argument("--implementation-control", action="store_true")
    p.add_argument("--optimizer-state", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    if a.score_only and a.implementation_control:
        raise ValueError("frozen targets always use the reference implementation")
    a.output.mkdir(parents=True, exist_ok=False)
    report = identity()
    report.update(
        role=a.role,
        precision=a.precision,
        rollout_manifest_sha256=sha(a.rollout / "manifest.json"),
        implementation_control=IMPLEMENTATION_CONTROL if a.implementation_control else None,
    )
    if not a.score_only:
        report["optimizer_state"] = a.optimizer_state
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
            model = HFModel(a.checkpoint, a.role, dtype, a.implementation_control)
            named = model.canonical()
            # Independent FP32 master optimizer for both precision controls.
            optimizer = ReferenceAdamW(named, dtype, a.optimizer_state)
            artifacts = {}
            for update in range(CONFIG.epochs):
                before = {n: p.detach().float().cpu().clone() for n, p in named.items()}
                optimizer.zero_grad()
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
                    optimizer.accumulate()
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
                gradients = optimizer.gradients()
                for n, g in gradients.items():
                    if g is None or not torch.isfinite(g).all():
                        raise ValueError(f"invalid HF gradient: {n}")
                    norm += g.double().square().sum()
                    artifacts[f"{update}/gradient/{n}"] = write_tensor(
                        a.output, f"{update}/gradient/{n}", g
                    )
                norm = norm.sqrt()
                artifacts[f"{update}/grad_norm"] = write_tensor(
                    a.output, f"{update}/grad_norm", norm
                )
                scale = (CONFIG.max_grad_norm / (norm + 1e-6)).clamp(max=1)
                for g in gradients.values():
                    g.mul_(scale)
                artifacts[f"{update}/clipped_norm"] = write_tensor(
                    a.output,
                    f"{update}/clipped_norm",
                    sum(g.double().square().sum() for g in gradients.values()).sqrt(),
                )
                del gradients
                optimizer.step()
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
