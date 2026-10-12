# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Frozen inputs for real VIME policy-loss/optimizer recovery acceptance."""

import hashlib
import json
from dataclasses import dataclass

import torch

from rl_engine.validation.common.tensor_identity import assert_tensor_bitwise_equal

POLICY_SETTINGS = {
    "loss_type": "policy_loss",
    "advantage_estimator": "grpo",
    "use_rollout_logprobs": True,
    "disable_compute_advantages_and_returns": True,
    "normalize_advantages": False,
    "eps_clip": 0.2,
    "eps_clip_high": 0.2,
    "eps_clip_c": None,
    "entropy_coef": 0.0,
    "kl_coef": 0.0,
    "kl_loss_coef": 0.0,
    "opd_kl_coef": 0.0,
    "use_kl_loss": False,
    "use_tis": False,
    "use_opsm": False,
    "get_mismatch_metrics": False,
    "custom_pg_loss_reducer_function_path": None,
    "global_batch_size": 4,
    "micro_batch_size": 1,
    "use_dynamic_batch_size": False,
    "rollout_temperature": 1.0,
    "rollout_top_p": 1.0,
    "rollout_top_k": -1,
    "seed": 1234,
}
OPTIMIZER_SETTINGS = {
    "optimizer": "adam",
    "lr": 1e-6,
    "lr_decay_style": "constant",
    "weight_decay": 0.1,
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "adam_eps": 1e-8,
    "clip_grad": 1.0,
    "optimizer_cpu_offload": True,
    "optimizer_offload_fraction": 1.0,
    "overlap_cpu_optimizer_d2h_h2d": False,
    "use_precision_aware_optimizer": True,
    "use_distributed_optimizer": True,
    "use_stateless_adam": False,
    "no_save_optim": False,
    "no_save_rng": False,
}


def _tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PolicyFixture:
    tokens: torch.Tensor
    old_logprobs: torch.Tensor
    baseline_identity: str
    content_sha256: str

    @classmethod
    def freeze(cls, tokens, old_logprobs, *, baseline_identity):
        if tokens.shape != (4, 80) or tokens.dtype != torch.int64:
            raise ValueError("Policy fixture requires four int64 prompt64+response16 samples")
        if old_logprobs.shape != (4, 16) or old_logprobs.dtype != torch.float32:
            raise ValueError("Baseline selected logprobs must be FP32 [4, 16]")
        if not torch.isfinite(old_logprobs).all() or old_logprobs.requires_grad:
            raise ValueError("Baseline logprobs must be finite, frozen rollout values")
        if not baseline_identity or not isinstance(baseline_identity, str):
            raise ValueError("A verified baseline run/content identity is required")
        if bool(((tokens < 0) | (tokens >= 151936)).any()):
            raise ValueError("Token ID is outside the official Qwen3-Next vocabulary")
        if torch.unique(tokens, dim=0).shape[0] != 4:
            raise ValueError("The four policy samples must be distinct")
        tokens = tokens.detach().cpu().clone()
        old_logprobs = old_logprobs.detach().cpu().clone()
        digest = cls._identity(tokens, old_logprobs, baseline_identity)
        return cls(tokens, old_logprobs, baseline_identity, digest)

    @staticmethod
    def _identity(tokens, old_logprobs, baseline_identity):
        payload = {
            "tokens": _tensor_digest(tokens),
            "old_logprobs": _tensor_digest(old_logprobs),
            "baseline": baseline_identity,
            "advantages": [1.0, -0.5, 0.75, -0.25],
            "response_mask": [1] * 16,
            "sample_order": [[0], [1], [2], [3]],
            "policy": POLICY_SETTINGS,
            "optimizer": OPTIMIZER_SETTINGS,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def validate(self):
        if (
            self._identity(self.tokens, self.old_logprobs, self.baseline_identity)
            != self.content_sha256
        ):
            raise ValueError("Frozen policy fixture was modified")

    def verify_initial_recompute(self, logprobs):
        self.validate()
        assert_tensor_bitwise_equal(
            logprobs.cpu(), self.old_logprobs, name="baseline policy logprobs"
        )

    def batch(self, sample_ids, *, device):
        """Produce native VIME loss fields; no rollout hidden/state tensors are used."""
        self.validate()
        if not sample_ids or any(type(i) is not int or i not in range(4) for i in sample_ids):
            raise ValueError("Policy sample IDs must be nonempty integers in [0, 4)")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("A microbatch cannot duplicate policy samples")
        advantages = [1.0, -0.5, 0.75, -0.25]
        return {
            "unconcat_tokens": [self.tokens[i].to(device=device) for i in sample_ids],
            "rollout_log_probs": [self.old_logprobs[i].to(device=device) for i in sample_ids],
            "advantages": [
                torch.full((16,), advantages[i], dtype=torch.float32, device=device)
                for i in sample_ids
            ],
            "loss_masks": [torch.ones(16, dtype=torch.int32, device=device) for _ in sample_ids],
            "rollout_mask_sums": torch.full(
                (len(sample_ids),), 16, dtype=torch.float32, device=device
            ),
            "total_lengths": [80] * len(sample_ids),
            "response_lengths": [16] * len(sample_ids),
            "partition": list(sample_ids),
        }


def require_policy_configuration(args):
    """Reject drift rather than changing an already constructed training configuration."""
    for name, expected in {**POLICY_SETTINGS, **OPTIMIZER_SETTINGS}.items():
        if not hasattr(args, name) or getattr(args, name) != expected:
            raise ValueError(f"Policy acceptance requires {name}={expected!r}")
