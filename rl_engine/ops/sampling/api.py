# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch
import torch.nn as nn

from rl_engine.platforms.constants import constants
from rl_engine.utils.logger import logger


class SamplerBackend(nn.Module):
    def __init__(self):
        super().__init__()
        self.backend = self._detect_backend()
        self._init_backend_assets()

    def _detect_backend(self):
        if torch.version.hip:
            logger.info_once("Detected AMD GPU (ROCm) - Using AITER backend")
            return constants.BackendLib.AITER.value
        else:
            logger.info_once("Detected NVIDIA GPU (CUDA) - Using FlashInfer backend")
            return constants.BackendLib.FLASHINFER.value

    def _init_backend_assets(self):
        """
        Preload backend-specific dependencies
        to avoid runtime import overhead.
        """
        if self.backend == constants.BackendLib.FLASHINFER.value:
            try:
                import flashinfer

                self.flashinfer = flashinfer
                logger.info("FlashInfer kernels loaded successfully.")
            except ImportError:
                logger.error("FlashInfer not found. Please install it for NVIDIA GPUs.")
        elif self.backend == constants.BackendLib.AITER.value:
            pass

    @torch.inference_mode()
    def sample(self, logits, top_k=None, top_p=None, temperature=1.0, deterministic=True):
        """
        Unified sampling interface.
        """
        logits = logits.contiguous()
        if temperature != 1.0:
            logits = logits / temperature

        if self.backend == constants.BackendLib.FLASHINFER.value:
            from flashinfer.sampling import top_k_renorm_probs, top_p_sampling_from_probs

            logits = logits.float().contiguous()
            probs = torch.softmax(logits, dim=-1)

            if top_k is None and top_p is None:
                return torch.multinomial(probs, num_samples=1).view(-1)

            if top_k is not None:
                probs = top_k_renorm_probs(probs, top_k)

            if top_p is not None:
                return top_p_sampling_from_probs(probs, top_p, deterministic=deterministic)

            return torch.multinomial(probs, num_samples=1).view(-1)

        elif self.backend == constants.BackendLib.AITER.value:
            # TODO: Connect to AITER's sampling operator
            # return aiter.ops.sample(logits, ...)
            pass

        # Fallback to native PyTorch sampling
        if top_k is not None:
            topk_values, _ = torch.topk(logits, top_k)
            min_topk = topk_values[..., -1, None]
            logits = torch.where(logits < min_topk, torch.full_like(logits, float("-inf")), logits)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.inference_mode()
    def compute_logp(self, logits, token_ids):
        """
        Pre-allocated block logic for computing log probabilities of selected tokens.
        """
        batch_size = logits.shape[0]
        out_logp = torch.empty(
            logits.shape[0], logits.shape[1], device=logits.device, dtype=logits.dtype
        )

        chunk_size = 4096  # Tune this based on GPU memory and performance characteristics

        for i in range(0, batch_size, chunk_size):
            c_logits = logits[i : i + chunk_size]
            c_token_ids = token_ids[i : i + chunk_size]

            out_logp[i : i + chunk_size] = (
                torch.log_softmax(c_logits, dim=-1)
                .gather(-1, c_token_ids.unsqueeze(-1))
                .squeeze(-1)
            )

        return out_logp.view(batch_size, -1)
