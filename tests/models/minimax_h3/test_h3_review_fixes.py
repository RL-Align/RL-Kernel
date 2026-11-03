# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Regression tests for the review findings on the block-0 replay (one case per finding)."""

from __future__ import annotations

import hashlib

import pytest
import torch

from rl_engine.validation.models import h3_block, h3_cases


class TestGoldenBackwardIsStrictFp32:
    def test_backward_runs_with_tf32_disabled_even_when_the_caller_enabled_it(self, monkeypatch):
        seen = {}

        def fake_graph(mode, ctx, ops=None):
            hidden = ctx["hidden"]
            return {node.name: hidden * 1.0 for node in h3_block.NODES}

        def spy_backward(tensors, grads=None, **kwargs):
            seen["matmul_tf32"] = torch.backends.cuda.matmul.allow_tf32
            seen["cudnn_tf32"] = torch.backends.cudnn.allow_tf32

        weights = {f"{h3_block.BLOCK}.{name}": torch.zeros(2) for name in h3_block.PARAM_LEAVES}
        ctx = {"weights": weights, "hidden": torch.ones(1, 3, 2), "temb": torch.ones(2)}
        monkeypatch.setattr(h3_block, "run_graph", fake_graph)
        monkeypatch.setattr(torch.autograd, "backward", spy_backward)
        saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            h3_block._backward("golden", ctx, None, torch.ones(1, 3, 2))
            assert torch.backends.cuda.matmul.allow_tf32 is True  # restored for the caller
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved
        assert seen == {"matmul_tf32": False, "cudnn_tf32": False}


class TestBlockLayout:
    @pytest.mark.parametrize("num_timesteps", [0, -1, 9])
    def test_a_timestep_count_the_layout_cannot_honour_is_rejected(self, num_timesteps):
        with pytest.raises(ValueError, match="num_timesteps"):
            h3_cases.h3_block_layout(8, num_timesteps, device="cpu")

    @pytest.mark.parametrize(("seq_len", "num_timesteps"), [(8, 1), (8, 8), (261, 3), (1, 1)])
    def test_every_requested_timestep_owns_a_run(self, seq_len, num_timesteps):
        timesteps, tags = h3_cases.h3_block_layout(seq_len, num_timesteps, device="cpu")
        assert timesteps.unique().numel() == num_timesteps
        assert tags.shape == timesteps.shape == (seq_len,)


class TestEvidenceDigest:
    def test_file_digest_is_the_sha256_of_the_bytes(self, tmp_path):
        from tools.validation.models.h3_evidence_integrity import digest

        path = tmp_path / "f.bin"
        payload = bytes(range(256)) * 40000  # larger than one read block
        path.write_bytes(payload)
        assert digest(path) == hashlib.sha256(payload).hexdigest()

    def test_the_block_diagnostic_needs_nothing_from_the_full_model_modules(self):
        import importlib
        import sys

        blocked = {
            "tools.validation.models.h3_full_model": None,
            "tools.validation.models.h3_full_generation": None,
        }
        saved = {k: sys.modules.get(k) for k in blocked}
        sys.modules.update(blocked)  # None makes the import raise ImportError
        sys.modules.pop("tools.validation.models.h3_block_diagnostic", None)
        sys.modules.pop("tools.validation.models.h3_evidence_integrity", None)
        try:
            importlib.import_module("tools.validation.models.h3_evidence_integrity")
            try:
                importlib.import_module("tools.validation.models.h3_block_diagnostic")
            except ModuleNotFoundError as exc:  # only the compiled extension may be missing
                assert "h3_full" not in str(exc), exc
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
            sys.modules.pop("tools.validation.models.h3_block_diagnostic", None)


class TestBlockContextDevice:
    def test_every_tensor_of_the_context_follows_the_weights(self, h3_weights_cpu):
        ctx = h3_block.make_block_context(h3_weights_cpu, layout="tiny", seed=0)
        assert ctx["hidden"].device.type == "cpu"
        assert ctx["temb"].device.type == "cpu"
