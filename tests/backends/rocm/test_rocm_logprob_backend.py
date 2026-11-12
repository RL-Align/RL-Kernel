# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from rl_engine.backends.rocm.logprob.vocab_parallel_logp import RocmVocabParallelLogprobOp
from rl_engine.contracts.operators.logprob import (
    LogprobContract,
    MaskSpec,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.reference.logprob.vocab_parallel_logp import VocabParallelLogprobOp
from rl_engine.runtime.registry import KernelRegistry, OpBackend


def test_rocm_backend_preserves_ws2_operator_surface():
    assert issubclass(RocmVocabParallelLogprobOp, VocabParallelLogprobOp)
    assert RocmVocabParallelLogprobOp.op_class == "logprob"
    assert RocmVocabParallelLogprobOp.is_batch_invariant


def test_rocm_backend_is_gated_by_native_extension(monkeypatch):
    registry = KernelRegistry()
    registry._platform = lambda: "rocm"
    candidates = registry._logprob_candidates["rocm"]
    if OpBackend.ROCM_VOCAB_PARALLEL_LOGP in candidates:
        assert candidates[0] is OpBackend.ROCM_VOCAB_PARALLEL_LOGP
        capability = registry._logprob_capabilities["rocm"][OpBackend.ROCM_VOCAB_PARALLEL_LOGP]
        assert capability.backend_id == "rocm-vocab-parallel-logp-ws2"
        assert capability.implementation_kind == "production"
    else:
        assert candidates[0] in {
            OpBackend.TRITON_VOCAB_PARALLEL_LOGP,
            OpBackend.PYTORCH_VOCAB_PARALLEL_LOGP,
        }


def test_rocm_native_tile_kernel_is_hip_guarded_and_registered():
    source = (Path(__file__).resolve().parents[3] / "csrc/bindings/ops.cpp").read_text(
        encoding="utf-8"
    )
    kernel = (
        Path(__file__).resolve().parents[3] / "csrc/cuda/logprob/deterministic_logp_kernel.cu"
    ).read_text(encoding="utf-8")
    assert "deterministic_logp_tile_stats" in source
    assert "deterministic_logp_tile_stats_kernel" in kernel
    assert "atomic" not in kernel.lower()
    hip_kernel = (
        Path(__file__).resolve().parents[3] / "csrc/rocm/logprob/hip_deterministic_logp_kernel.hip"
    ).read_text(encoding="utf-8")
    assert "hip_deterministic_logp_tile_stats" in hip_kernel
    assert "hip_deterministic_logp_backward" in hip_kernel
    assert "atomic" not in hip_kernel.lower()
    assert "hip_deterministic_logp_backward" in source
    assert "__HIPCC__" in source
    assert "deterministic_collective_all_gather" in source


def test_rocm_backend_import_does_not_require_native_extension():
    # Importing the wrapper must remain possible in CPU-only CI; capability
    # loading, not module import, decides whether the native fast path exists.
    op = RocmVocabParallelLogprobOp()
    assert isinstance(op, VocabParallelLogprobOp)


def test_explicit_native_backend_fails_closed_when_extension_is_missing(monkeypatch):
    import rl_engine.runtime.registry as registry_module

    monkeypatch.setattr(registry_module, "_rocm_vocab_logprob_native_available", lambda: False)
    registry = KernelRegistry()
    registry._platform = lambda: "rocm"
    contract = LogprobContract(
        role="train",
        dtype="fp32",
        mask=MaskSpec(num_tokens=1, active_mask=(True,)),
        sharding=ShardingSpec(
            tp_rank=0,
            tp_world_size=1,
            vocab_shard_bounds=((0, 8),),
            real_vocab_size=8,
            padded_vocab_size=8,
        ),
        reduction=ReductionSpec(),
    )

    with pytest.raises(RuntimeError, match="rocm-vocab-parallel-logp-ws2"):
        registry.get_logprob_op(
            contract,
            requested_backend="rocm-vocab-parallel-logp-ws2",
        )


# --------------------------------------------------------------------------- GPU cases


def _native_rocm_available() -> bool:
    import torch

    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    from rl_engine.runtime.registry import _rocm_vocab_logprob_native_available

    return _rocm_vocab_logprob_native_available()


def _triton_available() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _kernel_backends() -> list:
    """(id, op factory) for every fused kernel backend usable on this machine."""

    backends = []
    if _native_rocm_available():
        backends.append(pytest.param(RocmVocabParallelLogprobOp, id="rocm-hip"))
    if _triton_available():
        from rl_engine.backends.shared.triton.logprob.vocab_parallel_logp import (
            TritonVocabParallelLogprobOp,
        )

        backends.append(pytest.param(TritonVocabParallelLogprobOp, id="triton"))
    return backends


@pytest.mark.skipif(not _kernel_backends(), reason="requires a fused WS2 kernel backend")
@pytest.mark.parametrize("op_class", _kernel_backends())
class TestFusedKernelPath:
    """Every fused kernel backend must agree with the reference op on the same contract."""

    @staticmethod
    def _case(real_vocab, padded_vocab, num_tokens, dtype, *, seed=3):
        import torch

        device = torch.device("cuda", 0)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        logits = (torch.randn(num_tokens, padded_vocab, generator=gen) * 3).to(device, dtype)
        targets = torch.randint(0, real_vocab, (num_tokens,), generator=gen).to(device)
        active = tuple(i % 4 != 2 for i in range(num_tokens))
        contract = LogprobContract(
            role="train",
            dtype={torch.bfloat16: "bf16", torch.float32: "fp32", torch.float16: "fp16"}[dtype],
            mask=MaskSpec(num_tokens=num_tokens, active_mask=active),
            sharding=ShardingSpec(
                tp_rank=0,
                tp_world_size=1,
                vocab_shard_bounds=((0, padded_vocab),),
                real_vocab_size=real_vocab,
                padded_vocab_size=padded_vocab,
            ),
            reduction=ReductionSpec(),
        )
        return logits, targets, torch.tensor(active, device=device), contract

    @pytest.mark.parametrize("dtype_name", ["fp32", "bf16", "fp16"])
    @pytest.mark.parametrize(
        "real_vocab,padded_vocab,tiles",
        [(1000, 1024, 32), (13, 16, 8), (151936, 151936, 64)],
        ids=["partial-pad", "full-pad-tile", "qwen3"],
    )
    def test_forward_backward_match_reference(
        self, op_class, dtype_name, real_vocab, padded_vocab, tiles
    ):
        import torch

        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
        logits, targets, active, contract = self._case(real_vocab, padded_vocab, 24, dtype)
        ref_op, rocm_op = VocabParallelLogprobOp(), op_class()
        outputs = {}
        for name, op in (("ref", ref_op), ("rocm", rocm_op)):
            leaf = logits.clone().requires_grad_(True)
            logp, lse = op.apply(leaf, targets, contract=contract, num_vocab_tiles=tiles)
            ((logp * active).sum() + 0.25 * lse.sum()).backward()
            outputs[name] = (logp.detach(), lse.detach(), leaf.grad.detach())
        for ref, rocm in zip(outputs["ref"], outputs["rocm"]):
            assert torch.isfinite(rocm).all()
            torch.testing.assert_close(rocm.float(), ref.float(), rtol=2e-5, atol=2e-5)
        # Padding columns never receive gradient.
        grad = outputs["rocm"][2]
        assert torch.equal(grad[:, real_vocab:], torch.zeros_like(grad[:, real_vocab:]))
        # Inactive rows only carry the LSE gradient.
        inactive = ~active
        if inactive.any():
            row = inactive.nonzero()[0, 0]
            p = torch.softmax(logits[row, :real_vocab].float(), dim=-1)
            torch.testing.assert_close(
                grad[row, :real_vocab].float(), 0.25 * p, rtol=2e-2, atol=2e-5
            )
        # Repeat is bitwise.
        again = rocm_op.apply(logits, targets, contract=contract, num_vocab_tiles=tiles)
        assert torch.equal(again[0], outputs["rocm"][0]) and torch.equal(
            again[1], outputs["rocm"][1]
        )

    def test_logp_only_and_lse_only_gradients(self, op_class):
        import torch

        logits, targets, active, contract = self._case(1000, 1024, 12, torch.bfloat16)
        ref_op, rocm_op = VocabParallelLogprobOp(), op_class()
        for which in ("logp", "lse"):
            grads = []
            for op in (ref_op, rocm_op):
                leaf = logits.clone().requires_grad_(True)
                logp, lse = op.apply(leaf, targets, contract=contract, num_vocab_tiles=32)
                ((logp * active).sum() if which == "logp" else lse.sum()).backward()
                grads.append(leaf.grad.float())
            torch.testing.assert_close(grads[1], grads[0], rtol=2e-5, atol=2e-5)

    def test_tile_stats_read_input_dtype_exactly(self, op_class):
        """BF16 input straight into the kernel equals the FP32 upcast path bitwise."""
        import torch

        from rl_engine.reference.logprob.vocab_parallel_logp import _local_tile_stats

        if op_class is RocmVocabParallelLogprobOp:
            from rl_engine.backends.extension import _C

            tile_stats = _C.hip_deterministic_logp_tile_stats
        else:
            from rl_engine.backends.shared.triton.logprob.vocab_parallel_logp import (
                triton_vocab_tile_stats,
            )

            tile_stats = triton_vocab_tile_stats
        logits, _, _, _ = self._case(1000, 1024, 16, torch.bfloat16)
        direct = tile_stats(logits, 0, 1000, 32)
        upcast = tile_stats(logits.float(), 0, 1000, 32)
        assert all(torch.equal(a, b) for a, b in zip(direct, upcast))
        # Same tile maxima as the PyTorch tile loop; sums differ only by order.
        ids = torch.arange(1024, device=logits.device)
        masked = logits.float().masked_fill((ids >= 1000).unsqueeze(0), float("-inf"))
        ref_max, ref_sum = _local_tile_stats(masked, 32)
        assert torch.equal(direct[0], ref_max)
        torch.testing.assert_close(direct[1], ref_sum, rtol=1e-6, atol=0.0)
        # A tile that is entirely padding is the (-inf, 0) identity partial.
        tail = tile_stats(logits, 1024, 1000, 32)
        assert torch.isneginf(tail[0]).all() and torch.equal(tail[1], torch.zeros_like(tail[1]))

    def test_entropy_path_still_available(self, op_class):
        import torch

        logits, targets, active, contract = self._case(1000, 1024, 8, torch.float32)
        ref_op, rocm_op = VocabParallelLogprobOp(), op_class()
        ref = ref_op.apply_with_entropy(logits, targets, contract=contract, num_vocab_tiles=32)
        rocm = rocm_op.apply_with_entropy(logits, targets, contract=contract, num_vocab_tiles=32)
        for a, b in zip(ref, rocm):
            torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-5)


def test_triton_backend_registered_where_triton_runs():
    registry = KernelRegistry()
    import rl_engine.runtime.registry as registry_module

    for platform in ("cuda", "rocm"):
        candidates = registry._logprob_candidates[platform]
        if registry_module._triton_vocab_logprob_available():
            assert OpBackend.TRITON_VOCAB_PARALLEL_LOGP in candidates
            capability = registry._logprob_capabilities[platform][
                OpBackend.TRITON_VOCAB_PARALLEL_LOGP
            ]
            assert capability.backend_id == "triton-vocab-parallel-logp-ws2"
            assert capability.implementation_kind == "production"
            # The reference never outranks a production kernel backend.
            assert candidates.index(OpBackend.PYTORCH_VOCAB_PARALLEL_LOGP) > candidates.index(
                OpBackend.TRITON_VOCAB_PARALLEL_LOGP
            )
        else:
            assert OpBackend.TRITON_VOCAB_PARALLEL_LOGP not in candidates
    assert OpBackend.TRITON_VOCAB_PARALLEL_LOGP not in registry._logprob_candidates["cpu"]
