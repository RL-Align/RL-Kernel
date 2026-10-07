# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``timestep_mlp_fp32``: FP32 256 -> 5376 -> 2688 timestep MLP.

* accuracy: forward and every gradient against an FP64 golden, on the pinned
  checkpoint weights and on synthetic non-H3 sizes;
* invariance: a timestep's ``temb`` bytes (and its row-local ``dx``) do not
  depend on how many timesteps share the call or where it sits;
* determinism: repeated forward/backward runs are bitwise equal;
* contract: BF16 inputs (RFC probe H7) and malformed shapes fail closed.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rl_engine.kernels.ops.pytorch.h3.timestep_mlp import NativeH3TimestepMLPOp
from rl_engine.kernels.ops.pytorch.h3.timestep_sinusoid import NativeH3TimestepSinusoidOp
from rl_engine.testing.h3_cases import h3_timesteps

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
NAMES = ("x", "w1", "b1", "w2", "b2")


def _cuda_op():
    from rl_engine.kernels.ops.cuda.h3.det_linear import det_linear_available
    from rl_engine.kernels.ops.cuda.h3.timestep_mlp import H3TimestepMLPCudaOp

    if not det_linear_available():
        pytest.skip("rl_engine._C lacks h3_det_linear_*")
    return H3TimestepMLPCudaOp()


def _features(num, device="cuda", seed=0):
    return NativeH3TimestepSinusoidOp().forward(h3_timesteps(num, seed=seed, device=device))


def _synthetic_params(k_in, hidden, out, device="cuda", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w1 = torch.randn(hidden, k_in, generator=g) / k_in**0.5
    b1 = torch.randn(hidden, generator=g) * 0.1
    w2 = torch.randn(out, hidden, generator=g) / hidden**0.5
    b2 = torch.randn(out, generator=g) * 0.1
    return [t.to(device) for t in (w1, b1, w2, b2)]


def _real_params(weights):
    return [
        weights[f"time_embedder.linear_{i}.{p}"].cuda() for i in (1, 2) for p in ("weight", "bias")
    ]


def _fp64_grads(x, params, grad_out):
    leaves = [t.detach().double().requires_grad_(True) for t in (x, *params)]
    z = F.linear(leaves[0], leaves[1], leaves[2])
    F.linear(z * torch.sigmoid(z), leaves[3], leaves[4]).backward(grad_out.double())
    return [leaf.grad for leaf in leaves]


class TestReference:
    def test_provider_and_golden_agree(self):
        x = _features(5, device="cpu")
        params = _synthetic_params(256, 64, 32, device="cpu")
        op = NativeH3TimestepMLPOp()
        torch.testing.assert_close(
            op.forward(x, *params), op.forward_fp32(x, *params), atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize("which", range(5))
    def test_rejects_bf16_anywhere(self, which):
        # RFC probe H7: casting a declared FP32 path to BF16 early.
        args = [_features(2, device="cpu"), *_synthetic_params(256, 64, 32, device="cpu")]
        args[which] = args[which].bfloat16()
        with pytest.raises(TypeError, match="float32"):
            NativeH3TimestepMLPOp().forward(*args)

    def test_rejects_shape_mismatches(self):
        x = _features(2, device="cpu")
        w1, b1, w2, b2 = _synthetic_params(256, 64, 32, device="cpu")
        op = NativeH3TimestepMLPOp()
        with pytest.raises(ValueError):
            op.forward(x[:, :128], w1, b1, w2, b2)
        with pytest.raises(ValueError):
            op.forward(x, w1, b1[:10], w2, b2)
        with pytest.raises(ValueError):
            op.forward(x, w1, b1, w2[:, :10], b2)
        with pytest.raises(ValueError):
            op.forward(x[:0], w1, b1, w2, b2)


@requires_cuda
class TestCudaRealWeights:
    @pytest.mark.parametrize("num", [1, 2, 3, 4, 7])
    def test_forward_against_fp64_golden(self, h3_weights_cpu, num):
        params = _real_params(h3_weights_cpu)
        x = _features(num, seed=num)
        out = _cuda_op()(x, *params)
        gold = NativeH3TimestepMLPOp().forward_fp32(x, *params)
        assert out.dtype == torch.float32 and out.shape == (num, 2688)
        # tolerance_contract.json forward_accuracy / reduction / float32.
        torch.testing.assert_close(out, gold, atol=1e-4, rtol=1e-4)
        assert (out - gold).abs().max().item() < 1e-5

    def test_rows_are_batch_and_position_invariant(self, h3_weights_cpu):
        op = _cuda_op()
        params = _real_params(h3_weights_cpu)
        x_all = _features(9, seed=1)
        full = op(x_all, *params)
        for i in range(9):
            assert torch.equal(op(x_all[i : i + 1], *params)[0], full[i])
        for num in (2, 3, 4, 5, 8):
            for pos in (0, num - 1):
                x = _features(num, seed=10 + num)
                x[pos] = x_all[4]
                assert torch.equal(op(x, *params)[pos], full[4]), (num, pos)

    def test_backward_against_fp64_golden(self, h3_weights_cpu):
        params = _real_params(h3_weights_cpu)
        x = _features(3, seed=2)
        leaves = [t.detach().clone().requires_grad_(True) for t in (x, *params)]
        grad_out = torch.randn(
            3, 2688, device="cuda", generator=torch.Generator("cuda").manual_seed(0)
        )
        _cuda_op()(*leaves).backward(grad_out)
        for name, leaf, ref in zip(NAMES, leaves, _fp64_grads(x, params, grad_out)):
            # tolerance_contract.json gradient_accuracy / reduction / float32.
            torch.testing.assert_close(leaf.grad.double(), ref, atol=1e-4, rtol=1e-4, msg=name)


@requires_cuda
class TestCudaSynthetic:
    @pytest.mark.parametrize("k_in, hidden, out", [(8, 24, 16), (12, 40, 20), (256, 5376, 2688)])
    def test_other_sizes(self, k_in, hidden, out):
        params = _synthetic_params(k_in, hidden, out, seed=k_in)
        x = torch.randn(5, k_in, device="cuda")
        torch.testing.assert_close(
            _cuda_op()(x, *params),
            NativeH3TimestepMLPOp().forward_fp32(x, *params),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_rejects_unaligned_k(self):
        params = _synthetic_params(6, 8, 8)
        with pytest.raises(RuntimeError, match="multiple of 4"):
            _cuda_op()(torch.randn(2, 6, device="cuda"), *params)

    def test_rejects_cpu_and_mixed_devices(self):
        params = _synthetic_params(8, 24, 16)
        with pytest.raises(ValueError):
            _cuda_op()(torch.randn(2, 8), *[p.cpu() for p in params])
        with pytest.raises(ValueError):
            _cuda_op()(torch.randn(2, 8), *params)

    def test_repeat_and_backward_are_deterministic(self):
        op = _cuda_op()
        params = _synthetic_params(256, 5376, 2688, seed=3)
        x = _features(4, seed=3)
        grad_out = torch.randn(4, 2688, device="cuda")
        runs = []
        for _ in range(3):
            leaves = [t.detach().clone().requires_grad_(True) for t in (x, *params)]
            out = op(*leaves)
            out.backward(grad_out)
            runs.append([out.detach(), *[leaf.grad for leaf in leaves]])
        for later in runs[1:]:
            for a, b in zip(runs[0], later):
                assert torch.equal(a, b)

    def test_dx_rows_are_batch_invariant(self):
        op = _cuda_op()
        params = _synthetic_params(256, 5376, 2688, seed=4)
        x = _features(6, seed=4)
        grad_out = torch.randn(6, 2688, device="cuda")

        def dx(rows):
            leaf = x[rows].detach().clone().requires_grad_(True)
            op(leaf, *params).backward(grad_out[rows])
            return leaf.grad

        full = dx(slice(0, 6))
        for i in range(6):
            assert torch.equal(dx(slice(i, i + 1))[0], full[i])

    def test_registry_dispatches_cuda(self):
        from rl_engine.kernels.registry import KernelRegistry

        _cuda_op()
        op = KernelRegistry().get_op("timestep_mlp_fp32", device="cuda")
        assert type(op).__name__ == "H3TimestepMLPCudaOp"
