# SPDX-License-Identifier: Apache-2.0
"""CPU-safe API, independent semantic and partial-gradient regression tests."""

import math
import unittest

import torch
import torch.nn.functional as F

from rl_engine.kernels.gtest.tolerance import load_contract, resolve_tolerance
from rl_engine.kernels.ops.pytorch.timestep_embed_mlp import NativeTimestepEmbedMLPOp
from rl_engine.kernels.ops.timestep_embed_mlp import TimestepEmbedMLPOp


class TimestepOfficialTests(unittest.TestCase):
    def values(self, dtype=torch.float32, batch=3):
        gen = torch.Generator().manual_seed(386)
        return (
            torch.randn(batch, generator=gen),
            torch.randn(17, 256, generator=gen).to(dtype) / 16,
            torch.randn(17, generator=gen).to(dtype) / 16,
            torch.randn(17, 17, generator=gen).to(dtype) / math.sqrt(17),
            torch.randn(17, generator=gen).to(dtype) / math.sqrt(17),
        )

    def test_independent_standard_semantics_and_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            v = tuple(x.requires_grad_() for x in self.values(dtype))
            t, w1, b1, w2, b2 = v
            # Separate conventional expression, independent of the candidate's primitives.
            f = (-math.log(10000) * torch.arange(128).float() / 128).exp()
            p = (t[:, None].float() * f) * 1000
            e = torch.cat((p.cos(), p.sin()), 1)
            y = F.linear(F.silu(F.linear(e, w1.float(), b1.float())), w2.float(), b2.float())
            gold = NativeTimestepEmbedMLPOp().forward_fp32(*v)
            dy = torch.randn(y.shape, generator=torch.Generator().manual_seed(9386))
            ga = torch.autograd.grad(y, v, dy, retain_graph=True)
            gb = torch.autograd.grad(gold, v, dy)
            for judgment, a, b in [
                ("forward_accuracy", y, gold),
                *[("gradient_accuracy", a, b) for a, b in zip(ga, gb, strict=True)],
            ]:
                tol = resolve_tolerance(
                    load_contract(), judgment=judgment, op_class="reduction", dtype=dtype
                )
                torch.testing.assert_close(a, b, atol=tol.atol, rtol=tol.rtol)

    def test_fallback_is_explicit_and_observable(self):
        v = self.values()
        with self.assertRaisesRegex(RuntimeError, "CUDA device"):
            TimestepEmbedMLPOp("cuda")(*v)
        y, trace = TimestepEmbedMLPOp("cuda", allow_fallback=True)(*v, return_trace=True)
        self.assertEqual(trace.actual_backend, "pytorch")
        self.assertIsNotNone(trace.fallback_reason)
        self.assertTrue(torch.equal(y, NativeTimestepEmbedMLPOp()(*v)))

    def test_empty_backward_and_partial_gradients(self):
        for batch in (0, 2):
            for index in range(5):
                v = list(self.values(batch=batch))
                v[index].requires_grad_()
                y = NativeTimestepEmbedMLPOp()(*v)
                (grad,) = torch.autograd.grad(y, (v[index],), torch.ones_like(y))
                self.assertEqual(grad.shape, v[index].shape)
                self.assertTrue(torch.isfinite(grad).all())

    def test_shapes_dtypes_and_layout_validation(self):
        v = self.values()
        op = TimestepEmbedMLPOp("pytorch")
        for kw in (
            {"chunk_size": 0},
            {"sample_ids": torch.zeros(3, dtype=torch.int64)},
            {"active_mask": torch.ones(3)},
            {"sample_ids": torch.arange(2)},
        ):
            with self.assertRaises(ValueError):
                op(*v, **kw)
        with self.assertRaises(ValueError):
            op(v[0].view(3, 1), *v[1:])
        with self.assertRaises(TypeError):
            op(*[x.half() for x in v])

    def test_trace_is_owned_by_call(self):
        op = TimestepEmbedMLPOp("pytorch")
        _, first = op(*self.values(), return_trace=True)
        _, second = op(*self.values(), return_trace=True)
        self.assertIsNot(first, second)
        self.assertEqual(first.kernels, ["pytorch.product_sum_gold"])


@unittest.skipUnless(torch.cuda.is_available(), "requires an NVIDIA CUDA GPU")
class TimestepGpuEdgeTests(unittest.TestCase):
    values = TimestepOfficialTests.values

    def test_gpu_empty_partial_and_graph_lifecycle(self):
        for backend in ("cuda", "triton"):
            for dtype in (torch.float32, torch.bfloat16):
                op = TimestepEmbedMLPOp(backend)
                for batch in (0, 2):
                    for index in range(5):
                        values = [x.cuda().detach() for x in self.values(dtype, batch=batch)]
                        values[index].requires_grad_()
                        y, trace = op(*values, return_trace=True)
                        (grad,) = torch.autograd.grad(y, (values[index],), torch.ones_like(y))
                        self.assertEqual(grad.dtype, values[index].dtype)
                        self.assertEqual(grad.shape, values[index].shape)
                        self.assertTrue(torch.isfinite(grad).all())
                        if batch == 0:
                            self.assertTrue((grad == 0).all())
                        self.assertEqual(trace.actual_backend, backend)
                # Two live graphs on one dispatcher must retain separate traces.
                first_values = [x.cuda().detach().requires_grad_() for x in self.values(dtype)]
                second_values = [x.cuda().detach().requires_grad_() for x in self.values(dtype)]
                first, first_trace = op(*first_values, return_trace=True)
                second, second_trace = op(*second_values, return_trace=True)
                second_launches = list(second_trace.kernels)
                first_grads = torch.autograd.grad(
                    first, first_values, torch.ones_like(first), retain_graph=True
                )
                self.assertEqual(second_trace.kernels, second_launches)
                again = torch.autograd.grad(first, first_values, torch.ones_like(first))
                for a, b in zip(first_grads, again, strict=True):
                    self.assertTrue(torch.equal(a, b))
                torch.autograd.grad(second, second_values, torch.ones_like(second))
                self.assertIsNot(first_trace, second_trace)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    third_values = [x.detach().clone().requires_grad_() for x in first_values]
                    third = op(*third_values)
                    third_grads = torch.autograd.grad(third, third_values, torch.ones_like(third))
                torch.cuda.current_stream().wait_stream(stream)
                for a, b in zip(first_grads, third_grads, strict=True):
                    self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
