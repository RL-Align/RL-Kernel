# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Synthetic forward -> saved tensors -> direct CUDA backward, without an ABI.

python -m unittest discover -s tests -p 'test_p3_router_backward_*.py' -v
"""

import unittest
from unittest.mock import patch

import torch

from examples.p3_router_backward.forward_reference import (
    hash_forward_reference,
    learned_forward_reference,
)
from examples.p3_router_backward.prototype import (
    _load_extension,
    diagnostic_reference,
    fixed_tree6,
    route_backward_core,
)


def hash_case(tokens=5, *, device="cpu", dtype=torch.float32):
    scores = torch.linspace(0.25, 3, tokens * 256, dtype=dtype, device=device).reshape(tokens, 256)
    table = torch.tensor(
        [[255, 7, 7, 3, 7, 3], [4] * 6, [0, 31, 32, 127, 128, 255]],
        dtype=torch.int32,
        device=device,
    )
    token_ids = torch.arange(tokens, device=device) % 3
    active = torch.ones(tokens, dtype=torch.bool, device=device)
    return scores, token_ids, table, active


def learned_case(tokens=5, *, device="cpu", dtype=torch.float32):
    scores = torch.linspace(0.25, 3, tokens * 256, dtype=dtype, device=device).reshape(tokens, 256)
    bias = torch.zeros(256, dtype=dtype, device=device)
    # Force experts with small pre-bias scores into the selected set.
    bias[[0, 4, 17, 31, 100, 127]] = 8
    active = torch.ones(tokens, dtype=torch.bool, device=device)
    return scores, bias, active


def upstream_gradient(tokens, *, device="cpu", dtype=torch.float32):
    return torch.tensor([1, -2, 3, -4, 5, -6], device=device, dtype=dtype).repeat(tokens, 1)


def independent_score_gradient(scores, ids, g, active):
    """FP64 calculus from the original scores; no explicit backward formula."""
    source = scores.detach().cpu().double().requires_grad_()
    rows = active.cpu().nonzero(as_tuple=True)[0]
    a = source[rows].gather(1, ids.cpu()[rows].long())
    w = 1.5 * a / (a.sum(dim=1, keepdim=True) + 1e-20)
    return torch.autograd.grad((w * g.cpu()[rows].double()).sum(), source)[0]


class ForwardSemanticsTests(unittest.TestCase):
    def test_hash_preserves_table_slots_and_duplicate_score_gradients(self):
        scores, token_ids, table, active = hash_case(dtype=torch.float64)
        scores.requires_grad_()
        weights, saved = hash_forward_reference(scores, token_ids, table, active)
        self.assertTrue(torch.equal(saved.ids, table[token_ids]))
        g = upstream_gradient(len(scores), dtype=scores.dtype)
        (ds,) = torch.autograd.grad((weights * g).sum(), scores)
        expected = independent_score_gradient(scores, saved.ids, g, active)
        torch.testing.assert_close(ds, expected, rtol=1e-12, atol=1e-12)

    def test_learned_bias_changes_selection_but_not_weights_or_gradient_path(self):
        scores, bias, active = learned_case(dtype=torch.float64)
        scores.requires_grad_()
        bias.requires_grad_()
        weights, saved = learned_forward_reference(scores, bias, active)
        _, unbiased = learned_forward_reference(scores, torch.zeros_like(bias), active)
        self.assertFalse(torch.equal(saved.ids, unbiased.ids))
        expected_ids = torch.tensor([127, 100, 31, 17, 4, 0], dtype=torch.int32).repeat(5, 1)
        self.assertTrue(torch.equal(saved.ids, expected_ids))
        a = scores.gather(1, saved.ids.long())
        expected_weights = 1.5 * a / (a.sum(dim=1, keepdim=True) + 1e-20)
        torch.testing.assert_close(weights, expected_weights, rtol=1e-12, atol=1e-12)
        g = upstream_gradient(len(scores), dtype=scores.dtype)
        ds, dbias = torch.autograd.grad((weights * g).sum(), (scores, bias), allow_unused=True)
        self.assertIsNone(dbias)
        expected = independent_score_gradient(scores, saved.ids, g, active)
        torch.testing.assert_close(ds, expected, rtol=1e-12, atol=1e-12)
        # A deliberately wrong post-bias normalization must be observable.
        wrong_a = (scores + bias).gather(1, saved.ids.long())
        wrong_w = 1.5 * wrong_a / (wrong_a.sum(dim=1, keepdim=True) + 1e-20)
        self.assertFalse(torch.allclose(weights, wrong_w, rtol=1e-5, atol=1e-7))
        (wrong_dbias,) = torch.autograd.grad((wrong_w * g).sum(), bias)
        self.assertGreater(wrong_dbias.abs().max().item(), 0)

    def test_learned_ties_and_one_ulp_crossing_keep_logical_expert_order(self):
        scores = torch.full((2, 256), 0.25)
        scores[:, :5] = 2
        scores[:, 5:8] = 1
        scores[1, 7] = torch.nextafter(torch.tensor(1.0), torch.tensor(float("inf")))
        _, saved = learned_forward_reference(scores, torch.zeros(256), torch.ones(2).bool())
        self.assertEqual(saved.ids.tolist(), [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 7]])

    def test_saved_snapshots_have_no_graph_or_input_alias(self):
        scores, bias, active = learned_case()
        scores.requires_grad_()
        bias.requires_grad_()
        weights, saved = learned_forward_reference(scores, bias, active)
        before = [t.clone() for t in saved]
        for tensor in saved:
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)
        with torch.no_grad():
            scores.fill_(float("nan"))
            bias.fill_(float("nan"))
            active.zero_()
            weights.zero_()
        for actual, expected in zip(saved, before):
            self.assertTrue(torch.equal(actual, expected))


@unittest.skipUnless(torch.version.cuda and torch.cuda.is_available(), "NVIDIA CUDA required")
class CudaHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _load_extension()

    def assert_bits_equal(self, actual, expected):
        self.assertTrue(
            torch.equal(
                actual.detach().cpu().view(torch.int32), expected.detach().cpu().view(torch.int32)
            )
        )

    def check_handoff(self, forward, inputs):
        weights, saved = forward(*inputs)
        g = upstream_gradient(len(weights), device="cuda")
        # The tested entry must reach the real extension once for active rows.
        extension = _load_extension()
        with patch(
            "examples.p3_router_backward.prototype._load_extension", return_value=extension
        ) as loader:
            ds = route_backward_core(g, *saved)
        loader.assert_called_once_with()
        expected = diagnostic_reference(g.cpu(), *(tensor.cpu() for tensor in saved))
        self.assert_bits_equal(ds, expected)
        mathematical = independent_score_gradient(inputs[0], saved.ids, g, saved.row_active)
        torch.testing.assert_close(ds.cpu().double(), mathematical, rtol=2e-5, atol=3e-7)
        # CPU and GPU producers must agree on both the selection and saved bytes.
        cpu_weights, cpu_saved = forward(*(value.cpu() for value in inputs))
        self.assert_bits_equal(weights, cpu_weights)
        for actual, reference in zip(saved, cpu_saved):
            if actual.is_floating_point():
                self.assert_bits_equal(actual, reference)
            else:
                self.assertTrue(torch.equal(actual.cpu(), reference))
        return ds, saved

    def test_hash_forward_to_cuda_backward(self):
        self.check_handoff(hash_forward_reference, hash_case(device="cuda"))

    def test_learned_forward_to_cuda_backward(self):
        self.check_handoff(learned_forward_reference, learned_case(device="cuda"))

    def test_learned_tie_and_near_tie_forward_to_backward(self):
        scores = torch.full((2, 256), 0.25, device="cuda")
        scores[:, :5] = 2
        scores[:, 5:8] = 1
        scores[1, 7] = torch.nextafter(scores[1, 7], torch.tensor(float("inf"), device="cuda"))
        bias = torch.zeros(256, device="cuda")
        active = torch.ones(2, dtype=torch.bool, device="cuda")
        ds, saved = self.check_handoff(learned_forward_reference, (scores, bias, active))
        self.assertEqual(saved.ids[:, -1].tolist(), [5, 7])
        self.assertEqual(ds[0, 7].item(), 0)
        self.assertEqual(ds[1, 5].item(), 0)

    def test_backward_uses_saved_forward_after_live_inputs_change(self):
        for forward, inputs in (
            (hash_forward_reference, hash_case(device="cuda")),
            (learned_forward_reference, learned_case(device="cuda")),
        ):
            with self.subTest(mode=forward.__name__):
                _, saved = forward(*inputs)
                g = upstream_gradient(len(inputs[0]), device="cuda")
                before = route_backward_core(g, *saved)
                snapshot = [t.clone() for t in saved]
                for value in inputs:
                    if value.is_floating_point():
                        value.fill_(float("nan"))
                    elif value.dtype == torch.bool:
                        value.zero_()
                    else:
                        value.fill_(-1)
                # Any attempt to select experts again is an error.
                with patch("torch.argsort", side_effect=AssertionError("backward reselected")):
                    after = route_backward_core(g, *saved)
                self.assert_bits_equal(after, before)
                for actual, expected in zip(saved, snapshot):
                    self.assertTrue(torch.equal(actual, expected))

    def test_forward_and_backward_padding_batch_and_launch_invariance(self):
        for forward, inputs in (
            (hash_forward_reference, hash_case(device="cuda")),
            (learned_forward_reference, learned_case(device="cuda")),
        ):
            with self.subTest(mode=forward.__name__):
                inputs[-1][1::2] = False
                inputs[0][1::2] = float("nan")
                if forward is hash_forward_reference:
                    inputs[1][1::2] = -1
                ds, saved = self.check_handoff(forward, inputs)
                g = upstream_gradient(len(ds), device="cuda")
                for threads in (128, 256):
                    self.assert_bits_equal(route_backward_core(g, *saved, threads=threads), ds)
                for row in (0, 2, 4):
                    one = (inputs[0][row : row + 1],)
                    if forward is hash_forward_reference:
                        one += (inputs[1][row : row + 1], inputs[2])
                    else:
                        one += (inputs[1],)
                    one += (inputs[-1][row : row + 1],)
                    _, single_saved = forward(*one)
                    self.assert_bits_equal(
                        route_backward_core(g[row : row + 1], *single_saved), ds[row : row + 1]
                    )
                self.assert_bits_equal(ds[~inputs[-1]], torch.zeros((2, 256)))

    def test_empty_and_all_padding_forward_handoff(self):
        for tokens in (0, 3):
            for forward, inputs in (
                (hash_forward_reference, hash_case(tokens, device="cuda")),
                (learned_forward_reference, learned_case(tokens, device="cuda")),
            ):
                with self.subTest(tokens=tokens, mode=forward.__name__):
                    inputs[-1].zero_()
                    inputs[0].fill_(float("nan"))
                    if forward is hash_forward_reference:
                        inputs[1].fill_(-1)
                    weights, saved = forward(*inputs)
                    g = torch.full((tokens, 6), float("nan"), device="cuda")
                    with patch("examples.p3_router_backward.prototype._load_extension") as loader:
                        ds = route_backward_core(g, *saved)
                    loader.assert_not_called()
                    self.assert_bits_equal(weights, torch.zeros((tokens, 6)))
                    self.assert_bits_equal(ds, torch.zeros((tokens, 256)))

    def test_forward_based_fixture_detects_wrong_backward_reduction_orders(self):
        scores = torch.full((1, 256), 0.5e-20, device="cuda")
        token_ids = torch.zeros(1, dtype=torch.long, device="cuda")
        active = torch.ones(1, dtype=torch.bool, device="cuda")
        g = torch.tensor([[2**27, 8, -(2**27), -8, 1, -1]], dtype=torch.float32, device="cuda")
        for slots in ([0, 1, 2, 3, 4, 5], [7] * 6):
            with self.subTest(slots=slots):
                table = torch.tensor([slots], dtype=torch.int32, device="cuda")
                _, saved = hash_forward_reference(scores, token_ids, table, active)
                ds = route_backward_core(g, *saved).cpu()
                expected = diagnostic_reference(g.cpu(), *(t.cpu() for t in saved))
                self.assert_bits_equal(ds, expected)
                gp = g.cpu() * saved.p.cpu()
                factor = torch.div(torch.full_like(saved.z.cpu(), 1.5), saved.z.cpu())
                if slots[0] == 0:
                    wrong_c = torch.zeros(1)
                    for slot in range(6):
                        wrong_c = wrong_c + gp[:, slot]
                    wrong_da = factor[:, None] * (g.cpu() - wrong_c[:, None])
                    self.assertFalse(
                        torch.equal(ds[:, :6].view(torch.int32), wrong_da.view(torch.int32))
                    )
                else:
                    da = factor[:, None] * (g.cpu() - fixed_tree6(gp)[:, None])
                    self.assertNotEqual(ds[0, 7].item(), fixed_tree6(da).item())

    def test_missing_cuda_backend_never_falls_back_to_reference(self):
        _, saved = learned_forward_reference(*learned_case(device="cuda"))
        with patch(
            "examples.p3_router_backward.prototype._load_extension",
            side_effect=RuntimeError("test: CUDA extension unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA extension unavailable"):
                route_backward_core(upstream_gradient(5, device="cuda"), *saved)


if __name__ == "__main__":
    unittest.main()
