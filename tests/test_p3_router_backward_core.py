# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Local arithmetic tests, independent of the pending official P3 acceptance kit.

Run without installing the repository or pytest:
python -m unittest discover -s tests -p test_p3_router_backward_core.py -v
"""

import unittest
from unittest.mock import patch

import torch

from examples.p3_router_backward.prototype import (
    _load_extension,
    diagnostic_reference,
    fixed_tree6,
    route_backward_core,
)


def make_case(tokens, *, duplicate=False, dtype=torch.float32):
    rng = torch.Generator().manual_seed(106)
    scores = torch.rand((tokens, 256), generator=rng, dtype=dtype) + 0.05
    slots = [7, 7, 3, 255, 7, 3] if duplicate else [0, 31, 32, 127, 128, 255]
    ids = torch.tensor(slots, dtype=torch.int32).repeat(tokens, 1)
    selected = scores.gather(1, ids.long())
    denominator = fixed_tree6(selected) + 1e-20
    p = selected / denominator[:, None]
    g = torch.randn((tokens, 6), generator=rng, dtype=dtype)
    active = torch.ones(tokens, dtype=torch.bool)
    return g, ids, p, denominator, active


def cuda_case(case):
    return tuple(t.cuda() for t in case)


@torch.no_grad()
def finite_difference_scores(scores, ids, g, relative_step):
    """Differentiate the scalar forward loss numerically, with selection frozen.

    Perturb the original expert score, not a gathered slot: repeated IDs must
    change together. FP64 differences check real-valued calculus, not FP32 bits.
    No backward formula or autograd is used to construct this expectation.
    """

    def loss(values):
        selected = values[ids.long()]
        weights = 1.5 * selected / (selected.sum() + 1e-20)
        return (weights * g).sum()

    result = torch.empty_like(scores)
    for expert in range(scores.numel()):
        step = float(scores[expert]) * relative_step
        plus, minus = scores.clone(), scores.clone()
        plus[expert] += step
        minus[expert] -= step
        result[expert] = (loss(plus) - loss(minus)) / (2 * step)
    return result


def finite_difference_cases():
    # Epsilon is material at 1e-20. Other scales exercise ordinary and large
    # scores. The all-equal ID case detects missing duplicate contributions.
    for scale in (1e-20, 1e-3, 1.0, 1e3):
        for slots in ([0, 31, 32, 127, 128, 255], [7, 7, 3, 255, 7, 3], [4] * 6):
            scores = torch.linspace(0.5, 2.0, 256, dtype=torch.float64) * scale
            ids = torch.tensor(slots, dtype=torch.int32)
            g = torch.tensor([1, -2, 3, -4, 5, -6], dtype=torch.float64)
            yield scale, scores, ids, g


class ReferenceTests(unittest.TestCase):
    def test_formula_against_finite_differences(self):
        for scale, scores, ids, g in finite_difference_cases():
            a = scores[ids.long()]
            z = a.sum() + 1e-20
            actual = diagnostic_reference(
                g[None, :], ids[None, :], (a / z)[None, :], z[None], torch.tensor([True])
            )[0]
            for step in (1e-4, 3e-5):
                with self.subTest(scale=scale, ids=ids.tolist(), step=step):
                    expected = finite_difference_scores(scores, ids, g, step)
                    # Normalize units so the tolerance works at every scale,
                    # including near-zero derivatives for all-equal IDs.
                    torch.testing.assert_close(
                        actual * scale, expected * scale, rtol=2e-6, atol=2e-8
                    )

    def test_formula_against_independent_autograd(self):
        # Differentiate the original scores, so repeated IDs share a source.
        for slots in ([0, 1, 2, 3, 4, 5], [7, 7, 3, 255, 7, 3], [4] * 6):
            with self.subTest(slots=slots):
                scores = torch.linspace(0.1, 3, 256, dtype=torch.float64).requires_grad_()
                ids = torch.tensor([slots], dtype=torch.int32)
                a = scores[ids.long()]
                z = a.sum(dim=1) + 1e-20
                p = a / z[:, None]
                g = torch.tensor([[1, -2, 3, -4, 5, -6]], dtype=torch.float64)
                (expected,) = torch.autograd.grad((1.5 * p * g).sum(), scores)
                actual = diagnostic_reference(g, ids, p, z, torch.tensor([True]))
                torch.testing.assert_close(actual[0], expected, rtol=1e-12, atol=1e-12)

    def test_epsilon_at_observable_scale(self):
        a = torch.full((1, 6), 1e-20, dtype=torch.float64)
        z = a.sum(dim=1) + 1e-20
        ids = torch.arange(6, dtype=torch.int32)[None, :]
        result = diagnostic_reference(
            torch.ones_like(a), ids, a / z[:, None], z, torch.tensor([True])
        )
        expected = torch.full((6,), (1.5e-20 / z.square()).item(), dtype=torch.float64)
        torch.testing.assert_close(result[0, :6], expected, rtol=1e-12, atol=0)


@unittest.skipUnless(torch.version.cuda and torch.cuda.is_available(), "NVIDIA CUDA required")
class CudaCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Report a build failure once, before running numerical assertions.
        _load_extension()

    def assert_bits_equal(self, actual, expected):
        # torch.equal on floats considers +0 == -0. Compare all 32 bits instead.
        self.assertTrue(
            torch.equal(actual.cpu().view(torch.int32), expected.cpu().view(torch.int32))
        )

    def test_cuda_against_forward_finite_differences(self):
        for scale, scores, ids, g in finite_difference_cases():
            with self.subTest(scale=scale, ids=ids.tolist()):
                # Build saved FP32 state from representable original scores.
                scores32 = scores.float()
                a = scores32[ids.long()][None, :]
                z = fixed_tree6(a) + 1e-20
                case = (g.float()[None, :], ids[None, :], a / z[:, None], z, torch.tensor([True]))
                actual = route_backward_core(*cuda_case(case))[0].cpu().double()
                expected = finite_difference_scores(scores32.double(), ids, g, 1e-4)
                # This compares FP32 arithmetic with a numerical FP64 derivative;
                # byte equality is separately checked against the FP32 reference.
                torch.testing.assert_close(actual * scale, expected * scale, rtol=2e-5, atol=3e-7)

    def test_fp32_reference_multiple_sizes(self):
        for tokens in (1, 7, 33, 257, 1024):
            for duplicate in (False, True):
                with self.subTest(tokens=tokens, duplicate=duplicate):
                    case = make_case(tokens, duplicate=duplicate)
                    expected = diagnostic_reference(*case)
                    self.assert_bits_equal(route_backward_core(*cuda_case(case)), expected)

    def test_fixed_tree_and_duplicate_order_are_observable(self):
        # c's tree gives zero; sequential summation gives a different result.
        # p=1/8 is attainable with all a=epsilon/2 and Z=4*epsilon.
        g = torch.tensor([[2**27, 8, -(2**27), -8, 1, -1]], dtype=torch.float32)
        p = torch.full((1, 6), 0.125)
        z = torch.tensor([4e-20])
        active = torch.tensor([True])
        gp = g * p
        sequential_c = torch.zeros(1)
        for slot in range(6):
            sequential_c = sequential_c + gp[:, slot]
        self.assertFalse(torch.equal(fixed_tree6(gp), sequential_c))
        for ids in (
            torch.arange(6, dtype=torch.int32)[None, :],
            torch.full((1, 6), 7, dtype=torch.int32),
        ):
            case = (g, ids, p, z, active)
            expected = diagnostic_reference(*case)
            self.assert_bits_equal(route_backward_core(*cuda_case(case)), expected)
            if (ids == 7).all():
                factor = torch.div(torch.full_like(z, 1.5), z)
                da = factor[:, None] * (g - fixed_tree6(gp)[:, None])
                self.assertNotEqual(expected[0, 7].item(), fixed_tree6(da).item())

    def test_repeated_runs_and_launch_shape(self):
        case = cuda_case(make_case(129, duplicate=True))
        expected = route_backward_core(*case)
        for threads in (128, 256):
            for _ in range(3):
                self.assert_bits_equal(route_backward_core(*case, threads=threads), expected)

    def test_batch_and_row_permutation_invariance(self):
        case = cuda_case(make_case(9, duplicate=True))
        batch = route_backward_core(*case)
        singles = torch.cat([route_backward_core(*(t[i : i + 1] for t in case)) for i in range(9)])
        self.assert_bits_equal(batch, singles)
        order = torch.tensor([8, 1, 4, 0, 5, 3, 6, 2, 7], device="cuda")
        shuffled = route_backward_core(*(t[order].contiguous() for t in case))
        self.assert_bits_equal(shuffled, batch[order])

    def test_padding_is_never_used_in_arithmetic(self):
        case = list(make_case(7, duplicate=True))
        case[-1][1::2] = False
        for index in (0, 2, 3):
            case[index][1::2] = float("nan")
        case[1][1::2] = -1
        self.assert_bits_equal(route_backward_core(*cuda_case(case)), diagnostic_reference(*case))

    def test_zero_active_and_empty_do_not_load_or_launch_extension(self):
        for tokens in (0, 5):
            case = list(make_case(tokens))
            case[-1].fill_(False)
            case[0].fill_(float("nan"))
            case[1].fill_(-1)
            case[2].fill_(float("nan"))
            case[3].zero_()
            with patch("examples.p3_router_backward.prototype._load_extension") as loader:
                actual = route_backward_core(*cuda_case(case))
                loader.assert_not_called()
            self.assert_bits_equal(actual, torch.zeros((tokens, 256)))

    def test_kernel_overwrites_dirty_output(self):
        case = list(make_case(7, duplicate=True))
        case[-1][1::2] = False
        expected = diagnostic_reference(*case)
        inputs = cuda_case(case)
        for initial in (float("nan"), 123.0):
            out = torch.full((7, 256), initial, device="cuda")
            _load_extension().route_backward_core_out(*inputs, out, 256)
            self.assert_bits_equal(out, expected)

    def test_nondefault_stream(self):
        case = make_case(9, duplicate=True)
        expected = diagnostic_reference(*case)
        inputs = list(cuda_case(case))
        g = inputs[0].clone()
        inputs[0].zero_()
        out = torch.empty((9, 256), device="cuda")
        extension = _load_extension()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(5_000_000)
            inputs[0].copy_(g)
            extension.route_backward_core_out(*inputs, out, 256)
        stream.synchronize()
        self.assert_bits_equal(out, expected)

    def test_reject_bad_shapes_dtypes_layout_and_device(self):
        case = list(cuda_case(make_case(3)))
        replacements = [
            (0, case[0][:, :5]),
            (1, case[1].long()),
            (2, case[2].double()),
            (3, case[3][:, None]),
            (4, case[4].int()),
            (0, case[0].cpu()),
            (2, torch.ones((6, 3), device="cuda").t()),
        ]
        for index, value in replacements:
            with self.subTest(index=index, shape=value.shape, dtype=value.dtype):
                bad = case.copy()
                bad[index] = value
                with self.assertRaises(ValueError):
                    route_backward_core(*bad)
        with self.assertRaises(ValueError):
            route_backward_core(*case, threads=64)

    def test_reject_nonfinite_and_invalid_active_values(self):
        for index, value in (
            (0, float("nan")),
            (0, float("inf")),
            (1, -1),
            (1, 256),
            (2, float("nan")),
            (2, -0.1),
            (3, 0),
            (3, float("inf")),
        ):
            with self.subTest(index=index, value=value):
                case = list(cuda_case(make_case(2)))
                case[index][0] = value
                with self.assertRaises(ValueError):
                    route_backward_core(*case)

    def test_arithmetic_overflow_fails(self):
        case = list(cuda_case(make_case(1)))
        case[0].fill_(torch.finfo(torch.float32).max)
        case[2].fill_(1.0)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            route_backward_core(*case)

    def test_internal_binding_rejects_output_alias(self):
        case = list(cuda_case(make_case(1)))
        out = torch.empty((1, 256), device="cuda")
        out[:, :6].copy_(case[0])
        case[0] = out[:, :6]
        with self.assertRaises(RuntimeError):
            _load_extension().route_backward_core_out(*case, out, 256)


if __name__ == "__main__":
    unittest.main()
