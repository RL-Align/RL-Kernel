# SPDX-License-Identifier: Apache-2.0
"""h3-parallel-lora-v2: LoRA on the H3 TP / SP / CP projections (RFC #420)."""

from __future__ import annotations

import pytest
import torch

from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel as par
from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel_lora as pl
from rl_engine.validation.models import h3_lora

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def gpus(n):
    return pytest.mark.skipif(torch.cuda.device_count() < n, reason=f"{n} GPUs required")


def _case(k, n, t=40, r=64, seed=0, b_zero=False):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(t, k, generator=gen).bfloat16().cuda()
    w = (torch.randn(n, k, generator=gen) / k**0.5).bfloat16().cuda()
    a = (torch.randn(r, k, generator=gen) / r).bfloat16().cuda()
    b = torch.zeros(n, r) if b_zero else torch.randn(n, r, generator=gen) * 0.02
    dy = torch.randn(t, n, generator=gen).bfloat16().cuda()
    return x, w, a, b.bfloat16().cuda(), dy


def _leaf(t):
    return t.detach().clone().requires_grad_(True)


def _strict(x, w, a, b, dy, scale=2.0):
    from rl_engine.models.minimax_h3.runtime import strict_lora_linear

    xs, a32, b32 = _leaf(x), _leaf(a.float()), _leaf(b.float())
    y = strict_lora_linear(xs, w, a32, b32, scale)
    y.backward(dy)
    return y.detach(), xs.grad, a32.grad, b32.grad


def _fp64(x, w, a, b, dy, scale=2.0):
    x, w, a, b, dy = (t.double() for t in (x, w, a, b, dy))
    z = x @ a.t()
    y = x @ w.t() + scale * z @ b.t()
    dz = scale * dy @ b
    return y, dy @ w + dz @ a, dz.t() @ x, scale * dy.t() @ z


def _col(x, w, a, b, dy, scale=2.0):
    xs, al, bl = _leaf(x), _leaf(a), _leaf(b)
    (y,) = pl.column_parallel_lora_linear(xs, [w], [(al, [bl])], scale)
    y.backward(dy)
    return y.detach(), xs.grad, al.grad, bl.grad


def _row(x, w, a, b, dy, scale=2.0):
    xs, al, bl = _leaf(x), _leaf(a), _leaf(b)
    y = pl.row_parallel_lora_linear(xs, w, al, bl, scale)
    y.backward(dy)
    return y.detach(), xs.grad, al.grad, bl.grad


def _rel(got, ref):
    return ((got.double() - ref).norm() / ref.norm()).item()


@needs_gpu
class TestOneGPU:
    def test_column_forward_is_strict_lora_bitwise(self):
        pytest.importorskip("triton")
        x, w, a, b, dy = _case(5376, 7168)
        assert torch.equal(_col(x, w, a, b, dy)[0], _strict(x, w, a, b, dy)[0])

    @pytest.mark.parametrize("kind", ["col", "row"])
    def test_errors_against_fp64_are_strict_loras(self, kind):
        """Forward, dX, dA and dB are as close to FP64 as #574's strict_lora_linear (tp=1 is the
        canonical 8-unit run, not bitwise the one-GEMM strict LoRA). FP32 A / B here, so the
        arithmetic is compared, not the rounding of BF16 adapter gradients."""

        pytest.importorskip("triton")
        k, n = (5376, 7168) if kind == "col" else (7168, 5376)
        x, w, a, b, dy = _case(k, n, seed=3)
        a, b = a.float(), b.float()
        ours = (_col if kind == "col" else _row)(x, w, a, b, dy)
        strict = _strict(x, w, a, b, dy)
        ref = _fp64(x, w, a, b, dy)
        for name, o, s, r in zip(("y", "dx", "dA", "dB"), ours, strict, ref):
            e_ours, e_strict = _rel(o, r), _rel(s, r)
            assert e_ours <= 2 * e_strict, (name, e_ours, e_strict)

    def test_bf16_adapters_round_their_gradients_once(self):
        """With BF16 A / B (the recipe's) the gradients are the FP32 ones rounded to BF16."""

        pytest.importorskip("triton")
        x, w, a, b, dy = _case(5376, 7168, seed=6)
        _, _, da16, db16 = _col(x, w, a, b, dy)
        _, _, da32, db32 = _col(x, w, a.float(), b.float(), dy)
        assert torch.equal(da16, da32.bfloat16()) and torch.equal(db16, db32.bfloat16())

    @pytest.mark.parametrize("kind", ["col", "row"])
    def test_b_zero_keeps_the_base(self, kind):
        """With the recipe's B = 0 the output is the projection without LoRA (values, and bytes
        except the sign of a zero) and dA is zero."""

        pytest.importorskip("triton")
        k, n = (5376, 7168) if kind == "col" else (7168, 5376)
        x, w, a, b, dy = _case(k, n, b_zero=True, seed=5)
        y, _, da, db = (_col if kind == "col" else _row)(x, w, a, b, dy)
        base = (
            par.column_parallel_linear(x, [w])[0]
            if kind == "col"
            else par.row_parallel_linear(x, w)
        )
        assert torch.equal(y, base)  # torch.equal: -0.0 == 0.0
        nonzero = base != 0
        assert torch.equal(y[nonzero].view(torch.int16), base[nonzero].view(torch.int16))
        assert not da.any() and db.any()

    @pytest.mark.parametrize("rows", [9, 0])
    def test_a_column_width_off_the_canonical_units_is_refused(self, rows):
        """dX sums 8 / tp equal slices of each output: 9 outputs at tp=1 would drop one."""

        x = torch.zeros(4, 8)
        w, a, b = torch.zeros(rows, 8), torch.zeros(2, 8), torch.zeros(rows, 2)
        with pytest.raises(ValueError, match="canonical units"):
            par.column_parallel_linear(x, [w])
        with pytest.raises(ValueError, match="canonical units"):
            pl.column_parallel_lora_linear(x, [w], [(a, [b])], 2.0)

    def test_sequence_parallel_flags_are_identities_at_tp1(self):
        """At tp=1 one rank holds every row: gather / scatter must not reach the (absent)
        collective, and the results are the unflagged ones, LoRA and base alike."""

        x, w, a, b, dy = _case(256, 384, seed=7)

        def run(fn, *args, **kw):
            xs = _leaf(x)
            leaves = [_leaf(t) for t in args]
            out = fn(xs, *leaves, **kw)
            y = out[0] if isinstance(out, tuple) else out
            y.backward(dy)
            grads = [t.grad for t in leaves]  # dA and dB in fixed slots
            assert all(g is not None for g in grads), "a LoRA leaf got no gradient"
            return [y.detach(), xs.grad, *grads]

        def col(xs, al, bl, **kw):
            return pl.column_parallel_lora_linear(xs, [w], [(al, [bl])], 2.0, **kw)

        def row(xs, al, bl, **kw):
            return pl.row_parallel_lora_linear(xs, w, al, bl, 2.0, **kw)

        def base_col(xs, **kw):
            return par.column_parallel_linear(xs, [w], **kw)

        def base_row(xs, **kw):
            return par.row_parallel_linear(xs, w, **kw)

        for flag, lora_fn in (("gather", col), ("scatter", row)):
            plain = run(lora_fn, a, b)
            flagged = run(lora_fn, a, b, **{flag: True})
            assert all(torch.equal(p, q) for p, q in zip(plain, flagged, strict=True)), flag
        for flag, fn in (("gather", base_col), ("scatter", base_row)):
            plain, flagged = run(fn), run(fn, **{flag: True})
            assert all(torch.equal(p, q) for p, q in zip(plain, flagged, strict=True)), flag

    def test_gradients_are_the_parameter_dtype(self):
        pytest.importorskip("triton")
        x, w, a, b, dy = _case(5376, 7168, t=16)
        _, _, da, db = _col(x, w, a, b, dy)
        assert da.dtype == db.dtype == torch.bfloat16


class TestLayout:
    @pytest.mark.parametrize("tp", [1, 2, 4, 8])
    def test_shards_tile_the_full_lora(self, tp):
        full = h3_lora.make_block_lora(3, rank=8, seed=1)
        ranks = [h3_lora.shard_block_lora(full, tp, r) for r in range(tp)]
        for target, kind in pl.TARGETS.items():
            a, b = full[target]["A"], full[target]["B"]
            if kind == "row":
                assert torch.equal(torch.cat([s[target]["A"] for s in ranks], 1), a)
                assert all(torch.equal(s[target]["B"], b) for s in ranks)
            elif target == "ff.net.0.proj":
                half = b.shape[0] // 2
                assert torch.equal(torch.cat([s[target]["B.hidden"] for s in ranks]), b[:half])
                assert torch.equal(torch.cat([s[target]["B.gate"] for s in ranks]), b[half:])
                assert all(torch.equal(s[target]["A"], a) for s in ranks)
            else:
                assert torch.equal(torch.cat([s[target]["B"] for s in ranks]), b)
                assert all(torch.equal(s[target]["A"], a) for s in ranks)

    def test_a_subset_or_reordered_map_draws_the_recipes_tensors(self):
        full = h3_lora.make_block_lora(5, rank=4, seed=3)
        alone = h3_lora.make_block_lora(
            5, rank=4, seed=3, shapes={"attn.to_k": h3_lora.SHAPES["attn.to_k"]}
        )
        reordered = h3_lora.make_block_lora(
            5, rank=4, seed=3, shapes=dict(reversed(h3_lora.SHAPES.items()))
        )
        for t in ("attn.to_k",):
            assert torch.equal(alone[t]["A"], full[t]["A"]) and torch.equal(
                alone[t]["B"], full[t]["B"]
            )
        for t in full:
            assert torch.equal(reordered[t]["A"], full[t]["A"]), t

    def test_the_draw_does_not_depend_on_the_grid(self):
        one = h3_lora.make_block_lora(7, rank=8, seed=2)
        two = h3_lora.make_block_lora(7, rank=8, seed=2)
        other = h3_lora.make_block_lora(8, rank=8, seed=2)
        assert all(torch.equal(one[t]["A"], two[t]["A"]) for t in one)
        assert not torch.equal(one["attn.to_q"]["A"], other["attn.to_q"]["A"])
        assert not h3_lora.make_block_lora(0, rank=8, b_zero=True)["ff.net.2"]["B"].any()

    @pytest.mark.parametrize(("tp", "cp"), [(1, 1), (2, 1), (4, 2), (8, 1)])
    def test_each_tensor_is_counted_once(self, tp, cp):
        """Over all ranks, every element of every LoRA tensor enters the norm exactly once."""

        for target, kind in pl.TARGETS.items():
            for part in ["A", "B.hidden", "B.gate"] if target == "ff.net.0.proj" else ["A", "B"]:
                sharded = (kind == "col") != (part == "A")
                if sharded:  # each TP rank's shard once, over the CP replicas
                    for t in range(tp):
                        per_shard = sum(pl.lora_counted(target, part, t, c) for c in range(cp))
                        assert per_shard == 1, (target, part, t)
                else:  # one whole tensor, counted once over all ranks
                    counts = [
                        pl.lora_counted(target, part, t, c) for c in range(cp) for t in range(tp)
                    ]
                    assert sum(counts) == 1, (target, part)

    def test_rollout_layout_follows_upstream(self):
        full = h3_lora.make_block_lora(0, rank=4, seed=3)
        out = pl.to_rollout_layout(full)
        b = full["ff.net.0.proj"]["B"]
        half = b.shape[0] // 2
        assert torch.equal(out["mlp.fc1_0.lora_B"], b[half:])  # gate first
        assert torch.equal(out["mlp.fc1_1.lora_B"], b[:half])  # then up (hidden)
        assert out["mlp.fc1_0.lora_A"] is out["mlp.fc1_1.lora_A"] is full["ff.net.0.proj"]["A"]
        assert torch.equal(out["attn.qkv_proj.k.lora_B"], full["attn.to_k"]["B"])
        assert torch.equal(out["mlp.fc2.lora_A"], full["ff.net.2"]["A"])
        assert len(out) == 14

    def test_refuses_mismatched_shapes(self):
        x = torch.zeros(2, 8)
        with pytest.raises(ValueError):
            pl.column_parallel_lora_linear(x, [torch.zeros(4, 8)], [(torch.zeros(2, 8), [])], 2.0)
        with pytest.raises(ValueError):
            pl.row_parallel_lora_linear(
                x, torch.zeros(4, 8), torch.zeros(2, 4), torch.zeros(4, 2), 2.0
            )
        with pytest.raises(ValueError):
            pl.row_parallel_lora_linear(
                x, torch.zeros(4, 8), torch.zeros(2, 8), torch.zeros(3, 2), 2.0
            )


# --- Multi-process grids: every layout equals the canonical single-GPU run bitwise --------------

GRIDS = [
    pytest.param(2, 1, False, id="tp2", marks=gpus(2)),
    pytest.param(2, 1, True, id="tp2_sp", marks=gpus(2)),
    pytest.param(1, 2, False, id="cp2", marks=gpus(2)),
    pytest.param(4, 1, True, id="tp4_sp", marks=gpus(4)),
    pytest.param(2, 2, False, id="tp2_cp2", marks=gpus(4)),
    pytest.param(2, 2, True, id="tp2_cp2_sp", marks=gpus(4)),
]


@needs_gpu
@pytest.mark.parametrize("b_zero", [False, True], ids=["b_random", "b_zero"])
@pytest.mark.parametrize(("tp", "cp", "sp"), GRIDS)
def test_lora_block_equals_canonical_bitwise(tp, cp, sp, b_zero, h3_block_weights_cuda):
    """Block 0 with LoRA on all six targets: output, dX, every base dW shard and every LoRA
    gradient shard on the grid equal the canonical Grid(1, 1) run bitwise."""

    pytest.importorskip("triton")
    from rl_engine.validation.distributed.h3_ws2 import run_world
    from rl_engine.validation.models import h3_tp

    kw = dict(backward=True, lora_rank=64, lora_seed=4, lora_b_zero=b_zero)
    ranks = run_world(
        tp * cp, h3_tp.block_target, {"dummy": torch.zeros(1)}, tp=tp, cp=cp, sp=sp, **kw
    )
    ref = h3_tp.block_case(h3_block_weights_cuda, h3_tp.Grid(1, 1, 0, 0), **kw)
    pieces = {}
    for r in ranks:
        span = tuple(int(v) for v in r["rows"])
        for key in ("out", "dx"):
            if (span, key) in pieces:
                assert torch.equal(pieces[(span, key)], r[key])
            pieces[(span, key)] = r[key]
    for key in ("out", "dx"):
        got = torch.cat([pieces[(s, k)] for (s, k) in sorted(pieces) if k == key], dim=1)
        assert torch.equal(got.cpu(), ref[key].cpu()), key
    # base weight gradients: each rank's shard is the canonical gradient cut like its weight
    from rl_engine.validation.models import h3_parallel_model as pm

    for name in pm.SHARDED_WEIGHTS:
        canonical = ref["dw." + name].cpu()
        for r in ranks:
            want = pm.shard_of(canonical, name, tp, int(r["rank"]) % tp)
            assert torch.equal(r["dw." + name].cpu(), want), (name, int(r["rank"]))
    # LoRA gradients: each rank's shard is the canonical gradient cut like the parameter
    for target, kind in pl.TARGETS.items():
        for name in h3_lora.leaf_names(target):
            canonical = ref["dw." + name].cpu()
            is_a = name.endswith(".A")
            for r in ranks:
                tp_rank = int(r["rank"]) % tp
                if kind == "row":
                    want = par.shard_cols(canonical, tp, tp_rank) if is_a else canonical
                else:
                    want = canonical if is_a else par.shard_rows(canonical, tp, tp_rank)
                assert torch.equal(r["dw." + name].cpu(), want), (name, int(r["rank"]))


class TestShardGradientVerdict:
    """The G4 tool's verdict: every shard gradient in both runs, equal; no partial pass."""

    def _digests(self, blocks=2):
        from rl_engine.validation.models import h3_parallel_model as pm

        keys = [f"{b}.{n}" for b in range(blocks) for n in pm.SHARDED_WEIGHTS]
        keys += [
            f"{b}.{n}" for b in range(blocks) for t in h3_lora.SHAPES for n in h3_lora.leaf_names(t)
        ]
        return {k: [i, i + 1] for i, k in enumerate(keys)}

    def _verdict(self, expected, observed, lora=True):
        from tools.validation.models.h3_parallel_full_model import shard_gradient_verdict

        return shard_gradient_verdict(expected, observed, 2, lora=lora)

    def test_equal_runs_pass(self):
        from rl_engine.validation.models import h3_parallel_model as pm

        d = self._digests()
        v = self._verdict(d, dict(d))
        assert v["base_bitwise"] and v["lora_bitwise"] and not v["missing"]
        assert v["base_compared"] == 2 * len(pm.SHARDED_WEIGHTS)
        assert v["lora_compared"] == 2 * h3_lora.LEAVES_PER_BLOCK

    @pytest.mark.parametrize("side", ["expected", "observed"])
    def test_a_lora_shard_missing_from_either_run_fails(self, side):
        expected, observed = self._digests(), self._digests()
        gone = "1.lora.ff.net.0.proj.B.gate"
        (expected if side == "expected" else observed).pop(gone)
        v = self._verdict(expected, observed)
        assert not v["lora_bitwise"] and v["base_bitwise"] and v["missing"] == [gone]
        full = self._verdict(self._digests(), self._digests())
        assert v["lora_compared"] == full["lora_compared"] - 1
        assert v["base_compared"] == full["base_compared"]

    def test_a_differing_digest_fails_and_is_named(self):
        expected, observed = self._digests(), self._digests()
        observed["0.lora.attn.to_v.A"] = [0, 0]
        observed["1.attn.to_q.weight"] = [0, 0]
        v = self._verdict(expected, observed)
        assert v["lora_mismatched"] == ["0.lora.attn.to_v.A"]
        assert v["base_mismatched"] == ["1.attn.to_q.weight"]
        assert not v["lora_bitwise"] and not v["base_bitwise"]

    def test_a_missing_base_shard_or_an_empty_run_fails(self):
        expected, observed = self._digests(), self._digests()
        observed.pop("0.ff.gate")
        assert not self._verdict(expected, observed)["base_bitwise"]
        empty = self._verdict({}, {})
        assert not empty["base_bitwise"] and not empty["lora_bitwise"]
        assert not self._verdict(self._digests(), self._digests(), lora=False)["lora_bitwise"]
