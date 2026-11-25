# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``h3-grid-train-v1``: the LoRA training view of the H3 grid (``h3_grid_actor``) and the refiner
LoRA of ``h3_lora``. CPU only: parameters, counting, gradient accumulation, state, export, sync.
The bitwise training step against the single-GPU run is G5 (``h3_grid_train_step.py``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rl_engine.backends.cuda.model_specific.minimax_h3 import parallel_lora as pl
from rl_engine.integrations.engines.train import h3_grid_actor as ga
from rl_engine.validation.models import h3_lora
from rl_engine.validation.models import h3_parallel_model as pm
from rl_engine.validation.models import h3_tp

R = 2  # LoRA rank of the stubs


class _RefinerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        for name in ("to_q", "to_k", "to_v"):
            setattr(self.attn, name, nn.Linear(2, 2, bias=False))
        self.attn.to_out = nn.ModuleList([nn.Linear(2, 2, bias=False), nn.Dropout(0.0)])
        self.ff = nn.Module()
        proj = nn.Module()
        proj.proj = nn.Linear(2, 4, bias=False)
        self.ff.net = nn.ModuleList([proj, nn.Dropout(0.0), nn.Linear(2, 2, bias=False)])

    def forward(self, x):
        """Report which SDPA backends the attention would see."""
        cuda = torch.backends.cuda
        return {"math": cuda.math_sdp_enabled(), "flash": cuda.flash_sdp_enabled(), "x": x}


class _Model(nn.Module):
    """Blocks that only hold LoRA attributes, and a two-block token refiner."""

    def __init__(self, blocks=2, refiner=2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(nn.Identity() for _ in range(blocks))
        self.token_refiner = nn.Module()
        self.token_refiner.refiner_blocks = nn.ModuleList(_RefinerBlock() for _ in range(refiner))


def _grid(tp=4, cp=2, tp_rank=1, cp_rank=0):
    return h3_tp.Grid(tp, cp, tp_rank, cp_rank, None, None, "h3_full", "ring")


def _model(tp=4, tp_rank=1, refiner=True, seed=0):
    model = _Model()
    shards = [
        h3_lora.shard_block_lora(h3_lora.make_block_lora(i, rank=R, seed=seed), tp, tp_rank)
        for i in range(2)
    ]
    loras = [h3_lora.make_refiner_lora(i, rank=R, seed=seed) for i in range(2)] if refiner else None
    h3_lora.install_lora_shards(model, shards, tp, tp_rank, scale=2.0, device="cpu", refiner=loras)
    return model


def _actor(model=None, strict=False, **grid):
    return ga.H3GridActor(model or _model(), _grid(**grid), sp=True, strict_boundaries=strict)


class TestParameters:
    @pytest.mark.parametrize("bare", [0, 1])
    def test_a_refiner_lora_on_some_blocks_only_is_refused(self, bare):
        model = _model()
        h3_lora.refiner_blocks(model)[bare].__dict__.pop(h3_lora.REFINER_ATTR)
        with pytest.raises(ValueError, match=rf"refiner blocks \[{bare}\] hold no LoRA"):
            _actor(model)

    def test_every_lora_tensor_is_a_parameter_in_a_fixed_order(self):
        model = _model()
        actor = _actor(model)
        assert set(actor.names) == set(h3_lora.lora_digests(model))
        assert len(actor.names) == 2 * h3_lora.LEAVES_PER_BLOCK + 2 * 12
        assert actor.names == _actor(_model()).names  # the order does not depend on the object
        # the blocks read the parameters' storage
        p = actor.parameters()[0]
        assert isinstance(p, nn.Parameter)
        assert model.transformer_blocks[0].__dict__[pm.LORA_ATTR]["tensors"]["attn.to_q"]["A"] is p
        refiner = [n for n in actor.names if n.startswith("refiner.")]
        assert (
            refiner == sorted(refiner, key=actor.names.index)
            and actor.names[-len(refiner) :] == refiner
        )

    def test_block_shards_take_gradients_from_the_sink_only(self):
        actor = _actor()
        blocks = [
            p for n, p in zip(actor.names, actor.parameters()) if not n.startswith("refiner.")
        ]
        refiner = [p for n, p in zip(actor.names, actor.parameters()) if n.startswith("refiner.")]
        assert not any(p.requires_grad for p in blocks)
        assert all(p.requires_grad for p in refiner)

    # written out rather than taken from ``pl.lora_counted``: the tensors every TP rank holds whole
    TP_REPLICATED = {
        ("attn.to_q", "A"),
        ("attn.to_k", "A"),
        ("attn.to_v", "A"),
        ("ff.net.0.proj", "A"),
        ("attn.to_out.0", "B"),
        ("ff.net.2", "B"),
    }

    @pytest.mark.parametrize("tp_rank, cp_rank", [(0, 0), (1, 0), (0, 1), (3, 1)])
    def test_each_replicated_tensor_is_counted_once(self, tp_rank, cp_rank):
        actor = _actor(_model(tp_rank=tp_rank), tp_rank=tp_rank, cp_rank=cp_rank)
        for name, p in zip(actor.names, actor.parameters()):
            if name.startswith("refiner."):
                assert actor.counted(p)  # global rank 0 (no process group)
                continue
            _, rest = name.split(".", 1)
            target = next(
                t for t in sorted(pl.TARGETS, key=len, reverse=True) if rest.startswith(t)
            )
            part = rest[len(target) + 1 :]
            replicated = (target, part) in self.TP_REPLICATED
            assert actor.counted(p) is (cp_rank == 0 and (tp_rank == 0 or not replicated)), name

    def test_refuses_what_it_cannot_train(self):
        with pytest.raises(ValueError, match="strict refiner"):
            _actor(strict=True)
        _actor(
            _model(refiner=False), strict=True
        )  # block LoRA alone is fine with strict boundaries
        with pytest.raises(ValueError, match="holds no LoRA"):
            ga.H3GridActor(_Model(), _grid(), sp=True, strict_boundaries=False)
        with pytest.raises(ValueError, match="rank 2"):
            _actor(_model(tp_rank=2), tp_rank=1)
        model = _model()
        model.transformer_blocks[1].__dict__[pm.LORA_ATTR]["scale"] = 1.0
        with pytest.raises(ValueError, match="scales differ"):
            _actor(model)


class TestGradients:
    def test_each_pass_is_added_in_order_and_base_grads_are_dropped(self):
        actor = _actor()
        name = "lora.attn.to_q.A"
        p = actor._by_leaf[(0, name)]
        g1 = torch.full_like(p, 0.5)
        g2 = torch.full_like(p, 0.25)
        actor._sink(0, "attn.to_q.weight", torch.ones(3))  # a frozen base shard
        actor._sink(0, name, g1)
        assert p.grad is not g1 and torch.equal(p.grad, g1)
        actor._sink(0, name, g2)
        assert torch.equal(p.grad, (g1 + g2))
        assert actor._received == {(0, name)}  # one shard, however many sinks
        with pytest.raises(TypeError, match="gradient"):
            actor._sink(0, name, g1.float())
        actor.zero_grad()
        assert all(q.grad is None for q in actor.parameters())


class TestState:
    def test_lora_state_round_trips_in_place(self):
        actor = _actor()
        state = actor.lora_state()
        assert set(state) == {"tp", "tp_rank", "shards", "digests", "refiner", "scale"}
        assert state["scale"] == actor.scale
        assert all(
            t.device.type == "cpu"
            for b in state["shards"]
            for ps in b.values()
            for t in ps.values()
        )
        before = actor.digests()
        with torch.no_grad():
            for p in actor.parameters():
                p.add_(1)
        assert actor.digests() != before
        params = actor.parameters()
        actor.load_lora_state(state)
        assert actor.digests() == before
        assert all(a is b for a, b in zip(params, actor.parameters()))  # the same objects

    def test_load_refuses_another_scale(self):
        actor = _actor()
        state = dict(actor.lora_state(), scale=actor.scale + 1.0)
        with pytest.raises(ValueError, match="scale"):
            actor.load_lora_state(state)

    def test_load_refuses_another_ranks_state(self):
        other = _actor(_model(tp_rank=2), tp_rank=2).lora_state()
        with pytest.raises(ValueError, match="rank 2's"):
            _actor().load_lora_state(other)

    def test_checkpoint_carries_the_optimizer(self):
        actor = _actor()
        opt = ga.make_optimizer(actor.parameters(), lr=3e-4, weight_decay=1e-4)
        assert opt.defaults["foreach"] is False and opt.defaults["fused"] is False
        for p in actor.parameters():
            p.grad = torch.ones_like(p)
        opt.step()
        saved = actor.checkpoint_state(opt)
        digests = actor.digests()

        fresh = _actor()
        fopt = ga.make_optimizer(fresh.parameters(), lr=3e-4, weight_decay=1e-4)
        fresh.load_checkpoint_state(saved, fopt)
        assert fresh.digests() == digests
        assert fopt.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"]
        with pytest.raises(ValueError, match="checkpoint"):
            _actor(_model(tp_rank=2), tp_rank=2).load_checkpoint_state(saved)

    @pytest.mark.parametrize("failing", [0, 1])
    def test_a_failed_write_raises_on_every_rank_after_the_gather(
        self, tmp_path, monkeypatch, failing
    ):
        """This rank is 0 of two: whichever rank failed, it joins the gather and then raises."""

        actor = _actor()
        actor.world = 2
        gathered = []

        def all_gather_object(out, mine, group=None):
            gathered.append(mine)
            other = (mine[0], {}, "OSError: disk full") if failing == 1 else (mine[0], {}, None)
            out[:] = [mine, other]

        def fail(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(ga.dist, "all_gather_object", all_gather_object)
        if failing == 0:
            monkeypatch.setattr(ga.torch, "save", fail)
        with pytest.raises(
            RuntimeError, match=rf"export failed on ranks \[\({failing}, 'OSError: disk full'"
        ):
            actor.export(str(tmp_path))
        assert len(gathered) == 1
        assert (gathered[0][2] is not None) == (failing == 0)

    def test_export_is_what_the_rollout_installs(self, tmp_path):
        actor = _actor()
        out = actor.export(str(tmp_path))
        assert out["path"] == str(tmp_path / "cp0" / "rank1.pt")
        assert out["rollout_source"].endswith("cp0/rank{rank}.pt")
        sent = torch.load(out["rollout_source"].format(rank=1), weights_only=True)
        assert sent["scale"] == out["scale"] == actor.scale == 2.0  # rlk_h3_load_lora checks it
        receiver = _Model()
        got = h3_lora.install_lora_shards(
            receiver,
            sent["shards"],
            4,
            1,
            scale=sent["scale"],
            expected=sent["digests"],
            device="cpu",
            refiner=sent.get("refiner"),
        )
        assert got == actor.digests() == out["digests"]


class TestRefinerLoRA:
    def test_sync_installs_and_rolls_back_the_refiner(self):
        model = _model()
        before = h3_lora.lora_digests(model)
        assert sum(k.startswith("refiner.") for k in before) == 24
        shards = [
            h3_lora.shard_block_lora(h3_lora.make_block_lora(i, rank=R, seed=5), 4, 1)
            for i in range(2)
        ]
        loras = [h3_lora.make_refiner_lora(i, rank=R, seed=5) for i in range(2)]
        wrong = dict(before)
        wrong["refiner.1.ff.net.2.B"] = "0" * 64
        with pytest.raises(RuntimeError, match="digest mismatch"):
            h3_lora.install_lora_shards(
                model, shards, 4, 1, scale=2.0, expected=wrong, device="cpu", refiner=loras
            )
        assert h3_lora.lora_digests(model) == before  # the refiner LoRA came back too
        h3_lora.install_lora_shards(model, shards, 4, 1, scale=2.0, device="cpu")
        assert not h3_lora.has_refiner_lora(model)  # nothing sent, nothing kept

    @pytest.mark.parametrize("had_refiner", [True, False])
    @pytest.mark.parametrize("other_rank_fails", [False, True])
    def test_a_refused_worker_sync_puts_the_refiner_back(
        self, tmp_path, monkeypatch, had_refiner, other_rank_fails
    ):
        from rl_engine.integrations.engines.rollout.vllm import h3_omni_worker as wk

        sender = _actor(_model(seed=5))
        sent = torch.load(sender.export(str(tmp_path))["path"], weights_only=True)
        model = _model(refiner=had_refiner)
        before = h3_lora.lora_digests(model)
        assert h3_lora.has_refiner_lora(model) is had_refiner and before != sent["digests"]
        ext = wk.H3OmniWorkerExtension()
        ext._rlk_h3 = SimpleNamespace(
            binding=SimpleNamespace(
                model=model, grid=SimpleNamespace(tp=4, tp_rank=1), strict_boundaries=False
            )
        )

        def gather(mine):  # this rank installed the whole LoRA; another rank may refuse
            assert mine["ok"], mine
            assert h3_lora.lora_digests(model) == sent["digests"]
            return [mine["result"]] * 4, ({2: "x"} if other_rank_fails else {})

        monkeypatch.setattr(wk, "_gather", gather)
        source = str(tmp_path / "cp0" / "rank{rank}.pt")
        if other_rank_fails:
            with pytest.raises(RuntimeError, match="no rank changed"):
                ext.rlk_h3_load_lora(source, scale=sent["scale"])
            assert h3_lora.lora_digests(model) == before
            assert h3_lora.has_refiner_lora(model) is had_refiner
        else:
            ext.rlk_h3_load_lora(source, scale=sent["scale"])
            assert h3_lora.lora_digests(model) == sent["digests"]

    def test_attaching_without_the_refiner_removes_an_earlier_refiner_lora(self):
        model, kw = _Model(), dict(rank=R, alpha=2.0 * R, device="cpu")
        h3_lora.attach_lora(model, 4, 1, seed=1, refiner=True, **kw)
        assert sum(k.startswith("refiner.") for k in h3_lora.lora_digests(model)) == 24
        h3_lora.attach_lora(model, 4, 1, seed=2, **kw)
        fresh = _Model()
        h3_lora.attach_lora(fresh, 4, 1, seed=2, **kw)
        assert not h3_lora.has_refiner_lora(model)
        assert h3_lora.lora_digests(model) == h3_lora.lora_digests(fresh)

    def test_an_empty_refiner_list_fits_a_model_without_refiner_blocks(self):
        model = _Model(refiner=0)
        shards = [
            h3_lora.shard_block_lora(h3_lora.make_block_lora(i, rank=R), 4, 1) for i in range(2)
        ]
        h3_lora.install_lora_shards(model, shards, 4, 1, scale=2.0, device="cpu", refiner=[])
        assert not h3_lora.has_refiner_lora(model)
        with pytest.raises(ValueError, match="0 refiner LoRAs for 2 refiner blocks"):
            h3_lora.install_lora_shards(_Model(), shards, 4, 1, scale=2.0, device="cpu", refiner=[])

    def test_refuses_a_partial_or_mis_shaped_refiner_lora(self):
        model = _Model()
        good = [h3_lora.make_refiner_lora(i, rank=R) for i in range(2)]
        with pytest.raises(ValueError, match="2 refiner blocks"):
            h3_lora._set_refiner(model, good[:1], 2.0, "cpu")
        bad = [dict(g) for g in good]
        bad[0]["attn.to_q"] = {
            "A": good[0]["attn.to_q"]["A"][:, :5],
            "B": good[0]["attn.to_q"]["B"],
        }
        with pytest.raises(ValueError, match="whole"):
            h3_lora._set_refiner(model, bad, 2.0, "cpu")

    def test_refuses_a_refiner_lora_of_another_or_zero_rank(self):
        model = _model()
        before = h3_lora.lora_digests(model)
        shards = [
            h3_lora.shard_block_lora(h3_lora.make_block_lora(i, rank=R), 4, 1) for i in range(2)
        ]
        for rank in (R // 2, 0):
            loras = [h3_lora.make_refiner_lora(i, rank=rank) for i in range(2)]
            with pytest.raises(ValueError, match="refiner LoRA ranks"):
                h3_lora.install_lora_shards(
                    model, shards, 4, 1, scale=2.0, device="cpu", refiner=loras
                )
        assert h3_lora.lora_digests(model) == before  # refused before anything was installed

    def test_without_a_lora_the_linear_is_unchanged(self):
        model = _Model()
        block = model.token_refiner.refiner_blocks[0]
        x = torch.randn(3, 2)
        want = block.attn.to_q(x)
        h3_lora._set_refiner(
            model, [h3_lora.make_refiner_lora(i, rank=R) for i in range(2)], 2.0, "cpu"
        )
        h3_lora._set_refiner(model, None, 2.0, "cpu")
        assert torch.equal(block.attn.to_q(x), want)  # patched, but no LoRA: the original forward
        assert "_rlk_refiner_forward" in block.attn.to_q.__dict__

    def test_a_refiner_with_a_lora_runs_attention_on_the_math_backend(self):
        model = _Model()
        block = model.token_refiner.refiner_blocks[0]
        assert block(1)["flash"] == torch.backends.cuda.flash_sdp_enabled()
        h3_lora._set_refiner(
            model, [h3_lora.make_refiner_lora(i, rank=R) for i in range(2)], 2.0, "cpu"
        )
        seen = block(1)
        assert seen["math"] and not seen["flash"] and seen["x"] == 1
        h3_lora._set_refiner(model, None, 2.0, "cpu")
        assert block(1)["flash"] == torch.backends.cuda.flash_sdp_enabled()  # no LoRA: as before

    def test_the_gate_up_base_runs_the_rows_fixed_kernel(self, monkeypatch):
        """As in the blocks (h3-parallel-lora-v2): gate/up on the fixed kernel, Q/K/V not."""

        seen = {}

        def column(x, weights, loras, scale, *, kernel="projection", **kw):
            seen[weights[0].shape] = kernel
            return (torch.zeros(x.shape[0], weights[0].shape[0]),)

        monkeypatch.setattr(pl, "column_parallel_lora_linear", column)
        for target, want in (("ff.net.0.proj", "fixed"), ("attn.to_q", "projection")):
            seen.clear()
            linear = nn.Linear(4, 6, bias=False)
            forward = h3_lora._refiner_forward(_LoRAHolder(target), target)
            forward(linear, torch.zeros(2, 4))
            assert list(seen.values()) == [want], target

    def test_the_draw_is_its_own_stream(self):
        refiner = h3_lora.make_refiner_lora(0, rank=R, seed=1)
        block = h3_lora.make_block_lora(0, rank=R, seed=1)
        assert not torch.equal(refiner["attn.to_q"]["A"], block["attn.to_q"]["A"])
        again = h3_lora.make_refiner_lora(0, rank=R, seed=1)
        assert all(torch.equal(refiner[t][p], again[t][p]) for t in refiner for p in refiner[t])


class _LoRAHolder:
    """A refiner block holding a LoRA for ``target`` (only what ``_refiner_forward`` reads)."""

    def __init__(self, target):
        parts = {"A": torch.zeros(2, 4), "B": torch.zeros(6, 2)}
        self.__dict__[h3_lora.REFINER_ATTR] = {"scale": 1.0, "tensors": {target: parts}}


class TestG5Cut:
    @pytest.mark.parametrize("tp", [2, 4, 8])
    def test_cut_of_the_whole_lora_is_the_grid_shard(self, tp):
        from tools.validation.models.h3_grid_train_step import _cut

        whole = h3_lora.make_block_lora(3, rank=R, seed=2, b_zero=False)
        canonical = h3_lora.shard_block_lora(whole, 1, 0)
        for rank in range(tp):
            mine = h3_lora.shard_block_lora(whole, tp, rank)
            for target, parts in canonical.items():
                for part, t in parts.items():
                    assert torch.equal(_cut(f"3.{target}.{part}", t, tp, rank), mine[target][part])


class TestG5Export:
    def test_a_replica_mismatch_is_recorded_in_an_injected_run_only(self, tmp_path):
        from tools.validation.models.h3_grid_train_step import _export

        class Diverged:
            def export(self, directory):
                raise RuntimeError("the CP replicas of TP rank 1 hold different LoRA")

        got = _export(Diverged(), tmp_path, injected=True)
        assert "CP replicas" in got["error"]
        assert not got["file_digests_equal"] and not got["cp_replicas_agree"]
        with pytest.raises(RuntimeError, match="CP replicas"):
            _export(Diverged(), tmp_path, injected=False)

    def test_a_good_export_is_checked_against_its_file(self, tmp_path):
        from tools.validation.models.h3_grid_train_step import _export

        got = _export(_actor(), tmp_path, injected=False)
        assert got["file_digests_equal"] and got["cp_replicas_agree"] and "error" not in got


class TestG5Arguments:
    def _check(self, **kw):
        import argparse

        from tools.validation.models.h3_grid_train_step import _check_args

        args = dict(optimizer_steps=2, passes=3, inject=None, tp=4, cp=2)
        _check_args(argparse.ArgumentParser(), argparse.Namespace(**{**args, **kw}))

    @pytest.mark.parametrize("inject", [None, "17.attn.to_v.A:2", "3.ff.net.0.proj.B.gate:7"])
    def test_accepts(self, inject):
        self._check(inject=inject)

    @pytest.mark.parametrize(
        "kw",
        [
            dict(optimizer_steps=0),
            dict(passes=0),
            dict(inject="17.attn.to_v.A"),
            dict(inject="17.attn.to_v.A:x"),
            dict(inject="17.attn.to_v.A:8"),
            dict(inject="refiner.0.attn.to_q.A:1"),
        ],
    )
    def test_refuses_a_run_that_would_prove_nothing(self, kw):
        with pytest.raises(SystemExit):
            self._check(**kw)


class TestRefinerBroadcast:
    def _two_ranks(self, monkeypatch, disagree_on=None):
        actor = _actor()
        actor.world = 2
        calls = []

        def all_reduce(t, op=None, group=None):
            if disagree_on is not None:
                t[disagree_on] = 0 if op == ga.dist.ReduceOp.MIN else 1

        monkeypatch.setattr(ga.dist, "all_reduce", all_reduce)
        monkeypatch.setattr(ga.dist, "broadcast", lambda t, src, group=None: calls.append(t))
        return actor, calls

    def test_every_refiner_gradient_is_broadcast_when_ranks_agree(self, monkeypatch):
        actor, calls = self._two_ranks(monkeypatch)
        for p in actor._refiner:
            p.grad = torch.ones_like(p)
        actor.finish_grads()
        assert len(calls) == len(actor._refiner) > 0

    def test_ranks_that_disagree_on_a_gradient_raise_instead_of_hanging(self, monkeypatch):
        actor, calls = self._two_ranks(monkeypatch, disagree_on=0)
        for p in actor._refiner:
            p.grad = torch.ones_like(p)
        with pytest.raises(RuntimeError, match=r"refiner LoRA tensors \[0\] have a gradient"):
            actor.finish_grads()
        assert calls == []


def test_every_forward_hooks_its_own_leaf():
    """A recomputed forward cuts new leaves from the same weight, so no leaf has two sink hooks."""

    weight, got = nn.Parameter(torch.randn(4, 4)), []

    def forward():  # what ``_block_forward`` does with one shard
        leaves = {}
        leaf = h3_tp.shard_leaf("w", weight, leaves)
        assert leaves["w"] is leaf and leaf is not weight
        pm._sink_on_grad(SimpleNamespace(grad_sink=lambda *a: got.append(a)), 3, "w", leaf)
        return leaf

    first, again = forward(), forward()
    assert first is not again and first.data_ptr() == again.data_ptr()
    (first.sum() + 2 * again.sum()).backward()
    assert len(got) == 2 and all((i, n) == (3, "w") for i, n, _ in got)
    assert sorted(float(g[0, 0]) for _, _, g in got) == [1.0, 2.0]
    assert first.grad is None and again.grad is None


def test_a_shard_that_sinks_twice_counts_once():
    actor = _actor()
    actor._received = set()
    (index, name), p = next(iter(actor._by_leaf.items()))
    actor._sink(index, name, torch.ones_like(p))
    actor._sink(index, name, torch.ones_like(p))
    assert actor._received == {(index, name)}
    assert torch.equal(p.grad, 2 * torch.ones_like(p))
