# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The vLLM-Omni rollout DiT binding (``h3-omni-rollout-v1``).

CPU: vLLM-Omni's packed DiT kwargs are built here as ``MiniMaxH3DenoiseBranch.forward_kwargs``
(one request) and ``minimax_h3_batched_forward_kwargs`` (step mode) build them, from known
Diffusers inputs; the conversion must give those inputs back bit for bit, a batch must split
into the single-request calls, and every refusal must fire before anything changes.
GPU (with the H3 provider): the bound call equals the model's own call on the same inputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rl_engine.integrations.engines.rollout.vllm import h3_omni_model as om


def _request(
    seed: int, *, text: int = 3, video: int = 4, audio: int = 2, cond: int = 1, t=(0.25, 0.4)
):
    """Diffusers inputs of one request, laid out ``[text | video | audio]``."""

    g = torch.Generator().manual_seed(seed)
    tags = torch.tensor([1] * text + [0] * video + [2] * audio, dtype=torch.long)
    used = tags.numel()
    video_idx = torch.arange(text, text + video)
    audio_idx = torch.arange(text + video, used)
    text_idx = torch.arange(text)
    row_t = torch.full((used,), t[0], dtype=torch.float32)
    row_t[video_idx[:cond]] = max(t[0], 0.9)  # condition rows keep their own timestep
    row_t[audio_idx] = t[1]
    timestep, timestep_indices = torch.unique(row_t, sorted=True, return_inverse=True)
    update = torch.ones(video, dtype=torch.bool)
    update[:cond] = False
    return (
        {
            "hidden_states": torch.randn(1, video, 96, generator=g),
            "audio_hidden_states": torch.randn(1, audio, 32, generator=g),
            "encoder_hidden_states": torch.randn(1, text, 5120, generator=g).bfloat16(),
            "timestep": timestep,
            "timestep_indices": timestep_indices,
            "token_tags": tags,
            "position_ids": torch.randn(used, 3, generator=g).double(),
            "video_indices": video_idx,
            "audio_indices": audio_idx,
            "text_indices": text_idx,
            "return_dict": False,
        },
        update,
        row_t,
    )


def _omni(req, update, row_t, pad: int):
    """vLLM-Omni's single-request DiT kwargs for ``req`` with a padding tail of ``pad`` rows."""

    used = req["token_tags"].numel()
    seq = used + pad
    x = torch.zeros(1, seq, 96)
    x[0].index_copy_(0, req["video_indices"], req["hidden_states"][0])
    audio_x = torch.zeros(1, seq, 32)
    audio_x[0].index_copy_(0, req["audio_indices"], req["audio_hidden_states"][0])
    rows = torch.cat([row_t, torch.full((pad,), float(row_t[0]))])  # padding inherits the video t
    unique, inverse = torch.unique(rows, sorted=True, return_inverse=True)
    positions = torch.cat([req["position_ids"], torch.zeros(pad, 3, dtype=torch.float64)])
    text = req["text_indices"].numel()
    return {
        "x": x,
        "audio_x": audio_x,
        "img_position_ids": positions[None],
        "unique_timesteps": unique,
        "inverse_indices": inverse,
        "update_mask": update,
        "token_tags": torch.cat([req["token_tags"], torch.full((pad,), -1, dtype=torch.long)]),
        "skip_mask_out_condition": False,
        "prompt_embeds": req["encoder_hidden_states"][0],
        "img_pos_info": {"position_ids": req["video_indices"]},
        "audio_pos_info": {"position_ids": req["audio_indices"]},
        "text_pos_info": {"position_ids": req["text_indices"]},
        "img_pos_for_infer_output_info": {"position_ids": req["video_indices"]},
        "packed_seq_params": {
            "cu_seqlens_q": torch.tensor([0, used] + ([seq] if pad else []), dtype=torch.int32),
            "max_seqlen_q": max(used, pad),
        },
        "refiner_packed_seq_params": {
            "cu_seqlens_q": torch.tensor([0, text], dtype=torch.int32),
            "max_seqlen_q": text,
        },
    }


def _batched(singles):
    """``minimax_h3_batched_forward_kwargs``'s layout of several single-request kwargs."""

    seq_off = text_off = 0
    cu, refiner = [0], [0]
    parts = {
        k: []
        for k in ("x", "audio_x", "pos", "tags", "update", "text", "img", "aud", "txt", "rows")
    }
    for k in singles:
        seq = k["x"].shape[1]
        b = k["packed_seq_params"]["cu_seqlens_q"].tolist()
        used = b[1]
        parts["x"].append(k["x"][0])
        parts["audio_x"].append(k["audio_x"][0])
        parts["pos"].append(k["img_position_ids"][0])
        parts["tags"].append(k["token_tags"])
        parts["update"].append(k["update_mask"])
        parts["text"].append(k["prompt_embeds"])
        parts["img"].append(k["img_pos_info"]["position_ids"] + seq_off)
        parts["aud"].append(k["audio_pos_info"]["position_ids"] + seq_off)
        parts["txt"].append(k["text_pos_info"]["position_ids"] + seq_off)
        parts["rows"].append(k["unique_timesteps"][k["inverse_indices"]])
        cu.append(seq_off + used)
        if used != seq:
            cu.append(seq_off + seq)
        text_off += k["prompt_embeds"].shape[0]
        refiner.append(text_off)
        seq_off += seq
    unique, inverse = torch.unique(torch.cat(parts["rows"]), sorted=True, return_inverse=True)
    img = torch.cat(parts["img"])
    return {
        "x": torch.cat(parts["x"])[None],
        "audio_x": torch.cat(parts["audio_x"])[None],
        "img_position_ids": torch.cat(parts["pos"])[None],
        "update_mask": torch.cat(parts["update"]),
        "token_tags": torch.cat(parts["tags"]),
        "skip_mask_out_condition": False,
        "prompt_embeds": torch.cat(parts["text"]),
        "img_pos_info": {"position_ids": img},
        "audio_pos_info": {"position_ids": torch.cat(parts["aud"])},
        "text_pos_info": {"position_ids": torch.cat(parts["txt"])},
        "img_pos_for_infer_output_info": {"position_ids": img},
        "packed_seq_params": {
            "cu_seqlens_q": torch.tensor(cu, dtype=torch.int32),
            "max_seqlen_q": 0,
            "num_requests": len(singles),
        },
        "refiner_packed_seq_params": {
            "cu_seqlens_q": torch.tensor(refiner, dtype=torch.int32),
            "max_seqlen_q": 0,
        },
        "unique_timesteps": unique,
        "inverse_indices": inverse,
    }


def _same(a, b) -> bool:
    """Bitwise: dtype, shape and bytes (``torch.equal`` compares values, so ``-0.0 == 0.0``)."""

    from rl_engine.integrations.engines.rollout.vllm import h3_omni_model as om

    return om.tensor_digest(a) == om.tensor_digest(b)


def _equal(a, b):
    assert set(a) == set(b)
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert _same(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


class TestConversion:
    @pytest.mark.parametrize("pad", [0, 5])
    def test_single_request_gives_the_diffusers_inputs_back(self, pad):
        req, update, row_t = _request(1)
        got = om.omni_to_diffusers(_omni(req, update, row_t, pad))
        _equal(got, req)

    def test_padding_timestep_does_not_reach_the_model(self):
        # Padding rows with a timestep no real row has must not add a unique timestep.
        req, update, row_t = _request(2)
        k = _omni(req, update, row_t, 4)
        rows = torch.cat([row_t, torch.full((4,), 0.123)])
        k["unique_timesteps"], k["inverse_indices"] = torch.unique(
            rows, sorted=True, return_inverse=True
        )
        _equal(om.omni_to_diffusers(k), req)

    @pytest.mark.parametrize("pads", [(0, 0), (3, 0), (0, 2), (5, 7, 0)])
    def test_a_batch_splits_into_its_single_requests(self, pads):
        reqs = [
            _request(10 + i, text=2 + i, video=3 + i, t=(0.1 * (i + 1), 0.3))
            for i in range(len(pads))
        ]
        singles = [_omni(r, u, t, p) for (r, u, t), p in zip(reqs, pads)]
        split = om.split_requests(_batched(singles))
        assert len(split) == len(singles)
        for (req, _, _), part in zip(reqs, split):
            _equal(om.omni_to_diffusers(part), req)

    def test_outputs_are_vllm_omnis(self):
        req, update, row_t = _request(3)
        k = _omni(req, update, row_t, 2)
        video, audio = torch.randn(1, 4, 96), torch.randn(1, 2, 32)
        v, a = om.diffusers_to_omni(video, audio, k)
        assert _same(v, video[0] * update.unsqueeze(-1)) and _same(a, audio[0])
        k["skip_mask_out_condition"] = True
        v, _ = om.diffusers_to_omni(video, audio, k)
        assert _same(v, video[0])
        k["skip_mask_out_condition"] = False
        k["update_audio_mask"] = torch.tensor([True, False])
        _, a = om.diffusers_to_omni(video, audio, k)
        assert _same(a, audio[0] * k["update_audio_mask"].unsqueeze(-1))


class TestRefusals:
    def _k(self):
        req, update, row_t = _request(4)
        return _omni(req, update, row_t, 2)

    @pytest.mark.parametrize(
        "change",
        [
            {"reference_kv_state": object()},
            {"control_rows": torch.zeros(1)},
        ],
    )
    def test_calls_the_actor_cannot_reproduce(self, change):
        k = {**self._k(), **change}
        with pytest.raises(ValueError):
            om.check_call(k)

    def test_pdd_and_unknown_kwargs(self):
        with pytest.raises(ValueError):
            om.check_call(self._k(), pdd_step=0)
        with pytest.raises(TypeError):
            om.check_call({**self._k(), "new_kwarg": 1})

    def test_vsa_prefix_segments_are_layout_metadata(self):
        k = self._k()
        k["packed_seq_params"] = {**k["packed_seq_params"], "vsa_prefix_segments": (3, 4)}
        om.check_call(k)  # sent on every call; only a VSA DiT (refused at install) reads them

    def test_zero_scale_control_is_allowed(self):
        om.check_call({**self._k(), "control_rows": torch.zeros(1), "control_context_scale": 0.0})

    def test_layouts_the_conversion_refuses(self):
        k = self._k()
        bad_out = {**k, "img_pos_for_infer_output_info": {"position_ids": torch.arange(4)}}
        with pytest.raises(ValueError):
            om.omni_to_diffusers(bad_out)
        bad_pad = {**k, "token_tags": k["token_tags"].clone()}
        bad_pad["token_tags"][-1] = 0
        with pytest.raises(ValueError):
            om.omni_to_diffusers(bad_pad)
        with pytest.raises(ValueError):
            om.omni_to_diffusers(
                {**k, "packed_seq_params": {**k["packed_seq_params"], "num_requests": 2}}
            )

    def test_engine_config(self):
        class Par:
            tensor_parallel_size, ulysses_degree, ring_degree = 4, 1, 1

        class Cfg:
            parallel_config = Par()
            cache_backend = "none"

        assert om.check_engine_config(Cfg(), tp=4) == []
        assert om.check_engine_config(Cfg(), tp=2)
        Cfg.parallel_config.ulysses_degree = 2
        assert om.check_engine_config(Cfg(), tp=4)
        Cfg.parallel_config.ulysses_degree = 1
        for name, value in (
            ("cache_backend", "tea_cache"),
            ("quantization_config", "fp8"),
            ("lora_path", "x"),
        ):
            cfg = Cfg()
            setattr(cfg, name, value)
            assert om.check_engine_config(cfg, tp=4), name
        cfg = Cfg()
        cfg.diffusion_compile_granularity = "full"
        assert om.check_engine_config(cfg, tp=4)
        cfg.enforce_eager = True
        assert om.check_engine_config(cfg, tp=4) == []


class MiniMaxH3DiTModel(torch.nn.Module):  # stands in for vLLM-Omni's class, by name
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(4, 4))
        self.register_buffer("b", torch.zeros(4))
        self.vdn_config = None

    def forward(self, **kwargs):
        return "engine"


class _Binding:
    grid = type(
        "G", (), {"tp": 4, "cp": 1, "tp_rank": 0, "attention": "h3_full", "tp_collective": None}
    )()

    def forward(self, **kwargs):
        return "bound"


class TestInstall:
    def test_install_replaces_and_remove_restores(self):
        t = MiniMaxH3DiTModel()
        handle = om.install_h3_omni_model(t, _Binding())
        assert t() == "bound"
        handle.remove()
        assert t() == "engine" and "forward" not in t.__dict__

    def test_refusals_leave_the_module_untouched(self):
        class Other(MiniMaxH3DiTModel):
            pass

        for t in (Other(), MiniMaxH3DiTModel()):
            if type(t) is MiniMaxH3DiTModel:
                t.vdn_config = object()
            with pytest.raises(RuntimeError):
                om.install_h3_omni_model(t, _Binding())
            assert "forward" not in t.__dict__ and not t.w.is_meta
        t = MiniMaxH3DiTModel()
        t.to_gate_compress = torch.nn.Linear(2, 2)
        with pytest.raises(RuntimeError, match="sparse attention"):
            om.install_h3_omni_model(t, _Binding())
        t = MiniMaxH3DiTModel()
        t.forward = lambda **k: "compiled"
        with pytest.raises(RuntimeError):
            om.install_h3_omni_model(t, _Binding())

    def test_a_direct_install_checks_the_world_size(self, monkeypatch):
        """Without ``od_config`` (no engine config to read DP from) WORLD == TP is still checked."""

        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: 8)  # DP2 x TP4
        t = MiniMaxH3DiTModel()
        with pytest.raises(RuntimeError, match="world size 8 != TP 4"):
            om.install_h3_omni_model(t, _Binding())
        assert "forward" not in t.__dict__
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: 4)
        om.install_h3_omni_model(t, _Binding()).remove()

    def test_release_moves_the_engine_weights_to_meta(self):
        t = MiniMaxH3DiTModel()
        handle = om.install_h3_omni_model(t, _Binding(), release_weights=True)
        assert handle.released == 16 * 4 + 4 * 4
        assert t.w.is_meta and t.b.is_meta and t() == "bound"


class TestSDPA:
    def test_the_call_gets_torchs_default_backends_back(self):
        cuda = torch.backends.cuda
        before = om.sdpa_backends_enabled()
        cuda.enable_cudnn_sdp(False)  # what importing vLLM does
        try:
            with om.torch_default_sdpa():
                assert all(om.sdpa_backends_enabled().values())
            assert not cuda.cudnn_sdp_enabled()  # put back on exit
        finally:
            cuda.enable_cudnn_sdp(before["cudnn"])


class TestLoRASync:
    class _Model(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(n))

    def _shards(self, tp, rank, seed=0):
        from rl_engine.validation.models import h3_lora

        return [
            h3_lora.shard_block_lora(h3_lora.make_block_lora(i, rank=8, seed=seed), tp, rank)
            for i in range(2)
        ]

    def test_shard_to_shard_sync_checks_every_digest(self):
        from rl_engine.validation.models import h3_lora

        sender, receiver = self._Model(2), self._Model(2)
        shards = self._shards(4, 1)
        sent = h3_lora.install_lora_shards(sender, shards, 4, 1, scale=2.0, device="cpu")
        got = h3_lora.install_lora_shards(
            receiver, shards, 4, 1, scale=2.0, expected=sent, device="cpu"
        )
        assert got == sent and len(got) == 2 * h3_lora.LEAVES_PER_BLOCK

    def test_one_changed_element_is_found_and_nothing_is_kept(self):
        from rl_engine.validation.models import h3_lora
        from rl_engine.validation.models import h3_parallel_model as pm

        sender, receiver = self._Model(2), self._Model(2)
        sent = h3_lora.install_lora_shards(
            sender, self._shards(4, 1), 4, 1, scale=2.0, device="cpu"
        )
        old = h3_lora.install_lora_shards(
            receiver, self._shards(4, 1, seed=5), 4, 1, scale=2.0, device="cpu"
        )
        bad = self._shards(4, 1)
        bad[1]["ff.net.2"]["A"][0, 0] += 1
        with pytest.raises(RuntimeError, match="1 shards"):
            h3_lora.install_lora_shards(receiver, bad, 4, 1, scale=2.0, expected=sent, device="cpu")
        assert h3_lora.lora_digests(receiver) == old
        assert getattr(receiver.transformer_blocks[0], pm.LORA_ATTR)["tp_rank"] == 1

    def test_a_failed_transfer_or_digest_changes_no_block(self, monkeypatch):
        from rl_engine.validation.models import h3_lora

        receiver = self._Model(2)
        old = h3_lora.install_lora_shards(
            receiver, self._shards(4, 1, seed=5), 4, 1, scale=2.0, device="cpu"
        )
        real_to, calls = torch.Tensor.to, {"n": 0}

        def to(self, *args, **kwargs):  # fails in the second block's transfers
            calls["n"] += 1
            if calls["n"] > h3_lora.LEAVES_PER_BLOCK:
                raise torch.cuda.OutOfMemoryError("transfer")
            return real_to(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", to)
        with pytest.raises(torch.cuda.OutOfMemoryError):
            h3_lora.install_lora_shards(receiver, self._shards(4, 1), 4, 1, scale=2.0, device="cpu")
        monkeypatch.setattr(torch.Tensor, "to", real_to)
        assert h3_lora.lora_digests(receiver) == old

        def broken(model):
            raise RuntimeError("digest")

        monkeypatch.setattr(h3_lora, "lora_digests", broken)
        with pytest.raises(RuntimeError, match="digest"):
            h3_lora.install_lora_shards(receiver, self._shards(4, 1), 4, 1, scale=2.0, device="cpu")
        monkeypatch.undo()
        assert h3_lora.lora_digests(receiver) == old

    @pytest.mark.parametrize("file_scale", [None, 1.0])
    def test_worker_sync_refuses_a_scale_other_than_the_senders(
        self, monkeypatch, tmp_path, file_scale
    ):
        from types import SimpleNamespace

        from rl_engine.integrations.engines.rollout.vllm import h3_omni_worker as wk
        from rl_engine.validation.models import h3_lora

        model = self._Model(2)
        model.w = torch.nn.Parameter(torch.zeros(1))
        shards = self._shards(4, 1)
        sent = h3_lora.install_lora_shards(self._Model(2), shards, 4, 1, scale=2.0, device="cpu")
        payload = {"shards": shards, "digests": sent}
        if file_scale is not None:
            payload["scale"] = file_scale
        torch.save(payload, tmp_path / "rank1.pt")
        ext = wk.H3OmniWorkerExtension()
        ext._rlk_h3 = SimpleNamespace(
            binding=SimpleNamespace(model=model, grid=SimpleNamespace(tp=4, tp_rank=1))
        )
        errors = []

        def gather(mine):  # this rank only; its refusal is the failure
            if not mine["ok"]:
                errors.append(mine["error"])
            return [None, mine.get("result"), None, None], ({} if mine["ok"] else {1: "x"})

        monkeypatch.setattr(wk, "_gather", gather)
        with pytest.raises(RuntimeError, match="no rank changed"):
            ext.rlk_h3_load_lora(str(tmp_path / "rank{rank}.pt"), scale=2.0)
        assert "scale" in errors[0]
        assert not h3_lora.lora_digests(model)

    @pytest.mark.parametrize("strict", [True, False])
    def test_a_refiner_lora_is_refused_on_a_strict_binding_before_the_install(
        self, monkeypatch, tmp_path, strict
    ):
        from types import SimpleNamespace

        from rl_engine.integrations.engines.rollout.vllm import h3_omni_worker as wk
        from rl_engine.validation.models import h3_lora

        model = self._Model(2)
        model.w = torch.nn.Parameter(torch.zeros(1))
        payload = {"shards": [], "digests": {"x": "0"}, "scale": 2.0, "refiner": [{}, {}]}
        torch.save(payload, tmp_path / "rank1.pt")
        ext = wk.H3OmniWorkerExtension()
        ext._rlk_h3 = SimpleNamespace(
            binding=SimpleNamespace(
                model=model,
                grid=SimpleNamespace(tp=4, tp_rank=1),
                strict_boundaries=strict,
            )
        )
        errors, installs = [], []

        def gather(mine):  # this rank only
            if not mine["ok"]:
                errors.append(mine["error"])
            return [None, mine.get("result"), None, None], ({} if mine["ok"] else {1: "x"})

        monkeypatch.setattr(wk, "_gather", gather)
        monkeypatch.setattr(
            h3_lora,
            "install_lora_shards",
            lambda *a, **kw: installs.append(("install", kw["refiner"])),
        )
        monkeypatch.setattr(
            h3_lora, "attach_lora", lambda *a, **kw: installs.append(("attach", kw["refiner"]))
        )
        monkeypatch.setattr(h3_lora, "lora_digests", lambda m: {})
        source = str(tmp_path / "rank{rank}.pt")
        if strict:
            with pytest.raises(RuntimeError, match="no rank changed"):
                ext.rlk_h3_load_lora(source, scale=2.0)
            with pytest.raises(RuntimeError, match="rlk_h3_attach_seeded_lora failed"):
                ext.rlk_h3_attach_seeded_lora(refiner=True)
            assert installs == []
            assert len(errors) == 2 and all("strict refiner row" in e for e in errors)
        else:
            ext.rlk_h3_load_lora(source, scale=2.0)
            ext.rlk_h3_attach_seeded_lora(refiner=True)
            assert installs == [("install", [{}, {}]), ("attach", True)]
            assert errors == []

    def test_shards_without_a_are_refused_as_value_errors(self):
        from rl_engine.validation.models import h3_lora

        bad = self._shards(4, 1)
        del bad[1]["attn.to_q"]["A"]
        with pytest.raises(ValueError, match="have no A"):
            h3_lora.install_lora_shards(self._Model(2), bad, 4, 1, scale=2.0, device="cpu")
        bad = self._shards(4, 1)
        bad[0]["ff.net.2"]["B"] = [0.0]
        with pytest.raises(ValueError, match="not tensors"):
            h3_lora.install_lora_shards(self._Model(2), bad, 4, 1, scale=2.0, device="cpu")

    @pytest.mark.parametrize("other_rank_fails", [False, True], ids=["all-ok", "rank2-refused"])
    def test_worker_sync_is_all_ranks_or_none(self, monkeypatch, tmp_path, other_rank_fails):
        from types import SimpleNamespace

        from rl_engine.integrations.engines.rollout.vllm import h3_omni_worker as wk
        from rl_engine.validation.models import h3_lora

        model = self._Model(2)
        model.w = torch.nn.Parameter(torch.zeros(1))  # the shards go to the model's device
        old = h3_lora.install_lora_shards(
            model, self._shards(4, 1, seed=5), 4, 1, scale=2.0, device="cpu"
        )
        shards = self._shards(4, 1)
        sent = h3_lora.install_lora_shards(self._Model(2), shards, 4, 1, scale=2.0, device="cpu")
        torch.save({"shards": shards, "digests": sent, "scale": 2.0}, tmp_path / "rank1.pt")
        ext = wk.H3OmniWorkerExtension()
        ext._rlk_h3 = SimpleNamespace(
            binding=SimpleNamespace(model=model, grid=SimpleNamespace(tp=4, tp_rank=1))
        )

        def gather(mine):  # this rank is 1; another rank reports its refusal
            assert mine["ok"]
            failed = {2: "RuntimeError: LoRA sync digest mismatch"} if other_rank_fails else {}
            return [None, mine["result"], None, None], failed

        monkeypatch.setattr(wk, "_gather", gather)
        source = str(tmp_path / "rank{rank}.pt")
        if other_rank_fails:
            with pytest.raises(RuntimeError, match="no rank changed"):
                ext.rlk_h3_load_lora(source, scale=2.0)
            assert h3_lora.lora_digests(model) == old  # installed here, then put back
        else:
            assert ext.rlk_h3_load_lora(source, scale=2.0)[1] == sent
            assert h3_lora.lora_digests(model) == sent

    def test_zero_or_mixed_rank_adapters_are_refused(self):
        from rl_engine.validation.models import h3_lora

        zero = [
            h3_lora.shard_block_lora(h3_lora.make_block_lora(i, rank=0), 4, 1) for i in range(2)
        ]
        with pytest.raises(ValueError, match="r > 0"):
            h3_lora.install_lora_shards(self._Model(2), zero, 4, 1, scale=2.0, device="cpu")
        mixed = self._shards(4, 1)
        mixed[1] = h3_lora.shard_block_lora(h3_lora.make_block_lora(1, rank=4), 4, 1)
        with pytest.raises(ValueError, match="one rank"):
            h3_lora.install_lora_shards(self._Model(2), mixed, 4, 1, scale=2.0, device="cpu")

    @pytest.mark.parametrize("digests", [None, {}, ["not", "a", "mapping"]])
    def test_a_file_without_sender_digests_is_refused_on_every_rank(
        self, monkeypatch, tmp_path, digests
    ):
        from types import SimpleNamespace

        from rl_engine.integrations.engines.rollout.vllm import h3_omni_worker as wk
        from rl_engine.validation.models import h3_lora

        model = self._Model(2)
        model.w = torch.nn.Parameter(torch.zeros(1))
        old = h3_lora.install_lora_shards(
            model, self._shards(4, 1, seed=5), 4, 1, scale=2.0, device="cpu"
        )
        torch.save({"shards": self._shards(4, 1), "digests": digests}, tmp_path / "rank1.pt")
        ext = wk.H3OmniWorkerExtension()
        ext._rlk_h3 = SimpleNamespace(
            binding=SimpleNamespace(model=model, grid=SimpleNamespace(tp=4, tp_rank=1))
        )

        def gather(mine):  # this rank is 1, the others accept theirs
            failed = {} if mine["ok"] else {1: mine["error"]}
            return [None, mine.get("result"), None, None], failed

        monkeypatch.setattr(wk, "_gather", gather)
        with pytest.raises(RuntimeError, match="no rank changed"):
            ext.rlk_h3_load_lora(str(tmp_path / "rank{rank}.pt"), scale=2.0)
        assert h3_lora.lora_digests(model) == old

    def test_shards_of_another_rank_layout_are_refused(self):
        from rl_engine.validation.models import h3_lora

        with pytest.raises(ValueError):
            h3_lora.install_lora_shards(
                self._Model(2), self._shards(2, 0), 4, 0, scale=2.0, device="cpu"
            )
        with pytest.raises(ValueError):
            h3_lora.install_lora_shards(
                self._Model(3), self._shards(4, 0), 4, 0, scale=2.0, device="cpu"
            )


# --- GPU: the bound call is the actor's model call -----------------------------------------------


@pytest.fixture(scope="module")
def full_model():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("triton")
    pytest.importorskip("diffusers", reason="the H3 provider (Diffusers with MiniMax-H3)")
    from rl_engine.validation.models import h3_sharded_load as sl
    from rl_engine.validation.models.h3_weights import h3_shard_root

    root = h3_shard_root()
    if root is None:
        pytest.skip("released H3 shards ($RL_KERNEL_H3_WEIGHTS) required")
    model = sl.load_sharded_model(root, 1, 0, "cuda").eval()  # the harness runs the 50 blocks
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _cuda(d):
    return {
        k: (v.cuda() if isinstance(v, torch.Tensor) else _cuda(v) if isinstance(v, dict) else v)
        for k, v in d.items()
    }


def _direct(model, inputs, strict):
    from rl_engine.validation.models import h3_parallel_model as pm
    from rl_engine.validation.models import h3_tp
    from rl_engine.validation.models.h3_full_model import FullModelTrace

    grid = h3_tp.Grid(1, 1, 0, 0, None, None, "h3_full", "ring")
    with (
        torch.no_grad(),
        pm.parallel_model_execution(
            model, grid, sp=False, trace=FullModelTrace(), strict_boundaries=strict
        ),
    ):
        return model(**inputs)


@pytest.mark.parametrize(
    "with_lora,vllm_sdpa",
    [(False, False), (True, False), (False, True)],
    ids=["base", "lora", "base-cudnn-sdpa-off"],
)
def test_bound_call_is_the_models_call(full_model, with_lora, vllm_sdpa):
    from rl_engine.validation.models import h3_lora
    from rl_engine.validation.models import h3_parallel_model as pm
    from rl_engine.validation.models import h3_tp

    model = full_model
    strict = torch.cuda.get_device_capability() == (9, 0)  # strict video boundaries need SM90
    cudnn = torch.backends.cuda.cudnn_sdp_enabled()
    if with_lora:
        h3_lora.attach_lora(model, 1, 0, rank=64, seed=3)
    try:
        grid = h3_tp.Grid(1, 1, 0, 0, None, None, "h3_full", "ring")
        binding = om.H3OmniModelBinding(model, grid, strict_boundaries=strict)
        reqs = [
            _request(20 + i, text=5 + i, video=8 + 4 * i, audio=6, t=(0.3 + 0.1 * i, 0.5))
            for i in range(3)
        ]
        singles = [_cuda(_omni(r, u, t, p)) for (r, u, t), p in zip(reqs, (7, 0, 3))]
        alone = []
        for (req, update, _), k in zip(reqs, singles):
            torch.backends.cuda.enable_cudnn_sdp(not vllm_sdpa)  # vLLM's import turns it off
            v, a = binding.forward(**k)
            with om.torch_default_sdpa():  # the reference on torch's default backends
                dv, da = _direct(model, _cuda(req), strict)
            assert _same(v, dv[0] * update.cuda().unsqueeze(-1)) and _same(a, da[0])
            alone.append((v, a))
        v, a = binding.forward(**_batched(singles))
        assert _same(v, torch.cat([x for x, _ in alone]))
        assert _same(a, torch.cat([y for _, y in alone]))
    finally:
        torch.backends.cuda.enable_cudnn_sdp(cudnn)
        for block in model.transformer_blocks:
            block.__dict__.pop(pm.LORA_ATTR, None)


# --- CPU: install / remove lifecycle of the worker extension -------------------------------------


class TestInstallLifecycle:
    """``rlk_h3_install`` binds every TP rank or none; a released DiT is never run again."""

    class MiniMaxH3DiTModel(torch.nn.Module):  # the name check_engine_transformer accepts
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(3))

        def forward(self, **kwargs):
            return self.w * 2

    class _Bound(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity()])
            self.p = torch.nn.Parameter(torch.zeros(2))

    class _Manager:
        def __init__(self):
            self.calls = []

        def set_request_adapter(self, request, scale=1.0):
            self.calls.append((request, scale))

    def _ext(self, monkeypatch, failures, tp_rank=1, load_error=None):
        import sys
        from types import ModuleType, SimpleNamespace

        from rl_engine.integrations.engines.rollout.vllm import h3_omni_model as om
        from rl_engine.integrations.engines.rollout.vllm import h3_omni_worker as wk
        from rl_engine.validation.models import h3_sharded_load as sl

        group = SimpleNamespace(world_size=4, rank_in_group=tp_rank, device_group=None)
        ps = ModuleType("vllm.distributed.parallel_state")
        ps.get_tp_group = lambda: group
        for name in ("vllm", "vllm.distributed"):
            monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or ModuleType(name))
        monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", ps)
        grid = SimpleNamespace(tp=4, tp_rank=tp_rank, cp=1, attention="h3_full", tp_collective=None)
        monkeypatch.setattr(om, "tp_grid_from_group", lambda *a, **k: grid)
        monkeypatch.setattr(wk, "_preflight_checkpoint", lambda d: 1)

        def load(*args, **kwargs):
            if load_error:
                raise OSError(load_error)
            return self._Bound()

        monkeypatch.setattr(sl, "load_sharded_model", load)
        phases = iter(failures)  # the other ranks' failures, per gathered phase

        def gather(mine):
            failed = dict(next(phases, {}))
            if not mine["ok"]:
                failed[tp_rank] = mine["error"]
            return [mine.get("result")] * 4, failed

        monkeypatch.setattr(wk, "_gather", gather)
        ext = wk.H3OmniWorkerExtension()
        ext.model_runner = SimpleNamespace(
            pipeline=SimpleNamespace(transformer=self.MiniMaxH3DiTModel(), od_config=None)
        )
        ext.lora_manager = self._Manager()
        return ext

    def test_a_refusal_on_any_rank_changes_nothing(self, monkeypatch):
        ext = self._ext(monkeypatch, [{2: "RuntimeError: cannot bind"}])
        dit = ext.model_runner.pipeline.transformer
        with pytest.raises(RuntimeError, match="no rank changed"):
            ext.rlk_h3_install("ckpt", release_weights=True)
        assert ext._rlk_h3 is None and "forward" not in dit.__dict__ and not dit.w.is_meta

    def test_a_failed_load_without_release_can_be_retried(self, monkeypatch):
        ext = self._ext(monkeypatch, [{}, {3: "OSError: missing shard"}])
        dit = ext.model_runner.pipeline.transformer
        with pytest.raises(RuntimeError, match="no rank changed"):
            ext.rlk_h3_install("ckpt", release_weights=False)
        assert ext._rlk_h3 is None and "forward" not in dit.__dict__ and not dit.w.is_meta
        assert len(ext.rlk_h3_install("ckpt")) == 4  # a retry on the same worker binds
        assert ext._rlk_h3 is not None and dit.__dict__["forward"] is not None

    @pytest.mark.parametrize("release", [False, True])
    def test_a_failed_bind_says_whether_to_retry_or_restart(self, monkeypatch, release):
        """Check, load, grid and bind are gathered in turn; here another rank fails the bind."""

        ext = self._ext(monkeypatch, [{}, {}, {}, {2: "RuntimeError: report failed"}])
        dit = ext.model_runner.pipeline.transformer
        with pytest.raises(RuntimeError, match="restart the engine" if release else "be retried"):
            ext.rlk_h3_install("ckpt", release_weights=release)
        assert ext._rlk_h3 is None and dit.w.is_meta == release
        if release:
            with pytest.raises(RuntimeError, match="no usable DiT"):
                dit()
        else:
            assert "forward" not in dit.__dict__
            assert len(ext.rlk_h3_install("ckpt")) == 4  # and it can

    @pytest.mark.parametrize("where", ["here", "another rank"])
    def test_a_failed_load_after_release_disables_the_dit(self, monkeypatch, where):
        if where == "here":
            ext = self._ext(monkeypatch, [{}, {}], load_error="bad checkpoint")
        else:
            ext = self._ext(monkeypatch, [{}, {0: "OSError: bad checkpoint"}])
        dit = ext.model_runner.pipeline.transformer
        with pytest.raises(RuntimeError, match="restart the engine"):
            ext.rlk_h3_install("ckpt", release_weights=True)
        assert ext._rlk_h3 is None and dit.w.is_meta
        with pytest.raises(RuntimeError, match="no usable DiT"):
            dit()  # not vLLM-Omni's forward on meta tensors

    @pytest.mark.parametrize("release", [False, True])
    def test_a_failed_grid_on_another_rank_fails_every_rank(self, monkeypatch, release):
        """The grid and its collective are built in a gathered phase: a rank that fails there
        (out of memory, collective setup) is reported on every rank, and no rank binds."""

        ext = self._ext(monkeypatch, [{}, {}, {3: "OutOfMemoryError: staging"}])
        dit = ext.model_runner.pipeline.transformer
        with pytest.raises(RuntimeError, match=r"TP ranks \{3: 'OutOfMemoryError: staging'\}"):
            ext.rlk_h3_install("ckpt", release_weights=release)
        assert ext._rlk_h3 is None and dit.w.is_meta == release
        if not release:
            assert "forward" not in dit.__dict__ and len(ext.rlk_h3_install("ckpt")) == 4

    def test_the_grid_collective_staging_is_what_it_allocates(self):
        from rl_engine.distributed.algorithms import collectives as col
        from rl_engine.integrations.engines.rollout.vllm import h3_omni_model as om

        assert om.grid_collective_staging_bytes(1) == 0
        per_frame = om.GRID_COLLECTIVE_BYTES + col._COLLECTIVE_FRAME_METADATA_BYTES
        assert om.grid_collective_staging_bytes(4) == col._COLLECTIVE_STAGING_FRAMES * per_frame

    def test_bound_engine_refuses_request_lora_and_remove_after_release(self, monkeypatch):
        ext = self._ext(monkeypatch, [])
        dit = ext.model_runner.pipeline.transformer
        replies = ext.rlk_h3_install("ckpt", release_weights=True)
        assert len(replies) == 4 and replies[1]["engine_parameter_bytes_released"] == 12
        manager = ext.lora_manager
        manager.set_request_adapter(None)  # deactivating is fine
        assert manager.calls == [(None, 1.0)]
        with pytest.raises(ValueError, match="request LoRA"):
            manager.set_request_adapter(object(), 1.0)
        with pytest.raises(RuntimeError, match="released"):
            ext.rlk_h3_remove()
        assert ext._rlk_h3 is not None and dit.__dict__["forward"] is not None  # nothing changed

    def test_remove_without_release_restores_everything(self, monkeypatch):
        ext = self._ext(monkeypatch, [])
        dit = ext.model_runner.pipeline.transformer
        ext.rlk_h3_install("ckpt", release_weights=False)
        ext.rlk_h3_remove()
        assert ext._rlk_h3 is None and "forward" not in dit.__dict__
        assert "set_request_adapter" not in ext.lora_manager.__dict__
        assert torch.equal(dit(), torch.full((3,), 2.0))

    def test_a_failed_capture_is_reported_not_raised(self, monkeypatch, tmp_path):
        ext = self._ext(monkeypatch, [], tp_rank=0)
        ext.rlk_h3_install("ckpt", record_dir=str(tmp_path), record_calls=(0,))
        (tmp_path / "call0.pt").mkdir()  # torch.save cannot write there
        binding = ext._rlk_h3.binding
        binding.record({}, {"x": torch.zeros(1)}, (torch.zeros(1), torch.zeros(1)))
        assert binding.record_errors and binding.record_errors[0].startswith("call0:")
        assert ext._rlk_h3.report()["record_errors"] == binding.record_errors

    def test_a_failed_input_copy_is_reported_and_the_next_call_recorded(
        self, monkeypatch, tmp_path
    ):
        class _NoCopy(torch.Tensor):
            def cpu(self, *args, **kwargs):
                raise RuntimeError("copy failed")

        ext = self._ext(monkeypatch, [], tp_rank=0)
        ext.rlk_h3_install("ckpt", record_dir=str(tmp_path), record_calls=(0, 1))
        binding = ext._rlk_h3.binding
        out = (torch.zeros(1), torch.zeros(1))
        binding.record({}, {"x": torch.zeros(1).as_subclass(_NoCopy)}, out)
        assert binding.record_errors == ["call0: RuntimeError: copy failed"]
        binding.record({}, {"x": torch.zeros(1)}, out)
        assert (tmp_path / "call1.pt").exists() and len(binding.record_errors) == 1

    def test_a_record_dir_that_cannot_be_made_is_a_gathered_refusal(self, monkeypatch, tmp_path):
        (tmp_path / "file").write_bytes(b"")
        ext = self._ext(monkeypatch, [], tp_rank=0)
        dit = ext.model_runner.pipeline.transformer
        with pytest.raises(RuntimeError, match="no rank changed"):
            ext.rlk_h3_install(
                "ckpt", release_weights=True, record_dir=str(tmp_path / "file" / "calls")
            )
        assert ext._rlk_h3 is None and not dit.w.is_meta  # refused before the release

    def test_data_parallelism_is_refused(self):
        from types import SimpleNamespace

        from rl_engine.integrations.engines.rollout.vllm import h3_omni_model as om

        config = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=4, data_parallel_size=2)
        )
        assert any("data_parallel_size" in r for r in om.check_engine_config(config, tp=4))
        ok = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=4, data_parallel_size=1)
        )
        assert om.check_engine_config(ok, tp=4) == []


class TestRolloutReportGates:
    """The G3 tool's gates: every batched call has its alone twin; replays read this run only."""

    def test_every_batched_call_needs_an_alone_twin(self):
        from tools.validation.models.h3_omni_rollout_report import batch_twins

        def call(inputs, n):
            return {"inputs": inputs, "num_requests": n}

        ok = [call("a", 1), call("b", 1), call("a", 2), call("b", 2)]
        assert batch_twins(ok) == (2, 2)
        drifted = [
            call("a", 1),
            call("b", 1),
            call("a", 2),
            call("b'", 2),
        ]  # B's batched input differs
        assert batch_twins(drifted) == (2, 1)
        assert batch_twins([call("a", 1)]) == (0, 0)

    def test_replays_read_only_this_runs_listed_captures(self, tmp_path):
        from types import SimpleNamespace

        from rl_engine.integrations.engines.rollout.vllm import h3_omni_model as om
        from tools.validation.models import h3_omni_rollout_report as tool

        calls = tmp_path / "calls"
        calls.mkdir()
        for n in range(3):  # an earlier run's captures
            (calls / f"call{n}.pt").write_bytes(b"old")
        assert tool._fresh_dir(calls) == calls and not list(calls.glob("call*.pt"))
        video = torch.arange(4.0)
        torch.save({"video": video, "audio": torch.zeros(2)}, calls / "call0.pt")
        args = SimpleNamespace(out=tmp_path)
        listed = [
            {
                "call": "call0",
                "video_digest": om.tensor_digest(video),
                "sha256": tool._file_sha256(calls / "call0.pt"),
            }
        ]
        ((path, rec),) = tool._listed_captures(args, {"pass": True, "captures": listed})
        assert path == calls / "call0.pt" and torch.equal(rec["video"], video)
        with pytest.raises(RuntimeError, match="not a passing"):
            tool._listed_captures(args, {"pass": False, "captures": listed})
        with pytest.raises(RuntimeError, match="lists no captures"):
            tool._listed_captures(args, {"pass": True})
        with pytest.raises(RuntimeError, match="missing"):
            tool._listed_captures(args, {"pass": True, "captures": listed + [{"call": "call1"}]})
        torch.save({"video": video, "audio": torch.ones(2)}, calls / "call0.pt")  # same video
        with pytest.raises(RuntimeError, match="does not match"):
            tool._listed_captures(args, {"pass": True, "captures": listed})
        torch.save({"video": video + 1}, calls / "call0.pt")  # a later run's capture
        with pytest.raises(RuntimeError, match="does not match"):
            tool._listed_captures(args, {"pass": True, "captures": listed})
        # the valid capture again: only the manifest's missing sha256 can refuse it now
        torch.save({"video": video, "audio": torch.zeros(2)}, calls / "call0.pt")
        assert tool._listed_captures(args, {"pass": True, "captures": listed})
        without_sha = [{k: v for k, v in listed[0].items() if k != "sha256"}]
        with pytest.raises(RuntimeError, match="does not match"):  # an engine.json before sha256
            tool._listed_captures(args, {"pass": True, "captures": without_sha})

    def test_captures_are_hashed_as_loaded_and_compared_by_bytes(self, tmp_path):
        from tools.validation.models import h3_omni_rollout_report as tool

        path = tmp_path / "call0.pt"
        torch.save({"audio": torch.tensor([0.0, 1.0])}, path)
        rec, sha256 = tool._load_capture(path)
        assert sha256 == tool._file_sha256(path) and torch.equal(
            rec["audio"], torch.tensor([0.0, 1.0])
        )
        # torch.equal says these are equal; their bytes are not (the sign of zero)
        assert torch.equal(torch.tensor([-0.0]), torch.tensor([0.0]))
        assert not tool._same_bytes(torch.tensor([-0.0]), torch.tensor([0.0]))
        assert not tool._same_bytes(torch.zeros(2), torch.zeros(2, dtype=torch.float64))
        assert tool._same_bytes(torch.tensor([1.0, -0.0]), torch.tensor([1.0, -0.0]))

    def test_native_inputs_compare_every_tensor_kwarg(self):
        from tools.validation.models import h3_omni_rollout_report as tool

        a = {"x": torch.ones(2), "prompt_embeds": torch.zeros(3), "flag": True}
        assert tool._same_tensor_inputs(a, dict(a))
        assert not tool._same_tensor_inputs(a, {**a, "prompt_embeds": torch.ones(3)})
        assert not tool._same_tensor_inputs(a, {k: v for k, v in a.items() if k != "prompt_embeds"})

    @pytest.mark.parametrize(
        "argv, ok",
        [
            (["--lora-rank", "2"], False),
            (["--lora-rank", "-1"], False),
            (["--lora-rank", "4"], True),
            (["--lora-rank", "0"], True),
            (["--stage", "native"], False),
        ],
    )
    def test_arguments_are_checked_before_an_engine_starts(self, tmp_path, argv, ok):
        import argparse

        from tools.validation.models import h3_omni_rollout_report as tool

        ap = argparse.ArgumentParser()
        ap.add_argument("--stage", default="engine")
        ap.add_argument("--lora-rank", type=int, default=0)
        ap.add_argument("--compare-calls", type=Path)
        args = ap.parse_args(argv)
        if ok:
            tool.check_args(ap, args)
        else:
            with pytest.raises(SystemExit):
                tool.check_args(ap, args)
        args = ap.parse_args(["--stage", "native", "--compare-calls", str(tmp_path)])
        tool.check_args(ap, args)  # an existing directory is accepted

    def test_an_engine_run_removes_the_earlier_reports(self, tmp_path):
        from tools.validation.models import h3_omni_rollout_report as tool

        for name in tool.STAGE_REPORTS:
            (tmp_path / name).write_text("{}")
        tool._invalidate_reports(tmp_path)
        assert not any((tmp_path / name).exists() for name in tool.STAGE_REPORTS)
        tool._invalidate_reports(tmp_path)  # nothing to remove is fine
