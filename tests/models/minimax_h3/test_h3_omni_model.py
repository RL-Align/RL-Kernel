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


def _equal(a, b):
    assert set(a) == set(b)
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k
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
        assert torch.equal(v, video[0] * update.unsqueeze(-1)) and torch.equal(a, audio[0])
        k["skip_mask_out_condition"] = True
        v, _ = om.diffusers_to_omni(video, audio, k)
        assert torch.equal(v, video[0])
        k["skip_mask_out_condition"] = False
        k["update_audio_mask"] = torch.tensor([True, False])
        _, a = om.diffusers_to_omni(video, audio, k)
        assert torch.equal(a, audio[0] * k["update_audio_mask"].unsqueeze(-1))


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
            torch.backends.cuda.enable_cudnn_sdp(cudnn)  # the actor's process: torch's default
            dv, da = _direct(model, _cuda(req), strict)
            assert torch.equal(v, dv[0] * update.cuda().unsqueeze(-1)) and torch.equal(a, da[0])
            alone.append((v, a))
        v, a = binding.forward(**_batched(singles))
        assert torch.equal(v, torch.cat([x for x, _ in alone]))
        assert torch.equal(a, torch.cat([y for _, y in alone]))
    finally:
        torch.backends.cuda.enable_cudnn_sdp(cudnn)
        for block in model.transformer_blocks:
            block.__dict__.pop(pm.LORA_ATTR, None)
