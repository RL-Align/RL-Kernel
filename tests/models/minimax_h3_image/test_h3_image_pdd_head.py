# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``h3_image_pdd_head_contract`` (RFC #738): six FP32 heads, head ``i`` at step ``i`` only.

Pins:

* the fused file and the Diffusers layout load to the released heads, hash for hash
  (``RL_KERNEL_H3_IMAGE_ROOT``), and survive a serialization round trip (I-C11);
* probe I1 (previous / next / shared head) and probe I2 (missing, reordered, cast, altered
  heads) fail closed;
* the head index does not drift with batch position, rank, activation checkpointing or loop
  state: each has a test where the naive code would drift;
* the projection equals the FP32 reference within the row's tolerance and is invariant to
  batch position on CUDA.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch

from rl_engine.models.minimax_h3_image import fingerprint as fp
from rl_engine.models.minimax_h3_image import pdd_head as ph
from rl_engine.models.minimax_h3_image import pdd_schedule as ps

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
ROOT_ENV = "RL_KERNEL_H3_IMAGE_ROOT"


def _tensors(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        fp.PDD_WEIGHT: torch.randn(ph.WEIGHT_SHAPE, generator=g) * 0.02,
        fp.PDD_BIAS: torch.randn(ph.BIAS_SHAPE, generator=g) * 0.02,
    }


def _heads(seed=0):
    return ph.from_fused(_tensors(seed), sealed=False)


def _bits(t):
    return t.detach().cpu().contiguous().view(torch.int32)


def _same(a, b):
    return torch.equal(_bits(a), _bits(b))


@pytest.fixture(scope="module")
def root():
    value = os.environ.get(ROOT_ENV)
    if not value:
        pytest.skip(f"{ROOT_ENV} not set (a MiniMax-H3-Image checkout)")
    return value


# --- artifact ----------------------------------------------------------------------------------


def test_shapes_match_the_fingerprint_table():
    table = fp.expected_pdd_table(fp.transformer_config())
    assert table[fp.PDD_WEIGHT] == ("F32", ph.WEIGHT_SHAPE)
    assert table[fp.PDD_BIAS] == ("F32", ph.BIAS_SHAPE)
    assert len(set(ph.released_digests())) == ph.NUM_HEADS == 6


def test_released_file_loads_sealed(root):
    heads = ph.from_fused_file(root)
    assert heads.sealed and heads.digests == ph.released_digests()
    heads.verify()
    artifact = heads.artifact()
    assert [h["head"] for h in artifact["heads"]] == list(range(6))
    assert artifact["sha256"] == ph.from_fused_file(root).artifact()["sha256"]


def test_serialization_round_trip_keeps_every_byte(root, tmp_path):
    from safetensors.torch import save_file

    heads = ph.from_fused_file(root)
    save_file(heads.fused_state_dict(), str(tmp_path / fp.PDD_FILE))
    again = ph.from_fused_file(tmp_path)
    assert again.digests == heads.digests
    assert again.artifact()["sha256"] == heads.artifact()["sha256"]


def test_diffusers_layout_is_the_same_bank(root):
    heads = ph.from_fused_file(root)
    mapping = {f"pdd_proj_out.{i}.weight": w for i, w in enumerate(heads.weights)}
    mapping.update({f"pdd_proj_out.{i}.bias": b for i, b in enumerate(heads.biases)})
    mapping["proj_in.weight"] = torch.zeros(1)
    bank = ph.from_diffusers(mapping)
    assert bank.digests == heads.digests
    assert all(a is b for a, b in zip(bank.weights, heads.weights))  # no copies


def test_released_reordered_heads_are_named(root):
    tensors = ph.from_fused_file(root).fused_state_dict()
    order = [1, 0, 2, 3, 4, 5]
    swapped = {k: v[order].contiguous() for k, v in tensors.items()}
    with pytest.raises(ph.HeadArtifactError, match="head 0 .*released head 1"):
        ph.from_fused(swapped)


@CUDA
def test_released_heads_round_trip_the_gpu(root):
    assert ph.from_fused_file(root).to("cuda").digests == ph.released_digests()


def test_unreleased_heads_are_refused_when_sealed():
    with pytest.raises(ph.HeadArtifactError, match="head 0 is not released head 0"):
        ph.from_fused(_tensors())


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda t: t.pop(fp.PDD_BIAS), "holds"),
        (lambda t: t.update(extra=torch.zeros(1)), "holds"),
        (lambda t: t.update({fp.PDD_WEIGHT: t[fp.PDD_WEIGHT][:5]}), "must be FP32"),
        (lambda t: t.update({fp.PDD_WEIGHT: t[fp.PDD_WEIGHT].transpose(1, 2)}), "must be FP32"),
        (lambda t: t.update({fp.PDD_WEIGHT: t[fp.PDD_WEIGHT].bfloat16()}), "must be FP32"),
        (lambda t: t.update({fp.PDD_BIAS: t[fp.PDD_BIAS].double()}), "must be FP32"),
    ],
    ids=["missing-bias", "extra-tensor", "five-heads", "transposed", "bf16-weight", "fp64-bias"],
)
def test_malformed_fused_files_fail(edit, message):
    tensors = _tensors()
    edit(tensors)
    with pytest.raises(ph.HeadArtifactError, match=message):
        ph.from_fused(tensors, sealed=False)


@pytest.mark.parametrize("change", ["gap", "seventh", "no-bias", "bf16"])
def test_malformed_diffusers_layouts_fail(change):
    heads = _heads()
    mapping = {f"pdd_proj_out.{i}.weight": w for i, w in enumerate(heads.weights)}
    mapping.update({f"pdd_proj_out.{i}.bias": b for i, b in enumerate(heads.biases)})
    if change == "gap":
        del mapping["pdd_proj_out.2.weight"], mapping["pdd_proj_out.2.bias"]
    elif change == "seventh":
        mapping["pdd_proj_out.6.weight"] = heads.weights[0]
    elif change == "no-bias":
        del mapping["pdd_proj_out.5.bias"]
    else:
        mapping["pdd_proj_out.0.weight"] = heads.weights[0].bfloat16()
    with pytest.raises(ph.HeadArtifactError):
        ph.from_diffusers(mapping, sealed=False)


# --- selection (I1) ----------------------------------------------------------------------------


def test_step_selects_its_own_head_and_interval():
    heads = _heads()
    sigmas = heads.schedule.sigmas
    for step in range(6):
        s = heads.select(step)
        assert s.head == step and _same(s.weight, heads.weights[step])
        assert (s.sigma, s.sigma_next) == (float(sigmas[step]), float(sigmas[step + 1]))
        assert not s.sealed and s.artifact()["head"] == step


@pytest.mark.parametrize("step", [-1, 6, True, 0.0, 1.0, "0", None, torch.tensor(1)])
def test_only_plain_grid_steps_select(step):
    with pytest.raises(ps.OffGridTimestep):
        _heads().select(step)


def test_timestep_is_a_cross_check_and_never_a_lookup():
    heads = _heads()
    for step, t in enumerate(heads.schedule.timesteps):
        assert heads.select(step, timestep=t).head == step
    t2 = heads.schedule.timesteps[2]
    with pytest.raises(ph.HeadDrift, match="timestep of step 2"):
        heads.select(1, timestep=t2)  # previous head for step 2's timestep
    with pytest.raises(ph.HeadDrift, match="timestep of step 2"):
        heads.select(3, timestep=t2)  # next head
    near = torch.nextafter(t2, torch.tensor(2.0))
    with pytest.raises(ps.OffGridTimestep):
        heads.select(2, timestep=near)  # one ulp off the grid is not "nearest head 2"


def test_sigma_interval_is_checked_bit_for_bit():
    heads = _heads()
    s = heads.schedule.sigmas
    assert heads.select(3, sigmas=(s[3], s[4])).head == 3
    with pytest.raises(ph.HeadDrift, match="covers sigmas"):
        heads.select(3, sigmas=(s[2], s[3]))
    with pytest.raises(ph.HeadDrift, match="covers sigmas"):
        heads.select(3, sigmas=(s[3], torch.nextafter(s[4], torch.tensor(0.0))))


# --- batch -------------------------------------------------------------------------------------


def test_one_forward_runs_one_head():
    assert ph.common_step([4, 4, 4]) == 4
    with pytest.raises(ph.HeadDrift, match=r"steps \[1, 2\]"):
        ph.common_step([1, 2, 1])
    with pytest.raises(ph.HeadDrift, match="empty"):
        ph.common_step([])
    with pytest.raises(ps.OffGridTimestep):
        ph.common_step([0, True])


# --- checkpointing -----------------------------------------------------------------------------


def test_recompute_runs_the_bound_head_after_the_loop_moved_on():
    """A late-binding closure would recompute every step with the last head."""

    from torch.utils.checkpoint import checkpoint

    heads = _heads()
    x = torch.randn(3, ph.HIDDEN, generator=torch.Generator().manual_seed(1))

    def grads(wrap, late_binding):
        leaf = x.clone().requires_grad_()
        total = torch.zeros(())
        projection = ph.head_projection("cpu", "reference")
        current = {}
        for step in range(6):
            bound = heads.bind(step, projection)
            current["bound"] = bound

            def call(h, bound=bound):
                return (current["bound"] if late_binding else bound)(h)

            out = checkpoint(call, leaf, use_reentrant=False) if wrap else call(leaf)
            total = total + (out * (step + 1)).square().sum()
        total.backward()
        return leaf.grad

    plain = grads(False, False)
    assert _same(grads(True, False), plain)
    assert not _same(grads(True, True), plain)  # the drift the contract prevents


def test_head_changed_between_forward_and_recompute_is_refused():
    heads = _heads()
    bound = heads.bind(2, ph.head_projection("cpu", "reference"))
    x = torch.ones(1, ph.HIDDEN)
    bound(x)
    with torch.no_grad():
        heads.weights[2].add_(1e-3)
    with pytest.raises(ph.HeadDrift, match="modified after it was bound"):
        bound(x)


def test_sealed_bank_refuses_a_written_head(monkeypatch):
    tensors = _tensors()
    digests = _heads().digests
    monkeypatch.setattr(ph, "released_digests", lambda: digests)
    heads = ph.from_fused(tensors)
    heads.select(0)
    with torch.no_grad():
        heads.biases[5][0] += 1.0
    with pytest.raises(ph.HeadDrift, match="head 5 changed"):
        heads.select(0)


def test_unsealed_bank_reports_the_new_digest_after_a_training_update():
    heads = _heads()
    before = heads.select(1).weight_sha256
    with torch.no_grad():
        heads.weights[1].mul_(0.5)
    after = heads.select(1)
    assert after.weight_sha256 != before and after.weight_sha256 == ph.head_sha256(after.weight)


# --- loop state --------------------------------------------------------------------------------


def test_cursor_hands_out_each_step_once_in_order():
    cursor = ph.HeadCursor(_heads())
    assert [cursor.take(i).head for i in range(6)] == list(range(6))
    cursor.finish()
    with pytest.raises(ph.HeadDrift):
        cursor.take(5)


@pytest.mark.parametrize("sequence", [[0, 0], [0, 2], [1], [0, 1, 0]])
def test_cursor_refuses_repeated_skipped_and_restarted_steps(sequence):
    cursor = ph.HeadCursor(_heads())
    with pytest.raises(ph.HeadDrift, match="expected step"):
        for step in sequence:
            cursor.take(step)
    with pytest.raises(ph.HeadDrift, match="of 6 steps"):
        cursor.finish()


# --- rank --------------------------------------------------------------------------------------


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rank_main(rank, world, port, case, queue):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    try:
        # "weights": right index, another head bank on rank 1
        heads = _heads(seed=1 if case == "weights" and rank == 1 else 0)
        step = 3 if case == "step" and rank == 1 else 2
        try:
            queue.put((rank, "ok", ph.check_ranks(heads.select(step))))
        except ph.HeadDrift as exc:
            queue.put((rank, "drift", str(exc)))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["same", "step", "weights"])
def test_ranks_agree_on_step_and_head_bytes(case):
    ctx = torch.multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_rank_main, args=(r, 2, port, case, queue)) for r in range(2)]
    for p in procs:
        p.start()
    results = {rank: (kind, value) for rank, kind, value in (queue.get(timeout=180) for _ in procs)}
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    if case == "same":
        assert results == {r: ("ok", [(2, 2), (2, 2)]) for r in range(2)}
    else:  # every rank raises, not only the odd one out
        assert all(kind == "drift" for kind, _ in results.values())
        assert all("rank 0" in text and "rank 1" in text for _, text in results.values())


def test_single_process_check_is_local():
    assert ph.check_ranks(_heads().select(4)) == [(4, 4)]


# --- projection --------------------------------------------------------------------------------


def test_reference_projection_equals_linear_to_fp32_rounding():
    heads = _heads()
    x = torch.randn(17, ph.HIDDEN, generator=torch.Generator().manual_seed(3))
    out = heads.bind(1, ph.head_projection("cpu"))(x)
    expected = x.double() @ heads.weights[1].double().t() + heads.biases[1].double()
    assert (out.double() - expected).abs().max() < 1e-4


def test_head_input_must_be_fp32():
    bound = _heads().bind(0, ph.head_projection("cpu"))
    with pytest.raises(TypeError, match="FP32"):
        bound(torch.ones(1, ph.HIDDEN, dtype=torch.bfloat16))


def test_projection_binding_is_named():
    assert ph.head_projection("cpu").provenance()["status"] == "reference"
    with pytest.raises(ValueError, match="binding"):
        ph.head_projection("cpu", "fast")
    with pytest.raises(RuntimeError, match="needs CUDA"):
        ph.head_projection("cpu", "row")


@CUDA
def test_cuda_binding_is_the_row_on_sm90_and_interim_elsewhere():
    sm90 = torch.cuda.get_device_capability() == (9, 0)
    assert ph.head_projection("cuda").status == ("row" if sm90 else "interim")
    if not sm90:
        with pytest.raises(RuntimeError, match="SM90"):
            ph.head_projection("cuda", "row")


@CUDA
def test_cuda_projection_is_accurate_and_row_invariant():
    heads = _heads().to("cuda")
    bound = heads.bind(4)
    x = torch.randn(2, 257, ph.HIDDEN, generator=torch.Generator().manual_seed(5)).cuda()
    out = bound(x)
    golden = x.double() @ heads.weights[4].double().t() + heads.biases[4].double()
    assert (out.double() - golden).abs().max() < 1e-4
    assert _same(bound(x), out)  # repeat
    assert _same(bound(x[1:]), out[1:])  # batch position
    assert _same(bound(x[:, 100:101]), out[:, 100:101])  # row count
    for other in range(6):
        if other != 4:
            assert not _same(heads.bind(other)(x), out)  # probe I1: another head is visible


@CUDA
def test_only_the_selected_head_gets_gradients():
    heads = _heads()
    weights = [w.cuda().requires_grad_() for w in heads.weights]
    biases = [b.cuda().requires_grad_() for b in heads.biases]
    bank = ph.PDDHeads(weights, biases, sealed=False)
    x = torch.randn(1, 64, ph.HIDDEN, generator=torch.Generator().manual_seed(7)).cuda()
    x.requires_grad_()
    keys = torch.stack([torch.zeros(64, dtype=torch.long), torch.arange(64)], dim=1).cuda()
    from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

    g = torch.randn(1, 64, 96, generator=torch.Generator().manual_seed(8)).cuda()
    with canonical_backward_session():
        out = bank.bind(3)(x, logical_keys=keys)
        out.backward(g)
    dw = g[0].double().t() @ x.detach()[0].double()
    assert (weights[3].grad.double() - dw).abs().max() < 1e-3
    assert (biases[3].grad.double() - g[0].double().sum(0)).abs().max() < 1e-3
    assert (x.grad[0].double() - g[0].double() @ weights[3].detach().double()).abs().max() < 1e-4
    for i in range(6):
        if i != 3:
            assert weights[i].grad is None and biases[i].grad is None


# --- readback ----------------------------------------------------------------------------------


class _Toy(torch.nn.Module):
    def __init__(self, n=6):
        super().__init__()
        self.pdd_proj_out = torch.nn.ModuleList([torch.nn.Linear(4, 2) for _ in range(n)])

    def forward(self, x, index):
        return self.pdd_proj_out[index](x)


def test_readback_names_the_head_that_ran():
    model = _Toy()
    with ph.HeadReadback(model) as readback:
        for step in range(6):
            model(torch.ones(1, 4), step)
            readback.check(step)
        model(torch.ones(1, 4), 2)
        with pytest.raises(ph.HeadDrift, match="step 3 must run head 3"):
            readback.check(3)
        with pytest.raises(ph.HeadDrift):
            readback.check(0)  # nothing ran
    model(torch.ones(1, 4), 0)
    assert readback.calls == [0] * 6  # hooks are gone


def test_readback_needs_six_heads():
    with pytest.raises(ph.HeadArtifactError):
        ph.HeadReadback(_Toy(5))


def test_pinned_diffusers_transformer_runs_head_i_at_step_i():
    """The released forward indexes ``pdd_proj_out`` with ``pdd_step_index`` and nothing else."""

    pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3_image")
    from diffusers import MiniMaxH3ImageTransformer2DModel

    model = MiniMaxH3ImageTransformer2DModel(
        num_layers=1,
        num_refiner_layers=1,
        hidden_size=32,
        num_attention_heads=2,
        attention_head_dim=16,
        ffn_dim=64,
        in_channels=4,
        text_dim=8,
        freq_dim=16,
        time_embed_dim=16,
        time_embed_hidden_dim=16,
        rope_freq_dim=2,
    ).eval()
    readback = ph.HeadReadback(model)
    text, image = 3, 4
    kwargs = dict(
        hidden_states=torch.randn(1, image, 16),
        encoder_hidden_states=torch.randn(1, text, 8),
        timestep=torch.tensor([0.5]),
        timestep_indices=torch.zeros(7, dtype=torch.long),
        token_tags=torch.tensor([1] * text + [0] * image),
        position_ids=torch.rand(7, 3, dtype=torch.float64),
        image_indices=torch.arange(text, 7),
        text_indices=torch.arange(text),
        return_dict=False,
    )
    with torch.no_grad(), readback:
        for step in range(6):
            model(pdd_step_index=step, **kwargs)
            readback.check(step)
