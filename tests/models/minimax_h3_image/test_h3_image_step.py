# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``ws1_one_h3_image_step`` (RFC #738): one strict step, layer by layer.

Full-width two-block model with synthetic weights (the released tensor table, cut to two
blocks); the released weights are covered by ``tools/validation/models/h3_image_step.py``. Pins:

* strict executions are bitwise equal at every node: repeated, batched with another request,
  physically permuted (I-C0, I-C1, I-C3), under autograd and under checkpointing (I-C9);
* ``first_drift`` names the node, the node inside the block, the row group and the packed row;
* the PDD head is the step's own (I1), and only that head gets gradients;
* the provider arm records the same nodes in the same order and reads back the head that ran.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.models.minimax_h3_image import fingerprint as fp
from rl_engine.models.minimax_h3_image import packed_layout as pl
from rl_engine.models.minimax_h3_image import pdd_head as ph
from rl_engine.models.minimax_h3_image import pdd_schedule as ps
from rl_engine.validation.models import h3_block
from rl_engine.validation.models import h3_full_model as hfm
from rl_engine.validation.models import h3_image_step as hs

BLOCKS = 2
STEP = 2
NODES = ["time_proj", "time_embedder", "proj_in", "context_embedder", "refiner.0", "refiner.1"] + [
    "token_refiner",
    "packed",
    "block.0",
    "block.1",
    "norm_out",
    "pdd_proj_out",
    "velocity",
]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _synthetic_weights():
    """The released table's tensors for two blocks: names, dtypes and shapes, seeded values."""

    config = {**fp.transformer_config(), "num_layers": BLOCKS}
    g = torch.Generator("cuda").manual_seed(0)
    dtypes = {"F32": torch.float32, "BF16": torch.bfloat16}
    weights = {}
    for name, (dtype, shape) in fp.expected_tensor_table(config).items():
        if "norm" in name and name.endswith(".weight") and len(shape) == 1:
            value = torch.ones(shape, device="cuda")
        else:
            value = torch.randn(shape, device="cuda", generator=g) * 0.02
        weights[name] = value.to(dtypes[dtype])
    return weights


@pytest.fixture(scope="module")
def weights():
    out = _synthetic_weights()
    yield out
    out.clear()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def ops():
    return h3_block.CandidateOps()


def _strict(weights, ops):
    heads = ph.from_diffusers(weights, sealed=False)
    return hs.StrictImageTransformer(weights, heads, num_layers=BLOCKS, ops=ops)


@pytest.fixture(scope="module")
def strict(weights, ops):
    return _strict(weights, ops)


def _layout(task):
    tags = torch.ones(40, dtype=torch.long)
    refs = []
    if task == "edit":
        tags[3:9] = 0
        refs = [(8, 12), (6, 4)]
    return pl.build_layout(tags, refs, (16, 20))


def _inputs(task="edit", seed=1, step=STEP):
    layout = _layout(task)
    unique, indices = ps.row_timestep_plan(layout)[step]
    g = torch.Generator().manual_seed(seed)
    return hs.StepInputs(
        torch.randn(1, layout.image_indices.numel(), 96, generator=g),
        torch.randn(1, 40, 5120, generator=g),
        layout,
        unique,
        indices,
        step,
    )


def _with(inputs, **changes):
    fields = ("image_rows", "prompt_embeds", "layout", "unique_timesteps")
    fields += ("timestep_indices", "step")
    return hs.StepInputs(**{**{f: getattr(inputs, f) for f in fields}, **changes})


# --- strict executions are equal ---------------------------------------------------------------


@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_repeat_batch_and_physical_permutation_are_bitwise(strict, task):
    inputs = _inputs(task)
    velocity, base = hs.run_strict(strict, inputs, keep_tensors=True, detail_blocks=frozenset({1}))
    assert [n for n in base.order if n.count(".") < 2] == NODES
    assert [n for n in base.order if n.count(".") == 2] == [
        f"block.1.{node}" for node in h3_block.NODE_NAMES
    ]
    assert velocity.shape == (1, inputs.image_rows.shape[1], 96)
    assert velocity.dtype == torch.float32 and bool(torch.isfinite(velocity).all())

    detail = {"keep_tensors": True, "detail_blocks": frozenset({1})}
    _, again = hs.run_strict(strict, inputs, **detail)
    assert hs.first_drift(base, again, inputs.layout) is None

    other = _inputs(task, seed=9)
    for position, batch in ((0, [inputs, other]), (1, [other, inputs])):
        _, batched = hs.run_strict(strict, hs.stack(batch), **detail)
        assert hs.first_drift(base, batched.row(position), inputs.layout) is None

    g = torch.Generator().manual_seed(3)
    order = torch.randperm(inputs.layout.sequence_length, generator=g)
    permuted = hs.permute_inputs(inputs, order)
    assert pl.layout_digest(permuted.layout) != pl.layout_digest(inputs.layout)
    _, moved = hs.run_strict(strict, permuted, **detail)
    assert hs.first_drift(base, moved, inputs.layout) is None


def test_every_head_runs_at_its_own_step_only(strict):
    digests = set()
    for step in range(6):
        inputs = _inputs("t2i", step=step)
        _, trace = hs.run_strict(strict, inputs)
        assert trace.head["head"] == trace.head["step"] == step
        digests.add(trace.digests["velocity"])
    assert len(digests) == 6


TRAINABLE = ["image_rows", "prompt_embeds", "proj_in.weight", "proj_in.bias"]
TRAINABLE += [f"pdd_proj_out.{i}.{p}" for i in range(6) for p in ("weight", "bias")]


def test_autograd_and_checkpointing_change_nothing(strict):
    inputs = _inputs()
    plain, base = hs.run_strict(strict, inputs)
    g = torch.Generator().manual_seed(5)
    upstream = torch.randn(plain.shape, generator=g)
    runs = {}
    for name, checkpointed in (("plain", False), ("again", False), ("checkpointed", True)):
        trace = hs.StepTrace()
        runs[name] = hs.training_step(
            strict, inputs, TRAINABLE, upstream, checkpoint_blocks=checkpointed, trace=trace
        )
        assert hs.first_drift(base, trace) is None  # rollout == training, per node
        assert torch.equal(runs[name]["velocity"], plain)
    grads = runs["plain"]["grads"]
    assert runs["plain"]["head"]["head"] == STEP
    for name, grad in grads.items():
        if name.startswith("pdd_proj_out.") and f".{STEP}." not in name:
            assert grad is None  # the other five heads took no part
            continue
        assert bool(torch.isfinite(grad).all()) and bool((grad != 0).any()), name
        for other in ("again", "checkpointed"):
            assert torch.equal(runs[other]["grads"][name], grad), (other, name)


# --- first drift -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weight, node",
    [
        ("time_embedder.linear_1.weight", "time_embedder"),
        ("context_embedder.weight", "context_embedder"),
        ("token_refiner.refiner_blocks.1.ff.net.2.weight", "refiner.1"),
        ("token_refiner.final_norm.weight", "token_refiner"),
        ("transformer_blocks.0.attn.to_v.weight", "block.0.v_proj"),
        ("transformer_blocks.1.ff.net.2.weight", "block.1.ffn_down"),
        ("norm_out.linear.bias", "norm_out"),
        (f"pdd_proj_out.{STEP}.bias", "pdd_proj_out"),
    ],
)
def test_first_drift_names_the_changed_node(strict, weights, ops, weight, node):
    inputs = _inputs()
    detail = {"detail_blocks": frozenset(range(BLOCKS))}
    _, base = hs.run_strict(strict, inputs, **detail)
    changed = dict(weights)
    changed[weight] = weights[weight].clone()
    changed[weight][: max(1, len(changed[weight]) // 2)] += 0.25
    _, other = hs.run_strict(_strict(changed, ops), inputs, **detail)
    assert hs.first_drift(base, other)["node"] == node


def test_another_head_drifts_at_the_head_and_nowhere_before(strict):
    """Probe I1: the next head at this step's timesteps."""

    inputs = _inputs()
    _, base = hs.run_strict(strict, inputs, keep_tensors=True)
    _, other = hs.run_strict(strict, _with(inputs, step=STEP + 1), keep_tensors=True)
    drift = hs.first_drift(base, other, inputs.layout)
    assert drift["node"] == "pdd_proj_out" and drift["row"] == 0


@pytest.mark.parametrize("group", ["reference", "target"])
def test_first_drift_names_the_row_group(strict, group):
    inputs = _inputs()
    references = inputs.layout.num_reference_rows
    row = 5 if group == "reference" else references + 7
    rows = inputs.image_rows.clone()
    rows[0, row] += 1.0
    _, base = hs.run_strict(strict, inputs, keep_tensors=True)
    _, other = hs.run_strict(strict, _with(inputs, image_rows=rows), keep_tensors=True)
    drift = hs.first_drift(base, other, inputs.layout)
    assert (drift["node"], drift["row"], drift["row_group"]) == ("proj_in", row, group)
    assert drift["differing_rows"] == 1


def test_first_drift_in_the_text_stream(strict):
    inputs = _inputs()
    embeds = inputs.prompt_embeds.clone()
    embeds[0, 11] += 1.0
    _, base = hs.run_strict(strict, inputs, keep_tensors=True)
    _, other = hs.run_strict(strict, _with(inputs, prompt_embeds=embeds), keep_tensors=True)
    drift = hs.first_drift(base, other, inputs.layout)
    assert (drift["node"], drift["row"], drift["row_group"]) == ("context_embedder", 11, "text")


def test_shared_timestep_for_the_references_drifts_at_the_first_block(strict):
    """Probe I21: reference rows at the target's timestep instead of 0.999."""

    inputs = _inputs()
    target = float(ps.released_schedule("cpu").timesteps[STEP])
    index = int((inputs.unique_timesteps == target).nonzero()[0])
    shared = torch.full_like(inputs.timestep_indices, index)
    _, base = hs.run_strict(strict, inputs, keep_tensors=True)
    _, other = hs.run_strict(strict, _with(inputs, timestep_indices=shared), keep_tensors=True)
    drift = hs.first_drift(base, other, inputs.layout)
    assert drift["node"] == "block.0"


def test_flipped_modality_tag_drifts_at_the_first_block(strict):
    """Probe I20: one text row tagged as an image row."""

    from dataclasses import replace

    inputs = _inputs()
    tags = inputs.layout.token_tags.clone()
    tags[0] = 1 - tags[0]
    flipped = _with(inputs, layout=replace(inputs.layout, token_tags=tags))
    detail = {"keep_tensors": True, "detail_blocks": frozenset({0})}
    _, base = hs.run_strict(strict, inputs, **detail)
    _, other = hs.run_strict(strict, flipped, **detail)
    drift = hs.first_drift(base, other, inputs.layout)
    assert (drift["node"], drift["row"], drift["row_group"]) == ("block.0.norm1", 0, "text")


def test_distance_splits_the_error_by_row_group():
    layout = _layout("edit")
    base = torch.ones(1, layout.sequence_length, 4)
    value = base.clone()
    value[0, 3, 2] = 9.0  # a text row
    row = hs.distance("block.0", value, base, layout)
    assert row["by_group"]["text"] > 0
    assert row["by_group"]["reference"] == row["by_group"]["target"] == 0.0
    expected = {"batch_row": 0, "row": 3, "row_group": "text", "channel": 2}
    assert row["largest_error"] == {**expected, "reference": 1.0, "value": 9.0}
    assert hs.distance("block.0", base, base, layout)["bitwise_equal"]


def test_traces_of_different_graphs_are_not_compared(strict):
    inputs = _inputs("t2i")
    _, base = hs.run_strict(strict, inputs)
    _, detailed = hs.run_strict(strict, inputs, detail_blocks=frozenset({0}))
    assert hs.first_drift(base, detailed)["reason"] == "different node sequences"


# --- inputs ------------------------------------------------------------------------------------


def test_malformed_steps_fail():
    inputs = _inputs()
    with pytest.raises(ps.OffGridTimestep):
        _with(inputs, step=6)
    with pytest.raises(hs.StepInputError, match="FP32"):
        _with(inputs, image_rows=inputs.image_rows.bfloat16())
    with pytest.raises(hs.StepInputError, match="one image row"):
        _with(inputs, image_rows=inputs.image_rows[:, 1:])
    with pytest.raises(hs.StepInputError, match="one prompt row"):
        _with(inputs, prompt_embeds=inputs.prompt_embeds[:, 1:])
    with pytest.raises(hs.StepInputError, match="one timestep index"):
        _with(inputs, timestep_indices=inputs.timestep_indices[1:])


def test_only_requests_at_one_step_and_layout_are_batched():
    with pytest.raises(ph.HeadDrift, match=r"steps \[2, 3\]"):
        hs.stack([_inputs(), _inputs(step=3)])
    with pytest.raises(hs.StepInputError, match="layout"):
        hs.stack([_inputs("edit"), _inputs("t2i")])


def test_missing_block_weights_fail(weights, ops):
    partial = {k: v for k, v in weights.items() if k != "transformer_blocks.1.norm2.weight"}
    with pytest.raises(ValueError, match="block 1 is missing"):
        _strict(partial, ops)


def test_reference_fallback_is_refused(weights, ops, monkeypatch):
    from rl_engine.reference.minimax_h3.final_adaln_out import NativeH3FinalAdaLNOutOp
    from rl_engine.runtime.registry import KernelRegistry

    original = KernelRegistry.get_op

    def get_op(self, op_type, *args, **kwargs):
        if op_type == "final_adaln_out":
            return NativeH3FinalAdaLNOutOp()
        return original(self, op_type, *args, **kwargs)

    monkeypatch.setattr(KernelRegistry, "get_op", get_op)
    with pytest.raises(hfm.CandidateBindingError, match="final_adaln_out"):
        _strict(weights, ops)


def test_attention_row_can_be_required(weights):
    interim = h3_block.CandidateOps(attention="interim")
    heads = ph.from_diffusers(weights, sealed=False)
    with pytest.raises(hfm.CandidateBindingError, match="RL_KERNEL_H3_STRICT"):
        hs.StrictImageTransformer(
            weights, heads, num_layers=BLOCKS, ops=interim, require_attention_row=True
        )


def test_bindings_name_every_operator(strict):
    sm90 = torch.cuda.get_device_capability() == (9, 0)
    for node in ("proj_in", "pdd_proj_out"):  # rows of the FP32 boundary are SM90 only
        assert strict.bindings[node]["status"] == ("row" if sm90 else "interim")
    assert strict.bindings["norm_out"]["status"] == "row"
    assert strict.bindings["velocity"]["status"] == "reference"
    assert {f"block.{n}" for n in h3_block.NODE_NAMES} <= set(strict.bindings)


def test_denoiser_is_the_same_step(strict):
    inputs = _inputs()
    denoiser = hs.StrictImageDenoiser(strict)
    velocity = denoiser(
        inputs.image_rows[0],
        inputs.prompt_embeds,
        inputs.layout,
        inputs.unique_timesteps,
        inputs.timestep_indices,
        inputs.step,
    )
    assert torch.equal(velocity, hs.run_strict(strict, inputs)[0])


# --- provider arm ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def provider(weights):
    pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3_image")
    from diffusers import MiniMaxH3ImageTransformer2DModel

    with torch.device("meta"):
        model = MiniMaxH3ImageTransformer2DModel(num_layers=BLOCKS)
    missing, unexpected = model.load_state_dict(weights, strict=True, assign=True)
    assert not missing and not unexpected
    model.rope = type(model.rope)(rope_freq_dim=16, rope_theta=10000.0).cuda()
    return model.eval()


@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_provider_arm_records_the_same_nodes(strict, provider, task):
    inputs = _inputs(task)
    _, ours = hs.run_strict(strict, inputs, keep_tensors=True)
    _, theirs = hs.run_provider(provider, inputs, keep_tensors=True)
    assert ours.order == theirs.order == NODES
    table = {row["node"]: row for row in hs.compare(ours, theirs)}
    assert table["time_proj"]["bitwise_equal"]
    for node, row in table.items():
        assert row["rel_frobenius"] < 0.05, (node, row)


def test_isolated_blocks_run_on_the_providers_inputs(strict, provider):
    inputs = _inputs()
    _, theirs = hs.run_provider(provider, inputs, keep_tensors=True)
    _, ours = hs.run_strict(strict, inputs, keep_tensors=True)
    isolated = dict(strict.isolated_blocks(inputs, theirs))
    assert list(isolated) == ["block.0", "block.1"]
    for node, value in isolated.items():
        local = hs.distance(node, value, theirs.tensors[node], inputs.layout)
        assert local["rel_frobenius"] < 0.02, local
        # the provider's input is not the strict one, so the outputs differ from the chained run
        assert not torch.equal(value.cpu(), ours.tensors[node])
    for node, value in hs.provider_isolated_blocks(provider, inputs, theirs):
        assert torch.equal(value.cpu(), theirs.tensors[node])  # same kernels: the run itself
    moved = hs.permute_inputs(inputs, torch.arange(inputs.layout.sequence_length).flip(0))
    with pytest.raises(hs.StepInputError, match="canonical"):
        next(iter(strict.isolated_blocks(moved, theirs)))


def test_provider_head_is_read_back(provider):
    for step in (0, 5):
        _, trace = hs.run_provider(provider, _inputs("t2i", step=step))
        assert trace.head == {"head": step, "step": step}


def test_provider_wrong_head_is_caught(provider, monkeypatch):
    original = type(provider).forward

    def forward(self, *args, pdd_step_index, **kwargs):
        return original(self, *args, pdd_step_index=1 - pdd_step_index, **kwargs)

    monkeypatch.setattr(type(provider), "forward", forward)
    with pytest.raises(ph.HeadDrift, match="step 0 must run head 0"):
        hs.run_provider(provider, _inputs("t2i", step=0))


class _LimitedRowOp:
    """A row-wise GEMM stand-in that refuses more than ``limit`` rows, like ``h3_ffn_down_gemm``."""

    def __init__(self, limit):
        self.limit, self.calls, self.last_execution = limit, [], {"op": "limited"}

    def __call__(self, x, weight, *, logical_keys=None, parameter_id="w"):
        if x.shape[0] > self.limit:
            raise ValueError(f"row count must be in [1,{self.limit}]")
        self.calls.append((x.shape[0], None if logical_keys is None else logical_keys.clone()))
        return (x.unsqueeze(1) * weight.unsqueeze(0)).sum(-1)


def _row_inputs(rows=10, *, x_grad=False, w_grad=False):
    g = torch.Generator().manual_seed(11)
    x = torch.randn(rows, 6, generator=g, requires_grad=x_grad)
    w = torch.randn(3, 6, generator=g, requires_grad=w_grad)
    keys = torch.stack([torch.zeros(rows, dtype=torch.int64), torch.arange(rows)], 1)
    return x, w, keys


def test_row_sliced_gemm_runs_slices_without_weight_gradient():
    x, w, keys = _row_inputs()
    whole = _LimitedRowOp(100)(x, w, logical_keys=keys)
    op = _LimitedRowOp(4)
    sliced = hs.RowSlicedGemm(op, 4)
    with torch.no_grad():
        out = sliced(x, w, logical_keys=keys, parameter_id="p")
    assert torch.equal(out, whole)
    assert [n for n, _ in op.calls] == [4, 4, 2] and sliced.slices == 1
    assert torch.equal(torch.cat([k for _, k in op.calls]), keys)
    assert sliced.last_execution == {"op": "limited"}  # the row's own attributes stay visible
    with torch.no_grad():
        sliced(x[:4], w, logical_keys=keys[:4])  # within the limit: one call, no slicing
    assert [n for n, _ in op.calls][-1] == 4 and sliced.slices == 1


def test_row_sliced_gemm_input_gradient_matches_and_weight_gradient_keeps_the_limit():
    x, w, keys = _row_inputs(x_grad=True)
    hs.RowSlicedGemm(_LimitedRowOp(4), 4)(x, w, logical_keys=keys).sum().backward()
    sliced_grad = x.grad.clone()
    x.grad = None
    _LimitedRowOp(100)(x, w, logical_keys=keys).sum().backward()
    assert torch.equal(sliced_grad, x.grad)
    x, w, keys = _row_inputs(w_grad=True)
    with pytest.raises(ValueError, match="row count"):
        hs.RowSlicedGemm(_LimitedRowOp(4), 4)(x, w, logical_keys=keys)


def test_row_limited_wraps_only_the_down_row_on_a_copy():
    class _Ops:
        down_gemm = None

    bare = _Ops()
    assert hs._row_limited(bare) is bare
    ops = _Ops()
    ops.down_gemm = _LimitedRowOp(8)
    wrapped = hs._row_limited(ops)
    assert wrapped is not ops and isinstance(wrapped.down_gemm, hs.RowSlicedGemm)
    assert wrapped.down_gemm.max_rows == 32768 and isinstance(ops.down_gemm, _LimitedRowOp)
    assert hs._row_limited(wrapped) is wrapped
