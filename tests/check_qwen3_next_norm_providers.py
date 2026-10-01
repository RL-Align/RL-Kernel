# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""How far apart vLLM's eager gated-RMSNorm paths are, and ours from each.

Named ``check_`` rather than ``test_`` on purpose, following
``tests/distributed/check_*.py``: this module imports real vLLM, and
``tests/test_framework_operator_integrations.py`` asserts that ``vllm`` is absent
from ``sys.modules`` -- an invariant any collected test importing vLLM would break
for the whole session. Run it explicitly:

    pytest tests/check_qwen3_next_norm_providers.py -v

RFC #428 section 2.1 defines L2 as "bitwise identical to vLLM rollout". For
Qwen3-Next's GDN gated norm that phrase is not well defined until a single
provider is named, because vLLM ships several and they do not agree bitwise with
each other.

This module records two things so a vLLM upgrade cannot move them silently:

1. **Provider facts** -- the GDN decode env defaults, and that ``RMSNormGated``
   has distinct ``forward_native`` and ``forward_cuda`` methods. It does NOT
   determine which path vLLM dispatches at runtime: the checks below call each
   method directly (eager), and vLLM's default compiled mode traces
   ``forward_native`` into an inductor graph instead.
2. **The size of the gap** -- a seed sweep that asserts an upper bound on the
   disagreement and on the mismatch rate between the eager paths. The bounds are
   provider-gap bounds, not ``tolerance_contract.json`` thresholds, and they
   deliberately do NOT assert equality.

Measured on 2x B200 (sm_100, torch 2.13.0+cu130, vllm 0.30.0), bf16,
``head_v_dim=128``, 512 rows, 40 seeds:

===============================  ==================  ================
comparison                       seeds not bitwise   worst max|diff|
===============================  ==================  ================
ours vs ``forward_native``       6 / 40              1.56e-2
ours vs ``forward_cuda``         18 / 40             3.91e-3
``forward_native`` vs ``cuda``   21 / 40             1.56e-2
===============================  ==================  ================
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.norm.qwen3_next_rms_norm import Qwen3NextRMSNormGatedOp

# vLLM is imported inside the checks, never at module scope, so that merely
# collecting this file does not pull it into sys.modules.

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

# Qwen3-Next-80B-A3B-Instruct: linear_value_head_dim / rms_norm_eps.
_HEAD_V_DIM = 128
_EPS = 1e-6
_ROWS = 512
_SEEDS = 40

# The gated norm is constructed by vLLM's GDN block with exactly these settings;
# see vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py.
_NORM_BEFORE_GATE = True
_GROUP_SIZE = None
_ACTIVATION = "silu"


@pytest.fixture(scope="module")
def vllm_config_ctx():
    """`RMSNormGated` is a CustomOp and refuses to build outside a config context."""
    pytest.importorskip("vllm", reason="vLLM is required to identify the provider")
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        yield


def _make_norm():
    from vllm.model_executor.layers.layernorm import RMSNormGated

    return RMSNormGated(
        _HEAD_V_DIM,
        eps=_EPS,
        group_size=_GROUP_SIZE,
        norm_before_gate=_NORM_BEFORE_GATE,
        activation=_ACTIVATION,
    )


def _inputs(seed: int, dtype: torch.dtype, rows: int = _ROWS):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, _HEAD_V_DIM, device="cuda", dtype=dtype, generator=g)
    gate = torch.randn(rows, _HEAD_V_DIM, device="cuda", dtype=dtype, generator=g)
    weight = torch.randn(_HEAD_V_DIM, device="cuda", dtype=dtype, generator=g)
    return x, gate, weight


def _disagreement(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """(worst absolute difference, fraction of elements that differ bitwise)."""
    bits_a = a.float().view(torch.int32)
    bits_b = b.float().view(torch.int32)
    mismatch = int((bits_a != bits_b).sum())
    worst = (a.float() - b.float()).abs().max().item()
    return worst, mismatch / a.numel()


# --------------------------------------------------------------------------- #
# 1. Provider facts -- recorded, not a runtime dispatch check
# --------------------------------------------------------------------------- #
def test_custom_op_has_distinct_native_and_cuda_paths(vllm_config_ctx):
    """The two methods are distinct. Which one runs depends on the vLLM mode: eager
    (``custom_ops="all"``) dispatches ``forward_cuda``; the default compiled mode
    traces ``forward_native``. This test does not check that choice."""
    norm = _make_norm()
    assert type(norm).forward_cuda is not type(norm).forward_native


def test_gdn_decode_provider_env_defaults_are_recorded():
    """Record the GDN decode env defaults.

    This records the defaults only; it does not assert which kernel runs. For
    Qwen3-Next the ``VLLM_GDN_DECODE_KERNEL="cuda"`` default does not take effect:
    vLLM builds its GDN layers with ``gqa_interleaved_layout=True`` and falls back
    to the Triton decode kernel. A change of either default still fails here, as a
    prompt to re-derive the decode path.
    """
    pytest.importorskip("vllm", reason="vLLM is required to identify the provider")
    import vllm.envs as envs

    observed = {
        "VLLM_GDN_DECODE_KERNEL": envs.VLLM_GDN_DECODE_KERNEL.strip().lower(),
        "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE": bool(
            envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE
        ),
    }
    assert observed == {
        "VLLM_GDN_DECODE_KERNEL": "cuda",
        "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE": True,
    }, (
        f"GDN decode provider defaults changed: {observed}. Re-derive which kernel "
        "the rollout decode path takes before relying on any exactness claim."
    )


# --------------------------------------------------------------------------- #
# 2. The gap, bounded -- never asserted to be zero
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "dtype, max_abs, max_mismatch_rate",
    [
        # Provider-gap bounds, not contract thresholds. bf16 rounding absorbs most
        # of the reduction-tree difference, so few elements move.
        (torch.bfloat16, 2e-2, 0.05),
        # In fp32 about 36% of elements differ, so the rate is left unbounded and
        # only the magnitude is bounded (no per-element ULP bound is asserted).
        (torch.float32, 1e-5, 1.0),
    ],
)
def test_vllm_paths_disagree_only_within_bounds(vllm_config_ctx, dtype, max_abs, max_mismatch_rate):
    """vLLM's own two paths differ; bound how much.

    A growing magnitude means the reduction trees have diverged semantically,
    which would invalidate treating either as the reference. A high mismatch
    *rate* at ULP magnitude does not -- it just means the tree shapes differ.
    """
    norm = _make_norm().to("cuda", dtype)
    worst_abs, worst_rate, differing = 0.0, 0.0, 0
    for seed in range(_SEEDS):
        x, gate, weight = _inputs(seed, dtype)
        norm.weight.data = weight.clone()
        native = norm.forward_native(x, gate)
        cuda = norm.forward_cuda(x, gate)
        abs_d, rate = _disagreement(native, cuda)
        worst_abs, worst_rate = max(worst_abs, abs_d), max(worst_rate, rate)
        differing += int(rate > 0.0)

    assert worst_abs <= max_abs, (
        f"vLLM forward_native vs forward_cuda worst |diff| {worst_abs:.3e} exceeds "
        f"{max_abs:.3e} over {_SEEDS} seeds ({differing} seeds differ)"
    )
    assert worst_rate <= max_mismatch_rate


@pytest.mark.parametrize("path", ["forward_native", "forward_cuda"])
def test_ours_tracks_each_vllm_path_within_bounds(vllm_config_ctx, path):
    """Our strict op follows vLLM's convention; bound the residual.

    Not an equality assertion. We reproduce the fp32 weight multiply and the
    single trailing cast, but our reduction is the repo's fixed 32-wide chunked
    sum rather than whatever tree the provider uses, so a few elements straddle
    a rounding boundary.
    """
    dtype = torch.bfloat16
    ours = Qwen3NextRMSNormGatedOp()
    norm = _make_norm().to("cuda", dtype)

    worst_abs, worst_rate = 0.0, 0.0
    for seed in range(_SEEDS):
        x, gate, weight = _inputs(seed, dtype)
        norm.weight.data = weight.clone()
        reference = getattr(norm, path)(x, gate)
        abs_d, rate = _disagreement(ours.forward(x, weight, gate), reference)
        worst_abs, worst_rate = max(worst_abs, abs_d), max(worst_rate, rate)

    # Provider-gap bounds, not contract thresholds.
    assert worst_abs <= 2e-2, f"worst |diff| vs {path} was {worst_abs:.3e}"
    assert worst_rate <= 0.05, f"mismatch rate vs {path} was {worst_rate:.3%}"


# --------------------------------------------------------------------------- #
# 3. Batch invariance -- the property we DO claim
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", ["forward_native", "forward_cuda"])
def test_vllm_gated_norm_is_batch_invariant(vllm_config_ctx, path):
    """If this ever fails, vLLM stops being a coherent L2 target.

    Kept as a guard rather than a claim about our code: a provider whose output
    depends on unrelated rows cannot anchor a bitwise contract.
    """
    dtype = torch.bfloat16
    norm = _make_norm().to("cuda", dtype)
    for seed in range(8):
        x, gate, weight = _inputs(seed, dtype)
        norm.weight.data = weight.clone()
        full = getattr(norm, path)(x, gate)
        for n in (1, 2, 8, 16, 32, 48, 64, 256):
            sliced = getattr(norm, path)(x[:n], gate[:n])
            assert torch.equal(sliced, full[:n]), f"{path} seed={seed} n={n}"


def test_our_gated_op_is_batch_invariant():
    """The L1 claim for this operator, bitwise."""
    dtype = torch.bfloat16
    ours = Qwen3NextRMSNormGatedOp()
    for seed in range(8):
        x, gate, weight = _inputs(seed, dtype)
        full = ours.forward(x, weight, gate)
        for n in (1, 2, 8, 16, 32, 48, 64, 256):
            assert torch.equal(ours.forward(x[:n], weight, gate[:n]), full[:n])
