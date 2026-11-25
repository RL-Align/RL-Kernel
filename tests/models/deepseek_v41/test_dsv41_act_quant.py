# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""MXFP8 activation rounding (mxfp8_fake_quant) is byte-equal to DeepSeek's reference
``act_quant(x, 32, "ue8m0", inplace=True)`` (TileLang, ``inference/kernel.py``), batch
-invariant, and leaves MXFP8 values unchanged."""

from __future__ import annotations

import importlib.util
import os
import pathlib

import pytest
import torch

from rl_engine.kernels.ops.cuda.matmul.bf16_linear import mxfp8_fake_quant

META = pathlib.Path(
    os.environ.get("DSV41_META", "/work/u4478697/dsv4-lora-e2e/models/DeepSeek-V4.1-Flash-meta")
)
CUDA = torch.cuda.is_available()


def _x(rows=300, cols=5120, seed=0):
    g = torch.Generator().manual_seed(seed)
    mag = torch.logspace(-6, 4, rows)[:, None]
    x = torch.randn(rows, cols, generator=g) * mag
    x[3, :64] = 0  # an all-zero block hits the 1e-4 floor
    return x.to(torch.bfloat16)


@pytest.mark.skipif(not CUDA or not (META / "inference/kernel.py").exists(), reason="reference")
def test_equals_the_reference_kernel():
    pytest.importorskip("tilelang")
    spec = importlib.util.spec_from_file_location("dsv41_ref_kernel", META / "inference/kernel.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    x = _x().cuda()
    want = ref.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True)
    got = mxfp8_fake_quant(x)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_row_local_and_idempotent():
    dev = "cuda" if CUDA else "cpu"
    x = _x(64, 1024, 1).to(dev)
    y = mxfp8_fake_quant(x)
    assert torch.equal(mxfp8_fake_quant(x[10:11]), y[10:11])
    assert torch.equal(mxfp8_fake_quant(y), y)
    rel = ((y.float() - x.float()).abs() / x.float().abs().clamp(min=1e-30)).nan_to_num()
    big = x.float().abs() > x.float().abs().amax(-1, keepdim=True) / 8
    assert float(rel[big].max()) <= 2**-4  # one E4M3 rounding
