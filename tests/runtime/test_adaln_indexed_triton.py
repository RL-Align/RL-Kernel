# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""L3: real missing-extension resolution and indexed Triton execution traces."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from rl_engine.runtime.registry import KernelRegistry
from tests.helpers.adaln_indexed import assert_bytes, check_small_case, small_case


@pytest.mark.parametrize("triton_available", [True, False])
def test_indexed_missing_extension_real_resolution(tmp_path, triton_available):
    if triton_available:
        pytest.importorskip("triton")
    if torch.version.hip is not None:
        pytest.skip("NVIDIA resolution policy")
    root = Path(__file__).resolve().parents[2]
    # A distinct real package has no extension; the current _C.so is untouched.
    shutil.copytree(
        root / "rl_engine",
        tmp_path / "rl_engine",
        ignore=shutil.ignore_patterns("*.so", "*.pyc", "__pycache__"),
    )
    script = (
        f"TRITON_AVAILABLE = {triton_available!r}\n"
        + """
import pathlib, torch, rl_engine, importlib.util
assert (importlib.util.find_spec("triton") is not None) == TRITON_AVAILABLE
from rl_engine.backends.extension import _EXT_AVAILABLE
from rl_engine.runtime.registry import KernelRegistry
assert pathlib.Path(rl_engine.__file__).resolve().parent.parent == pathlib.Path.cwd()
assert not _EXT_AVAILABLE
op, trace = KernelRegistry().get_adaln_modulation_op(
    "cuda", hidden=2, modulate_index=torch.zeros(1, 8, dtype=torch.int32))
assert type(op.implementation).__name__ == (
    "TritonAdaLNModulationOp" if TRITON_AVAILABLE else "NativeAdaLNModulationOp"), trace
assert trace["requested_backend"] == ("triton" if TRITON_AVAILABLE else "pytorch")
assert trace["arithmetic_profile"] == (
    "adaln-indexed-fp32-tree-v1" if TRITON_AVAILABLE else "adaln-select01-framework-v1")
assert trace["fallback"] is (not TRITON_AVAILABLE)
assert trace["fallback_reason"] == (
    None if TRITON_AVAILABLE else "modulate_index_requires_select01")
assert trace["actual_backend"] is None
x = torch.tensor([[[-3.,3.]]], requires_grad=True)
m = torch.zeros(2,6, requires_grad=True)
index = torch.zeros(1,1,dtype=torch.int32)
y,g = op(x,m,modulate_index=index,eps=7.)
dx,dm = torch.autograd.grad((y,g),(x,m),(torch.tensor([[[1.,0.]]]),torch.ones_like(g)))
assert torch.equal(dx,torch.tensor([[[7/128,-7/128]]]))
assert torch.equal(dm,torch.tensor([[1.,0.,-.75,0.,1.,1.],[0.]*6]))
assert trace["actual_backend"] == "pytorch"
assert trace["arithmetic_profile"] == "adaln-select01-framework-v1"
assert trace["fallback"] is True
assert trace["fallback_reason"] == (
    "indexed_triton_domain_unsupported" if TRITON_AVAILABLE else "modulate_index_requires_select01")
try:
    op(x,m,modulate_index=index+2,eps=7.)
except ValueError:
    pass
else:
    raise AssertionError("illegal route accepted")
assert trace["actual_backend"] is None and trace["observation"] is None
try:
    KernelRegistry().get_adaln_modulation_op("cuda",hidden=2,modulate_index=index,strict=True)
except RuntimeError:
    pass
else:
    raise AssertionError("indexed strict opened")
"""
    )
    env = dict(os.environ, PYTHONPATH=str(tmp_path), CUDA_VISIBLE_DEVICES="")
    # -S avoids the installed editable finder resolving absent submodules back
    # into the original checkout. Add dependency directories without .pth hooks.
    dependencies = [p for p in sys.path if "site-packages" in p]
    if not triton_available:
        dependency_view = tmp_path / "dependencies"
        dependency_view.mkdir()
        for directory in dependencies:
            for entry in Path(directory).iterdir():
                if (
                    entry.name.startswith(("triton", "__editable__"))
                    or entry.suffix == ".pth"
                    or (dependency_view / entry.name).exists()
                ):
                    continue
                (dependency_view / entry.name).symlink_to(entry)
        dependencies = [str(dependency_view)]
    script = "import sys; sys.path.extend(" + json.dumps(dependencies) + ")\n" + script
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def resolved_triton():
    from tests.helpers.adaln_indexed import require_triton_gpu

    require_triton_gpu()
    if os.environ.get("RLK_ADALN_TRITON_ISOLATED") != "1":
        pytest.skip("run in the prepared extension-free package for real Triton L3 selection")
    from rl_engine.backends.extension import _EXT_AVAILABLE

    assert not _EXT_AVAILABLE, "isolation must actually exclude the extension"
    index = torch.zeros(1, 8, dtype=torch.int32)
    op, trace = KernelRegistry().get_adaln_modulation_op("cuda", hidden=2, modulate_index=index)
    assert trace["selected_backend"] == "TRITON_ADALN_MODULATION"
    return op, trace


@pytest.mark.cuda_only
def test_triton_indexed_small_public_dispatch(resolved_triton):
    op, trace = resolved_triton
    check_small_case(op, "cuda")
    assert trace["actual_backend"] == "triton"
    assert trace["arithmetic_profile"] == "adaln-indexed-fp32-tree-v1"
    assert trace["kernel_fingerprint"] == "adaln_indexed_triton_v1"
    assert trace["fallback"] is False and trace["fallback_reason"] is None


@pytest.mark.cuda_only
@pytest.mark.parametrize("explicit_zero", [False, True])
def test_triton_dispatch_absent_vs_zero(resolved_triton, explicit_zero):
    from tests.helpers.adaln_indexed import check_absent_vs_zero_dy

    op, trace = resolved_triton
    check_absent_vs_zero_dy(op, "cuda", explicit_zero)
    assert trace["actual_backend"] == "triton"
    assert trace["fallback"] is False and trace["fallback_reason"] is None


@pytest.mark.cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize(
    "index_dtype", [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_triton_dispatch_route_domain(resolved_triton, dtype, broadcast, index_dtype):
    from tests.helpers.adaln_indexed import check_domain_routes

    op, trace = resolved_triton
    check_domain_routes(op, "cuda", dtype, index_dtype, broadcast)
    assert trace["actual_backend"] == "triton"
    assert trace["arithmetic_profile"] == "adaln-indexed-fp32-tree-v1"
    assert trace["fallback"] is False and trace["fallback_reason"] is None


@pytest.mark.cuda_only
@pytest.mark.parametrize("unsupported", ["H", "B", "eps_overflow", "eps_underflow", "cpu"])
@pytest.mark.parametrize("resolution", ["indexed", "shared"])
def test_triton_execution_fallback_and_failed_trace(resolved_triton, unsupported, resolution):
    from rl_engine.reference.norm.adaln_modulation import NativeAdaLNModulationOp

    op, trace = resolved_triton
    if resolution == "shared":
        op, trace = KernelRegistry().get_adaln_modulation_op("cuda", hidden=2)
    x, m, index, _, _ = small_case("cuda")
    op(x, m, modulate_index=index, eps=7.0)
    assert trace["actual_backend"] == "triton"
    if unsupported == "cpu":
        x, m, index = x.cpu(), m.cpu(), index.cpu()
    else:
        b, h = (65536, 2) if unsupported == "B" else (1, 4097 if unsupported == "H" else 2)
        x = torch.tensor([-3.0, 3.0], device="cuda").repeat(b, 1, (h + 1) // 2)[..., :h]
        m = torch.zeros(2 * b, 3 * h, device="cuda")
        index = torch.ones(1, 1, dtype=torch.int32, device="cuda")
    eps = {"eps_overflow": 1e300, "eps_underflow": 2**-150}.get(unsupported, 7.0)
    actual = op(x, m, modulate_index=index, eps=eps)
    expected = NativeAdaLNModulationOp()(x, m, modulate_index=index, eps=eps)
    for a, e in zip(actual, expected, strict=True):
        assert_bytes(a, e)
    assert trace["actual_backend"] == "pytorch"
    assert trace["arithmetic_profile"] == "adaln-select01-framework-v1"
    assert trace["fallback"] is True
    assert trace["fallback_reason"] == (
        "indexed_triton_domain_unsupported"
        if resolution == "indexed"
        else "modulate_index_requires_select01"
    )
    # A failed call after successful fallback must clear the old observation too.
    with pytest.raises(ValueError, match="values must be 0 or 1"):
        op(x, m, modulate_index=torch.full_like(index, 2), eps=eps)
    assert trace["actual_backend"] is None and trace["observation"] is None


@pytest.mark.cuda_only
@pytest.mark.parametrize("eps", [0.0, 2**-149, 2**-150, 1e300])
def test_triton_indexed_strict_closed(resolved_triton, eps):
    with pytest.raises(RuntimeError, match="strict AdaLN rejects"):
        KernelRegistry().get_adaln_modulation_op(
            "cuda", hidden=2, modulate_index=torch.zeros(1, 8, dtype=torch.int32), strict=True
        )
    op, trace = KernelRegistry().get_adaln_modulation_op("cuda", hidden=2, strict=True)
    x, m, index, _, _ = small_case("cuda")
    with pytest.raises(RuntimeError, match="strict AdaLN rejects"):
        op(x, m, modulate_index=index, eps=eps)
    assert trace["actual_backend"] is None
