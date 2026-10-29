# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""A build that leaves the H3 sources out has no ``h3_*`` symbols; no CUDA op may then construct.

``KERNEL_ALIGN_USE_FAST_MATH=1``, ROCm and relaxed ``NVCC_*_FLAGS`` builds omit the H3 kernels and
their bindings (build_tools/extensions.py). Every H3 CUDA op checks for its symbols when it is
constructed, so the registry falls back to the PyTorch reference instead of failing on the first
call with an AttributeError.
"""

from __future__ import annotations

import importlib

import pytest

PACKAGE = "rl_engine.backends.cuda.model_specific.minimax_h3"
# the modules that define the probes and own the ``_C`` they look in
PROBE_MODULES = (
    "timestep_sinusoid",
    "det_linear",
    "adaln_row_gather",
    "rmsnorm",
    "gate_residual",
    "sp_norm_adaln",
)
# module, class, constructor arguments
OPS = (
    ("timestep_sinusoid", "H3TimestepSinusoidCudaOp", ()),
    ("timestep_mlp", "H3TimestepMLPCudaOp", ()),
    ("adaln_projection", "H3AdaLNProjectionCudaOp", ()),
    ("adaln_row_gather", "H3AdaLNRowGatherCudaOp", ()),
    ("adaln_modulation", "H3AdaLNModulationCudaOp", ()),
    ("rmsnorm", "H3RMSNormCudaOp", ()),
    ("gate_residual", "H3GateResidualCudaOp", ()),
    ("final_adaln_out", "H3FinalAdaLNOutCudaOp", ()),
    ("sp_norm_adaln", "H3SPNormAdaLNCudaOp", (None, 1, 1)),
)


class ExtensionWithoutH3:
    """Stands in for a compiled ``rl_engine._C`` that has none of the H3 bindings."""


@pytest.fixture
def extension_without_h3(monkeypatch):
    for name in PROBE_MODULES:
        module = importlib.import_module(f"{PACKAGE}.{name}")
        monkeypatch.setattr(module, "_C", ExtensionWithoutH3())
        monkeypatch.setattr(module, "_EXT_AVAILABLE", True)


@pytest.mark.parametrize(("module", "op", "args"), OPS, ids=[o[1] for o in OPS])
def test_every_cuda_op_refuses_to_construct(extension_without_h3, module, op, args):
    cls = getattr(importlib.import_module(f"{PACKAGE}.{module}"), op)
    with pytest.raises(RuntimeError):
        cls(*args)


def test_the_sequence_parallel_op_checks_every_symbol_it_calls(monkeypatch):
    sp = importlib.import_module(f"{PACKAGE}.sp_norm_adaln")
    monkeypatch.setattr(sp, "_EXT_AVAILABLE", True)

    def extension(missing=None):
        names = {n: staticmethod(lambda *a, **k: None) for n in sp._SYMBOLS if n != missing}
        return type("Extension", (), names)()

    monkeypatch.setattr(sp, "_C", extension())
    assert sp.sp_norm_adaln_available() is True
    for name in sp._SYMBOLS:  # one missing symbol is enough to refuse: the op does not half-work
        monkeypatch.setattr(sp, "_C", extension(missing=name))
        assert sp.sp_norm_adaln_available() is False, name
