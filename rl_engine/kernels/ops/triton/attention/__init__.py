# SPDX-License-Identifier: Apache-2.0
"""Compatibility import; the canonical module is loaded below."""

from importlib import import_module


def __getattr__(name):
    if name == "TritonJointAttnSoftmaxOp":
        module = import_module("rl_engine.backends.cuda.attention.joint_attn_softmax_triton")
        return getattr(module, name)
    return getattr(import_module("rl_engine.backends.shared.triton.attention"), name)
