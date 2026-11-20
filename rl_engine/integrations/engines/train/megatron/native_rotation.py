# SPDX-License-Identifier: Apache-2.0
"""Bind the existing native ROCm rotation to this model's CSA indexer only."""

import types

from rl_engine.backends.rocm.attention.rotation import native_hadamard128


def _bind(module, method_name):
    original = getattr(module, method_name).__func__
    if "rotate_activation" not in original.__code__.co_names:
        raise ValueError(f"unsupported native CSA rotation ABI: {method_name}")
    namespace = dict(original.__globals__, rotate_activation=native_hadamard128)
    bound = types.FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    bound.__kwdefaults__ = original.__kwdefaults__
    setattr(module, method_name, types.MethodType(bound, module))


def bind_indexer_rotation(indexer):
    _bind(indexer, "forward_before_topk")
    _bind(indexer.compressor, "forward")
    return indexer
