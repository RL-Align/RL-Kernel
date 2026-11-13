# SPDX-License-Identifier: Apache-2.0
"""Scope learned score arithmetic to this native router's existing routing call."""

import types

import torch

from rl_engine.backends.rocm.moe.native_learned_route import native_learned_route


def bind_native_learned_route(router):
    if router.is_hash_layer or router.topk != 6 or router.score_function != "sqrtsoftplus":
        raise ValueError("expected original learned DSV4 Top6 router")
    original = router.routing.__func__
    if "topk_routing_with_score_function" not in original.__code__.co_names:
        raise ValueError("unsupported native learned routing ABI")

    def topk(
        logits,
        topk,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=None,
        score_function=None,
        expert_bias=None,
        fused=False,
        router_replay=None,
        topk_indices=None,
        dense_output=False,
    ):
        if (
            topk != 6
            or num_groups not in (None, 1)
            or group_topk not in (None, 1)
            or fused
            or router_replay is not None
            or topk_indices is not None
        ):
            raise ValueError("initial learned route requires ordinary ungrouped eager Top6")
        weights, ids = native_learned_route(logits, expert_bias, scaling_factor or 1.0)
        if not getattr(router, "_rlk_learned_executed", False):
            print(
                "RLK_NATIVE_LEARNED_ROUTING_EXECUTED",
                __file__,
                "vllm_topk_softplus_sqrt",
                flush=True,
            )
            router._rlk_learned_executed = True
        if dense_output:
            return weights, ids.long()
        return (
            torch.zeros_like(logits).scatter(1, ids.long(), weights),
            torch.zeros_like(logits, dtype=torch.bool).scatter(1, ids.long(), True),
        )

    cloned = types.FunctionType(
        original.__code__,
        dict(original.__globals__, topk_routing_with_score_function=topk),
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    cloned.__kwdefaults__ = original.__kwdefaults__
    router.routing = types.MethodType(cloned, router)
    return router
