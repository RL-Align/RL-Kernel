# SPDX-License-Identifier: Apache-2.0
"""Bind only real native hash layers; retain gate Parameters and native dispatch."""

import types

import torch

from rl_engine.backends.rocm.moe.native_hash_route import native_hash_route


def bind_native_hash_route(router):
    if not router.is_hash_layer or router.score_function != "sqrtsoftplus" or router.topk != 6:
        raise ValueError("expected original DSV4 Top6 hash-routing layer")
    if (
        router.config.moe_router_force_load_balancing
        or router.config.moe_router_force_biased is not None
    ):
        raise ValueError("benchmark routing overrides are unsupported")

    if router.router_replay is not None:
        raise ValueError("router replay override is unsupported")

    def route(self, logits, input_ids, dense_output=False):
        tokens = input_ids.T.reshape(-1)
        weights, ids = native_hash_route(
            logits, tokens, self.tid2eid, self.config.moe_router_topk_scaling_factor or 1.0
        )
        if not getattr(self, "_rlk_hash_executed", False):
            print(
                "RLK_NATIVE_HASH_ROUTING_EXECUTED",
                __file__,
                "vllm_topk_softplus_sqrt",
                str(weights.dtype),
                flush=True,
            )
            self._rlk_hash_executed = True
        if dense_output:
            return weights, ids.long()
        return (
            torch.zeros_like(logits).scatter(1, ids.long(), weights),
            torch.zeros_like(logits, dtype=torch.bool).scatter(1, ids.long(), True),
        )

    router._hash_routing = types.MethodType(route, router)
    return router
