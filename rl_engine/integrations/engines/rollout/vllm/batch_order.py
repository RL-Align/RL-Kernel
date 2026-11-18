# SPDX-License-Identifier: Apache-2.0
"""Instance-local eager prefill ordering for real AB/BA worker validation."""

import re
from types import FunctionType, MethodType


def requested_prefill_order(tokens, drafts, decode_query_len, requested, original):
    if len(requested) < 2:
        return original(tokens, drafts, decode_query_len)
    if len(set(requested)) != len(requested) or len(requested) != len(tokens):
        raise ValueError("requested real batch must contain exactly the scheduled request IDs")
    selected = []
    for name in requested:
        # Pinned InputProcessor adds exactly eight random hex characters to external IDs.
        matches = [
            key
            for key in tokens
            if key == name
            or (key.startswith(name + "-") and re.fullmatch(r"[0-9a-f]{8}", key[len(name) + 1 :]))
        ]
        if len(matches) != 1:
            raise ValueError(f"ambiguous/missing scheduled request: {name}; actual={list(tokens)}")
        selected.append(matches[0])
    if set(selected) != set(tokens):
        raise ValueError("requested-to-native request mapping is incomplete")
    if drafts or any(value <= decode_query_len for value in tokens.values()):
        raise ValueError(
            "explicit order supports only ordinary prefill batches, not decode/speculation"
        )
    return selected


def bind_prefill_order(runner, get_order):
    original = runner.gather_batch_req_state.__func__
    if "sort_batch_req_ids" not in original.__code__.co_names:
        raise ValueError("expected pinned vLLM native request-state gather boundary")
    native_sort = original.__globals__["sort_batch_req_ids"]

    def ordered(tokens, drafts, decode_query_len):
        requested = get_order()
        selected = requested_prefill_order(tokens, drafts, decode_query_len, requested, native_sort)
        if len(requested) > 1:
            runner._rlk_prefill_order_evidence.append(
                dict(requested=list(requested), actual=selected, num_scheduled_tokens=dict(tokens))
            )
        return selected

    cloned = FunctionType(
        original.__code__,
        dict(original.__globals__, sort_batch_req_ids=ordered),
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    cloned.__kwdefaults__ = original.__kwdefaults__
    runner._rlk_prefill_order_evidence = []
    runner.gather_batch_req_state = MethodType(cloned, runner)
