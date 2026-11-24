# SPDX-License-Identifier: Apache-2.0
"""INK-25: logical-visit top-8 replay for the fixed Inkling 2 Mini model."""

import torch

NUM_VISITS = 48
NUM_EXPERTS = 128
TOP_K = 8
ROUTE_SCALE = 8.0
ROUTE_NORM_EPS = 1e-20


def _check_ids(ids: torch.Tensor) -> None:
    if ids.dtype not in (torch.int32, torch.int64) or ids.shape[-1:] != (TOP_K,):
        raise ValueError("expected eight integer expert IDs per route")
    ordered = ids.sort(dim=-1).values
    if bool(((ids < 0) | (ids >= NUM_EXPERTS)).any()):
        raise ValueError("expert ID outside the released 128-expert range")
    if bool((ordered[..., 1:] == ordered[..., :-1]).any()):
        raise ValueError("duplicate expert IDs in a route")


class VisitRouteReplay:
    """Snapshot a rollout trace in explicit logical-visit order [T, 48, 8].

    Token keys must be unique logical token identities (not vocabulary IDs).
    The engine adapter must supply the 48-visit axis explicitly; a physical
    24-layer trace cannot be expanded by guessing. Weight version and token
    identity are checked before replay, including reordered microbatches.
    Metadata preparation is eager and can synchronize on an accelerator.
    """

    def __init__(
        self, token_keys: torch.Tensor, expert_ids: torch.Tensor, *, weight_version: str
    ) -> None:
        if not weight_version or not isinstance(weight_version, str):
            raise ValueError("weight version is required")
        if token_keys.ndim != 1 or token_keys.dtype != torch.int64 or not token_keys.numel():
            raise ValueError("expected nonempty int64 logical token keys")
        if expert_ids.shape != (token_keys.numel(), NUM_VISITS, TOP_K):
            raise ValueError("route trace must have shape [tokens, 48, 8]")
        if token_keys.device != expert_ids.device:
            raise ValueError("trace and token keys must share a device")
        _check_ids(expert_ids)
        keys, order = token_keys.sort()
        if bool((keys[1:] == keys[:-1]).any()):
            raise ValueError("duplicate logical token keys")
        self._keys = keys.detach().clone()
        self._routes = expert_ids[order].detach().long().clone()
        self._version = weight_version

    def for_visit(
        self, logical_visit: int, token_keys: torch.Tensor, *, weight_version: str
    ) -> torch.Tensor:
        """Return the exact recorded slot order for the requested logical tokens."""
        if type(logical_visit) is not int or not 0 <= logical_visit < NUM_VISITS:
            raise ValueError("logical visit must be in [0, 48)")
        if weight_version != self._version:
            raise ValueError("rollout weight version mismatch")
        if token_keys.ndim != 1 or token_keys.dtype != torch.int64:
            raise ValueError("expected int64 logical token keys")
        if token_keys.device != self._keys.device:
            raise ValueError("replay keys must share the trace device")
        indices = torch.searchsorted(self._keys, token_keys.contiguous())
        safe = indices.clamp(max=self._keys.numel() - 1)
        if bool(((indices >= self._keys.numel()) | (self._keys[safe] != token_keys)).any()):
            raise ValueError("logical token missing from rollout trace")
        return self._routes[safe, logical_visit]


def replay_route_weights(logits: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
    """Recompute differentiable FP32 scores on recorded routes, never reselect.

    Mirrors the score-only sigmoid and normalized/scale-8 router semantics;
    the fixed slot fold is a declared candidate, not native bitwise evidence.
    Bias is selection-only and does not belong in these recomputed weights.
    """
    if logits.ndim != 2 or logits.shape[1] != NUM_EXPERTS:
        raise ValueError("expected [tokens, 128] router logits")
    if expert_ids.shape != (logits.shape[0], TOP_K) or expert_ids.device != logits.device:
        raise ValueError("route IDs must match token count and device")
    if logits.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("router logits must be BF16 or FP32")
    _check_ids(expert_ids)
    scores = torch.sigmoid(logits.float()).gather(-1, expert_ids.long())
    denominator = scores[:, 0:1]
    for slot in range(1, TOP_K):
        denominator = denominator + scores[:, slot : slot + 1]
    return scores / (denominator + ROUTE_NORM_EPS) * ROUTE_SCALE
