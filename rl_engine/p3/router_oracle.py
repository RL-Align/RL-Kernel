"""Contract-mandated compatibility path for rl_engine.p3.oracle."""

from rl_engine.p3.oracle import finite as finite
from rl_engine.p3.oracle import hash_route_fwd as hash_route_fwd
from rl_engine.p3.oracle import learned_route_fwd as learned_route_fwd
from rl_engine.p3.oracle import normalize as normalize
from rl_engine.p3.oracle import require as require
from rl_engine.p3.oracle import route_bwd as route_bwd
from rl_engine.p3.oracle import router_sqrt_softplus_bwd as router_sqrt_softplus_bwd
from rl_engine.p3.oracle import router_sqrt_softplus_fwd as router_sqrt_softplus_fwd
from rl_engine.p3.oracle import stable_topk6 as stable_topk6
from rl_engine.p3.oracle import tensor as tensor

__all__ = [
    "finite",
    "hash_route_fwd",
    "learned_route_fwd",
    "normalize",
    "require",
    "route_bwd",
    "router_sqrt_softplus_bwd",
    "router_sqrt_softplus_fwd",
    "stable_topk6",
    "tensor",
]
