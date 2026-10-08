"""Independent T02–T06 CPU entry points and the synchronous reference wrapper."""

import numpy as np

from . import oracle
from .contract import SAVED_ROUTE_SCHEMA, SAVED_SCORE_SCHEMA, E, K, P3Error, P3OpResult, P3Verdict
from .provenance import cpu_provenance
from .provider import validate_saved


class ReferenceProvider:
    """Independent synchronous CPU operator references; no live kernel registration."""

    provenance = cpu_provenance()

    def _call(self, ctx, fn, *args, saved_version=None, specs=(), row_inputs=()):
        try:
            oracle.tensor(ctx.row_active, "bool", (len(ctx.row_active),))
            for index, dtype, shape in specs:
                oracle.tensor(args[index], dtype, shape)
            if saved_version:
                validate_saved(args[-1], saved_version, ctx)
                oracle.tensor(
                    args[0],
                    "float32",
                    (len(ctx.row_active), E if saved_version == SAVED_SCORE_SCHEMA else K),
                )
            if not ctx.row_active.any():
                return P3OpResult(P3Verdict.ZERO_ACTIVE_TOKENS)
            oracle.require(
                ctx.backend_tag == "recorded-cpu",
                P3Verdict.UNSUPPORTED_CAPABILITY,
                "recorded provider is never a CUDA fallback",
            )
            if saved_version:
                upstream, saved = args
                expected = (len(ctx.row_active), E if saved_version == SAVED_SCORE_SCHEMA else K)
                oracle.tensor(upstream, "float32", expected)
                active = ctx.row_active
                raw = {k: v[active] for k, v in saved.payload.items()}
                small = fn(upstream[active], raw)
                result = {}
                for key, value in small.items():
                    result[key] = np.zeros((len(active), *value.shape[1:]), dtype=value.dtype)
                    result[key][active] = value
            else:
                sanitized = list(args)
                for index in row_inputs:
                    sanitized[index] = args[index].copy()
                    sanitized[index][~ctx.row_active] = 0
                result = fn(*sanitized)
            return P3OpResult(P3Verdict.PASS, result, cpu_provenance(args))
        except P3Error as exc:
            # Discrete runner validation is handled by the checker, not operator ABI.
            verdict = exc.verdict if exc.verdict < 50 else P3Verdict.SCHEMA_MISMATCH
            return P3OpResult(verdict)

    def router_sqrt_softplus_fwd(self, ctx, z, logit_round_point):
        return self._call(
            ctx,
            oracle.router_sqrt_softplus_fwd,
            z,
            logit_round_point,
            specs=((0, "float32", (len(ctx.row_active), E)),),
            row_inputs=(0,),
        )

    def router_sqrt_softplus_bwd(self, ctx, ds, saved_score_sealed):
        return self._call(
            ctx,
            oracle.router_sqrt_softplus_bwd,
            ds,
            saved_score_sealed,
            saved_version=SAVED_SCORE_SCHEMA,
        )

    def hash_route_fwd(self, ctx, input_token_id, s, tid2eid):
        table_rows = len(tid2eid) if isinstance(tid2eid, np.ndarray) and tid2eid.ndim else 0
        return self._call(
            ctx,
            oracle.hash_route_fwd,
            input_token_id,
            s,
            tid2eid,
            specs=(
                (0, "int64", (len(ctx.row_active),)),
                (1, "float32", (len(ctx.row_active), E)),
                (2, "int32", (table_rows, K)),
            ),
            row_inputs=(0, 1),
        )

    def learned_route_fwd(self, ctx, s, b):
        return self._call(
            ctx,
            oracle.learned_route_fwd,
            s,
            b,
            specs=((0, "float32", (len(ctx.row_active), E)), (1, "float32", (E,))),
            row_inputs=(0,),
        )

    def stable_topk6_fwd(self, ctx, q):
        return self._call(
            ctx,
            lambda x: {"ids": oracle.stable_topk6(x)},
            q,
            specs=((0, "float32", (len(ctx.row_active), E)),),
            row_inputs=(0,),
        )

    def hash_route_bwd(self, ctx, dweights, saved_route_sealed):
        return self._call(
            ctx, oracle.route_bwd, dweights, saved_route_sealed, saved_version=SAVED_ROUTE_SCHEMA
        )

    def learned_route_bwd(self, ctx, dweights, saved_route_sealed):
        return self.hash_route_bwd(ctx, dweights, saved_route_sealed)


def router_sqrt_softplus_fwd(ctx, z, logit_round_point):
    return ReferenceProvider().router_sqrt_softplus_fwd(ctx, z, logit_round_point)


def router_sqrt_softplus_bwd(ctx, ds, saved_score_sealed):
    return ReferenceProvider().router_sqrt_softplus_bwd(ctx, ds, saved_score_sealed)


def hash_route_fwd(ctx, input_token_id, s, tid2eid):
    return ReferenceProvider().hash_route_fwd(ctx, input_token_id, s, tid2eid)


def hash_route_bwd(ctx, dweights, saved_route_sealed):
    return ReferenceProvider().hash_route_bwd(ctx, dweights, saved_route_sealed)


def stable_topk6_fwd(ctx, q):
    return ReferenceProvider().stable_topk6_fwd(ctx, q)


def learned_route_fwd(ctx, s, b):
    return ReferenceProvider().learned_route_fwd(ctx, s, b)


def learned_route_bwd(ctx, dweights, saved_route_sealed):
    return ReferenceProvider().learned_route_bwd(ctx, dweights, saved_route_sealed)
