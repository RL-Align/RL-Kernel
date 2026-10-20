"""P7 ``cross_config`` binding for the MoE router (T08).

P7 already owns the control plane: `Planner` builds the case grid,
`RuntimeTools` materializes a case, `PairedRunner` executes, and `_resume`
handles restart. The contract says T08 supplies a materializer hook and
RoutePlan evidence and does **not** write a second planner, runner or
tolerance, so this module is an adapter and nothing else.

Three pieces:

``P3_ROUTER_KNOBS``
    An *additive* extension of ``V1_KNOBS``, following the precedent set by
    the WS2 attention knobs. ``V1_KNOBS`` has no training-side parallel sizes
    beyond TP/CP and no expert parallelism at all, so the configurations the
    router has to be compared across cannot even be expressed in it. Nothing
    in ``V1_KNOBS`` is edited or deleted.

``P3RouterMaterializer``
    A ``RuntimeMaterializer`` that turns a normalized case into the router's
    view of it: a ``ParallelConfig`` and a ``PlacementMap``, both validated
    before anything is reported as applied. A knob the router does not model
    is reported ``UNOBSERVABLE`` rather than waved through as applied —
    claiming to have applied a rollout-engine knob would make the router
    materializer look like a whole-system one.

Consumer hooks
    ``rank_token_manifest`` and ``materialize_rank_rows`` connect P7's rank
    ownership manifest to ``mocks.materialize_tokens``, which is the hook the
    start kit left for exactly this.

No collective is implemented or wrapped here.
"""

from collections.abc import Mapping

from rl_engine.alignment.cross_config.adapters._common import application, flatten
from rl_engine.alignment.cross_config.planner import (
    _NORMALIZERS,
    V1_KNOBS,
    _positive_int,
    _strict_bool,
)
from rl_engine.alignment.cross_config.runtime import (
    AdapterMaterialization,
    RuntimeBinding,
)
from rl_engine.alignment.cross_config.schema import (
    IsolationScope,
    KnobDescriptor,
    MaterializationStatus,
)

from .collective_provenance import (
    CollectiveProvenance,
    collective_evidence,
    collective_problems,
)
from .contract import P3Error, P3Verdict
from .mocks import materialize_tokens
from .ownership import ParallelConfig, TokenLayout, expected_decisions
from .placement import (
    SEMANTICS,
    SEMANTICS_RANK_LOCAL,
    STRATEGY_BLOCK,
    STRATEGY_INTERLEAVE,
    PlacementMap,
    placement_problems,
)

BINDING_SCHEMA = "p3-p7-binding.v1"
RUNTIME_KIND = "p3_router"

#: Parallel sizes and router placement, added to the V1 allowlist rather than
#: edited into it. TP and CP already exist as training-side knobs in the WS2
#: attention set; PP, DP and EP do not exist anywhere, and the router is the
#: first consumer that needs them.
P3_ROUTER_KNOB_DESCRIPTORS = (
    KnobDescriptor("training.tensor_parallel_size", IsolationScope.PROCESS, ("training",)),
    KnobDescriptor("training.context_parallel_size", IsolationScope.PROCESS, ("training",)),
    KnobDescriptor("training.pipeline_parallel_size", IsolationScope.PROCESS, ("training",)),
    KnobDescriptor("training.data_parallel_size", IsolationScope.PROCESS, ("training",)),
    KnobDescriptor("training.expert_parallel_size", IsolationScope.PROCESS, ("training",)),
    KnobDescriptor("training.sequence_parallel", IsolationScope.PROCESS, ("training",)),
    KnobDescriptor(
        "router.placement_version",
        IsolationScope.ENGINE_CONSTRUCTION,
        ("training", "rollout"),
    ),
    KnobDescriptor(
        "router.placement_strategy",
        IsolationScope.ENGINE_CONSTRUCTION,
        ("training", "rollout"),
        allowed_values=(STRATEGY_BLOCK, STRATEGY_INTERLEAVE),
    ),
    KnobDescriptor(
        # Which reading of physical_expert_id the recorded evidence uses; see
        # docs/design/p3-placement-map.md. Carried as a knob so a comparison
        # can never silently span the two readings.
        "router.placement_semantics",
        IsolationScope.ENGINE_CONSTRUCTION,
        ("training", "rollout"),
        allowed_values=SEMANTICS,
    ),
)

P3_ROUTER_KNOBS = {
    **V1_KNOBS,
    **{descriptor.path: descriptor for descriptor in P3_ROUTER_KNOB_DESCRIPTORS},
}

P3_ROUTER_NORMALIZERS = {
    **_NORMALIZERS,
    "training.tensor_parallel_size": _positive_int,
    "training.context_parallel_size": _positive_int,
    "training.pipeline_parallel_size": _positive_int,
    "training.data_parallel_size": _positive_int,
    "training.expert_parallel_size": _positive_int,
    "training.sequence_parallel": _strict_bool,
}

#: Paths this materializer actually acts on. Everything else in a case is
#: another adapter's and is reported unobservable.
ROUTER_OWNED_PATHS = frozenset(descriptor.path for descriptor in P3_ROUTER_KNOB_DESCRIPTORS) | {
    "batch.size"
}

_PARALLEL_PATHS = {
    "tp": "training.tensor_parallel_size",
    "cp": "training.context_parallel_size",
    "pp": "training.pipeline_parallel_size",
    "dp": "training.data_parallel_size",
    "ep": "training.expert_parallel_size",
}


def parallel_config_from_knobs(flat, rank_order=None):
    """Build a ``ParallelConfig`` from a flattened case.

    Absent dimensions default to 1 rather than raising: a case that varies
    only EP should not have to restate the four sizes it leaves alone, and the
    planner emits exactly the paths the experiment declares.
    """
    sizes = {name: int(flat.get(path, 1)) for name, path in _PARALLEL_PATHS.items()}
    kwargs = dict(sizes, sp=bool(flat.get("training.sequence_parallel", False)))
    if rank_order is not None:
        kwargs["rank_order"] = tuple(rank_order)
    return ParallelConfig(**kwargs)


def placement_from_knobs(flat, num_experts=None):
    """Build the ``PlacementMap`` a case declares, or ``None`` if it declares none.

    A case that names no placement is not given a default one. Inventing a
    placement would make two configurations look comparable on a dimension
    neither of them declared.
    """
    version = flat.get("router.placement_version")
    if not version:
        return None
    ep_size = int(flat.get(_PARALLEL_PATHS["ep"], 1))
    kwargs = {}
    if num_experts is not None:
        kwargs["num_experts"] = num_experts
    return PlacementMap.from_ep_config(
        version,
        ep_size,
        flat.get("router.placement_strategy", STRATEGY_BLOCK),
        flat.get("router.placement_semantics", SEMANTICS_RANK_LOCAL),
        **kwargs,
    )


class P3RouterMaterializer:
    """Materialize the router's view of one cross-config case.

    The router is not an engine: this builds and validates the objects the
    router's ownership and placement contracts are expressed in, and reports
    the configuration that results. Engine construction stays with the
    Megatron and vLLM adapters.
    """

    runtime_kind = RUNTIME_KIND

    def __init__(
        self,
        *,
        layout=None,
        num_experts=None,
        rank_order=None,
        collective=None,
        backend_id="rl_engine.p3",
    ):
        self.layout = layout
        self.num_experts = num_experts
        self.rank_order = rank_order
        self.collective = collective
        self.backend_id = backend_id

    @property
    def implementation_fingerprint(self):
        from rl_engine.kernels.semantic_registry import implementation_fingerprint

        return implementation_fingerprint(
            type(self),
            instance=self,
            entrypoints=("materialize", "router_view"),
        )

    def router_view(self, flat):
        """``(config, placement, problems)`` for a flattened case.

        Problems are returned rather than raised so that each offending knob
        can be reported with its own status; a raise here would collapse the
        whole case into one opaque error.
        """
        problems = []
        config = None
        try:
            config = parallel_config_from_knobs(flat, self.rank_order)
        except P3Error as error:
            problems.append(str(error))

        placement = None
        if config is not None:
            try:
                placement = placement_from_knobs(flat, self.num_experts)
            except P3Error as error:
                problems.append(str(error))
            if placement is not None:
                problems.extend(placement_problems(placement))
        return config, placement, problems

    def materialize(self, normalized, descriptors):
        flat = flatten(normalized)
        config, placement, problems = self.router_view(flat)
        collective, collective_gap, collective_defects = self._collective_for(config)
        problems = problems + collective_defects
        applications = [
            self._application(descriptors, path, value, config, placement, problems, collective_gap)
            for path, value in sorted(flat.items())
        ]

        world_size = config.world_size if config is not None else 1
        side = {
            "schema_version": BINDING_SCHEMA,
            "parallel": config.describe() if config is not None else None,
            "rank_order": list(config.rank_order) if config is not None else None,
            "ownership": (
                {dim: config.ownership_of(dim) for dim in ("tp", "cp", "dp", "pp", "ep")}
                if config is not None
                else None
            ),
            "placement": placement.plan_fragment() if placement is not None else None,
            "problems": problems,
        }
        if collective is not None:
            side.update(collective_evidence(collective))
        if self.layout is not None:
            side["decisions"] = len(self.layout.decisions())
        return AdapterMaterialization(
            applications=tuple(applications),
            binding=RuntimeBinding(
                batch_size=int(flat.get("batch.size", 1)),
                side_configs={"training": side, "rollout": {}},
                topology={
                    "training": {
                        "world_size": world_size,
                        **(
                            {name: getattr(config, name) for name in _PARALLEL_PATHS}
                            if config is not None
                            else {}
                        ),
                        "sequence_parallel": bool(config.sp) if config is not None else False,
                    },
                    "rollout": {"world_size": 1},
                },
                scorer={"mode": "teacher_forcing", "framework": "rl_engine.p3", "unit": "decision"},
                operator_backends={"training": self.backend_id, "rollout": self.backend_id},
                runtime_kind=self.runtime_kind,
            ),
        )

    def _collective_for(self, config):
        """Resolve the gate-gradient collective record for a configuration.

        Returns ``(record, gap, defects)``. A *missing* record under DP is a
        gap, not a defect: the router materializer did not fail to build
        something, it was handed no observation of a reduction that P1 and the
        training framework own. A *malformed* record is a defect, because
        someone did hand over evidence and it does not hold together.

        At DP=1 no all-reduce runs, so the record is legitimately empty.
        """
        if config is None:
            return None, None, []
        if config.dp == 1:
            return CollectiveProvenance.none(dp_size=1), None, []
        if self.collective is None:
            return (
                None,
                f"dp={config.dp} runs a gate-gradient all-reduce but no collective "
                "provenance was supplied; its reduction order is unobserved",
                [],
            )
        return self.collective, None, collective_problems(self.collective)

    def _application(
        self, descriptors, path, value, config, placement, problems, collective_gap=None
    ):
        descriptor = descriptors.get(path)
        if descriptor is None:
            raise P3Error(
                P3Verdict.SCHEMA_MISMATCH,
                f"case path {path!r} has no descriptor; pass P3_ROUTER_KNOBS to RuntimeTools",
            )
        if path not in ROUTER_OWNED_PATHS:
            return application(
                descriptor,
                value,
                value,
                None,
                MaterializationStatus.UNOBSERVABLE,
                f"{path} belongs to an engine adapter; the router materializer cannot observe it",
            )
        if problems:
            return application(
                descriptor,
                value,
                value,
                None,
                MaterializationStatus.ERROR,
                "; ".join(problems),
            )
        if collective_gap is not None and path == _PARALLEL_PATHS["dp"]:
            # The DP size is the knob whose numeric effect the missing record
            # hides, so it is the knob that reports the hole.
            return application(
                descriptor,
                value,
                value,
                None,
                MaterializationStatus.UNOBSERVABLE,
                collective_gap,
            )
        return application(
            descriptor,
            value,
            value,
            value,
            MaterializationStatus.APPLIED,
            f"router view built: {config.describe()}"
            + (f", placement {placement.version}" if placement is not None else ""),
        )


# --- consumer hooks ----------------------------------------------------------


def rank_token_manifest(config, layout):
    """P7's rank ownership manifest for one configuration.

    ``{global_rank: {absolute_layer: (global_token_id, ...)}}`` — the shape
    ``mocks.materialize_tokens`` consumes, one entry per layer because a rank
    under ``pp=1`` owns several.
    """
    manifest = {}
    for rank in range(config.world_size):
        by_layer = {}
        for layer, token in sorted(expected_decisions(config, layout, rank)):
            by_layer.setdefault(layer, []).append(token)
        manifest[rank] = {layer: tuple(tokens) for layer, tokens in by_layer.items()}
    return manifest


def materialize_rank_rows(case, config, layout, rank):
    """Slice a recorded case down to the rows one rank owns at the case's layer.

    Delegates the actual slicing to ``mocks.materialize_tokens``, the hook the
    start kit left for T07/T08, so the row-selection semantics stay in one
    place.

    Two different empty results are kept apart. A rank that owns other layers
    but not this one is working on a different pipeline stage
    (``MISSING_RANK``). A rank that owns nothing at all is over-provisioned —
    more DP ranks than sequences, say — which is a degenerate but legal
    configuration the kit already names ``ZERO_ACTIVE_TOKENS``. Returning an
    empty slice for either would make "this rank has no work" look like
    "this rank completed its work".
    """
    expected = rank_token_manifest(config, layout).get(rank, {})
    if not expected:
        raise P3Error(
            P3Verdict.ZERO_ACTIVE_TOKENS,
            f"rank {rank} owns no decisions at all under {config.describe()}; "
            "the configuration provisions more ranks than the layout can fill",
        )
    tokens = expected.get(case.absolute_layer)
    if tokens is None:
        raise P3Error(
            P3Verdict.MISSING_RANK,
            f"rank {rank} owns layers {sorted(expected)} under {config.describe()}, "
            f"not layer {case.absolute_layer}",
        )
    return materialize_tokens(case, tokens)


def config_from_binding(binding):
    """Recover the ``ParallelConfig`` a materialization recorded.

    Reads the binding's topology rather than re-reading the case, so the
    comparison runs against what the adapter reported it built.
    """
    topology = binding.topology["training"]
    if not isinstance(topology, Mapping) or "tp" not in topology:
        raise P3Error(
            P3Verdict.UPSTREAM_CONTRACT_MISMATCH,
            "binding topology was not produced by the P3 router materializer",
        )
    return ParallelConfig(
        tp=int(topology["tp"]),
        cp=int(topology["cp"]),
        dp=int(topology["dp"]),
        pp=int(topology["pp"]),
        ep=int(topology["ep"]),
        sp=bool(topology.get("sequence_parallel", False)),
    )


def layout_from_case(case, layers):
    """A single-sequence ``TokenLayout`` over one recorded case's active tokens."""
    active = case.global_token_id[case.row_active].tolist()
    return TokenLayout.flat(tuple(dict.fromkeys(int(t) for t in active)), tuple(layers))
