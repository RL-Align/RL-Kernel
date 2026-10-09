# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Implemented native model entrypoints; availability is not acceptance evidence."""


def capabilities():
    from rl_engine.integrations import qwen3_next_native_recompute as recompute
    from rl_engine.integrations import qwen3_next_native_rollout as rollout

    ready = rollout.capabilities()
    if any(
        not callable(getattr(module, name, None))
        for module, names in (
            (rollout, ("run_rollout_case", "run_rollout_cases")),
            (recompute, ("run_recompute_case", "run_recompute_cases")),
        )
        for name in names
    ):
        raise RuntimeError("The native model worker implementation is incomplete")
    return {
        **ready,
        "recompute": True,
        "independent_processes_required": True,
        "full_model_l2_verified": False,
        "training_and_reload_verified": False,
    }


def run_rollout_cases(cases):
    from rl_engine.integrations.qwen3_next_native_rollout import run_rollout_cases as run

    # Preserve the lazy generator: baseline generation freezes response tokens
    # before the subsequent teacher-forced cases are constructed.
    return run(cases)


def run_rollout_case(request, output):
    return run_rollout_cases([(request, output)])


def run_recompute_cases(cases):
    from rl_engine.integrations.qwen3_next_native_recompute import run_recompute_cases as run

    return run(cases)


def run_recompute_case(request, output):
    return run_recompute_cases([(request, output)])
