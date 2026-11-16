# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Hydra config groups that select the ``rlkernel`` verl engine (``verl_engine.py``).

    python -m verl.trainer.main_ppo \\
        'hydra.searchpath=[pkg://rl_engine.integrations.dsv4.verl_config]' \\
        model_engine=rlkernel \\
        actor_rollout_ref.model.external_lib=rl_engine.integrations.dsv4.verl_engine ...

Each file extends verl's own group default (``actor``, ``ref``, ``dp_critic``).
"""
