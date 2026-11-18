# SPDX-License-Identifier: Apache-2.0
"""CPU launch contracts; no claim of GPU execution."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser

ROOT = Path(__file__).resolve().parents[3]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load("examples/operators/dsv4_4l/run_cluster_grpo.py", "cluster_launcher")
topology = load("rl_engine/integrations/orchestrators/verl/pipeline_topology.py", "pp2_topology")


@pytest.mark.parametrize("nodes,gpus,rank,ids", [(1, 2, 0, "4,5"), (2, 1, 0, "2"), (2, 1, 1, "7")])
def test_native_mp_local_slices(nodes, gpus, rank, ids):
    topology.validate_pp2_layout(
        nnodes=nodes, node_rank=rank, gpus_per_node=gpus, visible_devices=ids
    )


@pytest.mark.parametrize(
    "nodes,gpus,rank,ids", [(2, 2, 0, "0,1"), (2, 1, 2, "0"), (1, 2, 0, "0,0"), (2, 1, 0, "0,1")]
)
def test_invalid_local_slices(nodes, gpus, rank, ids):
    with pytest.raises(ValueError):
        topology.validate_pp2_layout(
            nnodes=nodes, node_rank=rank, gpus_per_node=gpus, visible_devices=ids
        )


def test_hydra_launch_preserves_pp2_and_attaches_cluster(tmp_path):
    (tmp_path / "extended-audit-prompts.json").write_text(json.dumps(["a", "b", "c"]))
    args = argparse.Namespace(
        actor_nodes="10.0.0.1,10.0.0.2",
        rollout_nodes="10.0.0.1,10.0.0.2",
        ray_address="auto",
        checkpoint=tmp_path,
        data=tmp_path,
        output=tmp_path / "out",
        stress_final_repeats=1000,
    )
    command, env = launcher.build_command(args)
    parsed = OverridesParser.create().parse_overrides(command[3:])
    values = {p.key_or_group: p.value() for p in parsed}
    assert values["trainer.nnodes"] == 2
    assert values["trainer.n_gpus_per_node"] == 1
    assert values["ray_kwargs.ray_init.address"] == "auto"
    assert values["ray_kwargs.ray_init.num_cpus"] is None
    assert values["rlk.multinode"] is True
    assert env["VLLM_PP_LAYER_PARTITION"] == "3,1"
    assert not any("num_gpus=" in a for a in command)
    args.ray_address = "local"
    with pytest.raises(ValueError, match="existing"):
        launcher.build_command(args)
