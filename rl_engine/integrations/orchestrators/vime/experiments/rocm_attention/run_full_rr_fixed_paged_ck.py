"""One PR377-workload R/R iteration with the fixed CK paged candidate."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rl_engine.integrations.orchestrators.vime.experiments.rocm_attention import (
    run_full_rr_single_arm_v90 as base,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tile", choices=["64", "128"], default="128")
    args = parser.parse_args()
    os.environ["RL_KERNEL_ROCM_FIXED_PAGED_TILE"] = args.tile
    os.environ["RL_KERNEL_ROCM_PAGED_KV_MAX_TOKENS"] = "8192"
    config_type = base.MatrixConfig

    def config(**kwargs):
        kwargs.update(
            num_rollout=1,
            rollout_batch_size=1,
            samples_per_prompt=8,
            global_batch_size=8,
            max_response_length=7168,
            max_tokens_per_gpu=4096,
            rollout_seed=1234,
        )
        return config_type(**kwargs)

    base.MatrixConfig = config
    base.RUN_DIR = args.run_dir
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
