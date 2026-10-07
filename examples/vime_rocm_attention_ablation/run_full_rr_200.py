from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/workspace/RL-Kernel-pr390")
from examples.vime_rocm_attention_ablation import (  # noqa: E402 -- checkout path above
    run_full_rr_single_arm_v90 as base,
)

if __name__ == "__main__":
    matrix_config = base.MatrixConfig
    base.MatrixConfig = lambda **kwargs: matrix_config(num_rollout=200, **kwargs)
    base.RUN_DIR = Path("/app/model/vime-runs/pr393-200round-r-r")
    sys.exit(base.main())
