# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""User-facing ROCm Qwen3-8B native/consistency launcher."""

from __future__ import annotations

from rl_engine.integrations.orchestrators.vime.experiments.rocm_attention.run_pr377_workload import (  # noqa: E501
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
