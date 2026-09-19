# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""User-facing ROCm Qwen3-8B native/consistency launcher."""

from __future__ import annotations

from .run_pr377_workload import main


if __name__ == "__main__":
    raise SystemExit(main())
