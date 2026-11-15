#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Add a strict bitwise log-prob metric to verl (idempotent).

verl's ``training/rollout_probs_diff_*`` compares ``exp(log_prob)`` and ``rollout_corr/k3_kl``
rounds tiny log-ratios to 0 in FP32, so "0" there does not prove equal bits. This patch adds,
next to them in ``verl/utils/debug/metrics.py::calculate_debug_metrics``:

    training/rollout_logprob_tokens              response tokens compared
    training/rollout_logprob_bitwise_mismatch    tokens whose FP32 bits differ
                                                 (rollout_log_probs vs old_log_probs)
    training/rollout_logprob_absdiff_max         max |rollout - old| (FP32)

Usage: python3 patch_verl_bitwise_metric.py /path/to/verl
"""

import pathlib
import sys

MARK = "# [dsv4-e2e] bitwise log-prob metric"
ANCHOR = "    response_mask_bool = response_mask.bool()\n"
COMPUTE = f"""    {MARK}
    _rl = rollout_old_log_probs.float().contiguous()
    _ol = actor_old_log_probs.float().contiguous()
    _bw = {{
        "training/rollout_logprob_tokens": int(response_mask_bool.sum().item()),
        "training/rollout_logprob_bitwise_mismatch": int(
            ((_rl.view(torch.int32) != _ol.view(torch.int32)) & response_mask_bool).sum().item()
        ),
        "training/rollout_logprob_absdiff_max": float(
            torch.where(response_mask_bool, (_rl - _ol).abs(), torch.zeros_like(_rl)).max().item()
        ),
    }}
"""
RET_OLD = '        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,\n    }\n'
RET_NEW = RET_OLD.replace("    }\n", "        **_bw,\n    }\n")


def main() -> int:
    path = pathlib.Path(sys.argv[1]) / "verl/utils/debug/metrics.py"
    src = path.read_text()
    if MARK in src:
        print(f"{path}: already patched")
        return 0
    if src.count(ANCHOR) != 1 or src.count(RET_OLD) != 1:
        print(f"{path}: anchors not found; verl changed, patch by hand", file=sys.stderr)
        return 1
    src = src.replace(ANCHOR, ANCHOR + COMPUTE).replace(RET_OLD, RET_NEW)
    path.write_text(src)
    print(f"{path}: patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
