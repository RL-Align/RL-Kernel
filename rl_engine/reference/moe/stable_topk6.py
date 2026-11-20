# SPDX-License-Identifier: Apache-2.0
"""Independent CPU lexicographic oracle; no packed-key implementation reuse."""

import torch


def stable_topk6_reference(scores, global_expert_ids=None):
    values = scores.detach().cpu()
    ids = (
        torch.arange(values.shape[1], dtype=torch.int64)
        if global_expert_ids is None
        else global_expert_ids.detach().cpu().long()
    )
    columns = []
    for row in values:
        # Python float represents every FP32 value exactly; +/-0 compare equal.
        chosen = sorted(range(values.shape[1]), key=lambda col: (-float(row[col]), int(ids[col])))[
            :6
        ]
        columns.append(chosen)
    column_tensor = torch.tensor(columns, dtype=torch.int64).reshape(values.shape[0], 6)
    return values.gather(1, column_tensor), ids[column_tensor], column_tensor
