# SPDX-License-Identifier: Apache-2.0
import torch

from rl_engine.contracts.operators.indexer_topk import TOPK_BACKEND_ID


def reference_topk(scores, lengths, k):
    rows = []
    for row, length in zip(scores.cpu().tolist(), lengths.cpu().tolist(), strict=True):
        candidates = [(-value, i) for i, value in enumerate(row[:length]) if value != float("-inf")]
        chosen = sorted((i for _, i in sorted(candidates)[:k]), reverse=True)
        rows.append(chosen + [-1] * (k - len(chosen)))
    return torch.tensor(rows, dtype=torch.int32).reshape(len(rows), k)


class ReferenceTopKBackend:
    backend_id = TOPK_BACKEND_ID
    device_type = "cpu"

    @staticmethod
    def plan(contract):
        return {"reference": True}

    def select(self, scores, lengths, k):
        return reference_topk(scores, lengths, k)
