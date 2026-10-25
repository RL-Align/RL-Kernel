"""Verify unchanged forward state and version the intentionally changed gradients."""

import argparse
import json
import torch

p = argparse.ArgumentParser()
p.add_argument("old")
p.add_argument("new")
a = p.parse_args()
old = torch.load(a.old, map_location="cpu", weights_only=True)
new = torch.load(a.new, map_location="cpu", weights_only=True)
assert old.keys() == new.keys()


def mismatch(x, y):
    assert x.shape == y.shape and x.dtype == y.dtype
    dtype = torch.int16 if x.dtype == torch.bfloat16 else torch.int32
    return (x.contiguous().view(dtype) != y.contiguous().view(dtype)).sum().item()


results = {}
for name in old:
    assert len(old[name]) == len(new[name]) == 6
    counts = [mismatch(x, y) for x, y in zip(old[name], new[name])]
    assert counts[:4] == [0, 0, 0, 0], (name, counts)
    results[name] = dict(forward_rstd_mismatches=counts[:4], gradient_changed_elements=counts[4:])
for local in ("False", "True"):
    counts = [mismatch(x, y) for x, y in zip(new[local + "/cuda"], new[local + "/triton"])]
    assert counts == [0] * 6, counts
assert any(any(r["gradient_changed_elements"]) for r in results.values())
print(json.dumps(results, indent=2))
