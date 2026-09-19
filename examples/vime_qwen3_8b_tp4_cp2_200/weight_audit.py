# SPDX-License-Identifier: Apache-2.0
"""Audit every exported parameter, not a sampled tensor."""
import hashlib
import json
import os
from pathlib import Path


def record_weight_update(args, version_dir, rollout_engines):
    import torch
    import torch.distributed as dist
    from safetensors import safe_open
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    root = Path(version_dir)
    tensors = {}
    for file in sorted(root.glob('*.safetensors')):
        with safe_open(file, framework='pt', device='cpu') as archive:
            for name in archive.keys():
                tensor = archive.get_tensor(name).contiguous()
                tensors[name] = {
                    'sha256': hashlib.sha256(memoryview(tensor.view(torch.uint8).numpy())).hexdigest(),
                    'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                }
    if not tensors:
        raise RuntimeError(f'No tensors to audit in {root}')
    digest = hashlib.sha256(json.dumps(tensors, sort_keys=True).encode()).hexdigest()
    output = Path(os.environ['RL_KERNEL_WEIGHT_AUDIT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{root.name}.json').write_text(json.dumps({
        'version': root.name, 'scope': 'all_exported_parameters',
        'sha256': digest, 'tensor_count': len(tensors), 'tensors': tensors,
    }, indent=2)+'\n')
