#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --partition=overflow
#SBATCH --account=wrd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00

set -euo pipefail
experiment_source=$1
trainer_python=$2
runtime_archive=$3
result_root=$4
experiment_results="$result_root/job-${SLURM_JOB_ID}-attempt-${SLURM_RESTART_COUNT:-0}"
mkdir -p "$experiment_results"
runtime_root=$(mktemp -d "${TMPDIR:-/tmp}/rl-kernel-ppo-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$runtime_root"' EXIT
tar -xf "$runtime_archive" -C "$runtime_root"
cd "$experiment_source"
export PYTHONPATH="$experiment_source:$runtime_root/frameworks"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

scontrol show job "$SLURM_JOB_ID" > "$experiment_results/slurm-job.txt"
nvidia-smi -q > "$experiment_results/nvidia-smi.txt"
nvidia-smi topo -m > "$experiment_results/gpu-topology.txt"
"$trainer_python" -c 'import torch; assert torch.cuda.device_count() == 4, torch.cuda.device_count(); print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(4)])'

for framework in megatron deepspeed; do
    srun --ntasks=1 "$trainer_python" -m torch.distributed.run \
        --standalone --nproc-per-node=4 \
        -m tests.distributed.tp.check_tp_ppo_rollout \
        --framework "$framework" --critic-tp 1 --gpu-layout dedicated \
        --vllm-python "$runtime_root/vllm/bin/python" \
        --output "$experiment_results/$framework" \
        > "$experiment_results/$framework.log" 2>&1
done

"$trainer_python" - "$experiment_results" <<'PY'
import json
import sys
from pathlib import Path
import torch

root = Path(sys.argv[1])
comparison = {}
for framework in ('megatron', 'deepspeed'):
    summaries = json.loads((root/framework/'summary.json').read_text())
    assert len(summaries) == 4
    assert {row['cuda_device'] for row in summaries} == {0, 1, 2, 3}
    assert all(row['gpu_layout'] == 'dedicated' for row in summaries)
    runtime = json.loads((root/framework/'vllm/report.json').read_text())
    assert runtime['installed'] == [0, 1, 2]
    assert runtime['sampled'] == [0, 1]
    actor = next(row for row in summaries if row['rank'] == 0)
    comparison[framework] = {
        'max_behavior_logprob_error': max(actor['behavior_logprob_max_errors']),
        'round_seconds': [r['metrics']['round_seconds'] for r in actor['rounds']],
        'engine_steps': actor['engine_steps'],
        'runtime': runtime,
    }
for role in ('actor', 'critic'):
    native = torch.load(root/'megatron'/f'{role}.pt', weights_only=True)
    engine = torch.load(root/'deepspeed'/f'{role}.pt', weights_only=True)
    for name in native:
        torch.testing.assert_close(native[name], engine[name], atol=2e-5, rtol=2e-4)
    comparison[f'{role}_max_difference'] = max(
        (native[name]-engine[name]).abs().max().item() for name in native
    )
(root/'comparison.json').write_text(json.dumps(comparison, indent=2))
(root/'PASS').write_text('Dedicated four-GPU Megatron/DeepSpeed/vLLM validation passed.\n')
print('PASS: dedicated Actor TP=2, Critic TP=1, vLLM TP=1', flush=True)
PY
