# Full H3 qualification (in progress)

The existing block-0 evidence remains a single-block result. This extension has
not yet qualified the complete model or generation quality.

`tools.validation.models.h3_full_model` runs the real 50 distinct transformer
layers, two text-refiner layers and input/conditioning/output components. Each
candidate block binds its own parameters to the existing graph. Row/interim/
reference labels are retained; boundary components stay provider/reference.
The runner checks every parameter/input gradient, checkpoints activations,
offloads saved tensors and persists/reclaims completed parameter gradients.

The `.sbatch` scripts request one GPU node and leave the partition, account and
job name to the submitting site, for example
`sbatch --partition=<partition> --account=<account> reference.sbatch ...`.

`reference.sbatch` first runs independent FP32 and BF16 provider passes, then freezes per-tensor error
bounds using the committed formula. It does not run a candidate. Candidate
acceptance compares immutable reference artifacts and records the first failing
tensor without changing bounds. Existing block-0 golden tolerances are unchanged.

Generation uses the pinned public reference image identified in
`generation-inputs.json`, the accompanying prompt and the three seeds/configuration
in `generation-contract.json`. The preflight is explicitly lower resolution.
Full generation keeps the official 124 frames, 1344×768, 24 fps and joint audio.
Every initial packed input and predicted velocity is retained. The comparator
checks identical initial noise, video/audio shape and quality gates and reports
PSNR/SSIM, temporal RMSE and audio differences. Query chunking bounds candidate
attention workspace while retaining all keys for each query; it is inference only.

An independent reference run from `2681cadb` completed the FP32/BF16 full-model
passes (638 parameter gradients each), three-seed generation and repeat comparisons.
`reference-manifest.json` anchors its 1,635 evidence files; `full-model-thresholds.json`
contains the unchanged 764 tensor bounds. These provider results do not establish
candidate acceptance.

`candidate.sbatch` builds its own H3 CUDA extension, records its identity, verifies
all reference/checkpoint hashes, and runs block-0 golden tests without skips before
full-model forward/backward. It requires calls to every candidate graph node in
every block and rejects CUDA backend fallback. The comparator orders first drift
by actual forward/backward execution. Only a passing numerical comparison permits
preflight and full three-seed candidate generation. Each job uses one B200 and
about 512 GB of host memory.

Outstanding: candidate backward qualification, three-seed candidate generation
and quality acceptance. No passed candidate result is claimed by CPU tests.

The generation environment must provide torchvision matching PyTorch (the B200
reference environment uses torch 2.9.1+cu130 and torchvision 0.24.1+cu130).
Qwen3VLProcessor needs its video processor even for the fixed image conditioning.
The runner loads this processor locally before model weights and rejects missing
required pipeline components instead of proceeding with a partially loaded pipeline.
