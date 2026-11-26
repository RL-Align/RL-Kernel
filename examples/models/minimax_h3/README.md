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

**Acceptance rule (2026-10-14).** The single-configuration bounds in
`full-model-thresholds.json` (1.5 × one provider pass + floor) accept only the provider
configuration they were frozen from: provider passes that differ only in SDPA backend
(flash, memory-efficient) or BLAS library (cuBLASLt) fail 36–66 of them. The first
candidate failed 54, all on single-element `max_abs` except two `relative_l2`. Acceptance
is therefore `full-model-ensemble-rule.json`, calibrated from four distinct provider
configurations with no candidate data (`tools/validation/models/h3_full_ensemble_rule.py`):
every tensor's `relative_l2` must be at most k × the largest provider error + floor,
with k = 2.6 the smallest factor at which every provider configuration held out passes
against the others; `max_abs` is reported but not gated, because held-out providers
exceed the others' maximum by up to 4.1× on it. Many gradient bounds are above 1, and a
gradient that is exactly zero has a relative error of exactly 1, so the check also rejects any
gradient that is exactly zero where the reference is not (the candidate comparison records both
norms as `tensor_l2`; a comparison without them is refused). A full-model node-swap diagnostic
(`--provider-nodes`) found no single node responsible for the candidate's high
tensors: swapping any group moves which tensors land high rather than removing them.
The single-configuration file is kept and still compared, as a record.

**Regenerated reference (2026-10-16).** The artifacts of the first reference run (about 200 GB)
were deleted to save disk space, so the reference was regenerated with the original commands at
the same source commit (`2681cadb`). 1,614 of the 1,635 pinned files are byte-identical to the
first run's: every parameter gradient, both `tensors.pt` and `gradients.json`, every initial
input, every per-step velocity file and the `video.npy` / `audio.npy` that the comparison
reads. The 21 that differ are the eight `report.json` files (timings, job metadata), the
thresholds file that embeds their hashes, and the twelve `video.mp4` / `audio.wav` containers,
which also differ between two runs of one job (the two WAVs differ in two header bytes).
The calibrated bounds are identical to the first run's, and the ensemble rule re-derived from
the regenerated reference (six provider runs compared with it) is identical in every numeric
field. `reference-manifest.json`, `full-model-thresholds.json` and `full-model-ensemble-rule.json`
are re-pinned to the regenerated run; the first run's SHA-256 were: manifest `1640d5c7fccb`,
thresholds `3064c7eb8924`, rule `2e2a4ac58471` and `fp32-reference/report.json` `4fd74feb00ad`
(first 12 hex digits).

`candidate.sbatch` builds its own H3 CUDA extension, records its identity, verifies
all reference/checkpoint hashes, and runs block-0 golden tests without skips before
full-model forward/backward. It requires calls to every candidate graph node in
every block and rejects CUDA backend fallback. The comparator orders first drift
by actual forward/backward execution. Only a passing numerical comparison permits
the generation preflight (the end of `candidate.sbatch`) and full generation.

Full candidate generation is its own job. With the candidate attention it takes
about 20 hours per seed (about 1500 s per sampling step), longer than the three
seeds fit in one 24-hour job, and a preemptible partition can evict a job every few
hours. `generation.sbatch` therefore runs one seed per array task (`--array=0-2`)
and `h3_full_generation --resume` continues after an eviction: every saved step is
replayed from disk instead of recomputed and each replayed call's inputs are checked
against their saved digest (`h3_generation_resume.py`). Recomputing the last saved step
and requiring a bitwise match (`--verify-last-step`) costs a full step per restart, which
on a heavily preempted partition is most of an attempt, so it is opt-in. A run is tied to
the git trees of the model paths, not to a commit (`h3_generation_adopt.py` moves an older
commit-based run over, only if those paths are identical). The task refuses
to start unless the gates of `candidate.sbatch` passed (`<gate job id>` argument) at a commit whose
model paths are byte-identical to this checkout's. After all seeds finish, `generation-compare.sbatch`
runs the quality comparison. Each GPU job uses one B200 and about 512 GB of host
memory.

Status of the full-model qualification (evidence in `reports/experiments/h3-full-model-b200/`):
numerical acceptance (50-block forward/backward under the provider-ensemble rule) has
passed; three-seed candidate generation has run, and its quality comparison has **not**
passed the bounds fixed before generation (seed 44 passes, seeds 42 and 43 do not). The
same bounds are also failed by the provider itself when one attention backend is forced
(flash and memory-efficient: 0 of 3 seeds each), and the first denoising step differs
from the provider's by the same amount (about 0.02 relative L2 on video) for the
candidate as for those provider variants, so the bounds, which had only been checked
against a bitwise-identical provider repeat, were never calibrated for a change of kernel.
They are unchanged here. No passed candidate result is claimed by CPU tests.

The generation environment must provide torchvision matching PyTorch (the B200
reference environment uses torch 2.9.1+cu130 and torchvision 0.24.1+cu130).
Qwen3VLProcessor needs its video processor even for the fixed image conditioning.
The runner loads this processor locally before model weights and rejects missing
required pipeline components instead of proceeding with a partially loaded pipeline.
