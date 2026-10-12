# Full H3 qualification (in progress)

The existing block-0 evidence remains a single-block result. This extension has
not yet qualified the complete model or generation quality.

`tools.validation.models.h3_full_model` runs the real 50 distinct transformer
layers, two text-refiner layers and input/conditioning/output components. Each
candidate block binds its own parameters to the existing graph. Row/interim/
reference labels are retained; boundary components stay provider/reference.
The runner checks every parameter/input gradient, checkpoints activations,
offloads saved tensors and persists/reclaims completed parameter gradients.

`reference.sbatch` is restricted to `overflow`, with job name `test`. It first
runs independent FP32 and BF16 provider passes, then freezes per-tensor error
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

Outstanding: actual full-model GPU/reference calibration, candidate backward
qualification, three-seed reference/candidate generation and quality acceptance.
No passed full-model result is claimed by the new CPU binding tests.
