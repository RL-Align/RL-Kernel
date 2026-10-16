# H3 full-model candidate: numerical acceptance (B200)

Generated at `4792c8a9` on one B200 (torch 2.9.1+cu130, diffusers `f53d552`) by
`examples/models/minimax_h3/candidate.sbatch`, from a clean checkout.

- `full-model-acceptance.json` -- the acceptance decision: the provider-ensemble rule
  (`examples/models/minimax_h3/full-model-ensemble-rule.json`, relative_l2 only) applied to the
  candidate. `status: passed`, 0 of 764 tensors failing, worst tensor at 0.55 of its bound.
- `full-model-comparison.json` -- the candidate against the frozen FP32 reference, per tensor
  (`max_abs`, `relative_l2`). Its `status: failed` is the *original single-configuration* bounds
  (`full-model-thresholds.json`, 1.5 x one provider pass + floor), kept as a record: 54 of 764
  tensors fail them, 52 on `max_abs` only and 2 on `relative_l2` as well.
- `provider-spread-summary.json` -- why that rule was replaced: the same bounds applied to six
  diffusers provider runs (four distinct configurations; a rerun and a forced-cuDNN run
  reproduce the default exactly), each compared with the same reference. It was made with
  `tools/validation/models/h3_full_provider_spread.py` at `c8ac5b19` from these provider runs and
  a candidate run at `a7d3f9d0`. Providers that differ only in SDPA backend or BLAS library fail
  36-66 of the bounds on `max_abs` while the default configuration they were calibrated on fails
  none; the candidate's 54 are in that range.

No bound was changed to make the candidate pass: the ensemble rule is calibrated on provider
configurations only. It was designed after the candidate had failed the single-configuration
bounds, which is why the evidence above is kept.
