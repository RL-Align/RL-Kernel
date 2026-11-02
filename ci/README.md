# CI entry points

Run `python ci/run.py layout -q` for CPU packaging and compatibility checks.
`ws1`, `ws1-chain`, and `ws1-ascend` invoke the existing device acceptance
scripts under `ci/scripts`. Their environment variables and fail-closed gates
are unchanged. Historical `ci/run_*.sh` paths remain forwarding wrappers.

GitHub triggers, fork restrictions, secrets handling and GPU provisioning remain
in `.github/workflows`. `ci/jobs`, `runners`, and `providers` reserve future
provider-independent configuration; they do not provision infrastructure.

Use the Dense CUDA/ROCm acceptance guide in `docs/validation/refactor-acceptance.md`
for this PR. Performance differences are reported for human review; no percentage
regression threshold has been agreed. This CPU suite does not certify GPU parity.
