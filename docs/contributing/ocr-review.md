# Automated PR review

[Open Code Review (OCR)](https://github.com/alibaba/open-code-review) provides an
automated first review of pull requests targeting `main`, `test`, any `test-*`
qualification branch, any `dsv4-*` development branch, or any `p6-*` development
branch. Opening, reopening, or pushing commits to a ready-for-review PR starts a
review. Draft PRs are skipped; marking a draft ready for review starts its first
OCR run. Fork PRs are reviewed when credentials are available. New pushes cancel
older runs for the same PR; the latest head receives the review. Multiple commits
in one push produce one review rather than one review per commit.

OCR posts actionable findings as GitHub review comments and maintains one summary
comment per PR. High and critical findings normally appear inline; medium and low
findings, plus style, documentation and maintainability categories, go into the
summary. Severity comes from the model and is not a guarantee of correctness.
Maintainers decide whether a finding is valid and whether the PR can merge.

After a configured OCR run completes, the workflow updates one reminder comment
on that PR and mentions the PR author from `github.event.pull_request.user.login`.
The reminder asks that contributor to address automated review findings before
maintainers begin manual review; it is resolved per PR and is never hard-coded to
one account.

## Maintainer setup

Merge `.github/workflows/ocr-review.yml`, `.opencodereview/rule.json` and this guide
into the upstream repository. Under **Settings > Secrets and variables > Actions**,
configure these values in the upstream repository, not a contributor's fork:

| Kind | Name | Value |
| --- | --- | --- |
| Secret | `OCR_LLM_URL` | The provider's OCR-compatible model API endpoint |
| Secret | `OCR_LLM_AUTH_TOKEN` | A credential for that endpoint |
| Variable | `OCR_LLM_MODEL` | The model identifier supported by that endpoint |
| Variable | `OCR_LLM_USE_ANTHROPIC` | `true` for Anthropic messages; `false` for OpenAI-compatible chat completions (default) |
| Variable, optional | `OCR_LLM_EXTRA_BODY` | Provider-specific request-body JSON; defaults to `{}` |
| Variable, optional | `OCR_REVIEW_ENABLED` | Set to `false` to disable the job; any other value leaves it enabled |

Follow the provider's endpoint and request-body requirements. OCR sends selected
PR source and repository context to the configured provider and incurs that
provider's API charges. Use a dedicated credential and the provider's spending
limit if available. No separate GitHub personal access token is needed: the job
uses `GITHUB_TOKEN` with `contents: read` and `pull-requests: write`. Repository or
organization Actions policy must allow the pinned OCR action and PR comments.

If the endpoint, credential or model is missing, the job records that review was
not performed in its Actions summary and makes no model call. This is not a clean
review result. Invalid protocol configuration or an OCR execution error fails the
job so the problem is visible. Dependabot PRs can have separate secret restrictions;
do not assume ordinary Actions secrets are available to them.

The integration PR does not review itself using its proposed workflow:
`pull_request_target` loads the workflow from the repository's default branch,
and the pinned action's checkout without an explicit `ref` also uses that
trusted default branch. Keep the workflow and `.opencodereview/rule.json` on
the default branch; the `branches` filter controls which PR target branches
receive the review, so those target branches do not need their own copies. A
rule change made only on `test-*`, `dsv4-*` or `p6-*` is not used until it is
merged into the default branch. After merging
and configuring the upstream, open a small PR and push another commit to verify
both the initial review and update behavior. For already-open PRs, push a new
commit or reopen the PR to trigger review. Maintainers can also re-run an existing
run from the Actions UI.

## Review policy

The project rules prioritize numerical contracts, kernel correctness, runtime
eligibility, distributed execution, train/rollout integration and validation
false passes. Findings should describe the trigger, affected code and observable
impact. A validation gap is distinct from a proven implementation defect. CPU
tests cannot establish GPU parity, and a static review cannot establish measured
speedups or bitwise agreement. Existing CPU/GPU CI and CODEOWNER review remain
the sources of execution evidence and merge decisions.

The rules explicitly include `.cu`, `.cuh` and `.hip` sources and test/validation
files, which OCR's default extension or path filters can otherwise omit.
Includes override those default filters; they are not an exclusive file list.
Rules use first-match order, with focused directories before the catch-all.
`merge_system_rule: false` replaces built-in language prompts with these focused
rules. In particular, OCR's built-in Python prompt discourages exact float
equality and includes style suggestions; intentional exact comparisons enforcing
RL-Kernel's numerical contracts must not generate such findings. Multiple project
rules do not combine with one another.

Every run reviews the full PR diff from its merge-base. `incremental: 'true'`
suppresses overlapping inline comments; it does not reduce review scope or
prove that an earlier finding was fixed. Cross-push checkpoints and automatic
thread resolution are disabled. The model may return no findings even when
defects remain. Large files, unsupported formats, timeouts or exhausted token
budget can leave parts of a PR unreviewed; inspect the summary and artifacts.

The workflow uses medium effort, two concurrent review tasks, a 30-minute job
timeout and a 500,000-token review budget. The token budget covers input and output
and is a dispatch limit, not a hard billing ceiling: in-flight work and final
submission rounds can exceed it. Inspect real usage before changing it. Partial
results must not be interpreted as full coverage. Do not configure OCR as a
required merge check solely because its workflow completed successfully.

## Trust boundary and maintenance

The pinned official composite action checks out the trusted base branch, fetches
the PR head as Git objects and reviews its contents without building or executing
the contributor's code. The explicit rule file comes from that trusted checkout.
Keep GPU provisioning and test execution in their separate workflows. Do not add
a PR-head checkout, dependency installation from the PR, custom executable review
tools, or tests to this privileged job. Repository text is untrusted model input;
review instructions ask the model to treat it as data, and this job grants no
repository contents write permission.

The action commit and OCR CLI version are independently pinned. Update and inspect
both intentionally, including changes to checkout behavior, available tools,
configuration loading and comment publication. The current pins are action
`375f64cd6f67a48a135da83cfcf59c52669a09cb` and CLI `1.13.3`.
The action uploads JSON results and stderr as workflow artifacts; their retention
follows repository policy. Review failures are visible in Actions and its logs.

When changing rules, use the pinned CLI to inspect file selection and matching
without an API key or a billable review:

```bash
ocr review --from origin/main --to HEAD --preview
ocr rules check csrc/cuda/logprob/fused_logp_kernel.cu
ocr rules check tests/runtime/test_dispatch.py
ocr rules check rl_engine/runtime/registry.py
```

For local rule changes, run the preview from the working tree containing those
rules. In production, changes to review rules take effect after they merge into
the trusted base. See the [upstream Action documentation](https://github.com/alibaba/open-code-review/blob/375f64cd6f67a48a135da83cfcf59c52669a09cb/examples/github_actions/README.md)
for provider configuration and publication behavior.
