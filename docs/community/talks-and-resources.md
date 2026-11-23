# Talks & Resources

A curated index of technical discussions, recordings, and presentation materials
related to RL-Kernel. These resources preserve the context behind engineering
decisions and help contributors find a starting point for further investigation.

**The [technical documentation](../contributing/README.md) remains the source of
truth.** Talks and slides describe the implementation and results at the time
they were presented; they are not a substitute for current contracts, operator
documentation, or validation evidence.

## Community meetings

The [weekly developer meeting](README.md#developer-meeting) takes place on
Saturdays at 21:00 Beijing time (UTC+8). It is an opportunity for kernel deep
dives, architecture discussions, benchmark reviews, and contributor walkthroughs.

No selected meeting recordings or slide decks are indexed here yet. Not every
meeting needs an archive entry: add one when it contains reusable technical
material, and include the date, topic, public recording or notes, status,
relevant documentation, and any version-specific limitations.

## External talks

Recordings can remain on the host community's platform; this page provides the
stable project-facing entry point.

| Date | Talk | Recording and host | Status | Related RL-Kernel docs | Context |
| --- | --- | --- | --- | --- | --- |
| 2026-10-20 | **RL-Kernel: Train–Rollout Consistency in LLM RL Post-Training** (Qingke Talk #160, Li Jiajie, Chinese) | [Recording (YouTube)](https://www.youtube.com/watch?v=rDly1tgf_cY) · [Qingke event page](https://www.qingkeai.online/blog/RL-Kernel) | **Historical** | [Numerical contract](../contracts/cross_config_logprob_drift_contract.md) · [Linear LogP](../operators/linear-logp.md) · [Reproduction guide](../usage/qwen3-vime-consistency.md) | Discusses v0.1.0, fixed replay, operator-level alignment, and 200-step CUDA/ROCm validation. Consult current docs for supported configurations. |

## Slides and materials

Slides shared only in private chats are not listed until a stable public link is available. A resource does not need an accompanying recording to be useful. Link directly
to public slides, diagrams, notes, or conference materials when they help
contributors understand a technical decision.

| Date | Material | Source | Status | Related RL-Kernel docs | Context |
| --- | --- | --- | --- | --- | --- |
| 2026-09 | [RL-Kernel v0.1.0 train–rollout consistency technical article](../blog/2026-09-18-rl-kernel-amd-vime-qwen3-8b-train-rollout-bitwise-consistency.md) | RL-Kernel documentation | **Historical** | [Train–rollout practical guide](../blog/2026-11-02-rl-kernel-practical-guide.md) · [CUDA/ROCm consistency audit](../usage/cuda-rocm-consistency-audit.md) | Project-maintained written background and validation results from v0.1.0; newer instructions and supported configurations are documented separately. |

## Contributing and maintaining entries

Add only materials directly relevant to RL-Kernel and suitable for public
sharing. Prefer durable links to the original host; do not commit large video
files or slide decks to the repository. Small, project-owned PDFs may be kept
under `docs/assets/` when appropriate.

For each new entry, include:

1. **Date and title:** when the meeting happened or material was published;
   identify the speaker, host, or language where helpful.
2. **Resource link:** a public recording, slide deck, notes, or conference page.
   Do not imply a recording or deck exists if it is not published.
3. **Project links:** relevant architecture, numerical contracts, operators,
   validations, benchmarks, issues, or PRs. Prefer links to current project docs.
4. **Status and context:** use **Current** for material aligned with the current
   documentation, **Historical** for a version-specific snapshot, or
   **Superseded** when a newer resource replaces it. Include a brief note
   explaining the version or limitation; preserve useful historical context
   rather than silently removing it.

When the implementation changes, review affected entries and update their
status or context note. If an external link is no longer publicly reachable,
replace it with an authoritative source or mark it unavailable rather than
pointing contributors to an unrelated resource.

To suggest an entry or correct a link, edit this page in a documentation PR or
open a [GitHub issue](https://github.com/RL-Align/RL-Kernel/issues).
