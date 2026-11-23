# Talks & Resources

A curated index of technical discussions, recordings, talks, and slides about
RL-Kernel. This page connects public presentations with the relevant technical
documentation so contributors can follow a discussion back to its implementation.

**The [technical documentation](../contributing/README.md) remains the source of
truth.** Talks, recordings, and slides describe the state of the project when
they were presented; review current contracts and validation guides before
reproducing results.

## Community meetings

The [weekly developer meeting](README.md#developer-meeting) takes place every
Saturday at 21:00 Beijing time (UTC+8). Meetings may cover kernel deep dives,
architecture discussions, benchmark reviews, and contributor walkthroughs.

No selected meeting recordings have been indexed yet. Not every meeting needs an
archive entry: add a meeting when it contains reusable technical material and
there is a stable public recording, slide deck, or technical summary to share.

## External talks

Talks are listed **chronologically (oldest first)**. A recap or event page is
not a recording; each link is labeled by its actual type.

| Date | Session | Speaker | Watch / event | Status and context |
| --- | --- | --- | --- | --- |
| **2026-09-26** | **vLLM Classroom #18** — RL-Kernel: Train–Rollout Consistency in LLM RL Post-Training (Chinese) | **Vensen Mu** | [Recording · Bilibili](https://www.bilibili.com/video/BV1Ech96cEzW) · [Companion technical article](https://princepride.github.io/live-streaming-tools/rl_kernel_train_infer_consistency/final/blog/) | **Historical** — RL-Kernel v0.1.0, numerical alignment, and CUDA/ROCm 200-step validation. |
| **2026-10-20** | **Qingke Talk #160** — RL-Kernel: Train–Rollout Consistency in LLM RL Post-Training (Chinese) | **Jiajie Li** | [Recording · YouTube](https://www.youtube.com/watch?v=rDly1tgf_cY) · [Qingke event page](https://www.qingkeai.online/blog/RL-Kernel) | **Historical** — v0.1.0, numerical contracts, fixed replay, cross-platform checks, and design discussion. |
| **2026-11-14** | **DevFest Kuala Lumpur 2026** — Practical Train–Inference Consistency in LLM RL Post-Training (English, in person) | **Vensen Mu** | [Event recap · LinkedIn](https://www.linkedin.com/posts/embedded-llm_devfestkl-reinforcementlearning-opensourceai-activity-7527001816417587200-EyG3) · [Official event page · GDG](https://gdg.community.dev/events/details/google-gdg-kuala-lumpur-presents-devfest-kuala-lumpur-2026/) | **Historical** — practical overview, reported results, and reproducibility. A public session recording is not linked. |

To explore the implementation beyond the talks, see the
[numerical contract](../contracts/cross_config_logprob_drift_contract.md),
[Linear LogP operator](../operators/linear-logp.md),
[reproduction guide](../usage/qwen3-vime-consistency.md), and
[practical guide](../blog/2026-11-02-rl-kernel-practical-guide.md).

## Slides and materials

Presentation decks are hosted externally on Google Slides. Use the
**View slides · Google Slides** links to read the original PPTX decks online.
Sharing permissions are controlled by the file owners; if a file prompts for
access, the owner should set **General access → Anyone with the link → Viewer**.
The documents remain version-specific references, not a replacement for the
latest technical guides.

| Date | Presentation | Slides | Related RL-Kernel docs |
| --- | --- | --- | --- |
| **2026-09-26** | **vLLM Classroom #18** — RL-Kernel v0.1.0 (Vensen Mu, Chinese, 27 slides) | [View slides · Google Slides](https://docs.google.com/presentation/d/1AY8a2UQKWMrLBpqHvaSVz2O6YSJFVJ2V/edit?usp=drive_link&ouid=102623941861710943111&rtpof=true&sd=true) | [Linear LogP](../operators/linear-logp.md) · [Reproduction guide](../usage/qwen3-vime-consistency.md) |
| **2026-10-20** | **Qingke Talk #160** — RL-Kernel v0.1.0 (Jiajie Li, Chinese, 33 slides) | [View slides · Google Slides](https://docs.google.com/presentation/d/1GjbPMtpERVjJuCRjL7ZWfNG_ZSZrxkqp/edit?usp=drive_link&ouid=102623941861710943111&rtpof=true&sd=true) | [Numerical contract](../contracts/cross_config_logprob_drift_contract.md) · [CUDA/ROCm audit](../usage/cuda-rocm-consistency-audit.md) |
| **2026-11-14** | **DevFest Kuala Lumpur** — Practical Train–Inference Consistency (Vensen Mu, English, 19 slides) | [View slides · Google Slides](https://docs.google.com/presentation/d/1tSeeLcYzb75hbX7KwR42k_ozXyV3jb4l/edit?usp=drive_link&ouid=102623941861710943111&rtpof=true&sd=true) | [Practical guide](../blog/2026-11-02-rl-kernel-practical-guide.md) · [CUDA/ROCm audit](../usage/cuda-rocm-consistency-audit.md) |

Other project materials:

- [RL-Kernel v0.1.0 train–rollout consistency technical article](../blog/2026-09-18-rl-kernel-amd-vime-qwen3-8b-train-rollout-bitwise-consistency.md)
  (**Historical**) — published numerical results; consult current validation
  documentation for the latest supported configurations.

## How to add another talk

This page is a **manually maintained index**, not an automatic event feed.
A new talk normally requires only a documentation pull request:

1. Confirm the presentation **date**, title, speaker, host, language, and its
   direct relevance to RL-Kernel.
2. Insert a row under **External talks** in chronological order. Label links
   accurately as a recording, event page, or recap. Do not invent a recording
   when only a social media post exists.
3. When slides are publicly available, add a matching entry under
   **Slides and materials**, again in chronological order. Prefer a stable
   external file link to committing large presentations into Git.
4. Link to relevant [architecture](../architecture/runtime-dispatch.md),
   [operator](../operators/README.md), [numerical contract](../contracts/cross_config_logprob_drift_contract.md),
   validation, benchmark, issue, or PR where applicable.
5. Mark each item **Current**, **Historical**, or **Superseded**, and provide
   a short note about version or implementation limitations.
6. Run `mkdocs build --strict -f mkdocs.yaml`, preview with `mkdocs serve`,
   and submit the documentation change for review.

### Hosting slides on Google Drive

Upload a public PPTX or PDF, open **Share → General access → Anyone with the
link → Viewer**, and copy its file-sharing URL. Check in a private browser
window that it opens without a sign-in prompt. Link to that URL from the
appropriate **Slides and materials** row; do not paste a local file path or a
link accessible only to a private organization.

If slides are revised, update the existing Google Drive file version when
possible to retain its URL, then revisit the status and context note here.
A different file with a new URL requires a small documentation edit and PR.

Do not upload confidential material or rely on short-lived chat attachments.
Slides and recordings are supplementary: the technical docs remain the
authoritative source for current behavior.
