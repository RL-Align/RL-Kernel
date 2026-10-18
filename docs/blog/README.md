# Blog

Release notes and engineering writeups for RL-Kernel.

<div class="grid cards" markdown>

-   **[RL-Kernel v0.1.0: Achieving Bitwise Consistency Between Training and Rollout](2026-09-18-rl-kernel-amd-vime-qwen3-8b-train-rollout-bitwise-consistency.md)**

    ---

    How vime and RL-Kernel align numerical execution between training and
    rollout, with operator ablations and 200-step ROCm and CUDA results.

-   **[DSV4 Flash MoE Train-Inference Consistency Roadmap](2026-08-09-dsv4-flash-moe-consistency-roadmap.md)**

    ---

    A roadmap for DeepSeek-V4-Flash MoE consistency, covering expert routing,
    tensor layouts, and reduction order across training and inference.

-   **[Announcing RL-Kernel for vime: Faster and Leaner `linear_logp` for Full RL Rollouts](2026-07-08-announcing-rl-kernel-linear-logp-for-vime.md)**

    ---

    Introducing the fused tensor-parallel `linear_logp` integration for vime
    to reduce CUDA time and memory use in full RL rollout training.

</div>
