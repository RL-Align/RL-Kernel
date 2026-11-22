# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Select H3 video and audio tokens after their respective output projections."""

import torch

from rl_engine.runtime.registry import kernel_registry


def main() -> None:
    """Run both projection outputs through selection and propagate gradients back."""
    # Both projections still have one row per mixed-sequence position.
    video_projected = torch.randn(2, 17, 96, device="cuda", requires_grad=True)
    audio_projected = torch.randn(2, 17, 32, device="cuda", requires_grad=True)
    select = kernel_registry.get_op("index_select", device=video_projected.device)
    # These are sequence positions. Within each index vector every position occurs once.
    video_indices = torch.tensor([1, 3, 5, 7, 9], device=video_projected.device)
    audio_indices = torch.tensor([2, 4, 6, 8], device=audio_projected.device)
    video = select(video_projected, video_indices, dim=1, unique_indices=True)
    audio = select(audio_projected, audio_indices, dim=1, unique_indices=True)
    (video.square().sum() + audio.square().sum()).backward()
    print(f"video={tuple(video.shape)}, audio={tuple(audio.shape)}")
    print(f"backend={type(select).__name__}")


if __name__ == "__main__":
    main()
