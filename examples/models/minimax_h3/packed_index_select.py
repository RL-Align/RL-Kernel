# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Run with PYTHONPATH=. python examples/models/minimax_h3/packed_index_select.py."""

import torch

from rl_engine.runtime.registry import kernel_registry


def main():
    select = kernel_registry.get_op("packed_index_select", device="cuda")
    # Stand-ins for proj_out(hidden_states) and audio_proj_out(hidden_states).
    video_projected = torch.randn(2, 17, 96, device="cuda", requires_grad=True)
    audio_projected = torch.randn(2, 17, 32, device="cuda", requires_grad=True)
    video_indices = torch.tensor([1, 3, 5, 7, 9], device="cuda")
    audio_indices = torch.tensor([2, 4, 6, 8], device="cuda")
    video = select(video_projected, video_indices)
    audio = select(audio_projected, audio_indices)
    (video.square().sum() + audio.square().sum()).backward()
    print(f"video={tuple(video.shape)}, audio={tuple(audio.shape)}")
    print(f"backend={type(select).__name__}, kernel={select.kernel_id}")


if __name__ == "__main__":
    main()
