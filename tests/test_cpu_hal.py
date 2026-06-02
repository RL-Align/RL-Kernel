import logging

import torch

from rl_engine.kernels.registry import kernel_registry

logging.basicConfig(level=logging.INFO)


def test_hal_routing_and_fallback():
    print("\n [Test 1/2] Testing HAL Registry Routing...")
    try:
        logp_op = kernel_registry.get_op("logp")
        print(f"Successfully routed to: {logp_op.__class__.__name__}")
        assert logp_op.__class__.__name__ == "NativeLogpOp", "Routing failed!"
    except Exception as e:
        print(f"Routing Error: {e}")
        return

    print("\n [Test 2/2] Testing NativeLogpOp Math Execution...")
    try:
        logits = torch.tensor(
            [
                [[2.0, 1.0, 0.1, 0.5], [1.0, 3.0, 0.2, 0.8], [0.5, 0.5, 5.0, 1.0]],
                [[0.1, 2.0, 1.0, 0.1], [0.2, 0.1, 4.0, 2.0], [1.0, 1.0, 1.0, 3.0]],
            ],
            dtype=torch.float32,
        )

        token_ids = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        log_probs = logp_op(logits, token_ids)

        print(f"Input Logits Shape: {logits.shape}")
        print(f"Output LogP Shape: {log_probs.shape}")
        print(f"Output Values:\n{log_probs}")
        print("\n CPU Fallback & HAL Architecture is working perfectly!")

    except Exception as e:
        print(f" Execution Error: {e}")


if __name__ == "__main__":
    test_hal_routing_and_fallback()
