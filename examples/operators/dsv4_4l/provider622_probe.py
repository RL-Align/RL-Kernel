# SPDX-License-Identifier: Apache-2.0
"""Evaluate PR 622 collectors on the real checkpoint, without GPU claims."""

import argparse
import json
from rl_engine.contracts.strict_provider import StrictProviderError
from rl_engine.runtime.provenance.provider_identity import checkpoint_identity, file_inventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    result = dict(
        pr_head="468bf2ddf6eef54f2e77e16a89b830892633c917",
        evidence_kind="disk_checkpoint",
        actual_worker_integration=False,
        actual_gpu_execution_verified=False,
    )
    try:
        result["indexed_checkpoint"] = checkpoint_identity(
            args.checkpoint,
            model_id="DeepSeek-V4-Flash-4l",
            revision="60d8d70770c6776ff598c94bb586a859a38244f1",
        )
    except StrictProviderError as error:
        result["indexed_collector_blocker"] = str(error)
    result["single_file_inventory"] = file_inventory(
        args.checkpoint,
        ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"],
    )
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
