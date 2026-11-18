# SPDX-License-Identifier: Apache-2.0
"""Build committed fork sources on the exact pinned ROCm dependency image."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rlk", type=Path)
    parser.add_argument("verl", type=Path)
    parser.add_argument("context", type=Path)
    parser.add_argument("tag")
    parser.add_argument("--base", default="rlk/dsv4-4l-rocm:306c0ba-verl-grpo-0d5a434")
    parser.add_argument(
        "--base-id",
        default="sha256:c426b78eff37ff3830829bd53e23d8adb1f570321d46f156c940037fc469a968",
    )
    args = parser.parse_args()
    if run("docker", "image", "inspect", "-f", "{{.Id}}", args.base) != args.base_id:
        raise ValueError("base image ID differs from the recorded dependency image")
    args.context.mkdir(parents=True, exist_ok=False)
    revisions = {}
    for name, repo, base, bundle in [
        ("RL-Kernel", args.rlk, "306c0bafe5be667d06989a163304c45496cc7daf", "rlk.bundle"),
        ("verl", args.verl, "0d5a434cfefd1b83e543a4061590e36a36b513df", "verl.bundle"),
    ]:
        if run("git", "-C", str(repo), "status", "--porcelain"):
            raise ValueError(f"{name}: commit all intended changes before building")
        revisions[name] = run("git", "-C", str(repo), "rev-parse", "HEAD")
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "bundle",
                "create",
                str(args.context.resolve() / bundle),
                "HEAD",
                "^" + base,
            ],
            check=True,
        )
    shutil.copy2(
        args.rlk / "examples/operators/dsv4_4l/Dockerfile.grpo", args.context / "Dockerfile"
    )
    command = [
        "docker",
        "build",
        "--network=none",
        "--pull=false",
        "--progress=plain",
        "--build-arg",
        "BASE_IMAGE=" + args.base,
        "--build-arg",
        "RLK_SHA=" + revisions["RL-Kernel"],
        "--build-arg",
        "VERL_SHA=" + revisions["verl"],
        "-t",
        args.tag,
        str(args.context),
    ]
    subprocess.run(command, check=True)
    record = dict(
        tag=args.tag,
        id=run("docker", "image", "inspect", "-f", "{{.Id}}", args.tag),
        source_shas=revisions,
        base_tag=args.base,
        base_id=args.base_id,
        build_command=command,
    )
    (args.context / "image.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
