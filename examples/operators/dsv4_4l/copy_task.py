# SPDX-License-Identifier: Apache-2.0
"""Actual reference-based copying-task reward, without stochastic shaping."""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source != "rlk/exact-copy":
        raise ValueError(data_source)
    reference = ground_truth
    previous = list(range(len(reference) + 1))
    for i, char in enumerate(solution_str, 1):
        current = [i]
        for j, target in enumerate(reference, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != target))
            )
        previous = current
    distance = previous[-1]
    score = 1.0 - distance / max(len(reference), len(solution_str), 1)
    return dict(score=score, edit_distance=distance, exact_match=float(solution_str == reference))


def prepare_data(destination):
    from pathlib import Path
    import json
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq

    r = Path(destination)
    r.mkdir(parents=True, exist_ok=True)
    (r / "replay-fixed-sequences.json").write_bytes(
        Path(__file__).with_name("acceptance-cases.json").read_bytes()
    )
    references = [
        "the quick brown fox jumps over the lazy dog",
        "pack my box with five dozen liquor jugs",
        "a red balloon floats above the green field",
        "the small bird sings a cheerful song",
        "bring the blue book to the quiet room",
        "we walked along the river before sunset",
    ]
    rows = [
        dict(
            data_source="rlk/exact-copy",
            prompt=[
                dict(
                    role="user", content="Copy this text exactly, without any explanation:\n" + text
                )
            ],
            ability="text-copy",
            reward_model=dict(style="rule", ground_truth=text),
            extra_info=dict(index=i, split="train"),
        )
        for i, text in enumerate(references)
    ]
    for name, subset in [("train", rows), ("val", rows[:2])]:
        p = r / (name + ".parquet")
        pq.write_table(pa.Table.from_pylist(subset), p)
    (r / "manifest.json").write_text(
        json.dumps(
            dict(
                task=(
                    "Exact text copying; normalized character Levenshtein similarity "
                    "to the explicitly supplied reference."
                ),
                source=(
                    "Deterministic six instruction/reference pairs, constructed for functional "
                    "integration, not a benchmark or training-benefit claim."
                ),
                rows=rows,
                sha256={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in r.glob("*.parquet")
                },
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    prepare_data(args.output)
