"""The fixed-rollout candidate and the memory probe run the TP sizes in fixed_matrix.sh."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = Path("tools/validation/models/qwen3_ppo")
ROWS = [
    f"{engine}-{precision}-{role}"
    for engine in ("megatron", "deepspeed")
    for precision in ("fp32", "bf16")
    for role in ("actor", "critic")
]
TP4_ONLY = {"megatron-bf16-actor", "deepspeed-bf16-actor", "deepspeed-bf16-critic"}


def _sizes(row: str) -> list[int]:
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{ROOT / SCRIPTS}/fixed_matrix.sh"; fixed_tp_sizes "$@"',
            "-",
            *row.split("-"),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [int(tp) for tp in out.split()]


@pytest.mark.parametrize("row", ROWS)
def test_bf16_actors_and_deepspeed_bf16_critic_are_tp4_only(row):
    assert _sizes(row) == ([4] if row in TP4_ONLY else [2, 4])


@pytest.mark.parametrize("row", ROWS)
def test_python_row_sizes_equal_fixed_matrix_sh(row):
    """The replay and comparison checks use the same per-row sizes as the batch scripts."""
    from tools.validation.models.qwen3_ppo.fixed_contract import fixed_tp_sizes

    assert list(fixed_tp_sizes(*row.split("-"))) == _sizes(row)


def _stage(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """A clean git tree holding the scripts, a python stand-in that logs every call, and an env."""
    source = tmp_path / "src"
    shutil.copytree(ROOT / SCRIPTS, source / SCRIPTS)
    git = ["git", "-C", str(source), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "scripts"], check=True)
    calls = tmp_path / "calls.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text(
        "#!/usr/bin/env bash\n"
        f'if [[ "$1" == - ]]; then exec "{sys.executable}" "$@"; fi\n'
        f'echo "$*" >> "{calls}"\n'
    )
    (bin_dir / "nvidia-smi").write_text("#!/usr/bin/env bash\n")
    for tool in bin_dir.iterdir():
        tool.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    (tmp_path / "out").mkdir()
    return source, calls, env


def _launches(calls: Path, probe: bool) -> list[tuple[str, int]]:
    rows = []
    for line in calls.read_text().splitlines():
        words = line.split()
        if "tests.distributed.tp.check_qwen3_fixed" not in words:
            continue
        assert ("--probe" in words) == probe
        tp = int(next(w for w in words if w.startswith("--nproc_per_node=")).split("=")[1])
        row = "-".join(words[words.index(k) + 1] for k in ("--framework", "--precision", "--role"))
        rows.append((row, tp))
    assert len(rows) == len(set(rows))
    return rows


@pytest.mark.parametrize(
    ("sizes", "expected"),
    [
        ((), {(row, 4) for row in TP4_ONLY} | {(row, 2) for row in ROWS if row not in TP4_ONLY}),
        (("2",), {(row, 2) for row in ROWS if row not in TP4_ONLY}),
        (("4",), {(row, 4) for row in ROWS}),
    ],
)
def test_probe_runs_each_row_at_its_sizes(tmp_path, sizes, expected):
    source, calls, env = _stage(tmp_path)
    args = [str(source), "python", "overlay", "ckpt", "ref", str(tmp_path / "out"), *sizes]
    subprocess.run(
        ["bash", str(source / SCRIPTS / "probe_trainer_memory.sh"), *args], check=True, env=env
    )
    assert set(_launches(calls, probe=True)) == expected


def test_probe_summary_lists_launches_that_wrote_no_report(tmp_path):
    """A launch that fails before its checker makes the output directory still gets a row."""
    source, _, env = _stage(tmp_path)
    args = [str(source), "python", "overlay", "ckpt", "ref", str(tmp_path / "out")]
    subprocess.run(
        ["bash", str(source / SCRIPTS / "probe_trainer_memory.sh"), *args], check=True, env=env
    )
    (summary,) = (tmp_path / "out").glob("probe-*/summary.tsv")
    rows = summary.read_text().splitlines()[1:]
    assert len(rows) == len(ROWS)
    assert all("\tno report (launch rc=0" in row for row in rows)


def test_candidate_skips_tp2_of_tp4_only_rows_and_their_tp_comparison(tmp_path):
    source, calls, env = _stage(tmp_path)
    args = [
        str(source),
        "python",
        "vllm",
        "overlay",
        "ckpt",
        "ref",
        "acc.json",
        str(tmp_path / "out"),
    ]
    subprocess.run(
        ["bash", str(source / SCRIPTS / "qualify_candidate.sbatch"), *args], check=True, env=env
    )
    expected = {(row, tp) for row in ROWS for tp in (2, 4)} - {(row, 2) for row in TP4_ONLY}
    assert set(_launches(calls, probe=False)) == expected
    comparisons = [line for line in calls.read_text().splitlines() if "--tp-comparison" in line]
    assert len(comparisons) == len(ROWS) - len(TP4_ONLY)
    assert not any(f"{row}-tp-comparison" in line for line in comparisons for row in TP4_ONLY)
