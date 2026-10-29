import pytest
import torch

from rl_engine.backends.rocm.attention.joint_backward import (
    AiterJointBackwardBackend,
    load_upstream_csr,
)
from rl_engine.contracts.operators.joint_attention import JointAttentionError


@pytest.mark.parametrize("rows", [0, 1, 31, 32, 33, 65])
@pytest.mark.parametrize("heads", [1, 13, 128])
def test_declared_sink_reduction_tree(rows, heads):
    torch.manual_seed(25)
    values = torch.randn(rows, heads)
    got = AiterJointBackwardBackend.reduce_sink_rows(values)
    padded = torch.zeros(((rows + 31) // 32, 32, heads))
    padded.flatten(0, 1)[:rows].copy_(values)
    expected = padded.sum(1).sum(0)
    assert torch.equal(got, expected)


def test_nonfinite_sink_is_masked():
    delta = torch.ones(3, 2)
    lse = torch.tensor([[1.0, 2.0], [float("-inf"), float("-inf")], [2.0, 3.0]])
    sink = torch.tensor([0.0, float("-inf")])
    out = AiterJointBackwardBackend.sink_contributions(delta, lse, sink)
    assert torch.isfinite(out).all() and out[:, 1].eq(0).all() and out[1].eq(0).all()


def test_csr_source_guard(tmp_path):
    with pytest.raises(JointAttentionError, match="source"):
        load_upstream_csr(tmp_path / "absent")
    source = tmp_path / "wrapper.py"
    source.write_text("raise RuntimeError('not executed')")
    with pytest.raises(JointAttentionError, match="hash"):
        load_upstream_csr(source)


def test_workspace_formula():
    backend = object.__new__(AiterJointBackwardBackend)
    backend.device = torch.device("cpu")
    expected = backend.workspace_bytes(3, 13, 128, 6, 20)
    ws = backend.allocate(m=3, h=13, d=128, selected=6, n_kv=20, max_bytes=expected)
    actual = sum(t.numel() * t.element_size() for t in vars(ws).values())
    assert actual == expected
    with pytest.raises(JointAttentionError, match="budget"):
        backend.allocate(m=3, h=13, d=128, selected=6, n_kv=20, max_bytes=expected - 1)


def test_engine_exports_shared_vjp_classes():
    from rl_engine.integrations.engines.train.megatron.joint_backward import (
        JointAttentionAutograd,
        JointBackwardPlan,
    )
    from rl_engine.models.deepseek_v4.joint_autograd import JointAttentionAutograd as Shared
    from rl_engine.models.deepseek_v4.joint_backward import JointBackwardPlan as SharedPlan

    assert JointAttentionAutograd is Shared and JointBackwardPlan is SharedPlan


@pytest.mark.parametrize("mode", ["download", "existing", "wrong_existing", "wrong_download"])
def test_explicit_provider_provisioning(tmp_path, monkeypatch, mode):
    import hashlib
    import io
    from pathlib import Path
    import runpy

    provision = runpy.run_path(
        str(Path(__file__).resolve().parents[3] / "scripts/prepare_dsv4_sparse_provider.py")
    )["provision"]
    payload = b"pinned example"
    digest = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "source.py"
    if mode in ("existing", "wrong_existing"):
        target.write_bytes(payload if mode == "existing" else b"other")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: io.BytesIO(b"wrong" if mode == "wrong_download" else payload),
    )
    if mode.startswith("wrong"):
        with pytest.raises(SystemExit, match="hash"):
            provision(target, "https://example.invalid/source", digest, "TEST_SOURCE")
        if mode == "wrong_download":
            assert not target.exists()
        else:
            assert target.read_bytes() == b"other"
    else:
        provision(target, "https://example.invalid/source", digest, "TEST_SOURCE")
        assert target.read_bytes() == payload
