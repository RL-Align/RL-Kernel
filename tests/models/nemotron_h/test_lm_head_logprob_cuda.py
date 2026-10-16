# SPDX-License-Identifier: Apache-2.0
"""Nemotron-H strict embedding + LM-head selected-token logprob: forward.

Two orthogonal properties, per RFC section 4.1:

* batch invariance: a row's ``logp``/``lse`` bits do not depend on batch size,
  position, leading shape, repetition, or train-vs-rollout entry (``torch.equal``);
* correctness: ``allclose`` against the independent FP64 oracle at the WS1
  logprob ``forward_accuracy`` tolerance.

The geometry-gate tests run on CPU with meta tensors.
"""
from __future__ import annotations

import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.models.nemotron_h.lm_head_contract import (
    NemotronHGeometryError,
    validate_embedding_geometry,
    validate_lm_head_logprob_geometry,
)
from rl_engine.models.nemotron_h.reference import embedding_reference, selected_logprob_fp64

from .conftest import HIDDEN, VOCAB, make_hidden, make_targets, requires_nemotron_sm90

_TOL = resolve_tolerance(
    load_contract(), judgment="forward_accuracy", op_class="logprob", dtype=torch.bfloat16
)


def _meta(*shape, dtype=torch.bfloat16):
    return torch.empty(*shape, dtype=dtype, device="meta")


class TestGeometryGate:
    def _valid(self):
        return _meta(4, HIDDEN), _meta(VOCAB, HIDDEN), _meta(4, dtype=torch.long)

    def test_accepts_pinned_geometry(self):
        hidden, weight, target = self._valid()
        validate_lm_head_logprob_geometry(hidden, weight, target, embed_weight=_meta(VOCAB, HIDDEN))

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("hidden", _meta(4, HIDDEN + 64), "hidden last dim"),
            ("hidden", _meta(4, HIDDEN, dtype=torch.float32), "hidden must be"),
            ("weight", _meta(VOCAB + 64, HIDDEN), "lm_head_weight must be"),
            ("weight", _meta(VOCAB, HIDDEN, dtype=torch.float16), "lm_head_weight must be"),
            ("weight", _meta(HIDDEN, VOCAB).t(), "contiguous"),
            ("target", _meta(5, dtype=torch.long), "one id per hidden row"),
            ("target", _meta(4, dtype=torch.float32), "target_ids must be int"),
        ],
    )
    def test_rejects_off_contract_inputs(self, field, value, match):
        hidden, weight, target = self._valid()
        args = {"hidden": hidden, "weight": weight, "target": target}
        args[field] = value
        with pytest.raises(NemotronHGeometryError, match=match):
            validate_lm_head_logprob_geometry(args["hidden"], args["weight"], args["target"])

    def test_rejects_bias_softcap_and_tied_embedding(self):
        hidden, weight, target = self._valid()
        with pytest.raises(NemotronHGeometryError, match="no bias"):
            validate_lm_head_logprob_geometry(hidden, weight, target, bias=_meta(VOCAB))
        with pytest.raises(NemotronHGeometryError, match="softcap"):
            validate_lm_head_logprob_geometry(hidden, weight, target, final_logit_softcap=30.0)
        with pytest.raises(NemotronHGeometryError, match="untied"):
            validate_lm_head_logprob_geometry(hidden, weight, target, embed_weight=weight)

    def test_rejects_bad_temperature_and_embedding_ids(self):
        hidden, weight, target = self._valid()
        with pytest.raises(NemotronHGeometryError, match="float32"):
            validate_lm_head_logprob_geometry(
                hidden, weight, target, temperature=_meta(4, dtype=torch.bfloat16)
            )
        with pytest.raises(NemotronHGeometryError, match="scalar or one value"):
            validate_lm_head_logprob_geometry(
                hidden, weight, target, temperature=_meta(3, dtype=torch.float32)
            )
        with pytest.raises(NemotronHGeometryError, match="token_ids must be int"):
            validate_embedding_geometry(_meta(4, dtype=torch.float32), weight)


def _temperature(rows: int) -> torch.Tensor:
    return torch.linspace(0.6, 1.4, rows, device="cuda", dtype=torch.float32)


@requires_nemotron_sm90
class TestForwardCorrectness:
    @pytest.mark.parametrize("use_temperature", [False, True])
    def test_matches_fp64_reference(self, strict_op, lm_head_weight, use_temperature):
        hidden = make_hidden(96, seed=10)
        target = make_targets(96, seed=11)
        temperature = _temperature(96) if use_temperature else None

        logp, lse = strict_op.logprob(hidden, lm_head_weight, target, temperature=temperature)
        ref_logp, ref_lse = selected_logprob_fp64(
            hidden, lm_head_weight, target, temperature=temperature
        )

        assert logp.dtype == torch.float32 and lse.dtype == torch.float32
        torch.testing.assert_close(logp.double(), ref_logp, atol=_TOL.atol, rtol=_TOL.rtol)
        torch.testing.assert_close(lse.double(), ref_lse, atol=_TOL.atol, rtol=_TOL.rtol)
        assert bool((logp <= 0).all())

    def test_embedding_is_a_bitwise_untied_row_copy(self, strict_op, embed_weight):
        token_ids = make_targets(2 * 37, seed=12).reshape(2, 37)

        out = strict_op.embed(token_ids, embed_weight)

        assert out.shape == (2, 37, HIDDEN) and out.dtype == torch.bfloat16
        assert torch.equal(out, embedding_reference(token_ids, embed_weight))

    def test_rejects_tied_weights_and_out_of_range_targets(self, strict_op, lm_head_weight):
        hidden = make_hidden(4, seed=13)
        target = make_targets(4, seed=14)
        with pytest.raises(NemotronHGeometryError, match="untied"):
            strict_op.logprob(hidden, lm_head_weight, target, embed_weight=lm_head_weight)
        bad = target.clone()
        bad[2] = VOCAB
        with pytest.raises(NemotronHGeometryError, match="target_ids must be in"):
            strict_op.logprob(hidden, lm_head_weight, bad)


@requires_nemotron_sm90
class TestForwardBatchInvariance:
    _ROWS = 300

    @pytest.fixture(scope="class")
    def canonical(self, strict_op, lm_head_weight):
        hidden = make_hidden(self._ROWS, seed=20)
        target = make_targets(self._ROWS, seed=21)
        temperature = _temperature(self._ROWS)
        logp, lse = strict_op.logprob(hidden, lm_head_weight, target, temperature=temperature)
        return hidden, target, temperature, logp, lse

    @pytest.mark.parametrize("batch", [1, 2, 7, 64, 255, 257])
    def test_rows_are_invariant_to_batch_size_and_position(
        self, strict_op, lm_head_weight, canonical, batch
    ):
        hidden, target, temperature, ref_logp, ref_lse = canonical
        gen = torch.Generator(device="cpu").manual_seed(batch)
        index = torch.randperm(self._ROWS, generator=gen)[:batch].to("cuda")

        logp, lse = strict_op.logprob(
            hidden[index], lm_head_weight, target[index], temperature=temperature[index]
        )

        assert torch.equal(logp, ref_logp[index])
        assert torch.equal(lse, ref_lse[index])

    def test_leading_shape_and_repeat_runs_do_not_change_bits(
        self, strict_op, lm_head_weight, canonical
    ):
        hidden, target, temperature, ref_logp, ref_lse = canonical

        logp, lse = strict_op.logprob(
            hidden.reshape(3, 100, HIDDEN),
            lm_head_weight,
            target.reshape(3, 100),
            temperature=temperature,
        )

        assert logp.shape == (3, 100)
        assert torch.equal(logp.reshape(-1), ref_logp)
        assert torch.equal(lse.reshape(-1), ref_lse)

    def test_train_and_rollout_entries_agree_bitwise(self, strict_op, lm_head_weight, canonical):
        hidden, target, temperature, ref_logp, ref_lse = canonical

        with torch.no_grad():
            logp, lse, logits = strict_op.logprob_with_logits(
                hidden, lm_head_weight, target, temperature=temperature
            )

        assert torch.equal(logp, ref_logp)
        assert torch.equal(lse, ref_lse)
        assert logits.shape == (self._ROWS, VOCAB) and logits.dtype == torch.float32
        sampled = torch.log_softmax(logits / temperature[:, None], dim=-1)
        sampled = sampled.gather(1, target[:, None]).squeeze(1)
        torch.testing.assert_close(sampled, logp, atol=_TOL.atol, rtol=_TOL.rtol)

    def test_scalar_temperature_matches_expanded_temperature(self, strict_op, lm_head_weight):
        hidden = make_hidden(33, seed=22)
        target = make_targets(33, seed=23)
        scalar = torch.tensor([0.75], device="cuda", dtype=torch.float32)

        scalar_out = strict_op.logprob(hidden, lm_head_weight, target, temperature=scalar)
        expanded_out = strict_op.logprob(
            hidden, lm_head_weight, target, temperature=scalar.expand(33).contiguous()
        )

        assert torch.equal(scalar_out[0], expanded_out[0])
        assert torch.equal(scalar_out[1], expanded_out[1])
