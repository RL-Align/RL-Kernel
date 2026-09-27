# SPDX-License-Identifier: Apache-2.0
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from rl_engine.p6.contract import (
    Context,
    ContractError,
    CombinePlan,
    SavedForward,
    production_provider,
)
from rl_engine.p6.fixtures import Generator, RecordedProvider, evaluate, load_golden, make_case
from rl_engine.p6.oracle import (
    add,
    backward,
    bf16_bits,
    f32,
    forward,
    gradient_dispatch,
    matrix_hex,
    mock_return,
)

ROOT = Path(__file__).resolve().parent


class TestArithmetic(unittest.TestCase):
    def test_fp32_rounding_each_step(self):
        self.assertEqual(add(add(2**24, 1), -(2**24)), 0)
        self.assertEqual(add(2**24, add(1, -(2**24))), 1)

    def test_bf16_even_tie(self):
        self.assertEqual(bf16_bits(1 + 2**-8), 0x3F80)

    def test_bf16_odd_tie(self):
        self.assertEqual(bf16_bits(1 + 3 * 2**-8), 0x3F82)

    def test_bf16_signed_zero(self):
        self.assertEqual(bf16_bits(-0.0), 0x8000)
        self.assertEqual(bf16_bits(0.0), 0)

    def test_nonfinite_and_overflow(self):
        for value in (float("nan"), float("inf"), -float("inf"), 1e100):
            with self.subTest(value=value), self.assertRaises(ContractError):
                f32(value)

    def test_subnormal_explicitly_unsupported(self):
        with self.assertRaisesRegex(ContractError, "UNSUPPORTED_CAPABILITY"):
            f32(2**-149)


class TestPlanAndOracle(unittest.TestCase):
    def setUp(self):
        self.case = make_case("unit", n=3, h=7, mode="mixed")
        self.plan = CombinePlan.from_dict(self.case["plan"])

    def forward(self, case=None, plan=None, context=None):
        c = self.case if case is None else case
        p = self.plan if plan is None else plan
        return forward(
            p, c["rows"], c["shared"], c["residual"], p.context if context is None else context
        )

    def test_plan_json_roundtrip(self):
        p = CombinePlan.from_dict(json.loads(json.dumps(self.plan.to_dict())))
        p.validate(self.plan.context)
        self.assertEqual(p, self.plan)
        self.assertEqual(p.fingerprint, self.plan.fingerprint)

    def test_unknown_schema(self):
        with self.assertRaisesRegex(ContractError, "SCHEMA_MISMATCH"):
            self.forward(plan=replace(self.plan, schema="old"))

    def test_unknown_field(self):
        p = self.plan.to_dict()
        p["weight_again"] = True
        with self.assertRaises(ContractError):
            CombinePlan.from_dict(p)

    def test_policy_drift(self):
        with self.assertRaisesRegex(ContractError, "SCHEMA_MISMATCH"):
            self.forward(plan=replace(self.plan, policy_hash="0" * 64))

    def test_wrong_run_microbatch_forward_checkpoint(self):
        for field in Context.__dataclass_fields__:
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ContractError, "STALE_RUN_METADATA"),
            ):
                self.forward(context=replace(self.plan.context, **{field: "other"}))

    def test_duplicate_token(self):
        with self.assertRaisesRegex(ContractError, "INVALID_DISCRETE_PLAN"):
            self.forward(plan=replace(self.plan, token_ids=(self.plan.token_ids[0],) * 3))

    def test_duplicate_or_missing_route(self):
        indices = [i for i, r in enumerate(self.plan.inverse_map) if r[2]]
        inv = list(self.plan.inverse_map)
        inv[indices[1]] = inv[indices[0]]
        for rows in (tuple(inv), self.plan.inverse_map[:-1]):
            with self.subTest(rows=rows), self.assertRaises(ContractError):
                self.forward(plan=replace(self.plan, inverse_map=rows))

    def test_out_of_range_or_foreign_token(self):
        inv = list(self.plan.inverse_map)
        index = next(i for i, r in enumerate(inv) if r[2])
        for row in ((self.plan.token_ids[0], 6, True), (0, 0, True)):
            bad = list(inv)
            bad[index] = row
            with self.subTest(row=row), self.assertRaises(ContractError):
                self.forward(plan=replace(self.plan, inverse_map=tuple(bad)))

    def test_bool_not_integer_mask(self):
        with self.assertRaises(ContractError):
            self.forward(plan=replace(self.plan, valid_slots=((1,) * 6,) * 3))

    def test_padding_sentinel(self):
        inv = list(self.plan.inverse_map)
        index = next(i for i, r in enumerate(inv) if not r[2])
        inv[index] = (0, 0, False)
        with self.assertRaises(ContractError):
            self.forward(plan=replace(self.plan, inverse_map=tuple(inv)))

    def test_invalid_payload_not_loaded(self):
        changed = copy.deepcopy(self.case)
        for i, row in enumerate(self.plan.inverse_map):
            if not row[2]:
                changed["rows"][i] = [float("nan")] * 7
        self.assertEqual(self.forward(changed)["stages"], self.forward()["stages"])

    def test_active_nan_rejected(self):
        changed = copy.deepcopy(self.case)
        index = next(i for i, r in enumerate(self.plan.inverse_map) if r[2])
        changed["rows"][index][0] = float("nan")
        with self.assertRaisesRegex(ContractError, "NON_FINITE"):
            self.forward(changed)

    def test_non_bf16_input_rejected(self):
        changed = copy.deepcopy(self.case)
        changed["shared"][0][0] = 1 + 2**-10
        with self.assertRaisesRegex(ContractError, "DTYPE_MISMATCH"):
            self.forward(changed)

    def test_payload_shape_rejected(self):
        changed = copy.deepcopy(self.case)
        changed["rows"][0] = [1.0]
        with self.assertRaisesRegex(ContractError, "UNSUPPORTED_GEOMETRY"):
            self.forward(changed)

    def test_arrival_invariance(self):
        baseline = self.forward()
        for seed in range(5):
            case = copy.deepcopy(self.case)
            case["rows"] = mock_return(
                self.plan,
                case["rows"],
                Generator(seed).shuffle(range(len(case["rows"]))),
                self.plan.context,
            )
            result = self.forward(case)
            self.assertEqual(result["stages"], baseline["stages"])
            self.assertEqual(result["trace"], baseline["trace"])

    def test_duplicate_arrival_rejected(self):
        with self.assertRaises(ContractError):
            mock_return(
                self.plan, self.case["rows"], [0] * len(self.case["rows"]), self.plan.context
            )

    def test_physical_permutation_preserves_logical_order(self):
        order = list(reversed(range(len(self.case["rows"]))))
        p = replace(self.plan, inverse_map=tuple(self.plan.inverse_map[i] for i in order))
        c = copy.deepcopy(self.case)
        c["rows"] = [c["rows"][i] for i in order]
        self.assertEqual(self.forward(c, p)["stages"], self.forward()["stages"])
        self.assertEqual(p.order_hash, self.plan.order_hash)
        self.assertNotEqual(p.fingerprint, self.plan.fingerprint)

    def test_saved_forward_roundtrip_backward(self):
        fwd = self.forward()
        saved = SavedForward(**json.loads(json.dumps(fwd["saved"].__dict__)))
        result = backward(
            saved,
            self.case["dx_rows"],
            self.case["dx_shared"],
            self.plan.context,
            self.plan.fingerprint,
            self.plan.gradient_boundary,
        )
        self.assertEqual(result["trace"]["plan_fingerprint"], self.plan.fingerprint)
        self.assertEqual(result["output_boundary"], self.plan.gradient_boundary)

    def test_saved_corruption_and_wrong_fingerprint(self):
        saved = self.forward()["saved"]
        for bad, expected in (
            (replace(saved, fingerprint="0" * 64), self.plan.fingerprint),
            (replace(saved, plan_json="{"), self.plan.fingerprint),
            (saved, "0" * 64),
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(ContractError, "CORRUPT_METADATA"):
                bad.restore(self.plan.context, expected)

    def test_backward_rejects_other_input_boundary(self):
        with self.assertRaisesRegex(ContractError, "GRADIENT_BOUNDARY_MISMATCH"):
            backward(
                self.forward()["saved"],
                self.case["dx_rows"],
                self.case["dx_shared"],
                self.plan.context,
                self.plan.fingerprint,
                "raw-residual-input",
            )

    def test_backward_rejects_stale_forward(self):
        with self.assertRaisesRegex(ContractError, "STALE_RUN_METADATA"):
            self.forward()["saved"].restore(
                replace(self.plan.context, forward_id="other"), self.plan.fingerprint
            )

    def test_gradient_dispatch_same_dy_per_slot(self):
        result = gradient_dispatch(
            self.forward()["saved"], self.case["dy"], self.plan.context, self.plan.fingerprint
        )
        table = dict(zip(self.plan.token_ids, self.case["dy"], strict=True))
        for row, (t, _s, valid) in zip(result, self.plan.inverse_map, strict=True):
            self.assertEqual(row, table[t] if valid else [0.0] * 7)

    def test_production_provider_fail_closed(self):
        with self.assertRaisesRegex(ContractError, "UNSUPPORTED_CAPABILITY"):
            production_provider()


class TestGoldenAndArtifact(unittest.TestCase):
    def setUp(self):
        self.golden = load_golden(ROOT / "data" / "golden.v1.json")

    def test_all_golden_intermediates(self):
        for record in self.golden["payload"]["cases"]:
            with self.subTest(case=record["input"]["name"]):
                self.assertEqual(evaluate(record["input"]), record["expected"])

    def test_cancellation_distinguishes_presum(self):
        record = next(
            r
            for r in self.golden["payload"]["cases"]
            if r["input"]["name"] == "cancellation-slot-order"
        )
        self.assertEqual(record["expected"]["forward"]["output_bf16"], "0040")
        values = [float(2**24), 1.0, -float(2**24), 3.0, -3.0, 2.0]
        rank_a = add(add(values[0], values[2]), values[4])
        rank_b = add(add(values[1], values[3]), values[5])
        self.assertNotEqual(matrix_hex([[add(rank_a, rank_b)]], "bf16"), "0040")

    def test_known_bf16_rounding_outputs(self):
        record = next(
            r
            for r in self.golden["payload"]["cases"]
            if r["input"]["name"] == "one-round-bf16-ties"
        )
        self.assertEqual(record["expected"]["forward"]["output_bf16"], "803f823f813f")

    def test_signed_zero_raw_bytes(self):
        record = next(
            r for r in self.golden["payload"]["cases"] if r["input"]["name"] == "signed-zero"
        )
        self.assertEqual(record["expected"]["forward"]["output_bf16"], "00800000")

    def test_recorded_stub_binding(self):
        record = self.golden["payload"]["cases"][0]
        provider = RecordedProvider(record["input"], record["expected"])
        self.assertEqual(provider.run(record["input"]), record["expected"])
        changed = copy.deepcopy(record["input"])
        changed["shared"][0][0] += 1
        with self.assertRaisesRegex(ContractError, "IDENTITY_DRIFT"):
            provider.run(changed)

    def test_corrupt_golden_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            bad = copy.deepcopy(self.golden)
            bad["payload"]["cases"][0]["input"]["shared"][0][0] += 1
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ContractError, "CORRUPT_ARTIFACT"):
                load_golden(path)

    def test_recorded_corrupt_expected_rejected(self):
        record = self.golden["payload"]["cases"][0]
        bad = copy.deepcopy(record["expected"])
        bad["forward"]["output_bf16"] = "0000"
        with self.assertRaisesRegex(ContractError, "CORRUPT_ARTIFACT"):
            RecordedProvider(record["input"], bad)

    def test_early_rounding_changes_result(self):
        from rl_engine.p6.oracle import bf16_value

        reference = bf16_bits(add(add(1.0, 2**-8), 2**-8))
        premature = bf16_bits(add(bf16_value(add(1.0, 2**-8)), 2**-8))
        self.assertNotEqual(reference, premature)

    def test_empty_and_all_invalid_cases(self):
        for name in ("zero-tokens", "zero-routes"):
            record = next(r for r in self.golden["payload"]["cases"] if r["input"]["name"] == name)
            with self.subTest(name=name):
                result = evaluate(record["input"])
                self.assertEqual(result, record["expected"])


if __name__ == "__main__":
    unittest.main()
