# SPDX-License-Identifier: Apache-2.0
"""Long-context copy reward: exact token edit distance, bounded, and no length/repetition exploit."""

import itertools
import random
import time

import pytest

from examples.operators.dsv4_4l.long_context_copy import compute_score, edit_distance

SOURCE = "rlk/long-context-copy"
EOS = 1
TARGET = [10 + (i * 7) % 23 for i in range(60)]


def dp_distance(a, b):
    """Independent plain O(n*m) Levenshtein reference."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def score(response, target=TARGET, cap=None):
    info = dict(response_ids=response, target_ids=target, eos_token_id=EOS, max_tokens=cap)
    return compute_score(SOURCE, "ignored text", "ignored", info)


def test_bit_parallel_matches_dp_random_and_edge_cases():
    rng = random.Random(0)
    cases = [([], []), ([], [1]), ([1], []), ([1], [1]), ([1], [2]), ([1, 2], [2, 1])]
    for _ in range(600):
        alphabet = rng.choice([2, 3, 8, 50])
        a = [rng.randrange(alphabet) for _ in range(rng.randrange(0, 40))]
        b = [rng.randrange(alphabet) for _ in range(rng.randrange(0, 70))]
        cases.append((a, b))
    for a in itertools.product(range(3), repeat=3):
        cases.append((list(a), [0, 1, 2, 0]))
    big = [rng.randrange(100) for _ in range(130)]  # more than one 64-bit word
    cases += [(big, big[::-1]), (big[3:], big), (big + big, big)]
    for a, b in cases:
        assert edit_distance(a, b) == dp_distance(a, b), (a, b)
        assert edit_distance(b, a) == dp_distance(b, a)


def test_exact_copy_is_one_with_and_without_eos():
    for response in (TARGET, TARGET + [EOS]):
        result = score(response)
        assert result["score"] == 1.0 and result["exact_match"] == 1.0 and result["edit_distance"] == 0
    assert score(TARGET + [EOS])["eos_terminated"] and not score(TARGET)["eos_terminated"]


def test_only_one_trailing_eos_is_stripped_and_other_tokens_kept():
    assert score(TARGET + [EOS, EOS])["edit_distance"] == 1
    assert score([EOS] + TARGET + [EOS])["edit_distance"] == 1
    assert score([5, 6, EOS], [5, 6])["score"] == 1.0


def test_empty_and_eos_only_output_score_zero():
    for response in ([], [EOS]):
        result = score(response)
        assert result["score"] == 0.0 and result["out_len"] == 0 and result["exact_match"] == 0.0


def test_single_token_repetition_does_not_win():
    best = max(score([t] * n)["score"] for t in set(TARGET) for n in (1, 5, len(TARGET), 7168))
    assert best < 0.5
    assert score([TARGET[0]] * 7168)["score"] < 0.01


def test_repeating_target_twice_is_worse_than_once_and_bounded():
    twice = score(TARGET + TARGET)
    assert twice["score"] == pytest.approx(0.5)
    assert twice["score"] < score(TARGET)["score"]
    assert score(TARGET * 10)["score"] < twice["score"]


def test_prefix_truncation_is_monotone_and_shorter_never_beats_complete_copy():
    scores = [score(TARGET[:n])["score"] for n in range(0, len(TARGET) + 1)]
    assert scores == sorted(scores) and scores[-1] == 1.0
    assert score(TARGET[:30])["score"] == pytest.approx(0.5)


def test_truncation_at_cap_flag_and_score():
    cap = 40
    truncated = score(TARGET[:cap], cap=cap)
    assert truncated["truncated"] and not truncated["eos_terminated"]
    assert truncated["score"] == pytest.approx(cap / len(TARGET))
    finished = score(TARGET[: cap - 1] + [EOS], cap=cap)
    assert finished["eos_terminated"] and not finished["truncated"]


def test_early_eos_with_gibberish_is_below_aligned_output():
    gibberish = [900 + i for i in range(5)]
    early = score(gibberish + [EOS])
    assert early["score"] == 0.0 and early["eos_terminated"]
    assert score(TARGET[:20] + [EOS])["score"] > early["score"]
    assert score([900 + i for i in range(7168)])["score"] == 0.0


def test_score_in_unit_interval_and_deterministic():
    rng = random.Random(1)
    for _ in range(200):
        response = [rng.randrange(40) for _ in range(rng.randrange(0, 200))]
        a, b = score(response), score(response)
        assert a == b and 0.0 <= a["score"] <= 1.0


def test_rejects_other_sources():
    with pytest.raises(ValueError):
        compute_score("rlk/exact-copy", "a", "a", dict(response_ids=[], target_ids=[]))


def test_large_batch_is_fast():
    rng = random.Random(2)
    target = [rng.randrange(129000) for _ in range(1000)]
    began = time.monotonic()
    for _ in range(128):
        response = [rng.randrange(129000) for _ in range(7168)]
        assert 0.0 <= score(response, target, cap=7168)["score"] <= 1.0
    assert time.monotonic() - began < 30.0
