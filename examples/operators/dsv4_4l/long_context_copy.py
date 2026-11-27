# SPDX-License-Identifier: Apache-2.0
"""Long-context copy: deterministic token-level reference reward.

reward = 1 - EditDistance(out_ids, target_ids) / max(len(out_ids), len(target_ids), 1)

* ``out_ids`` are the actually generated response token ids (not decoded text), passed as
  ``extra_info["response_ids"]``; ``target_ids`` are the tokenizer encoding of the passage
  (``add_special_tokens=False``) made at request-preparation time (``extra_info["target_ids"]``).
* EOS handling: exactly one trailing EOS id (``extra_info["eos_token_id"]``) is stripped before
  scoring; every other token, special tokens included, is scored. ``eos_terminated`` records whether
  that EOS was present; ``truncated`` is ``len(response) == max_tokens`` without EOS.
* Nothing is sampled, shaped or rank-clipped: the score is a pure function of (response, target).
* The distance is exact Levenshtein (insert/delete/substitute, unit cost) computed with the
  Myers/Hyyro bit-parallel algorithm on Python big ints: O(len(out) * ceil(len(target)/word)).
  ``tests/examples/test_dsv4_long_context_copy.py`` cross-checks it against a plain O(n*m) DP.
"""

DATA_SOURCE = "rlk/long-context-copy"


def edit_distance(out_ids, target_ids):
    """Exact Levenshtein distance between two integer sequences (global alignment)."""
    m = len(target_ids)
    if m == 0:
        return len(out_ids)
    if not out_ids:
        return m
    mask = (1 << m) - 1
    high = 1 << (m - 1)
    peq = {}
    for i, symbol in enumerate(target_ids):
        peq[symbol] = peq.get(symbol, 0) | (1 << i)
    pv, mv, score = mask, 0, m
    for symbol in out_ids:
        eq = peq.get(symbol, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & high:
            score += 1
        elif mh & high:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = mh | (~(xv | ph) & mask)
        mv = ph & xv
    return score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source != DATA_SOURCE:
        raise ValueError(data_source)
    info = extra_info or {}
    response = [int(t) for t in info["response_ids"]]
    target = [int(t) for t in info["target_ids"]]
    eos = info.get("eos_token_id")
    eos_terminated = bool(response) and eos is not None and response[-1] == eos
    cap = info.get("max_tokens")
    truncated = bool(cap is not None and len(response) == cap and not eos_terminated)
    out = response[:-1] if eos_terminated else response
    distance = edit_distance(out, target)
    return dict(
        score=1.0 - distance / max(len(out), len(target), 1),
        edit_distance=distance,
        out_len=len(out),
        target_len=len(target),
        eos_terminated=eos_terminated,
        truncated=truncated,
        exact_match=float(out == target),
    )
