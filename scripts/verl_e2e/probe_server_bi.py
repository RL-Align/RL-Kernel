#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Black-box batch-invariance probe for a running vLLM OpenAI server.

Eight fixed probe prompts are decoded greedily (a) one at a time, then (b) concurrently with
K filler requests of random lengths (several K, shuffled order). With batch invariance the
generated tokens AND their log-probs must be identical in every setting. Run it against the
BI server (expect PASS) and the non-BI server (a control: expect differences).

    python3 probe_server_bi.py --port 8000 --out probe_bi.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROBES = [
    "Prove that the sum of the first n odd numbers is n^2.",
    "Let f(x) = x^3 - 3x + 1. Find the number of real roots of f and justify your answer.",
    "A bag has 5 red and 7 blue balls. Two are drawn without replacement. What is the "
    "probability that both are the same color?",
    "Explain why the harmonic series diverges.",
    "Find all integers n such that n^2 + 2n + 12 is a perfect square.",
    "What is the remainder when 7^2026 is divided by 13?",
    "Compute the integral of x * e^x from 0 to 1.",
    "In triangle ABC, AB = 13, BC = 14, CA = 15. Find the radius of the inscribed circle.",
]
WORDS = ("alpha beta gamma delta epsilon theorem lemma integer prime modulo triangle circle "
         "probability expectation matrix vector limit series sequence function derivative").split()


def complete(port, model, prompt, max_tokens, timeout=1800):
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
        "logprobs": 1, "seed": 0, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        c = json.load(r)["choices"][0]
    lp = c["logprobs"]
    return {"tokens": lp["tokens"], "logprobs": lp["token_logprobs"]}


def compare(ref, got):
    """(same, first differing position or None, kind)."""
    pairs = zip(ref["tokens"], got["tokens"], ref["logprobs"], got["logprobs"])
    for i, (ta, tb, la, lb) in enumerate(pairs):
        if ta != tb:
            return False, i, "token"
        if la != lb:
            return False, i, f"logprob {la!r} vs {lb!r}"
    if len(ref["tokens"]) != len(got["tokens"]):
        return False, min(len(ref["tokens"]), len(got["tokens"])), "length"
    return True, None, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="dsv4")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--fillers", default="16,64,192")
    ap.add_argument("--out", default="probe.json")
    a = ap.parse_args()
    rng = random.Random(0)

    alone = [complete(a.port, a.model, p, a.max_tokens) for p in PROBES]
    alone2 = [complete(a.port, a.model, p, a.max_tokens) for p in PROBES]
    results = {"alone_rerun": [compare(x, y)[0] for x, y in zip(alone, alone2)]}
    for k in [int(v) for v in a.fillers.split(",")]:
        jobs = [("probe", i, p, a.max_tokens) for i, p in enumerate(PROBES)]
        for j in range(k):
            text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 1500)))
            jobs.append(("filler", j, text, rng.randint(16, 512)))
        rng.shuffle(jobs)
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            futs = [
                (kind, i, ex.submit(complete, a.port, a.model, p, mt)) for kind, i, p, mt in jobs
            ]
            got = {i: f.result() for kind, i, f in futs if kind == "probe"}
        cmp = [compare(alone[i], got[i]) for i in range(len(PROBES))]
        results[f"with_{k}_fillers"] = [
            {"same": s, "first_diff": pos, "kind": kind} for s, pos, kind in cmp
        ]
    ok = all(results["alone_rerun"]) and all(
        c["same"] for key, v in results.items() if key.startswith("with_") for c in v
    )
    results["PASS"] = ok
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))
    print("BATCH-INVARIANT (tokens and log-probs identical)" if ok else "NOT batch-invariant")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
