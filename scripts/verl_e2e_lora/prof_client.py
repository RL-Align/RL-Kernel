# SPDX-License-Identifier: Apache-2.0
"""Decode-latency probe for one running vLLM server (``profile_rollout.sh``).

Timed phases (no profiler): C concurrent requests, each a distinct short prompt that
generates exactly N tokens (``ignore_eos``). ms/step = wall / N, tok/s = C * N / wall.
Then, with ``--profile``: ``/start_profile``, the ``PROF_PHASES`` phases (default
"1:16,64:16", concurrency:tokens; few steps, since node-level graph tracing makes the
stop's flush slow), ``/stop_profile`` and a wait for the flush. ``PROF_ONLY=1`` skips the
timed phases.
"""

import argparse
import concurrent.futures as cf
import json
import os
import random
import time
import urllib.request

TIMED = [(1, 256), (8, 256), (64, 256), (512, 256)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    def post(path, body=None, timeout=3600):
        data = json.dumps(body or {}).encode()
        req = urllib.request.Request(
            base + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    def one(i, n):
        prompt = (
            f"Problem {random.randint(0, 10**9)}-{i}: compute the sum of the first "
            f"{i + 7} odd numbers and explain each step."
        )
        body = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": n,
            "min_tokens": n,
            "ignore_eos": True,
            "temperature": 1.0,
        }
        return post("/v1/completions", body)["usage"]["completion_tokens"]

    def phase(conc, n):
        t = time.time()
        with cf.ThreadPoolExecutor(conc) as ex:
            toks = sum(ex.map(lambda i: one(i, n), range(conc)))
        wall = time.time() - t
        row = {
            "conc": conc,
            "tokens_each": n,
            "tokens": toks,
            "wall_s": round(wall, 2),
            "ms_per_step": round(wall / n * 1000, 1),
            "tok_per_s": round(toks / wall),
        }
        print(json.dumps(row), flush=True)
        return row

    rows = []
    phase(8, 32)  # warm-up
    for conc, n in [] if os.environ.get("PROF_ONLY") else TIMED:
        rows.append(phase(conc, n))
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    if not args.profile:
        return
    post("/start_profile")
    time.sleep(2)
    for spec in os.environ.get("PROF_PHASES", "1:16,64:16").split(","):
        conc, n = map(int, spec.split(":"))
        rows.append({"profiled": phase(conc, n)})
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    try:
        post("/stop_profile", timeout=1800)
    except Exception as e:  # the engine may time the RPC out while CUPTI flushes
        print("stop_profile:", e, flush=True)
    time.sleep(240)


if __name__ == "__main__":
    main()
