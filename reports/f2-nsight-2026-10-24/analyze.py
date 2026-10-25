"""Summarize completed NVTX ranges from the captured Nsight SQLite timeline."""

import json
import sqlite3
from pathlib import Path

root = Path(__file__).resolve().parent
conn = sqlite3.connect(root / "nsight-results/timeline.sqlite")
conn.row_factory = sqlite3.Row
strings = dict(conn.execute("select id,value from StringIds"))
rows = []
for span in conn.execute("select text,start,end from NVTX_EVENTS where end is not null"):
    label, start, end = span
    if not label or not label.startswith("T"):
        continue
    kernels = list(
        conn.execute(
            "select * from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?", (start, end)
        )
    )

    def fused(k):
        name = strings[k["shortName"]]
        return (
            "f2_forward" in name or "f2_backward" in name or name in ("_kernel", "_backward_kernel")
        )

    target = [k for k in kernels if fused(k)]
    assert len(target) == 10, (label, len(target))
    apis = list(
        conn.execute(
            "select * from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=?", (start, end)
        )
    )
    syncs = [a for a in apis if strings[a["nameId"]].split("_v")[0] == "cudaStreamSynchronize"]
    copies = list(
        conn.execute(
            "select * from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?", (start, end)
        )
    )
    rows.append(
        dict(
            case=label,
            calls=10,
            wall_us_per_call=(end - start) / 10000,
            fused_kernel_us_per_call=sum(k["end"] - k["start"] for k in target) / 10000,
            other_kernels_us_per_call=sum(k["end"] - k["start"] for k in kernels if not fused(k))
            / 10000,
            other_kernels_per_call=(len(kernels) - 10) / 10,
            stream_syncs_per_call=len(syncs) / 10,
            stream_sync_cpu_us_per_call=sum(a["end"] - a["start"] for a in syncs) / 10000,
            copies=[{"bytes": c["bytes"], "kind": c["copyKind"]} for c in copies],
            block_threads=target[0]["blockX"] * target[0]["blockY"] * target[0]["blockZ"],
            registers_per_thread=target[0]["registersPerThread"],
            shared_bytes=target[0]["staticSharedMemory"] + target[0]["dynamicSharedMemory"],
        )
    )
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
for r in rows:
    print(
        r["case"],
        *[
            round(r[k], 2)
            for k in (
                "wall_us_per_call",
                "fused_kernel_us_per_call",
                "other_kernels_per_call",
                "stream_syncs_per_call",
            )
        ],
    )
