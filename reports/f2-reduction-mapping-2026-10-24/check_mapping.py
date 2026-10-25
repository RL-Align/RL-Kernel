"""Compare every ordered FP32-add node, not just sets of contributing inputs."""

import json
from pathlib import Path

STRIDES = (64, 32, 16, 8, 4, 2, 1)


def shared_tree():
    values = [("leaf", i) for i in range(128)]
    nodes = {}
    for stride in STRIDES:
        previous = values.copy()
        for i in range(stride):
            values[i] = ("fadd_rn", previous[i], previous[i + stride])
            nodes[stride, i] = values[i]
    return values[0], nodes


def hybrid_tree():
    shared = [("leaf", i) for i in range(128)]
    nodes = {}
    # Two cross-warp stages remain in shared memory with block barriers.
    for stride in (64, 32):
        previous = shared.copy()
        for i in range(stride):
            shared[i] = ("fadd_rn", previous[i], previous[i + stride])
            nodes[stride, i] = shared[i]
    lanes = shared[:32]
    for stride in (16, 8, 4, 2, 1):
        previous = lanes.copy()
        # All 32 lanes execute the shuffle. Only these lanes execute the add.
        for lane in range(stride):
            assert lane + stride < 32
            lanes[lane] = ("fadd_rn", previous[lane], previous[lane + stride])
            nodes[stride, lane] = lanes[lane]
    return lanes[0], nodes


def main():
    original, original_nodes = shared_tree()
    hybrid, hybrid_nodes = hybrid_tree()
    assert len(original_nodes) == len(hybrid_nodes) == 127
    assert original_nodes == hybrid_nodes
    assert original == hybrid
    # Negative control: contiguous groups per warp give a different add tree.
    groups = []
    for start in range(0, 128, 32):
        lanes = [("leaf", i) for i in range(start, start + 32)]
        for stride in (16, 8, 4, 2, 1):
            old = lanes.copy()
            for lane in range(stride):
                lanes[lane] = ("fadd_rn", old[lane], old[lane + stride])
        groups.append(lanes[0])
    wrong = ("fadd_rn", ("fadd_rn", groups[0], groups[2]), ("fadd_rn", groups[1], groups[3]))
    assert wrong != original
    result = {
        "ordered_add_nodes_checked": 127,
        "depth": 7,
        "strides": list(STRIDES),
        "hybrid_tree_identical": True,
        "ordinary_warp_first_tree_rejected": True,
        "scope": "symbolic arithmetic DAG only; no compiled GPU validation",
    }
    Path(__file__).with_name("result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
