# SPDX-License-Identifier: Apache-2.0
"""Frozen Flash-0731 task-30 Hash routing geometry and checkpoint identities."""

CONTRACT_ID = "rlkernel.hash_route.fp32.slot_order.clamp_min.v1"
AITER_CONTRACT_ID = "aiter.hash_route.fp32.slot_order.clamp_min.v1"
AITER_REVISION = "326cffa38c9eda16bef19375ba55334e4d3acd59"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
HASH_LAYERS = (0, 1, 2)
EXPERTS = 256
SLOTS = 6
VOCABULARY = 129280
TABLE_BYTES = VOCABULARY * SLOTS * 8
CHECKPOINT_TABLE_SHA256 = {
    0: "b310d7aa41cf4967e3db21bb0b00b12357f4423b4347e71760a29d340031b897",
    1: "0e20c4b0af4a7c2c06a73ad62f575b31379200cf8247f99c4536273cc6d3c271",
    2: "cfd0cc9496950c0fc26ee82d1a919f85a3ba54c357ec7844b65654cc0d7c326d",
}
