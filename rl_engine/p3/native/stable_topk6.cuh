// SPDX-License-Identifier: Apache-2.0
// stable_topk6_device_abi.v1. T04 includes this header; no post-selection reorder.
#pragma once
#include "bitmath.h"
namespace p3 {
struct alignas(8) P3StatusRecord {
    int32_t device_status;
    int32_t reserved;
    unsigned long long invocation_echo;
};
static_assert(sizeof(P3StatusRecord)==16,"p3-op-abi.v4 status layout");
__device__ inline void stable_topk6_row(const float* q, int32_t* ids, P3StatusRecord* status) {
    for (int e=0;e<256;++e) {
        if (!finite(q[e])) { atomicMin(&status->device_status,1); return; }
    }
    for (int k=0;k<6;++k) {
        int best=-1;
        for (int e=0;e<256;++e) {
            bool selected=false;
            for (int j=0;j<k;++j) selected=selected || ids[j]==e;
            if (!selected && (best<0 || q[e]>q[best] || (q[e]==q[best] && e<best))) best=e;
        }
        ids[k]=best;
    }
}
} // namespace p3
