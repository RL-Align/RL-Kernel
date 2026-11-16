# F2 fixed-tree CUDA shuffle mapping proposal

Status: arithmetic mapping checked and integrated for SM80 after isolated and
standard-build validation. Other architectures retain the shared-memory tree.
See validation evidence (local experiment record; not included in this PR).
This proposal preserves the 128-thread block and all 127 ordered FP32 addition
nodes. It changes transport/storage only for the final five stages. It is not a
replacement with an unspecified warp-reduction primitive.

## Existing arithmetic contract

Let `R(a,b) = __fadd_rn(a,b)`, with the existing compiler controls and FP32 inputs.
Leaves are already rounded FP32 products: `__fmul_rn(x,x)` for forward, or
`__fmul_rn(g,u)` with FP32 `u=__fmul_rn(x,r)` for backward VJP v2, after the declared
gradient casts. The historical v1 leaf was `g*x`; the shuffle mapping applies to
either leaf definition and does not prescribe the surrounding gradient formula.
For each stride s in 64,32,16,8,4,2,1, active index i < s computes:

```text
A_next[i] = R(A_previous[i], A_previous[i+s])
```

The left/right operand order matters. Do not replace the tree with a differently
associated sum, FMA, wider accumulator, hardware reduce-add or a library reduction.

## Conservative mapping

Keep one 128-thread block per head, with d = threadIdx.x:

| Stage | Active indices | Transport | Arithmetic |
| --- | --- | --- | --- |
| 64 | d=0..63 | shared memory | R(leaf[d], leaf[d+64]) |
| 32 | d=0..31 | shared memory | R(stage64[d], stage64[d+32]) |
| 16 | lanes 0..15 in warp 0 | shuffle-down 16 | R(stage32[lane], stage32[lane+16]) |
| 8 | lanes 0..7 in warp 0 | shuffle-down 8 | same ordered pair as original stage 8 |
| 4 | lanes 0..3 in warp 0 | shuffle-down 4 | same ordered pair as original stage 4 |
| 2 | lanes 0..1 in warp 0 | shuffle-down 2 | same ordered pair as original stage 2 |
| 1 | lane 0 in warp 0 | shuffle-down 1 | same ordered pair as original stage 1 |

After stride 32, lane l holds exactly:

```text
R(R(leaf[l], leaf[l+64]), R(leaf[l+32], leaf[l+96]))
```

The remaining valid values fit in warp 0. Every lane in warp 0 executes every
shuffle; only lanes below the stage stride perform an addition. Values of inactive
lanes are irrelevant after that stage. For each consumed shuffle result,
`lane + stride < 32`, so it comes from a participating, in-range source lane.

Mapping pseudocode (the operator uses an SM80-gated equivalent):

```cuda
// Exactly 128 threads call this; each has stored its FP32 leaf in acc[d].
// Returns via shared acc[0], not via a per-thread read of acc[0].
__device__ __forceinline__ void fixed_tree_shared_shuffle(float* acc, int d) {
    __syncthreads();
    if (d < 64) acc[d] = __fadd_rn(acc[d], acc[d + 64]);
    __syncthreads();
    if (d < 32) acc[d] = __fadd_rn(acc[d], acc[d + 32]);
    __syncthreads();

    if (d < 32) {  // Whole warp 0 enters; other whole warps skip.
        float v = acc[d];
        #pragma unroll
        for (int stride = 16; stride; stride >>= 1) {
            float peer = __shfl_down_sync(0xffffffffu, v, stride, 32);
            if (d < stride) v = __fadd_rn(v, peer);
        }
        if (d == 0) acc[0] = v;
    }
    __syncthreads();
}
```

There must be no early return or partial-lane participation. The final block
barrier makes lane 0's shared write visible to all four warps. No `volatile` shared
memory tricks or assumed lockstep replace synchronization. Shuffle communicates
registers; it is not used as a substitute for the cross-warp shared-memory barriers.

Forward keeps its existing thread-0 variance/rsqrt computation, shared rstd write,
and following block barrier. Other threads must not read the shared sum while
thread 0 overwrites that slot with rstd. Backward reads the stable shared sum after
the helper. The separate local RoPE shared buffers and barriers are unchanged.

The reduction section alone decreases block barriers from eight (initial leaf
publication plus seven stages) to four (initial publication, two cross-warp stages,
final root publication). This is a static count, not a measured speedup.

## Equivalence argument

1. Leaves and the first two stages are identical to the existing implementation.
2. At the start of the first shuffle stage, register v in lane i equals the
   original shared accumulator at i for every i < 32.
3. Suppose this holds at a stage of stride s. For every active i < s, shuffle
   supplies the previous value from lane i+s. Applying `__fadd_rn(v, peer)` creates
   exactly the original ordered node, including its FP32 rounding boundary.
4. The invariant holds for indices i < s at the next stage. Induction covers
   strides 16,8,4,2,1; lane 0 therefore holds the same arithmetic root.

The proof is about the arithmetic DAG. It does not by itself prove emitted code,
synchronization correctness, absence of races, cross-GPU identity or performance.
NaN payload behavior and the broader overflow/nonfinite contract remain unresolved;
this proposal does not extend the declared numerical domain.

## Structural verification scope

A local experimental checker compared the nested ordered `fadd_rn` expressions
for all 127 nodes and rejected a contiguous-warp-first negative control. That
experimental script is not shipped in this PR; the node mapping and argument
above document the implementation contract.

## Implementation acceptance

The SM80 local validation described above completed the following checks: build the candidate in isolation; inspect generated
PTX/SASS for preserved FP32 operations and intended shuffle/barrier layout; test
cancellation-sensitive signed reductions and dynamic ranges; compare forward,
rstd and backward against the original CUDA/Triton/reference paths; run available
race/synchronization tooling; and benchmark small and large shapes. Keep the old
implementation if it changes arithmetic or provides no useful performance gain.
