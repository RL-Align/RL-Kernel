# MiniMax-H3 Packed Layout Contract

Status: CPU v1 implementation with a draft broader integration/closeout plan for
`h3_packed_layout_schema`, the foundation row in [RFC #420, section 5][rfc-work].
The module is `rl_engine/models/minimax_h3/packed_layout.py`; its CPU tests are
`tests/models/minimax_h3/test_h3_packed_layout.py`. Independent reference
artifacts remain under `tests/reference/minimax_h3/packed_layout/`.
Actual model/GPU H-C3 and the wider work-item closeout remain pending.

Three labels distinguish the strength of a statement:

- **Source requirement**: required by the RFC or established by the pinned
  released implementation. The evidence is linked alongside the statement.
- **v1 design rule**: a concrete implemented choice for the CPU schema,
  available for review; it is not a rule already prescribed by the RFC.
- **Open decision**: a choice or acceptance responsibility still to resolve
  before claiming that the work item is complete.

## 1. Purpose and handoff

A packed row is one feature row in the transformer's mixed text, video, and
audio sequence. Its identity must explain which batch item and source row it
belongs to, its current packed position, its modality, its timestep-table
reference, its three-axis position, and its output-selection position, if any.
Preserving those relationships is the [RFC section 3 requirement][rfc-identity].

```text
Released layout preparation + row-timestep preparation
  -> layout fields and source feature tensors
  -> CPU schema: validate, describe identity, transform layout, seal evidence
  -> model/adapters: scatter -> AdaLN/RoPE/block computation -> select outputs
  -> validation: compare the same logical rows and report differences
```

The schema receives prepared metadata plus the shape and identity context
needed to check it. It returns a checked layout, derived identity information,
a consistently permuted layout, or a field-specific error/difference. These
are implemented through the interface described in section 10. Comparison and
first-numerical-drift reporting belong to the later validation work.

For this draft, the proposed division of responsibilities is as follows.
The producer owns media/text encoding and the task-specific coordinate and
timestep construction. The model or adapter owns moving feature rows together
with the layout and passing fields to existing operators. Operator work items
own projections, scatter/select kernels, attention, RoPE arithmetic, scheduler
math, and gradients. Distributed adapters own communication and co-sharding.
This division does not waive the RFC's shared validation obligations.

**Source requirement:** batch is a replication axis: one structural layout is
shared across batch items. Packing rows from different batch items into one
document changes that meaning. The RFC also requires row identity to survive
TP/CP/SP, checkpointing, and runtime adaptation. This CPU draft does not claim
to implement or validate those execution paths. [Sources: section 2][rfc-fingerprint]
and [section 3][rfc-identity].

## 2. Evidence baseline

The sources for this draft were checked on 2026-10-31:

| Source | Fixed identity and use |
| --- | --- |
| RFC #420 | Proposed RFC; use sections 2-6 and 9 for semantics and acceptance obligations. |
| Checkpoint | `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`, pinned by the RFC. |
| Project baseline | `RL-Align/RL-Kernel@a88b5a0959b194253553086ba2e78ae212a4b15c` on `test-h3`. |
| Diffusers reference | `f53d552036a0d1bd5570782a39cd40cfabf112bc`, pinned by the project's [H3 manifest][manifest], rather than by the RFC text itself. |
| Transformer source | `src/diffusers/models/transformers/transformer_minimax_h3.py`; SHA256 `435873d9bc26b8770342961e18cb20324aeeaa4d045e85e0941638d4857d7db3` in that manifest. |
| Preparation source | `src/diffusers/modular_pipelines/minimax_h3/before_denoise.py` at the same Diffusers revision; see the linked layout and timestep methods below. |

**v1 design rule:** versioned fixtures record their actual source identity and generation
arguments. A later reference update needs an explicit evidence update; a
moving upstream branch is not a sealed expected result.

## 3. Field meanings and shapes

Let `B` be batch size, `S` the packed sequence length, `T` the number of distinct
timestep-table entries, and `N_text`, `N_video`, `N_audio` the source-stream
lengths. Metadata has no batch axis in the pinned model. The model inputs have
shapes `(B, N_text, 5120)`, `(B, N_video, 96)`, and `(B, N_audio, 32)`;
the packed hidden state has shape `(B, S, 5376)`. These feature tensors provide
shape context and are not themselves layout metadata. [Source: model forward][model-forward].

The following field names match the pinned model. The dtype column describes
the pinned preparation path and the v1 CPU representation, not an RFC
requirement to accept only those dtypes. Broader input compatibility is D3.

| Field/context | Shape / representation | Meaning and consumer |
| --- | --- | --- |
| Batch and length context | `B`, `S`, `N_text`, `N_video`, `N_audio` | `B` comes from `batch_size`. The layout derives `S` from the row metadata and the expected source lengths from the selectors. |
| `text_indices` | `(N_text,)`, int64 | Source text row `k` is scattered to packed row `text_indices[k]`. The text projection/refiner processes this source order before scatter. |
| `video_indices` | `(N_video,)`, int64 | Source video row `k` is scattered to this packed row; video output slot `k` selects the same packed position. |
| `audio_indices` | `(N_audio,)`, int64 | Source audio row `k` is scattered to this packed row; audio output slot `k` selects the same packed position. |
| `token_tags` | `(S,)`, int64 | Per-packed-row modality: video `0`, text `1`, audio `2`. Used for AdaLN conditioning. |
| `timestep` | `(T,)`, float32 | Distinct, unscaled denoising times in `[0, 1]`; the pinned timestep builder returns them sorted. |
| `timestep_indices` | `(S,)`, int64 | Per-row reference into `timestep`, not the time value itself. |
| `position_ids` | `(S, 3)`, float64 | Per-row `(t, h, w)` rotary coordinates. Fractional coordinates are meaningful. |
| `adaln_indices` | `(S,)`, derived int64 | `timestep_indices * 3 + token_tags`; avoid a second independently mutable authority for the same relationship. |
| Task and source identity | Checkpoint/runtime revisions, FL2VA or Ref2VA, preparation arguments | Required context for comparable replay; the artifact encoding is D4. |
| Condition/reference context | Producer's condition-video/audio counts or equivalent source-role information | Distinguishes condition/reference rows from generated rows. The representation is D2. |

Evidence: [model input shapes and consumers][model-forward], [FL2VA builder][fl2va],
[Ref2VA builder][ref2va], [row-timestep preparation][row-timesteps], and
[manifest conventions][manifest].

### Relationships that must remain explicit

**Source requirements:**

- Source stream and modality tag are separate facts. A visual conditioning
  block in the text feature stream can carry tag `0`. Do not derive the input
  stream from `token_tags`, or require every `text_indices` row to have tag `1`.
  [Sources: FL2VA][fl2va] and [Ref2VA][ref2va].
- **v1 design rule:** require video-source tags to be `0`, audio-source tags to
  be `2`, and text-source tags to be `0` or `1`. These are the combinations
  emitted by the pinned [pipeline tag constants][source-tags] and
  [text/vision presentation][text-tags], not a rule that every source name
  maps to exactly one tag.
- Rotary time in `position_ids[:, 0]` is a coordinate; denoising time in
  `timestep[timestep_indices]` describes noise level. They are not interchangeable.
  [Sources: coordinate construction][fl2va] and [timestep preparation][row-timesteps].
- Video/audio input and selected output rows can include conditions/references.
  Selection by itself does not mean that a row is a generation target. The
  caller retains the condition/reference information. [Source: layout returns][fl2va].
- Both output heads run on all packed rows and then select using
  `video_indices`/`audio_indices`. Selector order defines returned source order;
  sorting the selectors changes that order. [Source: output heads][model-outputs].
- `adaln_indices` has one entry per row and refers into the timestep-major,
  three-modality table using the formula above. [Source: model lookup][model-lookup].

**v1 design rule:** derive AdaLN indices from the primary fields; snapshots
store primary fields only. Preserve metadata dtype and values through validation,
permutation, and snapshot restore. Reject an unsupported representation with a
specific error instead of silently converting coordinates or integer indices.
Any normalization offered by a future adapter must be explicit and evidenced.
The schema does not move the arithmetic cast points of RoPE or other operators.

## 4. Row identity and output identity

For a single-source packed row `p = video_indices[k]`, the logical source is
`(video input stream, k)`, and its video output slot is `k`. Apply the same rule
to audio. A text source row has no selected video/audio output slot in the
pinned layout, including a text-stream visual row with tag `0`.

For each batch item `b`, a row description must recover:

```text
batch item b
current packed row p
source stream and source row k
token_tags[p]
timestep_indices[p] and the referenced timestep value
position_ids[p, :]
output stream and slot, or no selected output
condition/reference/target role when declared by the producer
```

This is a proposed way to expose the RFC identity tuple; it does not require
storing a redundant copy of every value. Batch item plus source stream and row
identify the same logical row across a packed permutation. Packed position is
its current address. The v1 profile requires single-source ownership;
overlap and uncovered-row rules are recorded in D1.

## 5. Legal permutation: direction and a worked example

**v1 convention:** `perm[new_p] = old_p`, so `new_rows = old_rows[perm]`.
Let `inverse[old_p] = new_p`. A legal permutation contains every integer from
`0` to `S - 1` exactly once and changes only packed placement.

```text
Per-row data/metadata: new_field[new_p] = old_field[perm[new_p]]
Source selectors:     new_indices[k] = inverse[old_indices[k]]
Packed restoration:   old_rows = new_rows[inverse]
```

Apply the per-row rule to the hidden row axis, tags, timestep indices, positions,
and any materialized derived row fields. A model adapter must apply it to the
actual hidden rows; transforming metadata alone is insufficient. Leave source
feature order, source-slot roles, and the timestep table unchanged. If the
timestep table is independently reordered, its references require a separate
remapping; that operation is not this packed-row permutation.

The following four-row example is hand-written to explain the mapping. It is
not a released-pipeline fixture or proof of a supported media geometry.

```text
B = 1; S = 4
timestep = [0.25, 0.5, 1.0]
text_indices = [0]; audio_indices = [1]; video_indices = [2, 3]
```

| Old packed row | Source identity | Role | Tag | Timestep index (value) | Position `(t,h,w)` | Selected output slot |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | text 0 | text condition | 1 | 0 (0.25) | `(0, 0, 0)` | none |
| 1 | audio 0 | target | 2 | 1 (0.5) | `(1, 0, -0.5)` | audio 0 |
| 2 | video 0 | condition | 0 | 2 (1.0) | `(1, 0, 0)` | video 0 |
| 3 | video 1 | target | 0 | 0 (0.25) | `(1, 0, 0.5)` | video 1 |

The original AdaLN indices are `[1, 5, 6, 0]`. The selected video list includes
the condition row; downstream task handling decides which rows to keep as
generated output.

Now choose a non-self-inverse permutation:

```text
perm       = [3, 0, 2, 1]
inverse    = [1, 3, 2, 0]
new order  = [video 1, text 0, video 0, audio 0]

new text_indices     = [1]
new audio_indices    = [3]
new video_indices    = [2, 0]
new token_tags       = [0, 1, 0, 2]
new timestep_indices = [0, 0, 2, 1]
new adaln_indices    = [0, 1, 6, 5]
new position_ids     = [old_position[3], old_position[0],
                        old_position[2], old_position[1]]
```

Use unique row payloads to check the index math without numerical kernels:

```text
original packed payload:                   [10, 20, 30, 40]
permuted packed payload:                   [40, 10, 30, 20]
select original video with [2, 3]:          [30, 40]
select permuted video with [2, 0]:          [30, 40]
```

Packed intermediate outputs need inverse permutation before comparison.
Already selected video/audio outputs are in source-slot order and are compared
in that order; do not inverse-permute them a second time. This gives a precise
layout-level interpretation of [H-C3][rfc-controls]. It does not prove the
floating-point model result under a changed attention reduction order.

## 6. Validity, semantic differences, and errors

The matrix below is the CPU v1 validation surface derived from the
source shapes and relationships. The RFC does not prescribe this exact input
acceptance matrix or error class.

| Case | v1 result | Reason / qualification |
| --- | --- | --- |
| Consistent metadata shapes, in-range fields, supported variant/context | Accept | The layout checks relationships among the supplied metadata. |
| Unsorted but otherwise valid source selectors, such as video `[2, 0]` | Accept | Packed location need not be increasing; source-slot order matters. |
| A text-stream visual row tagged `0` with matching source evidence | Accept | Released preparation permits it. |
| Fractional float64 positions | Accept and preserve exactly | Coordinates are not integer token offsets. |
| Invalid shape, non-integer selector, or out-of-range packed index | Reject | Cannot address the declared source/packed rows. |
| Tag outside `{0, 1, 2}` | Reject | Not a declared H3 modality. |
| Source/tag contradiction: video not `0`, audio not `2`, or text not `0`/`1` | Reject in v1 | Violates the pinned producer combinations; errors identify the packed row and source selector slot. |
| Timestep reference outside `[0, T)` | Reject | Missing referenced table row. |
| Duplicate values in a declared distinct timestep table | Reject | Contradicts the declared table identity; non-sorted distinct values are accepted under D3. |
| Non-finite coordinates/timesteps or timestep outside `[0, 1]` | Reject in v1 | Explicit proposed domain check. |
| An independent AdaLN field is supplied in a snapshot | Reject as an extra field | AdaLN indices are derived, so there is no second authority to contradict primary fields. |
| Repeated/missing/out-of-range entry in `perm` | Reject | Not a permutation of these `S` rows. |
| Duplicate/overlapping source destinations or uncovered packed rows | Reject under the single-source profile | D1; do not infer this restriction merely from zero-initialized upstream scatter. |
| One empty source stream | Accept at the metadata layer with a correctly shaped empty selector | D3; model execution support needs its own evidence. |
| `B <= 0`, `S == 0`, or `T == 0` | Reject in v1 | D3; no claim that the RFC explicitly forbids all empty layouts. |
| Non-sorted but distinct timestep values | Accept and preserve order | References address the supplied table; no implicit table sorting or remapping. |
| Non-contiguous dense metadata | Accept and copy into owned contiguous storage | Dtype, shape, and values remain exact; strides are not part of this profile. |
| Named dimensions in metadata or `perm` | Reject in v1 with `H3PackedLayoutError` | This profile uses positional axes and does not serialize dimension names; names are never silently removed. |
| Metadata requiring gradients | Reject | This CPU object represents static metadata, not a differentiable tensor transformation. |
| Unsupported dtype/device/variant/profile | Explicit unsupported error | No silent cast, transfer, fallback, or claimed coverage. |

The layout checks that each per-row metadata tensor has length `S`. Before
scatter, the adapter checks that each source feature tensor has as many rows
as its selector. The single-source profile requires the three source selector
sets to be disjoint and cover `[0, S)`.
This is an ownership rule for input scatter, not a blanket ban on repeated
indices in a general output gather operator. Independent output selectors and
their multiplicity, if added, require an explicit contract (D1/D2).

Errors should identify the field, offending position/value, and expected
domain. Snapshot construction should own its metadata so later mutation of a
caller's tensors cannot change sealed evidence. Exact ownership mechanics and
serialization are proposed implementation choices.

### Structural validity does not recover the caller's intent

Switching a text-source tag between `0` and `1`, pointing at a different valid
timestep, or swapping selector slots within the same modality may produce
another structurally valid layout. Cross-source/tag contradictions are rejected;
a validator still cannot infer the original intended value of every legal input.
Report these three questions separately:

1. Is this layout internally valid for its declared profile?
2. Does it describe the same logical rows as an independently sealed baseline?
3. Where does an actual consumer first produce a numerical difference?

This preserves the expected meanings of [H1/H2/H3/H14/H16/H17][rfc-probes]. A
schema-level mismatch can diagnose a field/row; the RFC's first-operator drift
signatures require execution evidence from the relevant consumers.

## 7. Snapshot and provenance

**Source requirement:** artifacts record exact checkpoint/runtime identity,
task family, dtype policy, and execution/backend identity. [Sources: section 2][rfc-fingerprint],
[section 4][rfc-numerical], and [section 5][rfc-work].

**v1 design rule:** a snapshot records its schema/profile version, field
dtype/shape/values, source lengths, preparation and variant context, and fixed
source references. A validation report additionally records the command,
environment, actual consumer/backend, comparator, and executed checks/results.
A CPU-only run reports that scope explicitly. Exact encoding is D4; round-trip
restoration must preserve the supported tensor representations exactly.

Distinguish the hash of a physical snapshot from logical equivalence. A packed
permutation changes physical field order and can change that hash while
preserving logical row relationships. A comparison must name which of those
claims it tests. Variant/checkpoint mismatches must be reported as identity
differences, rather than accepted as numerical drift of the same request.

## 8. Acceptance plan and current evidence limits

The RFC requires implementation, independent reference, invariance tests,
runtime provenance, and row-specific evidence to land together. The following
is the proposed work-item-specific test plan, not a claim of completed tests
or an exhaustive list printed in the RFC. [Source: section 5][rfc-work].

| Check | Planned evidence |
| --- | --- |
| Complete row identity | Hand-written small cases with independently specified source, packed, tag/time/position, and output identities. Include the four-row example above. |
| Validity and diagnostics | Positive/negative cases for the declared matrix; accept unsorted indices and the text-stream visual-tag exception; report field and row on rejection. |
| Permutation invariance | Unique-ID CPU scatter/select, non-self-inverse/random permutations, inverse restoration, and composition. Check per-row movement and selector-value remapping independently. |
| Semantic mismatch detection | Seal a baseline, then separately omit a metadata move, change a legal tag/timestep, leave selectors stale, or mix variants. Compare supplied distributed ownership records where available; actual co-sharding execution is integration evidence under D6. Separate rejection from baseline difference and actual numerical drift. |
| Batch and ownership | Same request alone/at different batch positions; unrelated payload changes do not affect it. External mutation does not alter sealed metadata. |
| Released preparation reference | Source-pinned independent fixtures including selectors and real row-timestep binding. Preserve existing gold; use candidates neither to generate nor to bless their sole expected result. |
| Precision and restore | Check dtype/shape/values and fractional coordinate preservation through permutation and snapshot round trip. |
| Actual field consumers | CPU row scatter/select plus existing PyTorch AdaLN lookup or equivalent reference consumer. A standalone type definition is insufficient evidence. |
| Automation and regression | New CPU tests run explicitly in CI; retain existing layout hashes and applicable packing/import/registry checks. A missing required reference/dependency is not a passing skip. |
| Block/model numerical control | Execute and report the agreed H-C3 integration evidence independently of metadata tests; closeout responsibility is D6. |

Existing [h3_cases.py][cases] provides random tag/time stress inputs and a
text-only FL2VA layout helper without keyframe anchors. The latter returns
positions/tags and stand-in timestep indices (text `0`, all media `1`), without
the real timestep table or the three source selectors. Its existing
[pinned-layout tests][layout-tests] seal position and tag bytes for 264, 925,
and 3160 rows; those hashes do not seal the missing identity fields.

The current [one-block replay documentation](../operators/h3-one-block.md)
labels H-C3 as reported rather than promised. Its recorded experiment observes
permutation drift with the interim attention reduction order. That is existing
evidence, not a new run for this draft. Exact metadata remapping does not prove
that the current numerical consumer passes H-C3, and this draft neither weakens
the RFC control nor assumes that every future attention implementation fails it.

The reference set now contains the hand-written four-row example and thirteen
source-pinned preparation cases. Its README records inputs, provenance, exact
floating-point encoding, reproduction commands, and limitations. Offline checks
validate reference integrity and semantic relationships independently of the
schema. Real fixtures remain in upstream's canonical order; the CPU schema
tests exercise their permutations separately.

The CPU implementation now has tests against these fixtures, scalar CPU
scatter/select consumers, and the existing PyTorch AdaLN consumer. These checks
do not establish GPU/full-model equivalence, the complete work-item acceptance
matrix, or actual model H-C3 success.

CPU negative controls reject source/tag contradictions, including cross-stream
stale selectors and tags left unmoved by the mixed-modality hand permutation.
Separate tests compare structurally valid changes against the sealed baseline:
unmoved timestep indices/positions, a legal text/visual tag or time reference
change, and stale source-slot order within a modality. They locate the field,
packed row, and expected logical source row; AdaLN/output-selection consumers
also demonstrate affected values and output slots. A separate test flags mixed
FL2VA/Ref2VA source declarations before numerical comparison. These are test
evidence, not the pending standalone comparison/report interface or a verification
of loaded model weights.

## 9. Implemented v1 decisions and remaining closeout questions

| ID | Question | Proposed starting point / required resolution |
| --- | --- | --- |
| D1 | Implemented ownership domain | CPU single-source/full coverage. Reject overlapping, duplicate or uncovered source destinations; accept unsorted selectors. Independent/repeated output selectors are not part of v1. This does not restrict a general gather operator. |
| D2 | Implemented role/output representation | Video/audio source-order prefixes record condition/reference counts; remaining source slots are targets. Text-stream rows are conditions and have no selected output slot, including visual-tagged rows. Output slots derive from source selectors. |
| D3 | Implemented representation domain | CPU int64 indices/tags, FP32 timesteps, FP64 positions. Empty source streams accepted; B/S/T positive. Distinct timesteps may be unsorted; supplied table order is preserved. Non-contiguous dense metadata is copied; named dimensions, non-CPU, gradients, or implicit dtype conversions are rejected. |
| D4 | Implemented interface and encoding | H3PackedLayout, H3PackedLayoutError, row_identity, permute_rows, to_dict/from_dict. Snapshot version h3-packed-layout.v1, profile cpu.single_source; exact little-endian tensor bytes. Constructor provenance declares full checkpoint/runtime revisions and FL2VA/Ref2VA identity. No RFC-prescribed dataclass/JSON Schema/hash format is implied. |
| D5 | Minimum released-variant fixture coverage | Plan FL2VA with no/first/last/both frame anchors and representative Ref2VA references, including real timestep binding. Agree the required case matrix before claiming variant support; metadata fixtures do not qualify a complete Phase B pipeline. |
| D6 | CPU work-item acceptance and model H-C3 ownership | Map the RFC's shared GPU/backward/benchmark/binding requirements to this CPU foundation row with explicit applicability explanations. Agree whether actual block/model H-C3 is this row's closeout prerequisite, who supplies it, and the execution profile/evidence required. Do not claim completion while an agreed required gate is missing. |

These are implementation choices, not maintainer approval of the whole row.
Changes to these decisions must update the validity matrix, fixture coverage,
and acceptance plan together. Unsupported cases remain explicit until they
are implemented and evidenced.

## 10. CPU v1 interface and a runnable example

`H3PackedLayout(...)` checks and owns the metadata immediately. Invalid inputs
raise `H3PackedLayoutError`, a `ValueError` subclass. It does not silently cast,
move metadata to CPU, or sort selectors/timesteps. Tensor properties and
`provenance` return independent copies. `source_lengths` gives the expected
row count for each source stream, derived from its selector. Before scatter,
the adapter checks the feature tensors' batch size and row counts against
`batch_size` and `source_lengths`.
Batch size defaults to one and condition counts default to zero; adapters must
pass the producer's actual condition/reference counts instead of inferring
them from tags or packed positions.

`row_identity(packed_row, batch_index=0)` returns a dictionary containing the
source stream/slot, modality, time reference/value, coordinate, output slot,
and condition/target role. Condition includes reference rows in Ref2VA.
Its return type is `H3RowIdentity`, a TypedDict whose values remain ordinary
dictionaries at runtime. Provenance uses `Mapping[str, Any]`: the layout stores
an independent copy of its JSON-compatible values. Runtime validation
requires `checkpoint_revision`, `runtime_revision`, and `task_family`.
`task_family` records `FL2VA` or `Ref2VA`, the same value stored as `variant` in
H3 trajectory artifacts. Optional `preparation` context remains extensible JSON.
The provenance property also exposes a Mapping interface and returns an
independent dictionary copy.

`permute_rows(perm)` returns a new layout and leaves the original unchanged.
It transforms metadata only: move an already-packed feature tensor separately
with `features.index_select(1, perm)`. Source feature streams keep their order.

Permutation code explicitly lists the source selectors, per-row fields, and
unchanged timestep table. Field-name suffixes do not determine their semantics.
Time validation and identity queries decode FP32 bit patterns without ambient
FP32 comparisons/conversions: positive subnormal times remain valid, negative
times are rejected, and positive/negative zero count as the same time for
uniqueness. The original tensor bits, including a single signed zero, are
preserved. Changing CPU flush-to-zero settings does not change these results.
Duplicate-time errors identify both zero-based table rows and the decoded value,
for example `timestep[2] duplicates timestep[0] (value 0.25)`.

The following synthetic four-row case is the independent hand-written example:

```python
import json
import torch
from rl_engine.models.minimax_h3.packed_layout import H3PackedLayout, H3PackedLayoutError

layout = H3PackedLayout(
    text_indices=torch.tensor([0]),
    audio_indices=torch.tensor([1]),
    video_indices=torch.tensor([2, 3]),
    token_tags=torch.tensor([1, 2, 0, 0]),
    timestep=torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32),
    timestep_indices=torch.tensor([0, 1, 2, 0]),
    position_ids=torch.tensor(
        [[0, 0, 0], [1, 0, -0.5], [1, 0, 0], [1, 0, 0.5]],
        dtype=torch.float64,
    ),
    num_condition_video_rows=1,
    provenance={
        "checkpoint_revision": "42ed227ee7df40d41602854ae760620d6eb651fe",
        "runtime_revision": "f53d552036a0d1bd5570782a39cd40cfabf112bc",
        "task_family": "FL2VA",
        "preparation": {"kind": "hand_written"},
    },
)
moved = layout.permute_rows(torch.tensor([3, 0, 2, 1]))
print(moved.video_indices.tolist())       # [2, 0]
print(moved.token_tags.tolist())          # [0, 1, 0, 2]
print(moved.row_identity(0, batch_index=0)["source_index"])  # 1: video source row 1
print(moved.row_identity(0)["role"])      # target

saved = json.loads(json.dumps(moved.to_dict()))
restored = H3PackedLayout.from_dict(saved)
assert torch.equal(restored.position_ids, moved.position_ids)

try:
    layout.permute_rows(torch.tensor([0, 0, 2, 3]))
except H3PackedLayoutError as error:
    print(error)  # perm: expected each packed row in [0, 4) exactly once
```

Provenance is finite JSON data with string object keys. Required checkpoint and
runtime revisions are full lowercase 40-hex strings; task family is FL2VA or
Ref2VA. These declarations do not themselves verify loaded weights or establish
full-pipeline support. Include actual preparation arguments and run context
when recording real evidence.

The layout preserves provenance floats in memory, including through `to_dict()`
and `from_dict()` with CPU flush-to-zero enabled. For JSON text round trips,
use an environment that preserves subnormal floats.

`to_dict()` returns a JSON-safe snapshot with version, profile, batch/condition
counts, provenance, and primary tensors. Every tensor carries dtype, shape,
and exact little-endian hex bytes; AdaLN indices are recomputed. `from_dict()`
rejects unknown versions, profiles, missing/extra fields, wrong dtypes/shapes,
or inconsistent byte lengths, then repeats normal layout validation.
Encoding uses a direct uint8 byte view, following the project's raw tensor-byte
pattern, without expanding each byte into a Python integer list or converting
floating values through decimal text.

Run the CPU checks with:

```bash
python -m pytest tests/models/minimax_h3/test_h3_packed_layout.py tests/reference/minimax_h3/packed_layout -q
```

The CI CPU H3 step includes the new schema tests explicitly. Import compatibility
tests allow this metadata child module while keeping other reserved models
empty and checking that importing H3 metadata does not change the registry,
including in a fresh process. A CPU metadata object does not certify a model
backend, GPU arithmetic, or model H-C3.

## 11. What to review in this interface

- Inputs and ownership: do the seven prepared tensor fields, batch/source lengths,
  provenance, and actual condition/reference counts express the upstream request?
  Callers own feature tensors; the layout owns metadata copies.
- Permutation and identity: per-row fields move together, source selectors address
  new packed positions, the timestep table and source order stay unchanged, and
  selected output slots still identify the original source rows.
- State and failure behavior: construction rejects unsupported/contradictory
  metadata with field-specific errors; permutation returns a new layout; snapshots
  recover exact supported values and source declarations. Review these behaviors
  separately from full-model numerical qualification.

[rfc-fingerprint]: https://github.com/RL-Align/RL-Kernel/issues/420#2-checkpoint-fingerprint
[rfc-identity]: https://github.com/RL-Align/RL-Kernel/issues/420#3-architecture-map
[rfc-numerical]: https://github.com/RL-Align/RL-Kernel/issues/420#4-numerical-contract
[rfc-work]: https://github.com/RL-Align/RL-Kernel/issues/420#5-kernel-and-work-item-table
[rfc-probes]: https://github.com/RL-Align/RL-Kernel/issues/420#62-h3-specific-mismatch-probes
[rfc-controls]: https://github.com/RL-Align/RL-Kernel/issues/420#63-invariant-controls
[manifest]: https://github.com/RL-Align/RL-Kernel/blob/a88b5a0959b194253553086ba2e78ae212a4b15c/rl_engine/validation/models/h3_manifest.json#L116-L134
[model-forward]: https://github.com/huggingface/diffusers/blob/f53d552036a0d1bd5570782a39cd40cfabf112bc/src/diffusers/models/transformers/transformer_minimax_h3.py#L529-L603
[model-lookup]: https://github.com/huggingface/diffusers/blob/f53d552036a0d1bd5570782a39cd40cfabf112bc/src/diffusers/models/transformers/transformer_minimax_h3.py#L605-L612
[model-outputs]: https://github.com/huggingface/diffusers/blob/f53d552036a0d1bd5570782a39cd40cfabf112bc/src/diffusers/models/transformers/transformer_minimax_h3.py#L622-L631
[fl2va]: https://github.com/huggingface/diffusers/blob/f53d552036a0d1bd5570782a39cd40cfabf112bc/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py#L268-L371
[ref2va]: https://github.com/huggingface/diffusers/blob/f53d552036a0d1bd5570782a39cd40cfabf112bc/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py#L560-L590
[row-timesteps]: https://github.com/huggingface/diffusers/blob/f53d552036a0d1bd5570782a39cd40cfabf112bc/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py#L1178-L1216
[source-tags]: https://github.com/huggingface/diffusers/blob/f53d552036a0d1bd5570782a39cd40cfabf112bc/src/diffusers/modular_pipelines/minimax_h3/modular_pipeline.py#L22-L25
[text-tags]: https://github.com/huggingface/diffusers/blob/f53d552036a0d1bd5570782a39cd40cfabf112bc/src/diffusers/modular_pipelines/minimax_h3/encoders.py#L571-L605
[cases]: https://github.com/RL-Align/RL-Kernel/blob/a88b5a0959b194253553086ba2e78ae212a4b15c/rl_engine/validation/models/h3_cases.py#L26-L164
[layout-tests]: https://github.com/RL-Align/RL-Kernel/blob/a88b5a0959b194253553086ba2e78ae212a4b15c/tests/models/minimax_h3/test_h3_one_block.py#L36-L63
