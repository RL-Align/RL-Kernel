# Layout migration

`layout.json` records the physical moves from upstream main at its
`baseline_commit`, retained compatibility files, and modules split across owners.
It is an audit map, not a script to rerun on an already-migrated checkout.

Historical workload/operator identifiers and archived evidence intentionally
retain their original values. New source imports should use the mapped canonical
module. Remove compatibility paths only in a separately announced migration
once downstream launchers, patches, imports and saved objects have migrated.

The MiniMax-H3 entries also record the open RFC #420 PR stack migration to
`test-h3`; those H3 files were not part of the original main baseline.
