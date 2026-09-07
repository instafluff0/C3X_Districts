# River, forest and jungle campaign

The user-directed goal remains active. Latest combined candidate:
**river-corridor-r3**, following **river-corridor-r2** and **canopy-variation-r1**. Sixteen matched frames
cover the three fixed gameplay regions and the new 100-tile forest/jungle
witness. This candidate has not replaced the complete preserved baseline below.

- [Inland headwater pool and forested channels, native gameplay size](out/river-corridor-r2/review/inland-h12-z1.png)
- [Inland rivers at night](out/river-corridor-r2/review/inland-h00-z1.png)
- [Fresh forest/jungle region and sea outlet](out/river-corridor-r2/review/freshcanopy-h12-z1.png)
- [Stable forest arrangement](out/canopy-variation-r1/review/coastal-h12-z1.png)

[First-loop findings](RIVER_VEGETATION_PASS_r2.md) record changed pixels, the
rejected offshore-pipe result, preserved source appearance, real-source crop
checks and the next visible gaps. Banks, irregular pool shape, bank-rock
placement and broader coverage remain active. No Civ VI-quality acceptance,
human approval or promotion is implied.

The [bank-rock follow-up](RIVER_BANK_ROCK_PASS_r3.md) moves existing source
rocks alongside the actual river and removes unsafe placements. It is a small
detail correction; [inland comparison](out/river-corridor-r3/review/inland-h12-z1.png).
The larger bank and pool-shape gaps remain open.

## Preserved terrain/lighting checkpoint — cleaner shadow receiving

2026-09-06. Sole lead, local Metal. Retained work in progress:
**shadow-receiver-r1**, including **combinedvolcano** as a separate synthetic
witness. This builds on the larger source bodies in `relief-size-r3`.

This is an incremental visual improvement, not Civ VI-level acceptance.
No human approval, milestone closure or Integration promotion is recorded.
Cities, units and improvements remain deferred. LQ0 remains ready/unaccepted.

- [Cleaner wilderness sand — native gameplay comparison](out/shadow-receiver-r1/review/wilderness-h12-z1-comparison.png)
- [Wilderness at night — full native zoom 2](out/shadow-receiver-r1/review/wilderness-h00-z2-comparison.png)
- [Inland mountains and forest shadows](out/shadow-receiver-r1/review/inland-h12-z1-comparison.png)
- [Fixed coast — full native zoom 2](out/shadow-receiver-r1/review/coastal-h12-z2-comparison.png)
- [Long coast — full native zoom 2](out/shadow-receiver-r1/review/longcoast-h12-z2-comparison.png)
- [New 100-tile desert/forest/mountain holdout](out/shadow-receiver-r1/review/freshshadow-h12-z2-comparison.png)
- [New holdout at night](out/shadow-receiver-r1/review/freshshadow-h00-z2-comparison.png)
- [Current combined volcano witness — explicitly synthetic](out/shadow-receiver-r1/review/combinedvolcano-h12-z1-comparison.png)
- [Full wilderness scene](out/shadow-receiver-r1/wilderness/h12-z1-pan00.png)
- [Full new holdout](out/shadow-receiver-r1/freshshadow/h12-z1-pan00.png)

Most thin dark mesh-edge lines across the wilderness sand are gone. Visible
forest and mountain cast shadows remain. The correction sizes the receiver
normal offset to its bounded shadow texel footprint and derives its plane from
unshifted geometry. It changes no terrain, source material, camera, vegetation
placement or shadow caster. The seven real regions and synthetic witness have
32 noon/midnight frames at both fixed zooms; all matched input packets are
byte-identical. Those invariants support comparison and do not grant acceptance.

Mountains still use 1.30 uniform source-body scale and volcanoes 1.60, with
bounded foothill overlap. The previous `relief-size-r3`, `coast-pass-rocks-r8`
and rejected attempts are preserved. Detailed changed-pixel locations,
diagnosis and reproduction are in [SHADOW_RECEIVER_PASS.md](SHADOW_RECEIVER_PASS.md).
[SHADOW_RECEIVER_r1_EVIDENCE.json](SHADOW_RECEIVER_r1_EVIDENCE.json) records
all matched outputs; [SHADOW_RECEIVER_DIAGNOSTICS.json](SHADOW_RECEIVER_DIAGNOSTICS.json)
records the rejected tests and numerical probe.

The three largest remaining gaps are:

1. Mountain/volcano projection, material detail and unproven source physical
   reconstruction. Direct inspection shows that the volcano height texture is
   incorrectly treated as a two-component normal. The first red-height normal
   reconstruction has too little visible benefit and is not selected.
2. Source dune reconstruction and residual facet artifacts. The inherited
   analytic dune body remains an unapproved proxy; cleaner shadows do not
   resolve that source-fidelity defect.
3. Soft shallow-water structure and abrupt cliff/grass joins.

The new [freshshadow benchmark](../../fixtures/beauty/shadow-receiver-foundation/freshshadow/BENCHMARKS.json)
was selected before viewing this candidate and received no local tuning. It is
now a regression witness, not an untuned witness for later acceptance.
Previous relief work is in [RELIEF_SIZE_PASS.md](RELIEF_SIZE_PASS.md); coastal
source work is in [COAST_SOURCE_JOIN_PASS.md](COAST_SOURCE_JOIN_PASS.md).

## Integration pickup

The user requested consolidation for implementation in C3X. The verified
[candidate preparation package](../../../../handoffs/candidates/lab_v2_terrain_lighting_r1/README.md)
contains the implementation map, pinned source snapshot, all retained frame
hashes, local asset inventory and explicit remaining gates. No additional
visual pass or native implementation was made during that preparation.
