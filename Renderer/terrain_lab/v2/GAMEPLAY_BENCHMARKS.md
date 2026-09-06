# Gameplay-first visual benchmarks

The user's primary quality target is a convincing, readable Civ III game view,
not an all-assets diorama. This policy changes evidence priority, not the
parallel schedule or immutable v1 baseline. It does not claim new fixtures
already exist or waive real-map acceptance.

## Three complementary fixture roles

1. **Gameplay context (primary beauty evidence).** Q8 owns deterministic
   placement recipes and camera/crop metadata under `fixtures/beauty/`.
   Start with one compact developed settlement-and-countryside view and one
   sparse frontier/coastal view. Use `Renderer/canonical/civ3_real_example.jpg`
   for visible composition relationships, not its old materials or compression.
   Include plausibly spaced settlements, connected routes with destinations,
   worked land concentrated around settlements, open terrain, and restrained
   terrain/domain-appropriate resources and units. Use a declared coherent era
   and culture context, not a sampler of every era and asset. Include a sandy
   shore and a rocky hill coast where source coverage permits; otherwise keep
   the missing case as a labeled synthetic witness. Do not reshape verified
   terrain to manufacture the ideal view.
2. **Focused diagnostics (fast iteration).** Each owner retains small seam,
   material, shoreline, city, network, lighting, and overlap fixtures. These
   isolate causes and exhaust edge cases but cannot alone prove composed beauty.
3. **Inventory and stress (regression coverage).** Preserve the frozen full
   all-assets scene and explicit crowded/unusual combinations. These test
   missing objects and robustness, not plausible placement or overall beauty.
   Do not delete hard cases to improve a score.

## View and comparison rules

- Judge actual output pixels at both Civ III zooms, with authoritative/pinned
  pixel projection and tile scale. Prefer a filled gameplay viewport with enough
  neighboring geometry for crop edges, rather than a distant floating diamond
  surrounded by black. A whole-map overview remains optional orientation evidence.
- Match before/after region, placement seed, source/layer hashes, crop, camera,
  output dimensions, phase, and settings except the declared variable. Do not
  enlarge the whole scene or cherry-pick a new camera to claim sharper assets.
- At candidate checkpoints include matched day/night views (the existing
  four-phase acceptance remains), both zooms, and a short deterministic camera
  scroll. Hydrology stays static: no new water animation requirement.
- Include Lab-only label/status envelopes and HUD clearance masks where relevant;
  these are layout witnesses, not proof of native UI integration.
- Report contextual before/after first, diagnosis/isolation second, stress
  coverage last. Record weaknesses at actual display size, not only enlarged crops.

## Ownership and nonblocking delivery

Q8's first deliverable is the smallest runnable versioned gameplay recipe and
preview using available candidates or frozen art, not a completed beauty gate.
Publish its exact Mac replay command, recipe schema, pinned inputs, seed, region
and crop coordinates, and provenance. Other owners consume it read-only and
render outputs in their own namespaces. Q8 owns placement; Q5 owns route
rendering, Q7 object/city presentation, Q6 lighting, and Q0 replay/platform and
verified source acquisition. Report missing interfaces to the owner; do not
duplicate shared backends or edit another owner's implementation.

Use verified cached `test.biq` regions as soon as available. Before that, Q8 and
the other owners may publish clearly labeled provisional contextual recipes
over available terrain and keep working on Mac. Do not call those actual
`test.biq` or captured gameplay. Added objects remain separate deterministic
Lab layers with augmentation-off controls. Replace provisional terrain and
rerun before real-map acceptance; preserve `REAL_MAP_VALIDATION.md` requirements.

Cache recipe geometry/assets and iterate on one small region, phase, and zoom.
Do not rerun the full gallery, all phases, or Windows parity on every change.
No owner waits for Q8 to finish before continuing useful work.
