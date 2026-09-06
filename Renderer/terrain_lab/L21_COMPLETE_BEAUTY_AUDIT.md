# L21 Complete Beauty Scene Audit

Status: complete; deterministic 192-tile alternate-skin release composition critically inspected and explicitly approved under the user's 2026-09-06 autonomous-review authorization.

## Scene and provenance

- `RUN_L21.bat` composes the frozen approved L9-L20 terrain, relief, water, vegetation, river, road, railroad, resource, city, mine, farm, goody-hut, colony, infrastructure, lighting, and unit handoffs without introducing another visual system.
- The terrain window is the authoritative 192-tile BIQ-derived L13 fixture. Objects and units remain deterministic Lab-only augmentations over unchanged BIQ terrain and are not represented as captured Civ III runtime state.
- Every frame uses only the approved alternate environment skin, source-authored materials and emissives, the shared environment evaluator, and the exact normalized source art already accepted by the per-system gates.

## Critical visual review

- Noon preserves continuous coastlines and rivers, connected mildly irregular routes, readable terrain families and elevation, dense but subordinate vegetation, coherent mountain chains, grounded cities/improvements, and clear resources and units across the complete scene. Land resources and every unit family now show explicit authored-normal face separation plus readable source-scaled cast/contact footprints.
- Sunset and sunrise preserve the canonical directional-light change, common cast-shadow direction, face shading, contact grounding, water response, and readable silhouettes without changing shadow geometry between object classes.
- Midnight remains deliberately dark but legible. Source-authored city and infrastructure emissives read clearly; water depth, shoreline contours, relief, routes, and non-emissive units remain distinguishable without invented bloom, smoke, streetlights, or animated effects.
- The reduced frame preserves strategic readability at the second Civ III zoom base. No severe clutter collapse, tile seam, shoreline cutoff, route break, floating body, mountain/hill clipping, or owner-color regression was found.
- The no-unit control isolates the corrected resource/object stack from units. Its deliberate delta from the earlier control is confined to the approved L16 land-resource face/cast correction; fish remain free of false water-surface shadows.
- The reopened unit-material pass removes near-white calibration bodies from the complete scene: named source tints produce olive tanks, khaki infantry, warm skin, and authored horse/wood browns before localized civilization color is applied. Noon, midnight, and reduced inspection found no white-unit recurrence or non-unit pixel regression.

## Deterministic evidence

Two final unchanged `python3 Renderer/tools/renderer_dev.py lab` runs passed 123 tests and produced byte-identical outputs:

- noon: `8814a86a6886b941a266da399eb9c53a6a88a20cfc6219c911bba5730614723e`
- sunset: `7d16ed27c5f7d8e6e9cd4f6786077225e3f9436ee19cca460b680cf869e48cd0`
- midnight: `c17d8d4f20a8732d4544a0b33d489f2cbde54e1359b561b21b2b6af8e5b580e8`
- sunrise: `911ca5f96e129639161bba0096d5f924216ee7cbbe33e159fd534d5ef80f4120`
- reduced: `e2703a52bc82f75f141b02aea94b7d1e219f33a0ac934be21c51aeafa6f53036`
- no units: `6582747ec96995cca34cc8de996305ec7ccf1d58732562a2042988aa488ddfd0`
- runner: `66c93895841c54e7af39db82260caf8d6d59b43f5497593cdaed8644eb6e78e5`
- authoritative terrain fixture: `1500c085e88425796fb150db4a96a904722d1f8b0cce1d614d46db58bc3c0b4b`
- Lab unit scenario: `b85d570959920564b2baa42f04ffc0abdd3ff7c528d5223864d55779aefe643b`

L21 is the final Renderer Lab appearance gate. Runtime capture, ownership, caching, suppression, compositing, and combined-release verification remain Integration responsibilities.
