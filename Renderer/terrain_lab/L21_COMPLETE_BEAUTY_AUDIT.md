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
- Direct canonical vehicle comparison is preserved in composition: Tank
  `TeamColor` geometry follows its normalized `tankAll` attachment and carries
  saturated localized owner panels while the body remains textured olive. This
  uses generic source/component semantics and instance owner state, not a
  runtime Tank-name coloring branch.
- Unit cast shadows are projected from every visible animated source component,
  so tank hulls and turrets, humanoid limbs and weapons, horses and riders,
  catapult frames, wings, hulls, sails, and Army members retain recognizable
  silhouettes. Their restrained penumbra, direction, and day-phase strength
  remain coherent with the shared terrain/object lighting rather than using a
  detached generic blob.

## Deterministic evidence

Two final unchanged `python3 Renderer/tools/renderer_dev.py lab` runs passed 124 tests and produced byte-identical outputs:

- noon: `96c1903ba0a3c772657f9abda9724dd778bfe4ca82623ff84520dc264a8a364c`
- sunset: `c867411726b1ed598216ee3d170ecc06f9d80a1c7480b768dfbaf96b24efdf26`
- midnight: `552f1e6bd10b45cc2b4a1f11f7066b1b9a3ddd159d41aa3639f2d39a7dc0241d`
- sunrise: `67b118240e5b5a8b31828d7ac36fd4974916d5957b0825e1baabc27beeec4f94`
- reduced: `c1faff8aa46c16a28aa35f6533f50c2a961896ad89f903253c2defb4a0e35365`
- no units: `6582747ec96995cca34cc8de996305ec7ccf1d58732562a2042988aa488ddfd0`
- runner: `9afa817dcf2276a7510a8c3ec3c4d12567fe5b898f3e1c34a64fce77b4127ec0`
- authoritative terrain fixture: `1500c085e88425796fb150db4a96a904722d1f8b0cce1d614d46db58bc3c0b4b`
- Lab unit scenario: `8066d641aa12ab7d42e62c83b77bc048bb003446eb97d975ede0e9814662d9b3`

L21 is the final Renderer Lab appearance gate. Runtime capture, ownership, caching, suppression, compositing, and combined-release verification remain Integration responsibilities.
