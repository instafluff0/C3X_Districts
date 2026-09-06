# L21 Complete Beauty Scene Audit

Status: complete; 192-tile alternate-skin release composition frozen by the user's explicit 2026-09-06 instruction to finish the Lab iteration immediately and waive the remaining final validation scripts.

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
- Civilization territory now follows tile ownership discontinuities with no
  same-owner internal seams. Each civilization uses one main owner color only;
  rival boundaries may place the two civilizations' separate single-color
  ribbons side by side, but no ribbon synthesizes a secondary outline color.
  Slight deterministic waviness and rounded same-color joins prevent the
  borders from reading as disconnected tile strokes.
- Cities, walls, mines, farm buildings, goody huts, colonies, fortresses,
  barricades, airfields, outposts, radar towers, and victory locations now use
  the same authored-normal face response and shared-light source-mesh cast
  projection. Flat territory ribbons, pollution, and crater art remain
  deliberately non-casting.

## Evidence and validation disposition

The pre-territory L21 composition passed two unchanged 124-test runs and
produced byte-identical outputs. The territory/unified-object-shadow revision
then passed 125 local contract tests and produced visually inspected noon,
sunset, midnight, and sunrise standalone frames. The user explicitly waived
the remaining full matrix and requested immediate Lab closure, so the hashes
below are retained only as the pre-territory deterministic baseline rather
than misrepresented as final-territory hashes:

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
