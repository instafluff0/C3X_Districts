# L21 Complete Beauty Scene Audit

Status: complete; deterministic 192-tile alternate-skin release composition critically inspected and explicitly approved under the user's 2026-09-06 autonomous-review authorization.

## Scene and provenance

- `RUN_L21.bat` composes the frozen approved L9-L20 terrain, relief, water, vegetation, river, road, railroad, resource, city, mine, farm, goody-hut, colony, infrastructure, lighting, and unit handoffs without introducing another visual system.
- The terrain window is the authoritative 192-tile BIQ-derived L13 fixture. Objects and units remain deterministic Lab-only augmentations over unchanged BIQ terrain and are not represented as captured Civ III runtime state.
- Every frame uses only the approved alternate environment skin, source-authored materials and emissives, the shared environment evaluator, and the exact normalized source art already accepted by the per-system gates.

## Critical visual review

- Noon preserves continuous coastlines and rivers, connected mildly irregular routes, readable terrain families and elevation, dense but subordinate vegetation, coherent mountain chains, grounded cities/improvements, and clear resources and units across the complete scene.
- Sunset and sunrise preserve the canonical directional-light change, common cast-shadow direction, face shading, contact grounding, water response, and readable silhouettes without changing shadow geometry between object classes.
- Midnight remains deliberately dark but legible. Source-authored city and infrastructure emissives read clearly; water depth, shoreline contours, relief, routes, and non-emissive units remain distinguishable without invented bloom, smoke, streetlights, or animated effects.
- The reduced frame preserves strategic readability at the second Civ III zoom base. No severe clutter collapse, tile seam, shoreline cutoff, route break, floating body, mountain/hill clipping, or owner-color regression was found.
- The no-unit control is byte-identical to the approved L19B/L20 no-unit noon frame, proving that final composition does not alter frozen earlier layers.

## Deterministic evidence

Two final unchanged `python3 Renderer/tools/renderer_dev.py lab` runs passed 121 tests and produced byte-identical outputs:

- noon: `cc13bf4fdfe0debaefca3c3ab1e8f4609f5370b40e75902118db510c90c7f6e7`
- sunset: `cc74becef3ccd4db03591e9b164c81aefded52eaa74b2347e71d343b0806ad22`
- midnight: `ed6ef1ec2d212442ed928cb935533cebd8255d2094d54d2f4bfe24d17022ae57`
- sunrise: `4c9d4120c988988b4034dfcdb573dd8646e1d5fee9f8b53b10dbb3cc4e504850`
- reduced: `12bc58fa16a53828358396da95633fad68d17847cc341943370f59154829f9b2`
- no units: `f9ede652e3eb47dbcbb8f0943ef8f2654a6f65735094a69dae35136a880d7fa1`
- runner: `66c93895841c54e7af39db82260caf8d6d59b43f5497593cdaed8644eb6e78e5`
- authoritative terrain fixture: `1500c085e88425796fb150db4a96a904722d1f8b0cce1d614d46db58bc3c0b4b`
- Lab unit scenario: `b85d570959920564b2baa42f04ffc0abdd3ff7c528d5223864d55779aefe643b`

L21 is the final Renderer Lab appearance gate. Runtime capture, ownership, caching, suppression, compositing, and combined-release verification remain Integration responsibilities.
