# L20 Unit and Animation Audit

Status: complete; deterministic 192-tile alternate-skin candidate critically inspected and explicitly approved under the user's 2026-09-06 autonomous-review authorization.

## Scene and provenance

- `fixtures/l20_units_192.csv` adds 51 deterministic Lab-only unit records over unchanged authoritative BIQ terrain: Archer, Swordsman, Infantry, Fighter, Galley, Horseman, Catapult, Tank, Classical Great General/Army, and Worker witnesses.
- Five ordinary families use normalized source meshes and sampled source clips. Horseman, Catapult, Tank, and Great General use the generic parent/child/socket compositions and remain atomic bodies.
- Worker witnesses use the prepared normalized Builder light-work, heavy-work, cutting, and capture clips on a restrained generic civilian body. No incorrect sword, shield, invented tool, projectile, particle, smoke, or work effect is rendered.
- Army presentation is exactly one mounted Classical Great General plus one ordinary selected member; it never invents a full loaded roster.

## Critical visual review

- The first pass was rejected because normal-scale humanoid silhouettes were too delicate and compound movement samples could interpolate toward water. Final scales were raised modestly, while compound and worker witnesses were moved to interior land with domain-correct endpoints.
- Close inspection confirmed readable three-body humanoid/worker formations, two-body mounted formations, single crewed/vehicle/naval/air silhouettes, stable tile grounding, and subordinate scale beside terrain and cities.
- The Tank composition uses the source-declared `hatchPivot` candidate because the extracted `GunnerAttach` helper resolves at vehicle origin. The exposed gunner is kept compact and attached to the hatch.
- All eight facings, four owner colors, movement interpolation, a two-body stack, four-phase move/attack/death samples, four Builder specialty states, and hidden-state suppression are represented. Catapult death remains truthfully absent rather than relabeling unrelated art.
- Units use the accepted shared face/contact/cast lighting. They remain non-emissive at night; existing source-authored city and infrastructure emission is preserved unchanged.
- The no-unit control is byte-identical to the approved L19B noon render, proving the unit pass does not alter earlier layers.

## Deterministic evidence

Two final unchanged `python3 Renderer/tools/renderer_dev.py lab` runs produced byte-identical outputs:

- noon complete: `cc13bf4fdfe0debaefca3c3ab1e8f4609f5370b40e75902118db510c90c7f6e7`
- midnight complete: `ed6ef1ec2d212442ed928cb935533cebd8255d2094d54d2f4bfe24d17022ae57`
- reduced: `12bc58fa16a53828358396da95633fad68d17847cc341943370f59154829f9b2`
- no units: `f9ede652e3eb47dbcbb8f0943ef8f2654a6f65735094a69dae35136a880d7fa1`
- unit isolation: `bd5347e8bda69362e5553060b3ff75a532cec640e851ec5f219d7df5a0c2b10e`
- eight-facing turntable: `de282e8225d6881417506115ab857b3e99148c8b7363241ce5aa1b4ddd29fdeb`
- action/worker matrix: `20647804ec000cf8f8764bae3a8121f239693bec7903fb68eae3a869562f645a`
- Lab scenario: `b85d570959920564b2baa42f04ffc0abdd3ff7c528d5223864d55779aefe643b`

The no-unit hash exactly matches approved L19B noon.
