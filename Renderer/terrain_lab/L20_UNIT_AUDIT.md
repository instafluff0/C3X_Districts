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

## Civilization-color refinement

The user reopened L20 on 2026-09-06 against the registered canonical infantry,
cavalry, naval, and vehicle references. The earlier Lab shader flattened every
active mask toward one hue and discarded component-authored strength, which made
some aircraft, ships, and clothing look dipped in player color. The accepted
revision keeps the source alpha mask per material, encodes strong (`0.82`),
medium (`0.58`), and restrained (`0.35`) authoring strengths, and maps neutral
source value through a dark-to-light owner-color ramp with a small source-detail
contribution. Skin, horse bodies, hair, wood, and steel remain neutral while
shields, armbands, barding, sails/hull markings, and dedicated vehicle panels
carry the readable owner cue.

The Great General's rider and mount armor receive explicit source-independent
pack-authoring masks because the source recipe supplied no active ownership cue;
the false-positive Catapult operator hair mask is disabled. These decisions live
in generic per-component metadata rather than unit-name shader branches. Close,
full-scene, night, action, eight-facing, and reduced-scale inspection confirmed
four distinct owner samples without changing unit scale, animation, placement,
lighting, shadows, or any non-unit layer. Two unchanged native runs were
byte-identical.

The first owner-color revision exposed a second source-material problem during
native inspection: several Civ VI base textures are intentionally near-white
calibration maps whose retained ArtDef `Tint` name supplies their actual color.
Ignoring that metadata left tanks and infantry looking unnaturally white. The
generic component pipeline now carries a separate source-tint marker for
`BaseMale_SkinColor_Caucasian`, `GreatPeople_Military`, `Horse_Default`,
`Horse_Secondary`, `Infantry_European`, `Vehicle_Woodland`, and `Wood`. The Lab
applies the exact selected installed `Units.artdef` RGB color to the textured
material before the independent civilization-color mask. Tanks therefore read
as olive vehicles, infantry as khaki, skin as warm skin, and horses/wood as
their authored browns while folds, recesses, owner panels, normals, and shadows
remain visible. A hierarchical marker decoder prevents lower decimal fields
from carrying into higher fields. Native close, complete, reduced, and night
inspection confirms the white-calibration failure is gone.

## Shared-shadow correction

The user reopened the final beauty gate on 2026-09-06 because unit lighting was
technically present but too subtle at map scale. Every ordinary, compound,
worker, and Army body now preserves its authored normals while receiving the
same horizontal face emphasis used for other small raised forms. Every visible
formation member has a source-scaled cast/contact footprint with a six-pixel
minimum and the exact shared L13A direction and time-of-day strength. The
corrected complete noon witness is
`0d516a393ed529eb839ef037a45999a0a224f29877739e1196cb49140c2300db`;
two unchanged L21 runs were byte-identical.

## Deterministic evidence

Two final unchanged `python3 Renderer/tools/renderer_dev.py lab` runs produced byte-identical outputs:

- noon complete: `8814a86a6886b941a266da399eb9c53a6a88a20cfc6219c911bba5730614723e`
- midnight complete: `c17d8d4f20a8732d4544a0b33d489f2cbde54e1359b561b21b2b6af8e5b580e8`
- reduced: `e2703a52bc82f75f141b02aea94b7d1e219f33a0ac934be21c51aeafa6f53036`
- no units: `6582747ec96995cca34cc8de996305ec7ccf1d58732562a2042988aa488ddfd0`
- unit isolation: `d2069f5030127ca551a358d80dd70e199ccfefcc2974f0e0c9686fa240ea502b`
- eight-facing turntable: `3ccef1c3dcc8c1fabe4126967e5484b2c4170759ce87691e8111604c8113cb98`
- action/worker matrix: `bf9caae116f76e751976592a01bdcb46e3b22ac391096fd2448add15fd7acb38`
- Lab scenario: `b85d570959920564b2baa42f04ffc0abdd3ff7c528d5223864d55779aefe643b`

The no-unit hash exactly matches approved L19B noon.
