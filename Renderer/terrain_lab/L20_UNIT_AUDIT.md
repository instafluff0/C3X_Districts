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

Direct comparison with `canonical/unit_texture_and_civ_colors3.png` then found
that the Tank's dedicated `TeamColor` component was still inheriting the
vehicle root transform and sitting largely below the ground plane. The generic
compound importer now resolves the component's declared `tankAll` attachment
bone, and the component-role contract maps dedicated `TeamColor` geometry to a
constant owner mask. This is source-semantic authoring: runtime code never
branches on the Tank name or hardcodes a color for a particular unit. The same
owner-color path supplies blue, red, gold, and green from instance state while
the neutral `Vehicle_Woodland` hull stays textured olive.

## Shared-shadow correction

The user reopened the final beauty gate on 2026-09-06 because unit lighting was
technically present but too subtle at map scale. Every ordinary, compound,
worker, and Army body preserves authored-normal face shading with the same
horizontal emphasis used for other small raised forms. Cast shadows no longer
use one radius/height quad per unit: every visible animated component projects
its actual source triangles along the shared L13A light direction. Light-facing
coverage plus a restrained five-sample penumbra preserves recognizable hull,
turret, limb, weapon, mount, sail, wing, and attachment silhouettes without
turning them into hard black decals. Direction and near-constant stylized length
remain shared across object families and day phases; only strength follows the
approved environment evaluator.

## Deterministic evidence

Two final unchanged `python3 Renderer/tools/renderer_dev.py lab` runs produced byte-identical outputs:

- noon complete: `96c1903ba0a3c772657f9abda9724dd778bfe4ca82623ff84520dc264a8a364c`
- midnight complete: `552f1e6bd10b45cc2b4a1f11f7066b1b9a3ddd159d41aa3639f2d39a7dc0241d`
- reduced: `c1faff8aa46c16a28aa35f6533f50c2a961896ad89f903253c2defb4a0e35365`
- no units: `6582747ec96995cca34cc8de996305ec7ccf1d58732562a2042988aa488ddfd0`
- unit isolation: `138de1c5f7207433432aaf3ec206b390e7bd15b975665d92a16b1340813e6390`
- eight-facing turntable: `65ae2adf03fe56eef6fd5ec710c4912cca0c92617cfd4aa4a7340e5946ce78e1`
- action/worker matrix: `497b9860aeb5b08b3b9e9008ea98ef4d29fe15032ad087ba8633e1746c346500`
- Lab scenario: `8066d641aa12ab7d42e62c83b77bc048bb003446eb97d975ede0e9814662d9b3`

The no-unit hash exactly matches approved L19B noon.
