# Unit asset conversion lab

Status: the first complete unit vertical slice is proven offline with `UNIT_WARRIOR`. No game/runtime integration is enabled.

The broader non-Warrior intake is also executable offline. It proves real
static art and raw animation curves for Archer, Swordsman, Infantry, Fighter,
and Galley without advancing L20 or enabling a unit runtime path.

Army presentation is now separately frozen as a generic composite contract.
It reuses the exact ordinary unit asset Civ III currently displays and places a
dedicated Civ VI Great General commander beside it; it never bakes combinations
or chooses a member in renderer code. Classical Great General intake confirms a
horse plus virtual Rider composition, while the Modern Great General exposes a
directly resolvable foot-officer variation. See
`army_rendering_strategy.md` and `army_render_strategy.json`.

## What the slice resolves

`unit_member_resolver.py` follows the Base `Units.artdef` and `Unit_Bins.artdef` data rather than treating a unit as one model. For the Base Warrior it resolves:

- four members using `UnitMemberTypes/Warrior` variation `Any/A`;
- the declared block formation and four combat offsets;
- body, head, hair/helmet, weapon, armor, and beard attachment bins;
- every candidate in the selected `Any` bins, their declaration order, scale, tint, package, and source entry;
- five non-empty deterministic first candidates: heavy male body, Caucasian head, African spear helmet, Warrior hammer, and Warrior armor;
- `USE_CIV_COLOR` on the armor attachment and the Caucasian skin-tint profile on body/head;
- idle, move, melee attack, and melee death action sources.

The complete source inventory remains in the ignored build report. The normalized runtime recipe contains only semantic `unit/warrior/*` IDs and no Civ VI package paths.

## Model extraction

`unit_model_extractor.py` reads `ModelPackageEntry` records from `units/units.blp` and emits a dedicated `c3x.unit_pack.v0` lab pack. It reuses the already-proven low-level static-package, mesh, material, texture, and skeleton decoders, but unit outputs use their own `units/`, `meshes/unit/`, `materials/unit/`, `textures/unit/`, and `skeletons/unit/` namespaces.

The first slice contains three skinned components and two rigid components. Body, head, and armor each reconstruct the source rest pose within the strict generic CPU-skin tolerance. Helmet and hammer preserve the ArtDef `Hat` and `WeaponPrimary` attachment semantics.

The decisive unit-specific detail is the mesh-local skin palette. Vertex joint bytes do not directly index the complete Granny skeleton. The selected Warrior components declare:

| Component | Local palette entries | Skeleton bones |
| --- | ---: | ---: |
| Body | 34 | 48 |
| Head | 5 | 75 |
| Armor | 22 | 48 |

The palette is the typed `int32` array referenced at offset `0x128` of the selected `ModelPackageEntry::BaseModelData_Entry`; the mesh container independently declares the same count at offset `0x20`. Both counts, every palette target, and every vertex-local palette index are fail-closed. Omitting this remap passes rest-pose reconstruction but produces visibly incorrect animated limbs, which is why the visual fixture is an essential gate.

## Actions and pose binding

The four standalone source animations are converted offline at `0.01` translation scale:

| Logical action | Source clip | Policy |
| --- | --- | --- |
| `idle` | `ANIMATION_Warrior_IdleB` | loop |
| `move` | `ANIMATION_UnitMedium_Run_SwordAndShieldA` | loop |
| `attack` | `ANIMATION_Warrior_AttackMeleeB` | clamp |
| `death` | `ANIMATION_Warrior_DeathMeleeA` | clamp |

`unit_action_validator.py` validates clip timing, the unique `Root` group, required humanoid core tracks, at least 30 matching tracks per skinned component, finite sampled deformation, and exact model-aware pose-cache/skeleton name binding. Untracked mesh/twist helpers retain their authored local rest relation; clip-only IK, cape, and accessory helpers are reported rather than silently treated as mesh bones.

Model-aware caches are emitted for all three skinned components and all four actions. The lab preview consumes those source-independent world-matrix caches and never loads Granny or Civ VI at render time.

## Current visual boundaries

- `Hat -> Head` and `WeaponPrimary -> Inven_R_Hand` are explicit lab socket aliases inferred from the ArtDef point names and matching humanoid rig bones. The eight-direction action fixtures show stable placement, but these aliases are not yet claimed as decoded engine attachment-point behavior.
- BC3 diffuse alpha is not transparency for these unit materials. The neutral fixture preserves source RGB and forces opaque sampling. The optional owner-color fixture treats inverse alpha as the `USE_CIV_COLOR` weight; the sentinel ArtDef has no parameters and the texture histogram strongly supports that interpretation, but the exact Firaxis tint equation remains inferred.
- The first formation fixture uses a four-corner block derived from the declared `SpacingX` and `SpacingY`. The exact general formation-layout algorithm remains a later cross-archetype task.

These boundaries are visible in reports and preview footers. They do not enable a production unit path.

## Non-Warrior family proof

`unit_family_strategy.json` selects five deliberately different mappings:

| Civ III unit | Source family | Archetype | Components | Unique clips / logical bindings |
| --- | --- | --- | ---: | ---: |
| Archer | `UNIT_ARCHER` | humanoid ranged | 5 | 8 / 8 |
| Swordsman | `UNIT_SWORDSMAN` | humanoid melee | 6 | 8 / 8 |
| Infantry | `UNIT_INFANTRY` | humanoid firearm | 6 | 8 / 8 |
| Fighter | `UNIT_FIGHTER` | aircraft | 1 | 8 / 12 |
| Galley | `UNIT_GALLEY` | naval | 1 | 8 / 8 |

`unit_family_asset_importer.py` follows each Unit -> Member -> attachment-bin
graph and compiles 19 components: 12 skinned and seven rigid attachments, with
45 deduplicated normalized textures. The unit-specific adapter accepts a real
cross-model quantization case that the landmark importer correctly continues
to reject: a small number of nonzero byte-weight tuples total 241--254 instead
of exactly 255. The adapter records source min/max/count evidence and normalizes
the tuple to one. Zero totals and invalid bone indices still fail closed.

Thirty-seven unique source clips convert into generic `.c3anim` files and provide
44 logical bindings. Every proof unit has idle, fidget, move, fortify, attack,
defend, victory, and death. Fighter reuses semantically safe existing clips for
fidget, fortify, defend, and victory, and additionally retains takeoff, landing,
left turn, and right turn. `unit_family_action_validator.py` chooses the source track
group with the greatest skeleton-name overlap, enforces a per-archetype minimum,
and samples every skinned component to prove finite deformation. The generic
`unit_family_pose_cache_builder.py` then bakes 93 unique component/model/clip
combinations for 100 logical component/action bindings; aliases reuse the same
cache bytes. Galley maps its stop clip to defend and aliases the three low-value
pose actions to idle. All five families now meet the same strict model-aware
pose-cache/skeleton-name boundary as the Warrior slice.

Galley also closes the first vehicle-container gap. Its source body contains
three meshes/primitives and two materials under one model, with three local
skin palettes containing 7, 7, and 10 entries. The normalized unit document now
emits a mesh list, material variants, and explicit draw bindings rather than
assuming one mesh/material per component. Legacy single-part components keep
their compact fields, so this is a generic format extension rather than a
Galley-specific branch.

`unit_action_conversion.json` is the deliberately small first action contract.
The installed 90-directory Civ III inventory has DEFAULT/RUN/DEATH on all 90,
FIDGET on 83, FORTIFY on 73, ATTACK1 on 44, VICTORY on 24, and ATTACK2 on only
nine. Therefore DEFAULT maps to idle, RUN to move, FIDGET to fidget, FORTIFY to
fortify, DEATH to death, and VICTORY to victory. ATTACK1/2/3 initially alias one
logical attack clip unless a unit receives an authored override. Defend is not
a Civ III FLC slot; it is a target-side reaction selected only when Civ III's
authoritative combat event reports one.

Converted resources are classified as `motion` or `pose`. A two-frame idle or
brace resource is preserved as a target pose. Fidget blends idle -> pose -> idle
over native FIDGET progress; fortify blends into the pose and holds it while the
native unit remains fortified. Motion clips sample directly by normalized
native progress. Victory returns to idle on the native transition, while death
holds its last pose until Civ III despawns the unit. No converted clip controls
gameplay duration, damage, removal, or combat outcome.

Attack conversion owns only body/weapon pose and recoil. Projectile release,
muzzle flash, impact, and damage visuals remain effect events with authored
normalized phase markers for M7.5. They are never inferred from rendered frame
count.

The checked visual fixtures show both REST/IDLE/MOVE/ATTACK/DEATH and the full
eight-action basic matrix for the first four non-Warrior families. Galley's
static body and all eight actions are converted and structurally validated;
Lab still owns its multi-part renderer fixture. Each rendered cell is auto-fit
to inspect source art, so it is not evidence for runtime scale. Equipment
separation visible in some extreme/death poses is recorded as a pose-cache/socket
task, not hidden by ad hoc preview offsets.
`Root`, `Hat -> Head`, `WeaponPrimary -> Inven_R_Hand`,
`WeaponSecondary -> Inven_L_Hand`, and the Infantry `ArmBand -> Lure` aliases
remain explicitly inferred.

Eight-direction presentation does not require eight converted mesh copies.
The eventual lab path rotates the normalized model about +Z in 45-degree steps,
using Civ III's authoritative facing and screen anchor. A per-unit authored
`facing_offset_degrees` calibration may correct a source model's declared
forward axis before those eight rotations; that is orientation metadata, not a
destructive mesh rewrite. Direction-specific authored clips should only replace
this rule when the source graph actually declares them.

The next composition families are now source-resolved and pass one generic
offline assembly/animation proof:

- Horseman and the Classical Great General expose independently selectable
  Horse and Rider variations at `RiderAttach`.
- Catapult exposes its vehicle, attached OperatorA, and separate two-figure
  defender recipe.
- Tank exposes its vehicle and attached TankGunner at `GunnerAttach`.

`unit_member_resolver.py --member-index/--variation` fetches each part exactly.
`compound_unit_asset_importer.py` compiles them as eight independent nodes and
four resolved socket joints. The paired path now validates 52 unique converted
clips serving 62 node/action bindings across 31 actions and bakes 52
model-aware pose caches. The runtime-facing recipe is an arbitrary acyclic tree
with no unit-name dispatch. The basic matrix is complete except for a truthful
Catapult death source. `unit_visual_calibration.json` and
`unit_formation_strategy.json` freeze all eight directions, both Civ III zoom
bases, single-body default, optional pack-authored humanoid triad, and the Army
two-body exception. L20 still owns the actual visual measurements and approval. See
`compound_unit_asset_conversion.md`.

Flattening any of those into a single proof body would lose authored structure
and give L20 false coverage.

## Civ III owner-color lab

`civ3_owner_palette_compiler.py` compiles `ntp00.pcx` through `ntp31.pcx`
into the source-independent `c3x.owner_color_pack.v0` lab format. Repeated
`--palette-root` arguments are resolved from low to high priority per filename,
so a scenario can override only the palettes it supplies while the remainder
fall back to the preceding root. Matching is case-insensitive, like Civ III's
Windows asset lookup. Missing tables, invalid PCX modes, truncated palettes,
and case-colliding files fail closed.

The pack preserves all 64 Civ III-replaced RGB entries for every
`Leader.Color_Table_ID`, plus a GPU-ready 64-by-32 RGBA8 lookup. This mirrors
the confirmed Civ III path: `Units_Image_Data` loads
`Art/Units/Palettes/ntp00.pcx` through `ntp31.pcx`, and unit animation palette
selection copies entries 0 through 63 from the table selected by
`leaders[civ_id].Color_Table_ID`. Machine-specific source paths remain only in
the ignored compiler report, not the normalized pack.

The current true-color units have no original Civ III FLC palette index per
texel. Entries 0 through 15 remain the coherent primary owner ramp and entry 6
is Civ III's representative display color. The first diagnostic incorrectly
selected a ramp shade from source luminance and replaced masked RGB at full
strength; that double-applied shading and erased source browns, metals, and
panel detail, producing implausibly saturated armor and aircraft. The corrected
lab fixture uses entry 6 to conservatively modulate the existing base color in
linear space, with a 0.82 maximum mask strength, 0.42 neutral floor, and 0.95
color gain. Those are provisional L20 visual-calibration values, not claimed
Firaxis constants.

The Civ III palette values and source ArtDef component assignments are exact;
the inverse-alpha mask interpretation and tint equation remain explicit visual
inferences. Installed binary strings confirm separate rigid/skinned tint shader
variants and primary/secondary color inputs, but do not expose the shader
equation. Entries 16 through 63 are preserved for future materials with exact
slot semantics and are not misused as a one-dimensional brightness ramp.

Coverage follows source composition rather than forcing every unit into a
uniform. Archer armor, Swordsman armor/hat/shield, and the Fighter body are
marked `USE_CIV_COLOR`. Infantry instead uses the fixed
`Infantry_European` profile on its uniform and reserves `USE_CIV_COLOR` for a
separate armband, so player-color variation is intentionally much subtler.
That source hint is not allowed to become an Infantry branch. Every normalized
component carries the same generic `owner_color` material record. Importers map
known source hints into it, while pack authors can supply a sidecar override by
stable logical asset ID for source art with no useful hint or insufficient
screen coverage. L20 measures changed pixels against the neutral render at both
Civ III scales; a unit below the readability threshold becomes
`needs_pack_authoring_override` instead of silently tinting its whole body.
The checked-in family-lab sidecar demonstrates that mechanism on Infantry: it
marks components and strengths in pack data only. The importer, renderer, and
runtime contract contain no Infantry name/type branch, so a scenario unit or
future source adapter uses the identical path with its own logical asset IDs.
The coverage renderer discovers unit IDs from the compiled manifest rather than
from a checked-in unit list; adding an arbitrary unit to an import profile also
adds it to the gate without a code change. Archetypes may likewise be shared by
any number of units.
Without that sidecar, the automated screen-space gate measured Infantry at
0 changed pixels at both normal and half scale and requested pack authoring.
With the component metadata applied it measures 583 normal-scale and 150
half-scale changed pixels, above the provisional 24/6 minimums. The other three
families continue to pass from source hints alone. These counts are deterministic
lab evidence for mask coverage, not a claim that visual taste is fully approved.

Owner color is selected at runtime, never baked into one texture per
civilization. `unit_owner_color_runtime.json` freezes the handoff: conversion
emits a neutral base plus a continuous `civ_color_weight`; the runtime uploads
one 64-by-32 lookup texture containing Civ III's effective color tables and
passes a `display_color_table_id` per unit instance. The provisional shader
uses primary index 6 to modulate, rather than replace, source RGB through the
material weight. Meshes, skeletons, clips, and neutral textures are therefore
shared by every civilization.

The selector is intentionally **display** color, not simply body owner. Civ
III's native `Unit::tick_anim` path can deliberately display civ 0 for a
hidden-nationality unit seen by another player; barbarian/neutral and other
viewer-conditioned cases must follow that same native decision. After the
display civilization is known, the row is
`leaders[display_civ_id].Color_Table_ID`. Captures, conversions, alternate
scenario colors, or any other owner/display-identity change update only that
small per-instance selector.

Production must populate the lookup from
`units_image_data.Color_Tables[*].JGL_Color_Table`, after Civ III has resolved
and loaded the active game/scenario. That automatically inherits scenario
search precedence and partial `ntpXX.pcx` overrides. The offline compiler's
repeated `--palette-root` behavior exists to reproduce and validate this in the
lab; it is not a second production asset resolver. The LUT is rebuilt only
when the effective palette generation changes, such as loading a different
scenario, rather than when an individual unit changes owner.

This requires no new production patch symbol. `Units_Image_Data.Color_Tables`,
`Leader.Color_Table_ID`, and the unit body owner are already represented in the
known native types; the later I20 body boundary must capture the native
viewer-conditioned display choice alongside the unit record. Production
activation remains deferred until L20 is approved.

## Reproduction

On the source development machine:

```sh
python3 Renderer/tools/asset_compiler/unit_member_resolver.py
python3 Renderer/tools/asset_compiler/unit_family_asset_importer.py \
  --owner-color-overrides Renderer/tools/asset_compiler/unit_family_owner_color_overrides.json
python3 Renderer/tools/asset_compiler/unit_family_action_validator.py \
  --pack Renderer/packs/UnitFamilyLab \
  --report Renderer/preview/out/units/family_action_validation.json
```

Convert the four source clips and bake the component pose caches with the checked-in Windows/VM converters, then run:

```sh
python3 Renderer/tools/asset_compiler/unit_action_validator.py \
  --pack Renderer/packs/UnitWarriorLab \
  --report Renderer/preview/out/units/warrior_actions.json

python3 Renderer/tools/asset_compiler/civ3_owner_palette_compiler.py \
  --palette-root /path/to/fallback/Art/Units/Palettes \
  --palette-root /path/to/scenario/Art/Units/Palettes \
  --output Renderer/packs/Civ3OwnerColorsLab \
  --report Renderer/preview/out/units/civ3_owner_color_compile.json

python3 Renderer/preview/render_unit_turntable.py \
  --pack Renderer/packs/UnitWarriorLab \
  --owner-palette-pack Renderer/packs/Civ3OwnerColorsLab \
  --color-table-id 6 \
  --output Renderer/preview/out/units/warrior_civ3_color06_action_sheet.png \
  --report Renderer/preview/out/units/warrior_civ3_color06_action_sheet.json

python3 Renderer/preview/render_owner_palette_sheet.py \
  --pack Renderer/packs/Civ3OwnerColorsLab \
  --output Renderer/preview/out/units/civ3_owner_color_tables.png \
  --report Renderer/preview/out/units/civ3_owner_color_tables.json

python3 Renderer/preview/render_unit_owner_color_sheet.py \
  --pack Renderer/packs/UnitFamilyLab \
  --owner-palette-pack Renderer/packs/Civ3OwnerColorsLab \
  --output Renderer/preview/out/units/family_runtime_owner_colors.png \
  --report Renderer/preview/out/units/family_runtime_owner_colors.json
```

For the non-Warrior family proof, build the static pack on macOS, convert all
37 unique clips through the Windows VM batch, then rebuild the manifest,
validate, bake pose caches, and render:

```sh
python3 Renderer/tools/asset_compiler/unit_family_asset_importer.py \
  --owner-color-overrides Renderer/tools/asset_compiler/unit_family_owner_color_overrides.json
# Windows 11 VM/shared checkout:
Renderer\tools\asset_compiler\CONVERT_UNIT_FAMILY_ANIMATIONS.bat
python3 Renderer/tools/asset_compiler/unit_family_asset_importer.py \
  --owner-color-overrides Renderer/tools/asset_compiler/unit_family_owner_color_overrides.json
python3 Renderer/tools/asset_compiler/unit_family_action_validator.py \
  --pack Renderer/packs/UnitFamilyLab \
  --report Renderer/preview/out/units/family_action_validation.json
python3 Renderer/tools/asset_compiler/unit_family_pose_cache_builder.py \
  --pack Renderer/packs/UnitFamilyLab \
  --report Renderer/preview/out/units/family_pose_caches.json
python3 Renderer/preview/render_unit_family_sheet.py \
  --pack Renderer/packs/UnitFamilyLab \
  --output Renderer/preview/out/units/family_static_animation_sheet.png \
  --report Renderer/preview/out/units/family_static_animation_sheet.json
python3 Renderer/preview/render_unit_family_sheet.py \
  --pack Renderer/packs/UnitFamilyLab \
  --basic-actions \
  --output Renderer/preview/out/units/family_basic_action_sheet.png \
  --report Renderer/preview/out/units/family_basic_action_sheet.json
```

Add `--formation` for the four-member sheet. Generated Firaxis-derived packs, clips, caches, reports, and previews remain ignored and must not be redistributed.
