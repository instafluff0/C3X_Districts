# Army Rendering Strategy

Status: offline L20/I20 investigation complete; no runtime ownership is enabled.

## Decision

An Army is one parent unit with two renderer-owned child bodies:

1. the exact member Civilization III currently chooses to display; and
2. a dedicated Civ VI Great General commander, offset beside that member.

This deliberately reproduces Civ III's readable visual grammar without baking
Army versions of every possible unit. A scenario-only or modded unit can become
the displayed member through the ordinary unit resolver with no Army-specific
asset or code branch. Loaded members other than the current displayed member
are not drawn in the basic presentation, because doing so would crowd a Civ III
tile and falsely turn roster size into formation size.

The commander source is not a generic infantry substitute. Ancient and Middle
Ages profiles target `UNIT_GREAT_GENERAL`; Industrial and Modern profiles target
`UNIT_GREAT_GENERAL_MODERN`. Packs may override the resulting generic commander
asset IDs, but the checked-in source adapter uses the dedicated Great General
art when converting the local prototype pack.

## What Civ III Actually Does

The decompiled installed-source audit confirms that the remembered behavior is
not an incidental stack draw:

- `Unit::tick_anim` detects an Army with a populated decompiled
  `field_1B0[1]` and enters a dedicated two-body path.
- That path resolves `field_1B0[1]` as a unit ID, draws the Army's own FLC body
  to the right, then draws that member's current FLC body at the ordinary
  anchor. The audited horizontal reference is 40 pixels at normal zoom and 20
  pixels at reduced zoom.
- `Animator::tick_all_unit_anims` and `Animator::update` advance both FLCs and
  include both bodies in dirty-region accounting.
- Army movement updates the displayed member's pixel target and direction with
  the containing Army.
- `Unit::load_into_army` initializes or may replace the field, and
  `Unit::select_army_member_for_combat` can change it during combat. C3X already
  patches the latter for its combat system.

The names `displayed_member_id` and `member_count` used in this renderer design
are inferred semantic aliases for decompiled `field_1B0[1]` and
`field_1B0[2]`; the field accesses and observed behavior are confirmed. The
renderer never independently selects the healthiest, strongest, first-loaded,
or most recently animated member.

Relevant navigation points in `ref/Civ3Conquests_master.exe.c` are
`Unit::FUN_005cbc30` near line 336591, `Unit::tick_anim` near line 336702, its
dedicated Army body routine immediately afterward, and the Animator routines
near lines 179298 and 180614. These line numbers are evidence aids, not patch
addresses.

## Source Art Result

The upgraded `unit_member_resolver.py` now understands explicit
`Category/Bin/#/Candidate` ArtDef bindings, reports virtual member attachments
instead of rejecting the recipe, and accepts an exact `--variation`. The
Classical `Horse` and attached `Rider` halves can therefore be fetched as two
explicit recipes without pretending the mounted compound is one flattened
mesh. The repeatable probe establishes:

| Civ VI target | Resolved member | First composition | Intake result |
| --- | --- | --- | --- |
| `UNIT_GREAT_GENERAL` | `GreatGeneral_Classical_Male` | Horse, saddle, mane, tail plus virtual `Rider` at `RiderAttach` | Dedicated source confirmed; requires the same compound mounted support already needed by Horseman |
| `UNIT_GREAT_GENERAL_MODERN` | `GreatGeneral_Modern_Male` | Foot `General` variation | Dedicated foot-officer components resolve now; static conversion and animation validation remain L20 work |

The modern member also declares driver/jeep variations. Those are an optional
later Modern-era presentation, not a reason to delay the foot commander. The
classical rider must remain a real horse+rider composition; flattening the
horse and omitting its virtual rider would produce exactly the wrong result.

No Civ VI package name enters the runtime strategy. Conversion emits generic
IDs such as `unit/army/commander/classical_mounted` and
`unit/army/commander/modern_foot`. C3X never depends on Civ VI files or formats
at runtime and does not redistribute the locally used source assets.

## Scene And Animation Contract

The immutable parent record is `kind: army` and carries the Army unit ID, era,
viewer-conditioned owner-color selector, native Army action/facing/progress,
Army pixel anchor, member count, and optional displayed-member record. The
member record carries its exact unit ID/type, its own native action/facing/
progress, display color, and pixel anchor.

Stable child IDs are:

- `army/<army_id>/commander`
- `army/<army_id>/member/<displayed_member_id>`

When Civ III changes the displayed member, only the member child identity and
ordinary unit asset change. The commander continues its parent event without a
pop or restart. An empty Army renders the commander alone.

Both native anchors are preferred. The 40/20-pixel offsets are reference
evidence for calibration and fallback, not coordinates baked into imported
geometry. Lab should express the fallback as a projection-relative right and
slightly rear offset so direction, both zooms, and future commander sizes remain
natural.

For the basic action set:

- idle/fidget/move use the two native action cursors, with the commander motion
  restrained so the member stays the primary silhouette;
- attack/defend let the displayed member carry the combat read while the
  commander uses a command/brace gesture;
- a displayed-member change never by itself kills or respawns the parent;
- commander death is played only for authoritative parent-Army death;
- Civ III remains authoritative for every action transition, duration, wait,
  outcome, anchor, direction, and despawn.

The commander uses the same runtime owner-color system as arbitrary units, but
on a bounded armband, sash, or small standard rather than washing the whole
figure in player color. The member uses its normal unit-specific mask. Civ III
retains one selection underlay, health bar, activity/status set, stack marker,
and HUD presentation for the parent Army; child bodies do not each get HUD.

## Ownership And Fallback

Before I20 owns Army bodies, an unresolved Army remains entirely native. After
ownership is enabled, missing commander/member art is a visible custom failure;
the renderer must not combine one custom body with one native FLC body.

The Army path fits the already identified I20 body-only interception problem.
It does not require a new Army-selection patch: current data and the existing
`Unit_select_army_member_for_combat` inlead expose authoritative identity, while
the eventual generic unit body boundary must suppress both native bodies and
retain the single native HUD. Exact I20 call-site auditing remains required
before any new patch request is made.

## L20 Proof Matrix

L20 must show empty, homogeneous-loaded, mixed-loaded, and displayed-member-
change Armies. It must cover all eight basic actions, eight facings, both Civ
III zooms, noon and midnight, at least two owner colors, and open-tile, stack,
city, horizontal-wrap, and viewport-clipping contexts. The active member cases
must include an arbitrary scenario-style unit key to prove there are no baked
member combinations.

The executable offline contract is
`tools/asset_compiler/army_render_strategy.json`; validate and sample it with:

```powershell
py Renderer\tools\asset_compiler\army_render_strategy.py
```
