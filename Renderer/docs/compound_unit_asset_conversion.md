# Compound Unit Asset Conversion

Status: generic parent/child/socket compilation, paired raw-clip validation,
model-aware pose caching, and first basic-action completion are complete
offline; L20 visual rendering/approval and runtime ownership remain pending.

`compound_unit_asset_importer.py` now turns any acyclic composition tree into a
source-independent `c3x.unit_composition.v0` recipe. A recipe names a root node,
independent animated nodes, parent/child joints, resolved skeleton bones,
child-local transforms, node-local clips, and one shared semantic action clock.
Nothing in the runtime-facing pack dispatches on Horseman, Catapult, Tank, or a
source-game identifier.

The local proof pack contains:

- four compositions: Horseman, Classical Great General, Catapult, and Tank;
- eight independently animated nodes and four real parent sockets;
- thirty normalized model components and fifty normalized textures;
- fifty-two converted unique node/action clips serving sixty-two logical
  bindings across thirty-one compound actions;
- fifty-two deterministic model-aware `.c3pose` caches containing complete
  ordered skeleton world matrices;
- CPU composition samples at normalized phases `0.0`, `0.37`, `0.61`, and
  `1.0`, with zero node-origin separation from every animated socket.

| Composition | Parent | Child | Resolved parent bone | Paired actions |
| --- | --- | --- | --- | --- |
| Horseman | horse | rider | `RiderAttach` | all eight basic actions |
| Classical Great General | horse | rider | `RiderAttach` | all eight; conservative idle/defend aliases for non-combat gestures |
| Catapult | vehicle | operator | `CatapultOperator` | seven basic actions; death remains explicitly unresolved |
| Tank | vehicle | gunner | `GunnerAttach` | all eight basic actions |

The Catapult's second two-defender member recipe remains optional composition
data. The first Civ III-like presentation should use vehicle plus one operator
instead of automatically crowding the tile. The same tree schema can add those
defenders, a chariot team, an arbitrary scenario passenger, or another nested
attachment without new runtime code.

## Transform and animation contract

Every node is evaluated in its own normalized coordinate space. The child node
frame is multiplied by its declared local transform and the current animated
parent-socket matrix. Child root motion remains local to that attached frame;
it is not cancelled every frame, which preserves authored attack, reaction,
and death motion. All nodes receive one authoritative Civ III action phase, but
each samples its own clip at that normalized phase. This handles exact-duration
pairs as well as deliberate pose or breathing clips with different durations.

The compiler fixed a shared bind-pose audit defect exposed by horse rigs:
Granny row-vector locals require `scale/shear * transposed rotation`, matching
`normalized_skin.py`. Uniform-scale assets concealed the previous reversed
order. Previously accepted compound and unit tests continue to pass.

Some shipped mounted attack/death curves contain non-finite, degenerate, or
implausibly large compressed samples. The Windows converter first retries an
adjacent sub-frame sample. If any sample in a channel remains unreadable, it
records that recovery and marks the whole channel absent so normalized skin
evaluation uses the corresponding skeleton-rest position, orientation, or
scale/shear while preserving every other decoded channel. The composition
validator also rejects any world transform outside a conservative normalized
envelope. These are conversion recoveries, not evidence that the L20 animation
looks are approved; L20 must inspect the resulting poses and may choose a
cleaner compatible clip.

## Atomic unit ownership

The whole tree remains one semantic Civ III unit:

- one authoritative anchor and eight-direction facing;
- one action/event identity and clock;
- one viewer-conditioned `display_color_table_id`, applied independently to
  each component's authored owner-color mask;
- one native selection/health/status/stack HUD instance;
- one complete-unit failure boundary.

If a required node, socket, clip, or component is unavailable, the compound
body fails atomically. The renderer must never show a rider without a mount,
duplicate the native HUD, or restore only part of a native unit body.

## Reproduction

The source-specific intake is declared in
`tools/asset_compiler/compound_unit_source_sets.json`. Static conversion is:

```text
python3 Renderer/tools/asset_compiler/compound_unit_asset_importer.py
```

On Windows, `CONVERT_COMPOUND_UNIT_ANIMATIONS.ps1` converts the declared node
clips through the existing offline CivNexus6 bridge. Rerun the importer with
`--require-animations`, bake deterministic model-aware pose caches, then
validate the generic tree with:

```text
python3 Renderer/tools/asset_compiler/compound_unit_composition_validator.py \
  --pack Renderer/packs/CompoundUnitLab \
  --report Renderer/preview/out/units/compound_unit_validation.json
```

`compound_unit_pose_cache_builder.py` performs the intervening bake. Direct
action aliases reuse the same cache rather than duplicating bytes, while each
action retains its own loop/completion semantics on Civ III's authoritative
clock.

The local licensed-source pack and converted payloads remain ignored and are
not redistributable. The compiler, source profile, validators, and synthetic
tests are tracked.

## Remaining L20 work

- Render and measure all eight facings at normal and reduced Civ III zoom using
  the checked-in calibration matrix.
- Select a truthful Catapult death only if a visually compatible source is
  proven; do not alias a reaction or idle merely to make the matrix green.
- Visually validate the frozen single-body default and optional restrained
  humanoid triad. Horseman, siege, armor, sea, and air remain single.
- Integrate the recipe through the Animator-owned dynamic unit plane only after
  L20 approval and the body-only native suppression boundary are proven.
