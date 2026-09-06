# Bombardment, Bombing, And Explosion Handoff

Status: offline intake, event design, pixel-only suppression, and nuclear
outcome boundaries are prepared for M7.5. Runtime effect ownership and native
suppression are not enabled.

## Decision

Explosions are not part of a unit's converted body clip. The unit animation
owns pose, weapon motion, and recoil; an effect event owns muzzle flash,
projectile, target impact, smoke, debris, splash, light, and any explicitly
authorized aftermath. A pack binds those two halves with normalized release
markers. This keeps the system usable for arbitrary scenario units without a
unit-name branch.

Civ III remains the director. It decides whether an action happens, its source
and target, visibility, each bombard damage roll, what was damaged, whether a
unit dies, whether a nuclear delivery is intercepted, and when control returns.
The renderer turns those facts into pixels using stable child event IDs and the
absolute M5.3 presentation clock. Missing rendered frames are skipped rather
than queued.

The executable offline form of this decision is
`tools/asset_compiler/combat_effect_contract.json`. Its validator and trace
sampler are `combat_effect_contract.py` and
`test_combat_effect_contract.py`.

## Confirmed Civ III bombard lifecycle

The installed executable separates firing from target impact more cleanly than
the old unit FLCs suggest:

1. `Unit::bombard_tile` starts the bombard and calls
   `Unit::play_bombard_fire_animation`. Ground and sea attackers play their
   attack FLC; air attackers use `Unit::play_bombing_animation`.
2. Only after that visual routine returns does `Fighter::do_bombard_tile`
   resolve city, improvement, population, and unit outcomes.
3. Each presented outcome calls `Unit::play_bombard_damage_animation(x, y,
   hit)`. A hit chooses one of `AE_Hit`, `AE_Hit2`, `AE_Hit3`, or `AE_Hit5`.
   A miss chooses `AE_Miss` or `AE_WaterMiss` from the target terrain.
4. That routine loads an `AnimatedEffect` at the target, queues its one-shot
   animation, and preserves a 500 ms offline wait. Fire Rate can therefore
   produce multiple impact calls from one attacker action.

This gives M7.5 an exact impact boundary. The existing
`Units_Image_Data_load_animated_effect` inlead observes effect IDs 3 through 8
at the same instant that Civ III creates the native effect. It does not by
itself identify what object was damaged, so the completed event records
before/after authoritative state around `Fighter_do_bombard_tile` as outcome
metadata. The renderer never infers damage from an explosion variant.

The existing bridge is sufficient to prototype the event stream:

| Phase | Existing boundary | Captured fact |
| --- | --- | --- |
| Begin | `patch_Unit_bombard_tile` | source, wrapped target, action identity |
| Release | `patch_Unit_play_anim_for_bombard_tile` plus the M7.4 action cursor | authored release marker and source socket |
| Impact | `patch_Units_Image_Data_load_animated_effect` for IDs 3--8 | exact native time, hit/land-miss/water-miss, impact ordinal |
| Outcome | return from `patch_Fighter_do_bombard_tile` / outer bombard call | authoritative state delta, death or destroyed-object identity |
| Cleanup | native completion, interruption, reset, or profile lifetime | no orphaned projectile, particles, or light |

Precision strike already has source/target boundaries in
`patch_Unit_do_precision_strike`, the bombing replacement calls, and the cruise
missile helper. Ordinary unit combat uses the existing `Fighter_begin` and
`Fighter_fight` lifecycle plus authored attack markers. Defensive bombard uses
the same effect event shape with reversed combat roles.

## Presentation rules

- Release markers belong to pack data in normalized native action progress.
  They are not guessed from a rendered frame number. One action may declare
  several markers.
- Native impact calls determine impact count. A Fire Rate of two may produce
  two impact events even if the unit clip has only one decorative release; the
  two counts are deliberately not treated as gameplay equivalents.
- One gameplay impact may create several deterministic visual children—flash,
  fireball, smoke, dust, debris, and light—but those children never mean extra
  damage rolls.
- Land hit, land miss, and water miss are distinct recipes. Water impacts use a
  splash/ring/mist family and never leave a ground decal. A hit may still choose
  target-material debris from authoritative city/unit/tile context.
- The initial profiles are ballistic shell, dropped bomb, guided missile, and
  nuclear detonation. Unit or attack mapping data chooses a profile and release
  sockets; source code never tests for Artillery, Bomber, Cruise Missile, or
  another unit name.
- Night impacts add a brief bounded analytic light and emissive flash through
  the shared environment contract. It expires with the event and cannot keep a
  static map redrawing.
- Projectiles and particles are world anchored, depth tested where solid, and
  clipped like other dynamic map objects. Reduced zoom receives authored
  density and size limits instead of scaling a huge cloud over adjacent tiles.
- Fog-hidden and off-screen events do not reveal information or keep the frame
  scheduler hot. Persistent scorch, smoke, fire, or craters require a separate
  authoritative map state or explicit presentation event; an impact cannot
  silently mutate terrain.

## Atomic ownership and audio

Many Civ III unit attack FLCs bake muzzle flashes or projectiles into the body
pixels. Once M7.4 owns a unit body those pixels disappear with the native body,
and M7.5 supplies its matching effect. If a unit has no custom mapping, both its
body and effects remain native. Mixing a custom body with replayed native impact
pixels is forbidden.

The target `AnimatedEffect` is different: Civ III creates it independently and
its INI/FLC can carry audio timing. Audio remains Civ III-owned initially, so
M7.5 must allow the native animation/audio state to load and advance while
suppressing only its pixels. Replacing the FLC with a transparent unrelated FLC
is not the accepted production plan because it may discard native sound cues.

The pixel-only boundary is resolved without intercepting the shared sprite
blitter. `Animator::update` calls `FLC_Animation::tick` first, computes the
effect rectangle, and then tests byte `0x184` before the direct FLC blit. It
still unions the dirty rectangle when that byte is zero. Native construction,
frame advancement, INI sound cues, the 500 ms bombard wait, dirty accounting,
and destruction therefore remain intact when only that byte is cleared.

`ref/Civ3Conquests.h` names the four-byte tail containing this flag
`FLC_Animation::Last`; the draw-enable flag is only its low byte. M7.5 must use
`*(byte *)&anim->Last = 0`, not `anim->Last = 0`, because offsets `0x185` and
`0x186` are independent native flags.

For ordinary IDs 3--8, the existing
`patch_Units_Image_Data_load_animated_effect` calls the native loader first and
can then clear the byte only after the renderer accepts the matching custom
event. Unmapped or rejected events leave it set and remain wholly native. This
needs no new draw hook and keeps the shared unit-body blitter untouched.

SDI interception uses a standalone FLC loaded through
`Units_Image_Data_load_animation`, not the animated-effect helper. That
existing CSV symbol must be upgraded from `define` to `inlead` so the same
post-load rule can recognize `Art\Animations\SDI\SDI.ini` and suppress its
pixels only when the renderer owns the nuclear-interception effect. Its known
addresses are GOG `0x4062A0`, Steam `0x406810`, and PCGames `0x4062D0`.

## Nuclear effects

Nuclear weapons are a separate, fail-closed family rather than a very large
ordinary explosion. The exact native order is now confirmed:

1. `Unit::nuke_tile` confirms wars and samples `rand_int(4)`.
2. It scans affected civilizations in leader order and keeps the first one
   owning `ITSW_Decreases_Success_Of_Missile_Attacks`.
3. `animate_nuclear_strike` plays the delivery animation before the result
   branch. This is not a detonation signal.
4. If an interceptor exists and the sampled value is nonzero, Civ III enters
   `Unit::get_intercepted_as_nuke`; otherwise it enters
   `Unit::do_nuke_tile`. Thus an eligible interceptor stops three of the four
   sampled values.
5. The multiplayer sync callback calls those same two methods from the
   transmitted `intercepting_civ_id` and roll result. They are therefore the
   authoritative offline and network-replay boundaries.

The custom lifecycle is:

```text
delivery -> intercepted + cleanup
         -> detonated -> flash -> sphere/shockwave -> cloud/scorch -> cleanup
```

No mushroom cloud is spawned merely because an ICBM attack clip started or a
missile unit disappeared. `Unit::do_nuke_tile` is the detonation signal even if
the strike happens to remove no unit; nuclear victim-despawn hooks remain only
outcome detail. `Unit::get_intercepted_as_nuke` is the interception signal.
The successful local visual uses the missile unit animation path; the
interception visual loads the standalone SDI FLC described above.

M7.5 therefore requires two new `inlead` entries. These entry points occur
before native visual/despawn and damage/popup work in offline play, and the
same methods are entered by network replay:

| Symbol | C signature | GOG | Steam | PCGames |
| --- | --- | ---: | ---: | ---: |
| `Unit_do_nuke_tile` | `void (__fastcall *)(Unit *, int, int, int, int)` | `0x5B4070` | `0x5C29C0` | `0x5B3D80` |
| `Unit_get_intercepted_as_nuke` | `void (__fastcall *)(Unit *, int, int, int, int, int)` | `0x5B4A00` | `0x5C3350` | `0x5B4710` |

The two leading integers after `Unit *` are the synthetic fastcall `edx` slot
and `tile_x`; the remaining arguments are `tile_y`, followed by the affected-
civilization array for detonation or `intercepting_civ_id` plus that array for
interception. The GOG entries and control flow were verified directly against
the installed executable. Steam and PCGames entries follow the exact regional
relocation proved by adjacent existing nuke-damage call sites: the first victim
despawn is `+0x304` from `Unit_do_nuke_tile` in all three builds, and the
surrounding named functions return to the same `+0xE950` Steam / `-0x2F0`
PCGames mapping.

Until those entries are added, M7.5 must leave the entire nuclear presentation
native. It must not synthesize an outcome from delivery, despawn, damage, or
fallout state.

## Prepared upstream art

The installed Civ VI Base packages provide real source evidence for the needed
families:

- `VFX.blp`: cannon flashes/smoke, debris, missile dust, water/crater waves,
  atomic sphere/cloud/flame/scorch resources;
- `VFX_A.blp`: bomber bombs, jet bombs, missile trails, shell hits, impact dust,
  explosion/debris families;
- `VFX_B.blp`: artillery shell, artillery/rifle/musket muzzle families,
  explosion plasma and smoke sheets;
- `VFX_C.blp`: a named water-explosion family.

`combat_effect_texture_sets.json` selects 22 conservative standalone
`SHARED_DATA` texture resources across muzzle, projectile, explosion, smoke,
debris, water, and nuclear categories. Running
`combat_effect_texture_importer.py` converts all 22 to a source-independent
`c3x.combat_effect_texture_pack.v0`; the current local intake is 3,172,048
bytes. Derived licensed pixels remain ignored and are not distributed.

This is intentionally texture-only. The BLP VFX packages also contain named
particle meshes/scripts such as bomber bomb, missile, explosion flash,
shockwave, smoke ring, sparks, and nuclear systems, but their emitter graphs,
bytecode, sprite layouts, blend constants, and timing are not decoded. The Lab
may reproduce behavior from the generic contract using the converted textures,
or a later decoder can prove more of the source data. Neither path makes the C3X
runtime depend on Civ VI formats or assets.

The first generic authored implementation now exists in
`effect_graph_profiles.json` and `effect_graph_compiler.py`. It provides bounded
land- and water-impact graphs, validates texture/atlas/blend references, applies
normal/reduced density limits, and deterministically samples stable events from
absolute time. It remains offline and does not claim to decode Firaxis emitter
bytecode; muzzle/projectile/nuclear expansion and final calibration remain M7.5
work.

## M7.5 acceptance additions

The two remaining native-boundary investigations are complete. The eventual
gate must replay at least ballistic land hit, land miss, water
miss, air bomb, guided missile, multiple Fire Rate impacts, interrupted flight,
skipped frames, fog/off-screen suppression, night flash, device reset, native
audio retention, and both nuclear branches. Traces must prove stable IDs,
ordered timestamps, exactly one visual owner, bounded cleanup, and zero
gameplay decisions made by the renderer.
