# Animation asset conversion proof

Status: the offline skeletal-clip conversion path, model-aware pose baking, and generic CPU skeleton binding are proven with real Civ VI cooked assets. Native/GPU rendering is intentionally outside this conversion proof.

## Boundary

The source-side importer may use Firaxis/Granny tooling offline. A `.c3anim` file does not contain Granny curves, pointers, type names, or source paths, and the reader does not load Firaxis libraries. This keeps the renderer-side contract independent of Civ VI's storage format.

The format preserves animation track groups. This is required: `ANIMATION_AT_Crew_AttackA` contains two different actors whose groups both use the name `Root` and whose tracks overlap by bone name. Flattening or picking the largest group would lose valid crew motion.

## Conversion

On the Windows conversion VM (or another 64-bit Windows environment with the checked-in CivNexus6 binaries), run:

```bat
Renderer\tools\asset_compiler\CONVERT_CIV6_ANIMATION.bat input-animation output.c3anim [translation-scale]
```

The input may be a standalone `CIVBIG` animation or a raw Granny payload extracted from a BLP animation table. The wrapper builds `export_civ6_animation.cs`, removes the wrapper when present, loads the animation with the Firaxis Granny implementation checked into `Renderer/third_party/CivNexus6`, samples authored curves at their declared timestep, and writes `.c3anim`. The temporary source payload is deleted. The converter deliberately retains the loader object until all native curve samples are complete; without that lifetime guard, long embedded clips can outlive their Granny memory arena.

`translation-scale` defaults to `1`. It is applied only to position curves, never quaternions or scale/shear. Use it when a source family has a proven spatial-unit conversion; embedded tile-base resources currently use `0.01` source units to one tile.

The importer fails closed for malformed wrappers, zero/invalid timing, multiple animations per source resource, missing groups/tracks, duplicate track names within one group, unexpected curve dimensions, non-finite samples, and currently unknown non-zero track flags.

## `.c3pose` model-aware cache

Some Granny clips cannot be composed correctly from their raw curves without model binding. `normalized_skeleton_to_cn6.py` creates a minimal offline sampling companion from the generic skeleton, `IMPORT_CN6_MODEL.bat` turns it into a temporary Granny model, and `CONVERT_CIV6_MODEL_POSE.bat` calls the same `SampleBone(model, name, time)` path used by CivNexus. The resulting `.c3pose` file is source-independent.

Version 1 has a 36-byte little-endian header (`C3XPOSE\0`, version, duration, sample rate, frame count, bone count, bone-record offset, sample offset), eight-byte UTF-8 name records, and frame-major row-vector 4x4 world matrices. Translation components are converted to tile units offline. `normalized_pose_cache.py` validates layout, timing, unique ordered names, finite affine matrices, exact file consumption, and exact skeleton-name binding before exposing frame sampling.

The model-aware caches are deliberately separate from `.c3anim`: the latter preserves authored local curves for ordinary clips and diagnostics, while the cache records the result of source-model semantics when those curves are insufficient.

## `.c3anim` version 1

All integers and IEEE-754 floats are little-endian. Names are un-terminated UTF-8 slices in a shared string table. Offsets are absolute file offsets.

The fixed 56-byte header contains:

| Field | Type |
| --- | --- |
| magic (`C3XANIM\0`) | 8 bytes |
| version, flags | `u32`, `u32` |
| duration seconds, sample rate | `f32`, `f32` |
| frame count, group count, track count | three `u32` values |
| string-table bytes | `u32` |
| group-table, track-table, data offsets | three `u32` values |
| data bytes | `u32` |

Each 16-byte group record stores its name range and a contiguous `(first track, track count)` range. Each 32-byte track record stores its name range, reserved flags, packed channel modes, three channel-data offsets, and a reserved word.

Every local transform has three channels:

- position: 3 floats
- orientation: 4-float `x, y, z, w` quaternion
- scale/shear: row-major 3x3 matrix (9 floats)

Each channel uses one of three modes: identity (no payload), constant (one value), or uniformly sampled (`frame count` values). That preserves full scale/shear without assuming all source transforms are simple TRS, while avoiding repeated storage for identity and constant channels. Frame zero and the exact clip endpoint are both stored.

Looping is not baked into the clip. The caller explicitly selects clamp or loop when sampling. The generic reader linearly interpolates position and scale/shear and uses normalized shortest-arc quaternion interpolation.

## Generic consumer

`normalized_animation.py` is both the strict reference reader and a tiny deterministic consumer:

```sh
python3 Renderer/tools/asset_compiler/normalized_animation.py output.c3anim \
  --time 1.25 --group 0 --track Head
```

It validates canonical layout, contiguous group/track ranges, channel packing, UTF-8 names, finite values, reserved fields, and complete data consumption before exposing the clip.

## Real-asset results

Four Base-game clips from both standalone and embedded sources were converted and then read/sampled on macOS without Firaxis libraries:

| Source | Source bytes | Result | `.c3anim` bytes |
| --- | ---: | --- | ---: |
| `ANIMATION_Warrior_IdleB` | 28,160 | 2.666667 s, 81 frames, 1 group, 49 tracks | 21,196 |
| `ANIMATION_AT_Crew_AttackA` | 51,200 | 4.0 s, 121 frames, 2 groups, 88 tracks | 111,084 |
| embedded `RES_Fish_ANIM` | 188,452 | 16.666668 s, 501 frames, 1 group, 28 tracks | 338,180 |
| embedded `Whale_Idle` | 22,164 | 13.333334 s, 401 frames, 1 group, 17 tracks | 154,932 |

The attack clip exercised identity, constant, and sampled position/orientation channels plus duplicate group and bone names across actors. All 137 tracks across both proof clips used identity scale/shear, but version 1 retains a sampled 3x3 scale/shear mode for assets that need it.

The resource clips were extracted structurally from `landmarks/tilebases.blp`, converted with a `0.01` translation scale, then loaded and sampled on macOS with the generic reader. Three independent conversions of `ANIMATION_Warrior_IdleB` produced the same SHA-256:

`bbc33a0cd909469b49248b1a091cf10152905cf28c519dcd71a00f2c7113622f`

The non-Warrior unit-family batch extends that format proof to 37 unique clips
and 44 logical bindings across Archer, Swordsman, Infantry, Fighter, and Galley.
All five provide the basic idle, fidget, move, fortify, attack, defend, victory,
and death contract, while Fighter also provides takeoff, landing, and left/right
turns. Safe aliases reuse an existing converted file with an independent logical
loop/clamp policy. The checked Windows batch converts all 37 at `0.01`
translation scale; the source-independent validator samples every skinned
component on macOS. A subsequent generic bake produces 93 deduplicated
component/model/clip `.c3pose` files serving 100 logical component/action
bindings, including Galley's multi-mesh body. L20 therefore consumes pose
caches rather than raw curves, while still owning their visual approval.

Nine logical bindings resolve to authored two-frame poses rather than motion
curves. The source-independent action contract preserves that evidence and
specifies native-progress crossfades for pose-form fidget/fortify instead of
pretending two samples are a looping animation. ATTACK1/2/3 share one logical
attack by default; defend is an event-derived target reaction, not an invented
Civ III FLC slot.

After canonicalizing sub-`1e-12` sampler noise to exact zero, two independent fish conversions also matched byte for byte:

`4dea27962ec58bf669ce1ad1f404fd0314f2653782da0751719b7f2b80e8aa33`

Two independent model-aware Fish pose bakes also matched byte for byte:

`da42fcc1c5ea18abd186860b49fe37d5497e70af2b98043d73c6ef71b571e295`

The model-aware Whale cache hash is `3c82dea9d7ab6dfc5aed941ed82b2db65503027e75dfab9488f872c34994ee5b`. Its five-frame rendered contact sheet has the same `d2dce831...e2caa1` hash as direct curve sampling, while the corrected Fish contact sheet repeats at `a1cd46e9...e831db`.

Generated proof clips and executables live under ignored `Renderer/preview/out/` and `Renderer/packs/` paths; no Firaxis assets are committed.

## Current binding boundary

This proves clip extraction, storage, validation, direct curve sampling, model-aware pose baking, normalized fish/whale skins and skeletons, exact rest-pose reconstruction, group-to-skeleton name binding, and a dependency-free CPU skinning consumer. Direct and model-aware Whale rendering produce the same deterministic contact-sheet hash. Fish direct curves are invalid when composed independently, but its model-aware cache produces a stable moving school and is the manifest-selected path. The native renderer still needs matching GPU buffers/shaders. Material animations such as ocean UV flow remain a separate animation type and should not be forced into either skeletal format.
