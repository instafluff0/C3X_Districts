# Animation integration: first GOG gameplay checkpoint

The animation DLL is compiled, verified offscreen and staged in `Renderer/bin`.
The human supplied the three GOG unit inleads, including the corrected reduced
signature, and the approved injected compilation passes. The installed mod path
is the shared checkout. **The agent did not launch Civ III or run INSTALL.bat.**
Use normal `INSTALL.bat`, then launch the game normally with:

```ini
enable_custom_rendering = true
enable_custom_rendered_units = true
```

No runtime environment variables or special launcher are required. Resources
animate by default in the pickup profile. The unit setting defaults off and
skips loading unit assets when disabled. This is a first user-run prototype,
not a claim that gameplay movement, fog and HUD behavior have been observed.
The exact staged DLL/source/asset identities are recorded in
`verification/animation/candidate-checkpoint.json`.

## Coverage

Ten animated resource families: bananas, cattle, furs, game, horses, ivory,
rubber, wheat, fish and whales. Source resources without an animation binding
continue to use their static custom bodies.

Nine custom unit families: Archer, Swordsman, Infantry, Fighter, Galley,
Warrior, Scout, Settler and Worker (also matched by PRTO_Builder).
Combat families have basic movement/combat clips. Settler and Worker have idle,
move, fidget, fortify/stop and capture; their combat, founding/build and Worker
specialty jobs retain native rendering in this checkpoint. Other families,
Armies and failed replacements also retain their complete native bodies.
The new bodies use a separate `UnitEarlyLab` offline intake, leaving frozen
Lab packs intact. One human member is selected explicitly for Scout/Settler;
companion animals and extra Builder formation members are outside this test.
Non-work Builder poses hide the mutually exclusive tools.

Settler's backpack-carrier clips leave its leader's walking staff behind.
The prototype instead binds compatible staff-humanoid Scout clips for its
idle/movement/fidget/stop and the source Settler leader capture clip. This is
an explicit pack mapping, not a source-specific runtime branch. Warrior/Scout
victory temporarily reuses fidget. These are initial presentation mappings,
not a claim of exact source-game animation-state reproduction.

The user reports resources look good so far, but a little dark. That is partial
in-game resource feedback; brightness adjustment remains separate. Edge speckling remains deferred. This checkpoint does not advance unrelated
Lab gates or M9–M11.

## Resource rendering and calibration

`native/animation_runtime.h` validates the generic binary skin-palette format
before allocation: exact byte length, dimensions, finite affine matrices, valid
indices and normalized weights. It preserves decoded assets on rejection.
The payload contains a C3XANM1 header, 64-byte rest vertices, u32 indices and
frame-major/bone-major row-vector skin matrices. Runtime data contains no
source-engine paths or types. Normals use the inverse transpose.

The offline compiler binds 26 subjects into 8,901,120 bytes of pose payloads
plus deduplicated textures. All authored frames are retained. Clutter clip
translations were exported at 1/12 while skeletons use 1/100; exact source clip
metadata supplies that conversion. Only constant source-scene root placement
is removed; animated deltas, rotations and scale remain.

`tools/asset_compiler/school_orientation.py` identifies 12 disconnected fish
and three whales, welding source position seams. At every authored frame it
aligns each head-minus-tail vector to +X (Civ III southeast), rotating about the
posed body centroid. Formation positions and complete deformation survive.
Tiny cross-rig influences receive the same body transform through duplicated
palettes rather than tearing toward a neighbor. This is entirely offline;
the DLL performs no source-specific or per-frame heading correction.

The actual C++ evaluator matches 104 resource poses against independent source
skinning plus calibration within 1.269e-7 tile units. All 7,215 marine body/frame
headings have positive X and zero Y in the compiled palettes. Every subject
moves across its clip. Synthetic checks cover cross-rig weights, seams, moving
centroids, ambiguous ownership, malformed data, timing and loop endpoints.
Evidence: `verification/animation/resource-payloads-portable_cpp.json`.

Animated resource pixels never enter static terrain mesh/pixel caches. Those
caches hold only authoritative anchors and ownership. A separate 32 MiB posed
vertex pool and 24 MiB exact MSAA4 color/depth backdrop cache support the dynamic
pass. Repeated callbacks in one 15 Hz absolute-time quantum reuse the published
bitmap. New phases preserve terrain preparation and use the existing native
redraw-demand scheduler. Animated shadows are outside this initial prototype;
static source shadows remain. The timing evidence is offscreen, not a claim
of vanilla-speed gameplay.

## Unit rendering and native bridge

`native/unit_body_renderer.h` renders on the existing D3D worker and copies a
completed premultiplied body bitmap to Civ III's Animator canvas. It owns no
window, gameplay timer or terrain layer. A fixed idle-stance fit is reused for
all actions. Planar root travel is removed consistently across each kit because
Civ III's body coordinates own movement; joint and vertical motion remain.
The generic full-clip exporter covers nine families and 70 actions in
27,934,624 payload bytes. The actual C++ evaluator matches 852 source poses,
including 270 socket poses, within 7.057e-8 tile units. See `verification/animation/unit-payloads-portable_cpp.json`.

The 8 MiB / 128-entry sprite cache includes action/cursor, direction, dimensions,
zoom, effective palette color and lighting. Unit identity and pixel position
are excluded, allowing exact pose reuse at new native anchors. Unit jobs preserve
the terrain worker's completed publication and queued preparation. No unit
animation state enters the terrain cache keys.

The material-number reader now preserves scientific notation and rejects
truncated tokens. The tiny negative Scout ground offset previously parsed as
a whole-unit negative displacement, hiding its entire body; an extracted
actual-reader regression covers that exact value.

The ground plane hides buried anatomy and spare equipment stored underground
by death clips. The bounds check covers the visible polygon, including ground
intersections, so rendering stays inside Civ III's native dirty rectangle.
There is no per-pose resizing or silent clipping of visible oversized bodies.

`injected_code.c` contains only capture/configuration/forwarding wrappers.
`Unit_tick_anim` establishes scoped unit/canvas identities and still calls the
complete original routine. Only its exact current-frame Sprite body is replaced
following a successful DLL draw. Native visibility, movement, timing, underlay
and HUD code continues. Effective palette color comes from the palette chosen
by the actual native draw, including hidden nationality. Native cursor and
current action are authoritative; a queued action cannot select the pose early.

The three GOG inleads are `Unit_tick_anim` at 0x005CBF50,
`Sprite_draw_unit_body_normal` at 0x005F88B0 and
`Sprite_draw_unit_body_reduced` at 0x005F8940. Reduced zoom has nine stack
arguments, confirmed by RET 0x24 and caller bytes. The CSV and unmodified
executable audit passes. Steam/PCGames.de counterparts remain unverified.
See `handoffs/animation_unit_hooks_gog.md` and the patch dependency ledger.
No further CSV action is required for this GOG checkpoint.

## Logging and verification

Normal runtime logging uses OutputDebugStringA only, with no automatic log
files. Injected code uses the existing `(*p_OutputDebugStringA)` import pointer.
`unit-config`, `unit-bind` and `unit-body` records report activation, mapping,
unit/key, current and queued action, native cursor/count, direction, anchor,
zoom, effective color, result/rejection reason, cache hit/bytes and elapsed time.
Resource records report binding, demand, phase/timing and cache costs. Explicit
headless verification commands may write their requested diagnostic logs.

`python3 Renderer/native/verify_units.py` builds and verifies the real DLL.
The day/night matrix covers 576 movement draws, both zooms and eight directions,
plus 912 action draws spanning idle, move, attack variants, death, fortify,
fidget, victory and civilian capture. Every movement row must contain visible
pixels and change between phases. Held endpoints, skipped cursors, immediate interruptions,
return to identical idle pixels, config-off preservation, unsupported actions,
RGB555/RGB565 clipped bodies and cached anchor translation all pass.
Resource phases build/upload zero terrain geometry; scrolling/removal and
terrain reconstruction after unit drawing meet unchanged parity thresholds.
See `verification/animation/units.json`.

The extracted actual injected bridge also passes portable and Windows x86
checks for argument preservation, both zooms, disabled/invisible/Army controls,
scoped restoration, effective palette capture and underlay/HUD ordering.
Approved injected compilation with active inleads, portable native smoke,
85 focused production contracts and the shared install-path check passed for
the original bridge checkpoint. This extension passes 45 focused tests,
including the actual number reader and all nine source-payload families;
injected sources and the three CSV entries are unchanged.

The required full workflow was run but is **not green**. Its source stage passes
374 tests and stops at the L19A frozen tile-object pack hash: the current Lab
pack differs from the approved fixture. The production DLL does not load that
pack. Separately, the previously documented legacy frozen-profile incremental
boundary check still fails (1,671 changed pixels; 47,588 channel error). Its
thresholds were not weakened. These unrelated failures remain open; this work
does not promote those gates. Reports: `verification/animation/full.json`,
`final-contracts.json` and `final-native.json`.

## User-run checkpoint

After normal installation, check the animated resources while idle and after
scrolling/minimap jumps. Check a supported unit moving in multiple directions
at both zooms, selection changes, native labels/status/rings, and fog boundaries.
Interrupt movement, fortify or fight, and check victory/death and return to idle.
Share the usual OutputDebugString output and the observed behavior. The optional
config-off control should restore native unit bodies. This is the single batched
manual checkpoint; the agent stops here and does not launch the game.


## Rebuilding the four early unit bindings

The strategy is `tools/asset_compiler/unit_early_strategy.json`. Its explicit
member selection and action scope keep the frozen five-family Lab intake intact.
Use the existing Windows dispatcher for `CONVERT_UNIT_EARLY_ANIMATIONS.bat`;
that batch uses the same offline converter and scale as the established family
pipeline. Then refresh the intake manifest and validate/bake the clips:

```sh
python3 Renderer/tools/asset_compiler/unit_family_asset_importer.py --strategy Renderer/tools/asset_compiler/unit_early_strategy.json --pack Renderer/packs/UnitEarlyLab --report Renderer/preview/out/units/early_build.json
# Windows VM: Renderer/tools/asset_compiler/CONVERT_UNIT_EARLY_ANIMATIONS.bat
# Repeat the importer above after conversion to refresh clip metadata.
python3 Renderer/tools/asset_compiler/unit_family_action_validator.py --pack Renderer/packs/UnitEarlyLab --report Renderer/preview/out/units/early_actions.json
python3 Renderer/tools/asset_compiler/unit_family_pose_cache_builder.py --pack Renderer/packs/UnitEarlyLab --report Renderer/preview/out/units/early_pose_caches.json
python3 Renderer/tools/asset_compiler/build_unit_animation_runtime.py
python3 Renderer/native/verify_units.py
```

The combined compiler defaults to `UnitFamilyLab` plus `UnitEarlyLab`. Source
art, converted clips and generated previews remain ignored local assets.
