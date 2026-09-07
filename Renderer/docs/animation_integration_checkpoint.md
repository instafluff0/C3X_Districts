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

Five custom unit families: Archer, Swordsman, Infantry, Fighter and Galley.
They use full clips, skinned parts and animated rigid socket attachments.
Other families, Armies, unsupported worker actions and failed replacements
retain their complete native bodies. Warrior clips are exported but its older
pack still lacks a resolved owner-material contract, so Warrior stays native
in this first prototype. Broader family coverage is subsequent work.

Edge speckling remains deferred. This checkpoint does not advance unrelated
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
The generic full-clip exporter covers six families and 48 actions. The actual
C++ evaluator matches 576 source poses, including 192 socket poses, within
2.256e-7 tile units. See `verification/animation/unit-payloads-portable_cpp.json`.

The 8 MiB / 128-entry sprite cache includes action/cursor, direction, dimensions,
zoom, effective palette color and lighting. Unit identity and pixel position
are excluded, allowing exact pose reuse at new native anchors. Unit jobs preserve
the terrain worker's completed publication and queued preparation. No unit
animation state enters the terrain cache keys.

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
The day/night matrix covers 320 movement draws, both zooms and eight directions,
plus 560 action draws spanning idle, move, attack variants, death, fortify,
fidget and victory. Held endpoints, skipped cursors, immediate interruptions,
return to identical idle pixels, config-off preservation, unsupported actions,
RGB555/RGB565 clipped bodies and cached anchor translation all pass.
Resource phases build/upload zero terrain geometry; scrolling/removal and
terrain reconstruction after unit drawing meet unchanged parity thresholds.
See `verification/animation/units.json`.

The extracted actual injected bridge also passes portable and Windows x86
checks for argument preservation, both zooms, disabled/invisible/Army controls,
scoped restoration, effective palette capture and underlay/HUD ordering.
Approved injected compilation with active inleads, portable native smoke,
85 focused production contracts and the shared install-path check pass.

The required full workflow was run but is **not green**. Its source stage passes
373 tests and stops at the L19A frozen tile-object pack hash: the current Lab
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
