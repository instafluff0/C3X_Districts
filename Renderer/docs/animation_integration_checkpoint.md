# Animation delivery and calibration

The current C3X checkout and category standards are authoritative. This document
is an operating/source guide, not an installation receipt or approval ledger.

Use the [visual workbench](../lab/README.md) for current commands. Ordinary Lab
work does not authorize staging, running INSTALL.bat or launching Civ III.
Use the normal C3X installation process and an actual game test at a strategic
delivery checkpoint. Current-code Integration itself neither installs nor claims
that game test.
The configuration controls are `enable_custom_rendering` and
`enable_custom_rendered_units`; the latter defaults off and avoids unit loading
when disabled. No special launcher or runtime environment switch is required.

## Authoritative animation boundaries

Civ III owns unit actions, cursors, movement, combat outcomes, visibility and
native canvas placement. The [native unit contract](i20_native_unit_animation_handoff.md)
documents the actual body hooks, palette capture, Army handling, background DC,
dirty rectangle expansion and fallback. The three supplied GOG inleads require
no further CSV action here; other-build addresses remain unverified.

The [current unit guide](../native/environment_refresh/UNIT_FIDELITY.md) describes
the 78-entry pack, source normals, materials and preparation. Nine-family native
tests are only a verification subset, not the current roster or complete coverage.
Compound/original kits have separate proof requirements.

Resources use elapsed-time presentation, independent of gameplay state. Animated
vertices do not enter static terrain geometry/pixel caches; those retain anchors
and ownership. Keep the bounded posed-vertex pool, exact MSAA backdrop reuse and
existing redraw-demand scheduler. Unit jobs likewise preserve the terrain
worker's completed publication and queued preparation. Neither system may create
a second game loop, camera or presenter.

## Resource source calibration

The generic animation decoder in `native/animation_runtime.h` validates byte
lengths, dimensions, finite affine matrices, indices and normalized weights
before replacing decoded assets. The format carries rest vertices, indices and
frame-major/bone-major row-vector skin matrices; normals use inverse transpose.
Runtime records contain no source-engine paths or types.

Current offline preparation preserves all authored resource frames. Clutter clip
translations use 1/12 export units while their skeletons use 1/100; the explicit
clip-unit recipe supplies that conversion. Remove only constant source-scene root
placement, retaining animated deltas, rotation and scale.

`tools/asset_compiler/school_orientation.py` identifies disconnected marine
bodies after welding source position seams. It aligns each posed head-minus-tail
vector to +X (Civ III southeast), around that body's centroid. Formation positions
and deformation survive; duplicated palettes give cross-rig influences the same
body transform. This is offline calibration, not a per-frame source-specific DLL
branch. Current source-reference checks cover 26 subjects, 104 poses and 7,215
marine body/frame headings. Their evidence does not establish live-game speed.

Category preparation uses the current static and animated sources and rejects
edited generated output. Source conversion recipes and necessary local art remain
intact; the old four-family import command sequence is not a rebuild recipe for
the current 78-entry unit pack.

## Unit presentation safeguards

- Preserve the fixed idle-stance fit across actions and the consistent removal
  of planar root travel; native coordinates own movement. Joint/vertical motion
  and explicit part/tool mappings remain. The worker/source mapping findings are
  in `worker_builder_animation_mapping.md`.
- Pose-only fortify sources use the existing baked local-joint transition,
  shortest-path normalized quaternions and smooth easing. Native cursor progress,
  interruption and held endpoints still determine presentation.
- The unit cache is bounded to 8 MiB/128 entries. Pose, cursor, direction, zoom,
  effective palette and lighting select reuse; screen translation is not a new
  pose. Preserve the actual number reader's scientific-notation regression.
- The native canvas's magenta key is not a blending background. Resolve partial
  body/shadow coverage against the forwarded underlay, honor clipping and reject
  invalid inputs before destination mutation. The bridge owns DC cleanup.
- Successful body replacement expands the existing native display-owner bounds;
  fallback leaves them unchanged. Do not resize each pose or silently clip
  visible anatomy to manufacture a pass.
- Keep the bounded input guard for ordinary UI clicks and the existing release
  interval. Bypass its click-decision delay immediately when Civ III's own
  selected-unit map/pathfinder-hold byte is set. Do not consume input messages
  or change native click/pathfinder/gameplay handlers.

Unit shadows currently use the bounded directional whole-kit field and a flat
unit-local receiving plane. Cross-unit and terrain/object receiving are not
established by that path; source cutout silhouettes use mesh geometry. Resource
animation's dynamic-shadow limitation is separate from static map shadows. The
guarded scene-linear backdrop path fixes the earlier black rectangles; keep its
compositing regression coverage.

## Verification and game checkpoint

Use `python3 Renderer/renderer.py test animation` for current category checks and
`integration animation` through the same interface for complete current-code checks.
Preserve decoder rejection, source-pose calibration, native cursor/interruptions,
held endpoints, both zooms, RGB555/RGB565 clipping, keyed backgrounds, input guard,
config-off and unchanged retained terrain. Tests that passed on a previous
candidate are not receipts for a new DLL. Keep current limitations in the owning
category or implementation guide, not a separate status ledger.

At an actual delivery checkpoint, batch the game checks: idle/scrolling resources,
minimap jumps, supported units moving in several directions at both zooms,
selection/status/HUD and fog boundaries, interrupted movement/fortify/combat,
death/victory/return to idle, and the config-off control. Record a material
observation in the relevant current notes without inferring a game pass from
headless images. Normal diagnostic output uses
the existing debugger channel; explicit offline runs may keep their requested
bounded logs. Do not restore historical game-file logging or repeatedly request
manual screenshots during ordinary Lab iteration.
