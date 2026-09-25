# Renderer sandbox reset

## Objective and starting point

Build a fresh x64 D3D11 visual client with current-production graphical quality,
using suitable 0 A.D. rendering techniques, while Civ III and the renderer make
independent progress. Civ III remains authoritative for gameplay and visibility.
Its present capture, camera, composition and presentation boundaries may change.

This brief replaces the previous assignment. The previous implementation and its
generated outputs have been removed from the active sandbox. Start fresh under
`Renderer/sandbox/`, using production visual code and existing production build/VM
tooling. Do not restore or import the discarded experimental renderer, adapters,
exporters or protocol. A historical archive is retained outside this directory
for recovery only; it is not an input to this assignment.

Production's working visual code is the starting material. A fresh architecture
does not require new terrain, coast, river, material or animation algorithms.
Read project instructions, `Renderer/README.md`, `Renderer/lab/README.md`, relevant
catalog entries and the pertinent `docs/visual_fidelity_playbook.md` findings.
Avoid an exhaustive historical-documentation tour. This README is the only plan.

## Current priority: implement the replacement runtime

The existing matched reference views have established the visual reuse path.
Preserve them and move on. The user explicitly prefers a substantial, coherent
implementation followed by validation over repeated pixel comparisons between
small changes. Stop full-frame pixel scans, exact-return-image checks and repeated
reference captures. Pixel identity is not an objective of this architecture experiment.

Write the new runtime through the real scene, camera, effects, animation and async
host path before the next detailed validation pass. During implementation, compile
and use short smoke checks for crashes, missing assets or a blank/broken frame;
debugging a concrete blocker is fine. Do not make each subsystem wait for an image
comparison, performance sweep or additional documentation. Then validate the
coherent result with a few inspected views, one motion sequence and focused timing.
Preserve the quality target and report remaining defects; do not hide omissions.

## Established visual inputs

Use the existing parser and unchanged `packs/RendererSourceStudies/maps/test.biq`.
The map has 5,000 tiles on a 100x100 staggered lattice and no placed units/cities.
Begin with the existing beach/river/forest view around (17,49): 2240x1260,
128x64 tiles, noon, matched against current production.

Trace that view through production asset loading, scene preparation and drawing.
Copy/reuse the cohesive visual implementation AND its setup, not isolated shaders
surrounded by invented replacements. A larger working visual module is preferable
to a small adapter that silently changes the rules. Preserve topology, loaded relief
fields, river connectivity, coast samples, material channels/formats, normals/UVs,
lighting, transforms and draw state. Missing required data is an error.

Start at these existing boundaries, relative to `Renderer/`:
- `native/biq_preview.cpp`: BIQ placement, coordinates and view construction.
- `native/c3x_renderer.cpp`: asset/relief loading, `make_ground_job`, visual bindings.
- `native/source_fidelity/ground_preparation.h`: `compile_selected_ground` assembles
  the actual preparation callbacks; prefer it over recreating them individually.
- `native/source_fidelity/terrain_compiler.h`, object preparation, and existing
  terrain/object/hydrology shaders: working visual algorithms and their inputs.

If direct reuse pulls in old runtime machinery, use a sandbox-only offline exporter
that invokes production loading/preparation to emit complete geometry, materials,
placements and effect inputs. Do not create a general extraction framework. The
new renderer draws real geometry and animates effects; baked camera images or
baked dynamic effects do not qualify. Production itself may generate references
or preparation data, but must not secretly render the benchmarked sandbox frames.

Keep production scheduling, RPC, image leases and raster caches outside the new
frame loop. Reuse visuals aggressively while giving their execution a fresh owner.
Preserve sculpted coasts, transitions, relief, vegetation, actual scene shadows and
reflections, and normal water motion. New aesthetic improvements are later work.

Use the already-established views as the final visual comparison targets; do not
repeat the reference milestone. If reuse stalls, identify the specific dependency
and take a more complete data/preparation boundary. Avoid inventing missing terrain
rules, but freely replace the runtime mechanisms that execute those visuals.

## Then: implement the useful 0 A.D. approach in D3D11

Review <https://gitea.wildfiregames.com/0ad/0ad>, focusing on `PatchRData.cpp`,
`TerrainRenderer.cpp`, `ModelRenderer.cpp`, `SceneRenderer.cpp` and `ShadowMap.cpp`.
The official GitHub mirror is an archived 2024 fallback; identify the revision used.
Record the adopted routines/ideas in a short paragraph, not a research document.
Consult the actual source implementations. Existing review notes are pointers;
their suggestions for incremental production changes do not constrain this reset.

The sandbox implementation takes scene ownership and pass organization from the
reviewed 0 A.D. revision `c3ace13b54f1d8a56557a6136814d1a6ceb66779`:
`PatchRData`/`TerrainRenderer` informed resident world geometry with camera-selected
visible records, `SceneRenderer` informed separate shadow, reflection, static and
water submissions, and `ShadowMap` informed one shared camera-framed light field.
The sandbox selects resident records independently for the main and reflected
cameras, owns geometry submission and complete terrain, object, cutout vegetation,
water and transparent pass state, and batches visible vegetation through one
instance stream. Prepared rigid objects retain shared meshes and instances.
Four synthetic units use resident authored meshes, exact source-frame GPU skinning,
and material shading directly in the scene depth target. The synthetic host
publishes moves and combat events while the client advances visible clips.
Static color and depth survive camera motion; short scrolls redraw exposed
strips, and unchanged reflections reuse their target. The main target is native
resolution with 2x MSAA, the reflection target is 0.375x resolution, and a
half-resolution HDR bloom feeds tone mapping directly into the swapchain.
Animated shoreline foam now comes from the water material's prepared coast depth
instead of submitting separate small wave meshes for this distant map view.
These are adaptations of the reviewed 0 A.D. scene and
batching structure, not a wholesale port.

Build a resident scene with terrain chunks, shared meshes/materials, compact
instances, camera/pass-specific culling, compatible draw batches and instancing.
Preserve transparency ordering. Rebuild geometry when inputs change, not when the
camera moves. Start with direct viewport drawing, a shared scene shadow pass and
reflection per relevant water plane. Adapt these algorithms to Civ III's projection
and D3D11. Port selected rendering techniques, not 0 A.D.'s simulation or whole engine.

Design now for stable world coordinates independent of camera position and zoom.
Camera changes must not require geometry regeneration or synchronous Civ III work.
Culling and selection of existing levels of detail may depend on the camera.
Supply lighting through shared environment parameters rather than baking lighting
into scene geometry; changing illumination may update shadow passes without
rebuilding the world. These are architecture requirements, not a request to begin
new visual effects during the current implementation.

Use this fresh execution structure from the outset. Do not rebuild production's
runtime first and then rewrite it again. Production visual modules may be adapted
as cohesive units. Mesh layouts and shader bindings may change; the visible quality
must remain comparable.

Specifically replace the production wrapped render atlas, regional reflection
cache/rebuild path, and receiver-page batching/32-page limit in the measured new
frame path. A new class around those mechanisms does not meet the objective.
Use resident scene geometry with camera transforms, scene-wide shadow maps and
reflections per relevant water plane; adapt the material bindings to those passes.
Production data loading/preparation and visual formulas remain reusable. Its runtime
structures and shader interfaces are not compatibility requirements. Keep the
reference path separate. Do not spend the implementation phase optimizing machinery
that this experiment is meant to replace. Measure alternatives after the coherent
new path runs, or earlier only to resolve a concrete implementation decision.

## Independent progress is mandatory

The x64 renderer owns its GPU resources, frame loop, presentation camera, input
and visual time. Civ III supplies copied authoritative snapshots/changes and
receives semantic input/actions. Neither side waits for the other's frames,
message pump or GPU work. A busy host leaves the client rendering coherent
last-known state; a stalled renderer leaves the host free to continue publishing.
Putting a synchronous request in a helper process does not meet this requirement.

Use bounded publication, coherent generations, latest-value camera updates and
ordered action handling with explicit overflow/reconciliation, including actions
in flight. Preserve unit incarnation, fog/viewer identity and displayed-view click
correctness. Civ III validates gameplay actions. Avoid stale camera-frame queues.
Native UI uses separate dirty surfaces for GPU composition; no map readback on the
game thread. Test one final presentation owner instead of assuming the old presenter.

Demonstrate independence with a synthetic x86 host: pause each process for two
seconds in turn while the other progresses, including real renderer-local camera
input. Recover coherently. Asset startup, device waits and OS scheduling are
separate from forbidden interprocess dependencies and must be reported honestly.

## A short realistic workload and useful measurements

Add deterministic units with existing models and authored clips. Show idle ambient
animation, multiple animated units, one accepted adjacent-tile move while the others
continue, scrolling during movement, then a distant jump and return. Add a
continuous zoom-in/zoom-out cycle in this same moving scene, including zoom while
units and water continue animating. At least one zoom leg should overlap scrolling
or follow the jump without resetting the scene. Keep normal water effects, shadows
and reflections on. Camera and zoom changes must not regenerate unchanged world
geometry, reset animation or lose movement. Check wrapping and a visibility change.
Increase density/world size after this sequence works at representative quality.

Report idle, scroll, jump and zoom costs separately: frame-time median/p95/worst,
fresh observed frames, input latency, geometry uploads/builds and memory. Include
the first zoom frame, ongoing zoom frames and return to the initial scale. Separate first
load, cold destinations and resident revisits. Target 60 Hz and 16.7 ms frame work,
camera-to-display below 33 ms p95, and host publication below 1 ms p95 for the stated
batch. Report misses. Present submissions are not displayed frames; existing VM
GPU timings are unreliable. Do not turn unavailable telemetry into another project.

At the end of the coherent implementation, use a few inspected comparisons, one
continuous clip and focused correctness checks. Preserve useful existing tests,
but do not add pixel assertions or rerun unrelated suites after every edit.
Start with short comparable runs.
The finish is a working representative scene, concise measurements/remaining gaps,
and a short recommendation naming the production responsibilities it replaces.
Full production migration is a separate assignment.

Zoom is part of the current moving workload, not a new visual-quality milestone.
New lighting and water aesthetics remain deferred while the measured moving scene
is above the 60 Hz target. Static color is now retained across unchanged frames;
moving water and units composite over it without a full color copy. A dedicated
water pass retains the prepared mesh, source textures, shadow and reflection
inputs with a bounded two-normal shader. Its fine ripples differ from the
production material. The next performance step is to replace the full-screen
multisample scroll restore with a larger resident camera region and redraw only
newly exposed geometry. Measure the moving workload with zoom included; do not
add pixel-comparison gates.

## Boundaries and useful prior findings

All new work stays in `Renderer/sandbox/`. Read/copy production visual code; do not
change shared production files in this assignment. Reuse existing VM/compiler
discovery and `Renderer/renderer.py` for production-category runs. Sandbox builds
and runs are authorized. Do not stage, install, launch Civ III, edit injected code
or patch tables, or begin deferred wonder/District rendering. Keep licensed assets
local. Do not create another plan, roadmap or approval ledger.

Prior experiments provided diagnostic measurements and synthetic independence
examples, not a visual-parity or sustained-60-Hz success. Historical evidence stays
in the archive; new claims require new captures and measurements.
Known bad assumptions include empty relief fields, disabled cliff/shadow inputs,
incomplete reflections, and 128x64 geometry placed at `tile_y * 16` instead of
production's `tile_y * 32`. Verify terrain/object/camera/depth/picking conventions
together. These findings are starting evidence, not a complete diagnosis.
