# C3X renderer

The [documentation index](docs/README.md) separates active work from preserved
contracts and history. Use [storage retention](docs/storage_retention.md) for
generated-output maintenance; ignored assets are not automatically disposable.

For renderer development, start with [current status](docs/retained_renderer_plan.md)
and the [Renderer64 scene and motion contract](docs/renderer64_scene_and_motion.md).
Use [validation](docs/benchmark_workflow.md) at the relevant integration gate and
the [deep architecture](docs/renderer_architecture.md) for subsystem detail.
The active destination is a retained GPU scene with copied authoritative game
changes, smooth renderer-owned presentation, and a cross-process map surface.
[Execution rules](docs/autonomous_renderer_execution.md) are a short reference.
Historical handoffs preserve evidence, not extra queues or current build identity.
The [input recording handoff](docs/input_recording_handoff.md) describes the current
recorder, visible Windows playback, measured controls and remaining qualification.
[Before/After playback](docs/realtime_replay_comparison.md) compares two renderers
at recorded input speed with independently generated animation frames.
[Short diagnostic capture](docs/short_diagnostic_capture.md) collects one brief
gameplay session for calibration, with explicit stop and memory-pressure handling.

The current C3X checkout is the implementation authority. Lab renders and tests
that code by category; Integration runs the affected current-code checks. Fixed
reference images are optional visual comparison aids, not numbered releases or
integration gates. Git holds history.

For the Windows game build, `native/BUILD_RENDERER64.bat` builds and stages the
32-bit bridge, Renderer64 DLL, and helper together under `bin/renderer64/`.
The existing `enable_custom_rendering` setting selects Renderer64 when true;
there is no separate backend setting. `INSTALL.bat` still installs C3X itself.
The game-facing direct surface and unit/UI behavior remain under integration
qualification as described in the [current status](docs/retained_renderer_plan.md).

Start with [the visual workbench](lab/README.md). The catalog has separate entries
for base terrains, relief, vegetation, water, map objects and animation. Shared
day/night and shadows live under lighting; tile-to-tile transitions live under
terrain. Their consumers share implementation, not copied category shaders.

```sh
python3 Renderer/renderer.py list
python3 Renderer/renderer.py lab grassland
python3 Renderer/renderer.py compare grassland
python3 Renderer/renderer.py test grassland
python3 Renderer/renderer.py integration grassland
python3 Renderer/renderer.py integration grassland --full
```

Category integration is intentionally focused and fast. `--full` adds the
cross-category tests and exhaustive behavior sweep used at strategic checkpoints.
Running these automated checks does not visually promote a changed category.
Before an agent describes a visual change as accepted, ready, promoted or
integrated, or ordinarily stages its DLL for game use, the agent must show the
user the relevant comparison and receive explicit visual acceptance. That
acceptance also authorizes staging the exact tested candidate into `Renderer/bin/`
for the user's game check; do so without asking for a second staging approval
unless the user says not to. An explicit request to stage before visual acceptance
authorizes an evaluation build only. Neither form of staging replaces fixed
references or authorizes `INSTALL.bat` or launching Civ III.

The Lab migration is complete. Current category previews use the production
D3D11 renderer; the generic Metal binding/compiler smoke test remains optional.
Retired milestone trees, release ledgers, cross-backend scene-parity experiments
and generated preview/verification archives have been removed.

## Architectural boundaries

The [architecture](docs/renderer_architecture.md) remains the deep reference for
world, asset, instance, view, pass and publication ownership. The short
[roadmap](docs/retained_renderer_plan.md) gives current status and order;
older presenter and clock details in deep references do not override the
Renderer64 cutover.
Shoreline waves, water motion and reflections are always enabled in the normal
performance workload. Optimize idle animation, scrolling and map jumps with all
three on. Effects-off runs are explicit diagnostic controls only; preserve the
user-facing switches.

Civ III/C3X owns game state, visibility, tile/object screen anchors, day/night and
seasons. The renderer produces resident map/pose textures and retains the native
composition operations over them. Its own visual clock schedules intervening
frames through the existing worker and HWND presenter, without requesting native
map redraws. Native captures share that clock; gameplay and directed-action
progress remain native. See [visual frame ownership](docs/visual_frame_ownership.md).
The renderer owns map fog/unseen coverage from copied native visibility. Admitted
map views also draw copied selection, route/turn-label and grid primitives through
the [tactical pass](docs/tactical_overlay_contract.md). Borders, city labels, unit
HUD and general UI retain native ownership.
Config-off preserves the original path. Custom-on map-plane failure must not
silently replay native terrain; custom-on map units are exclusively 3D, with explicit CPU 3D delivery at native
ownership barriers. UI portraits and renderer-off units remain native.

Integration owns capture, bounded caches, invalidation, dirty redraw, scrolling,
wrapping, zoom, compositing, device recovery and timing. Unit animation follows
Civ III's action director and native lifecycle; visuals do not drive gameplay.
Lab owns assets, shared geometry/shading, material response and visual approval.

Runtime packs are generic C3X data. Source-specific extraction remains offline.
Licensed source/derived art stays local and must not be redistributed with C3X.
Keep source normals, transforms, texture coordinates and material channels
unless an intentional, explicitly approved change calls for something else.

## Current references

- [Visual fidelity playbook](docs/visual_fidelity_playbook.md): preserved graphics breakthroughs.
- [Source-art findings](docs/source_art_findings.md): concrete recovered data, remaining inferences and selected local studies.
- [Natural geometry and materials](native/source_fidelity/README.md): current terrain adapters.
- [Cities](native/city_fidelity/CITY_FIDELITY.md): composition, lighting and known limits.
- [Units](native/environment_refresh/UNIT_FIDELITY.md): current source normals and materials.
- [Environment contract](docs/environment_lighting_and_ambient_effects.md) and
  [source lighting findings](docs/civ6_lighting_findings.md): confirmed data versus inference.
- [Patch dependency ledger](docs/civ3_patch_dependency_ledger.md): actual integration patch needs.
- [Configuration](docs/renderer_config_spec.md) and [visible scenes](docs/visible_scene_contract.md).
- [Mouse-wheel zoom](docs/custom_rendering_zoom.md): custom camera, input and fallback contract.

Natural wonders, constructed wonders and Districts remain deferred. Preserve
[natural-wonder contracts](docs/natural_wonder_rendering.md) and
[wonder/District contracts](docs/wonder_and_district_rendering.md): C3X's existing
definitions, placed instances and injected state remain authoritative. Multipart
art does not change one-tile gameplay identity; a constructed wonder gains no
map object without a C3X Wonder District tile. District kits must preserve
by-building/by-count state, topology and construction/pillage/abandonment. Do not
start this deferred work as part of the Lab cleanup.
