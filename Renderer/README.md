# C3X renderer

For the current navigation optimization continuation, start with the
[2026-09-09 handoff](docs/navigation_handoff.md). The navigation continuation is
not staged; the separate mountain evaluation has its own verified DLL and
rollback. Current binary identities, scoped measurements and remaining
performance/integration requirements are documented there.

The current C3X checkout is the implementation authority. Lab renders and tests
that code by category; Integration runs the affected current-code checks. Fixed
reference images are optional visual comparison aids, not numbered releases or
integration gates. Git holds history.

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

Civ III/C3X owns game state, visibility, tile/object screen anchors, time and
seasons. The renderer produces an off-screen map bitmap and inserts it at the
existing map boundary. It does not own a second presenter, game loop or camera.
Fog, borders, labels, selection, unit HUD and UI retain their native ownership.
Config-off preserves the original path. Custom-on map-plane failure must not
silently replay native terrain; preserve the existing separate unit fallback.

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
