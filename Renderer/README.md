# C3X renderer

The current production build is the approved visual baseline. Lab improves a
category; the user approves its affected appearance; Integration verifies that
revision in Civ III. Git holds history, not an ever-growing experiment ledger.

Start with [the visual workbench](lab/README.md). The catalog has separate entries
for base terrains, relief, vegetation, water, map objects and animation. Shared
day/night and shadows live under lighting; tile-to-tile transitions live under
terrain. Their consumers share implementation, not copied category shaders.

```sh
python3 Renderer/renderer.py list
python3 Renderer/renderer.py lab grassland
python3 Renderer/renderer.py compare grassland
python3 Renderer/renderer.py test grassland
python3 Renderer/renderer.py integration pending
```

Reorganization is in progress; [migration state](lab/MIGRATION.md) identifies
remaining work. Current catalog images use the actual production D3D11 renderer.
The Mac path now renders the grassland detail fixture from production inputs;
complete category parity is still unfinished. Legacy milestone files still
awaiting removal are not workflow authority.

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

Natural wonders, constructed wonders and Districts remain deferred. Preserve
[natural-wonder contracts](docs/natural_wonder_rendering.md) and
[wonder/District contracts](docs/wonder_and_district_rendering.md): C3X's existing
definitions, placed instances and injected state remain authoritative. Multipart
art does not change one-tile gameplay identity; a constructed wonder gains no
map object without a C3X Wonder District tile. District kits must preserve
by-building/by-count state, topology and construction/pillage/abandonment. Do not
start this deferred work as part of the Lab cleanup.
