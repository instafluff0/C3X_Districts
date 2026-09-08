# Production terrain helpers

The current C3X checkout is authoritative. This directory contains reusable
terrain, coastline, relief, shadow and compositing helpers; its older pickup
name is not a separate development workflow. Use `Renderer/renderer.py`.

## Current implementation map

- `terrain_query.h`: continuous material weights, world coordinates and wrap.
- `world_topology.h`, `world_coast.h`, `coast_index.h`: immutable captured
  topology, exact coastline queries and observed dependency revisions.
- `relief_query.h`: retained relief/shore/dune support and source-owner material
  coordinates. The selected natural terrain, mountain and forest layers also
  use `native/source_fidelity`; do not mistake the older underlying relief
  policy for the complete current appearance.
- `cliff_placement.h`: six source cliff bodies with world-stable placement,
  source geometry and complete material channels.
- `source_shadow.h` and `source_caster.hlsl`: source-geometry shadow pages,
  including alpha-cutout casters. World-aligned pages use physical light depth;
  their coordinates are independent of viewport anchors.
- `linear_target.h`: off-screen linear composition, MSAA resolve and display
  transfer. Exposure and transfer occur after reconstruction, not per material.

Current natural-pack decoding, height sampling and lighting-frame calculations
are shared in `lab/shared/natural/data.h`. Shader source preparation uses
`lab/shared/shaders`. These are shared implementations, not category copies.

## Findings and invariants worth preserving

**Material ownership.** JSON channel lookup must remain scoped to the selected
object. An earlier reader searched past object boundaries, letting a nested desert
layer supply an ordinary mountain's base color. Root channels and explicitly
selected nested channels must not borrow absent values from siblings. Tests use
the actual parser and selected local packs.

**Color destinations.** Known RGB555/RGB565 DIBs use world-anchored 8x8 ordered
rounding at the final copy. Full-color cached pixels remain unchanged; 32-bit and
unknown destinations retain their original path. Preserve clipping, outside
pixels, stable patterns across tile jumps and the destination-format diagnostic.
A screenshot alone is not proof of its destination pixel format.

**Exact query shortcuts.** A radius-r query around a point at coast distance d
needs only segments within d+2r. The bounded per-tile search preserves tie order,
wrapping and dependency certificates, and falls back to the exact index outside
its domain. The inland flat-ground certificate includes the normal-sampling
collar and a recorded 5x5 neighborhood; terrain edits or a closer coast revoke it.
Finite-difference height queries must not perform unrelated material work.

**Geometry and cache ownership.** Caster/prefetch-only records may contribute
geometry and shadows but must not publish map replacement flags. Clear ownership
for non-RENDER records without discarding their geometry. The injected validator
remains strict. Unchanged unit/UI redraws should reuse published terrain; camera
translation must not reseed world geometry. Preserve bounded mesh, viewport,
pixel-block and shadow caches, immutable worker inputs and atomic publication.

**Injected diagnostics.** Use the established game import
`(*p_OutputDebugStringA)`, not a direct injected API call that can retain an
installer-time address. Keep messages bounded and terminated. Normal gameplay
must not create a diagnostic file; standalone trace-file output is explicit opt-in.

## Assets and verification

`prepare_assets.py` builds the local generic `TerrainProfileR1` hill/cliff pack.
It reads the 26 preserved inputs under `packs/TerrainProfileSources/current`,
not a historical handoff manifest. Source mesh/index bytes and compressed texture
payloads stay exact; gloss uses a linear view. The optional paired
`--import-height`/`--import-cliffs` arguments preserve a selected normalized source
bundle and refuse to replace differing existing inputs. `--output` can select a
disposable rebuild directory; it cannot overlap the preserved inputs. Licensed
art stays local.

```sh
python3 Renderer/renderer.py build
python3 Renderer/renderer.py test grassland
python3 Renderer/renderer.py lab grassland
python3 Renderer/renderer.py compare grassland
python3 Renderer/renderer.py integration grassland
```

These commands do not install, stage or launch Civ III. Build writes the
candidate, not the production binary. Technical passes are not visual approval.
`verify_d3d11.cpp` retains focused hardware checks for source cutouts, shadow-page
reuse/invalidation, MSAA and linear transfer. The older `verify_native.py`
contains optional standalone minimap/default-logging diagnostics; its explicit
`pickup-r1` runs are not current-profile proof or category-workflow gates.

The current edit-reuse witness rebuilds all visible tiles and fails
its reuse assertion; do not replace that finding with earlier-profile passes.
Analytic dunes are not recovered source geometry, volcano BC5 semantics remain
unresolved, and historical screenshots or timings do not establish current game
parity. See `docs/civ3_patch_dependency_ledger.md` for patch needs and the
preserved `docs/visual_fidelity_playbook.md` for graphics guidance. Wonders and
Districts remain deferred.
