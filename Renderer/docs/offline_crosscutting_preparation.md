# Offline cross-cutting renderer preparation

Status: six requested head-start areas are implemented and executable. Victory
Location was explicitly excluded and remains set aside. Nothing here advances a
Lab gate, transfers native ownership, or enables injected rendering.

## Generic effect graphs

`tools/asset_compiler/effect_graph_compiler.py` validates source-independent
sprite/emitter graphs against the already normalized ambient and combat texture
manifests. The first bounded set contains six profiles and ten emitters: small
attached fire, smoke, and steam; persistent tile-local pollution radiation; and
transient land and water impacts.

Every profile has an absolute-time clock, finite tile bounds, exact normal and
reduced-zoom density/size policy, an explicit particle cap, deterministic stable
instance sampling, atlas layout, blend mode, cleanup, and activation authority.
The pollution profile has at most seven live particles and is driven by Civ III's
pollution bit. It deliberately describes a restrained generic layout rather than
claiming to decode Civ VI's city-scale nuclear-disaster emitter.

```sh
python3 Renderer/tools/asset_compiler/effect_graph_compiler.py
```

## Generic tile-fit calibration

`tools/asset_compiler/tile_fit_calibrator.py` reads normalized compound geometry,
grounds it at its minimum source Z, centers its footprint, projects all eight Civ
III facings at 128x64 and 64x32, and emits finite screen bounds plus a bounded
uniform scale for every cell. Enlargement is opt-in so an ordinary production
compile cannot silently inflate a naturally small object.

`preview/render_compound_tile_fit_sheet.py` supplies an isolated 32-cell
eight-facing/two-zoom/day-night inspection sheet. It is a candidate-inspection
tool, not a promotion renderer.

The radar observatory candidate required 11.22191796x enlargement merely to fill
the admissible tile envelope. Its isolated sheet shows a flat ornamental plaza
with paths and a central dome, but no tower or antenna silhouette. It therefore
remains structurally valid but is now a weak visual-semantic candidate that L19B
should expect to reject unless additional source-backed components change that
reading.

## State provenance

`integration/state_provenance.json` and
`tools/state_provenance_compiler.py` account for every field in the seven future
category contracts. The current audit covers 79/79 scene fields. It identifies
eight bounded gate questions around selected map art, visibility, construction,
tile flag precedence, and unit body/progress suppression. These are audit tasks,
not patch requests; `required_user_action` remains empty.

```sh
python3 Renderer/tools/state_provenance_compiler.py
```

## Content-addressed loader ABI

`tools/asset_compiler/pack_loader_abi.json` freezes canonical POSIX path rules,
SHA-256 object identity, exact object ranges, last-mount-wins path overrides,
cross-mount deduplication, pre-allocation budget checks, and fail-closed behavior.
`pack_loader_abi.py` is an executable reference loader that validates a bundle,
resolves partial overrides, produces a unique resident-object plan, verifies
objects before first use, and rejects missing, corrupt, unsafe, or over-budget
requests. It does not activate a native loader.

The rebuilt TileObjects pack now contains 1,353 logical files represented by
1,207 unique objects; deduplication saves 213,078 bytes. A real two-file
resident-set preflight verifies its manifest and catalog within a 1 MiB budget.

## Automated visual QA

`tools/visual_qa.py` reads dependency-free RGB8 PNG and uncompressed 24-bit BMP
outputs. Its reusable metrics cover silhouette bounds, coverage, contrast,
luminance span, edge clipping, allowed-region spill, ground-contact gap,
arbitrary Civ III palette coverage, day/night emissive strength, normal/reduced
zoom consistency, and temporal motion occupancy.

The functions and plan runner are tested independently so future Lab promotion
scripts can add thresholds without coupling them to one asset family.

## Industrial colony blocker closed

The old industrial-camp rejection was stale. The corrected row-vector local
composition (`scale/shear * transposed quaternion rotation`) now proves strict
bind/inverse-bind errors of `3.3737868e-7`, `7.9908300e-7`, and `3.3737868e-7`
for `IMP_Camp_IND_01..03`, all far below the unchanged `2e-5` threshold. A static
bake is therefore unnecessary and would discard valid component composition.

The normal recursive tile-object compiler now uses all three industrial roots
for Civ III eras 2 and 3. The rebuilt generic pack contains six colony roots,
91 components, 243 geometry parts, 179 materials, 79 textures, 71 emissive
materials, and 542 attachment points with zero optional dependency rejects.
Visual scale, density, resource coexistence, owner-color restraint, and final
selection still belong to L19A.

## Verification boundary

The focused unit tests are included in both the full discovery run and the
one-command Lab/Integration developer workflows. Derived local packs and preview
outputs remain ignored. The checked implementation contains no source-format
runtime dependency, no D3D activation, no injected-code change, and no new CSV
request.
