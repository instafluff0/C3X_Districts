# Mountains

Five authored macro height/blend variants form a continuous relief surface.
Connected mountains broaden along captured adjacency; shared-edge geometry and
shadow coverage match. Grass, plains, tundra or desert climbs the lower slope
before stone takes over. Terrain decals remain later.

## Flat-ground handoff (Lab candidate)

The mountain neighborhood uses one joined relief grid in place of ordinary
ground. Its zero-rise fringe now carries the ordinary ground normal, altitude
and support values, uses the native-prepared terrain material response, and
enters shared lighting as ground. The handoff fades out before the raised
rock and authored hill response, preserving adjacent mountain connections.
This handoff adjustment alone did not remove the straight grassland marks seen
in the `test.biq` crop; the separate material study below isolates them.
The original `test.biq` is unchanged. Review the real native captures under
`Renderer/lab/out/mountains/ground-handoff/after-v8/` against `before/`;
the marked plains/desert edges and nearby hill are collected in
`Renderer/lab/out/mountains/ground-handoff/seam-review.png`.
`python3 Renderer/renderer.py test mountains` passes (136 tests, one skip),
and the reviewed BIQ cameras render with zero fallback. This remains a Lab
candidate; no binary was staged and no fixed reference was replaced.

## Grassland triangle and texture-band study (Lab candidate)

At the fixed `test.biq` camera `(20,79)`, single-variable shader ablations
identified two overlapping marks. Grassland source-surface recipes draw many
disconnected three-vertex patches. A material-class render marks the visible
triangle outlines as the grass subtype (`material.y` between 4.5 and 5.5).
The Lab mesh now skips those optional grass patches; plains and desert patches
remain. The softer straight band that persists without those patches comes
from the single GrassColor lookup used by both ordinary terrain and the flat
mountain fringe. Averaging four full-resolution samples removed the band but
visibly flattened large grassland scenes. The current Lab material instead
keeps the original fine sample and replaces only its mip-7 coarse color with
an average of four offset coarse samples. This retains the source grass grit
and medium-scale variation while reducing the repeated band. Mountain
adjacency and rock materials are unchanged.

`Renderer/lab/studies/mountains/triangle_ablation.py` reproduces isolated
shader controls using an explicit candidate DLL and shader source root. The
fixed-camera ablations and the first exact user-crop comparison are under
`Renderer/lab/out/mountains/triangle-final-source/`. The wide-scene audit and
one-change-at-a-time captures are under `Renderer/lab/out/terrain-wide-audit/`.
The selected mip-7 comparison is under
`Renderer/lab/out/mountains/triangle-mip7/`, and native captures regenerated
from current Lab sources are under
`Renderer/lab/out/mountains/triangle-current-source/`. The candidate was built
without staging, and fixed references were not replaced. The Terrain task's
eight-view, 1600x900 native `test.biq` audit is under
`Renderer/lab/out/terrain-wide-audit/mip7-contact.png`: grass/plains/desert
blends, mountain collars, shorelines, and evening shadows show no new seams.
`test.biq` has no volcano tile, so this audit cannot assess volcanoes. Visual
acceptance is pending.

That material audit used the standard mountain mesh. A separate isolated Lab
build combines its corrected grass material and disabled grass triangles with
the `lower` mesh (68% height, 108% span). Six 1600x900 `test.biq` views and a
matched standard-versus-lower comparison are under
`Renderer/lab/out/mountains/lower-terrain-audit/`; all rendered with zero
fallback. Its build uses the committed city compiler while a concurrent city
edit is in progress; no shared native source or sandbox files were changed.

## Sandbox-based shape study (Lab only)

`Renderer/lab/studies/mountains/shape_study.py` copies the current sandbox x64
resident drawing path and native mountain inputs into an isolated Lab output
root. Its `sandbox` control uses those copied sources unchanged. The `lower`
proposal scales mountain height to 68% and footprint spans to 108%; `squat`
uses 55% height and 110% footprint spans. Both preserve the five authored
height/blend fields, material and lighting, adjacency detection, ridge union,
shared-edge geometry and shadow rendering. Neither proposal edits production or
sandbox code, stages a DLL or replaces a fixed reference.

Each proposal renders sixteen deterministic mountain placements on flat
grassland, with their five source-field IDs shown in a sprite-style atlas, and a
separate connected ridge. The images are actual sandbox D3D11 captures through
the copied client and renderer, rather than painted examples. Run with a Python
containing Pillow and the configured Windows VM:

```sh
python3 Renderer/lab/studies/mountains/shape_study.py
python3 Renderer/lab/studies/mountains/shape_study.py --sheets-only
```

Review `Renderer/lab/out/mountains/shape-study/comparison-variety.png`,
`comparison-connected.png`, and each shape's `variety-atlas.png`. The study
records input and DLL hashes alongside its ignored captures. These are
proposals for visual review; later sandbox or production promotion requires the
user to select and accept a shown result.

For map context, `--biq-examples` uses the saved Lab DLLs and the unchanged
`Renderer/packs/RendererSourceStudies/maps/test.biq` terrain. It captures the
lower proposal at raw tile centers (57,33), (63,58) and (17,49), and a matched
sandbox control at (57,33). The sandbox client adds its preview units/resources;
these are renderer captures rather than Civ III screenshots. The command writes
full-size images and source/DLL hashes under `lab/out/mountains/shape-study/`.

```sh
python3 Renderer/lab/studies/mountains/shape_study.py --biq-examples
```

## Forest and jungle mountain study (Lab only)

Civ III's `mountain forests.pcx` and `mountain jungles.pcx` do not indicate two
terrain categories stored on one tile. Its mountain draw routine checks a
terrain-6 tile's four diagonal neighbors and selects a forest or jungle
mountain sheet when the surrounding terrain qualifies. The separate vegetation
draw routine handles terrain-7 forest and terrain-8 jungle tiles. C3X captures
one visible category per tile, and the sandbox client and renderer likewise
select relief or vegetation from that single category. The original `test.biq`
has no combined terrain code; it does have two mountains with four forest
neighbors and three with four jungle neighbors.

`Renderer/lab/studies/mountains/canopy_study.py` renders an artistic analogue
using the accepted-for-review lower mountain geometry: a low forest or jungle
ring at the rocky foot. Sixteen isolated placements on flat grassland become
`forest-grassland-sheet.png` and `jungle-grassland-sheet.png`. Their canopy is a
Lab-authored visual marker, not a Civ III gameplay terrain type. The study also
exports the unchanged `test.biq` terrain, writes derived Lab CSV scenes marking
only the five fully surrounded mountains, and captures four context views plus
`test-biq-canopy-contexts.png`. Mountain geometry still joins adjacent
mountains in the existing mesh. Rendering uses a separately copied sandbox
client/native source and does not touch production, sandbox source, BIQ,
reference images, or staging.

Run with a Python containing Pillow and the configured Windows VM:

```sh
python3 Renderer/lab/studies/mountains/canopy_study.py
python3 Renderer/lab/studies/mountains/canopy_study.py --sheets-only
```

Captured images, CSVs, source receipts and DLL hashes are under
`Renderer/lab/out/mountains/canopy-study/`. All six captures completed with
`fallback=0`. The context views are actual sandbox D3D11 renders of terrain
from `test.biq`, with the documented Lab canopy addition; they are not Civ III
screenshots. This study is for visual review before any promotion.

## Accepted material

The user accepted the cleaner mountain body and slope-aware base projection on
2026-09-12: "Put it in production, please." The implementation preserves the
snow caps, rocky feet, geometry, material channels and prior zebra-band fix.
No fixed reference was replaced.

The Civ V Environment Skin's nine 2048×2048 base/top/snow color, height and
specular files match the installed package byte for byte. Base/top height and
specular happen to be identical in this skin; other packs keep independent
channels. Runtime materials remain source-independent.

Ground-to-rock color/detail/specular coverage still uses final rise 0.02–0.48
world units. Source footprint and face steepness cannot override coverage.
The patchy-snow top layer blends at normalized source height 0.52–0.68; full snow
blends at 0.62–0.78 with the existing 0.02–0.25 slope gate. Per-projection height
derivatives retain broad amplitude 0.04 and fine 3.7× detail amplitude 0.12.

Added grain, height-color and crevice contrast stays intact at the rocky foot,
then fades over rise 0.38–0.75, retaining 8% above that. It does not return near
the snow line. Full snow's color multiplier remains intact. These masks and
amplitudes are accepted C3X artistic choices, not recovered source equations.

The ground portion retains its original world-XY texture mapping on flat land.
On rising, steep slopes the 20 fine terrain/hill color, height and specular
samples blend toward three-axis projection, preserving each family's original
scale, rotation and offsets. This corrects the vertically stretched grass base.
The broad terrain color field and ground-to-rock coverage remain unchanged.
Both new calibrations use captured local volcano coverage to retain the existing
volcanic material response; there are no fixed Lab coordinates in production.

## Verification

`python3 Renderer/renderer.py integration mountains --renderer-only` checks the
current shader adapters, source/capture contracts and native terrain-edit reuse.
The executable material test checks neutral flat ground, monotonic projection,
volcano protection and relocation/wrap independence with and without volcano
bindings. Volcano Integration checks the shared surface's material lifecycle.

Accepted experiments remain under `lab/out/mountains/body-study/` and their
reproduction tools under `lab/studies/mountains/`. Current-code visual checks,
integration and staging identities are recorded under
`lab/out/mountains/body-promotion/`. Staging never launches Civ III.

Production staging is complete. The two mountain close-ups reproduce the accepted
Lab pixels exactly; volcano-context views differ only by one color step in 53
and 17 pixels. Mountain and volcano Integration passed 249 and 253 tests,
respectively (one skip each), including edit reuse and volcano lifecycle,
scrolling and wrap parity. The staged DLL matches the tested candidate hash.

Earlier source evidence and controlled snow/banding probes remain in the study
notes and `Renderer/docs/visual_fidelity_playbook.md`; Git preserves history.
