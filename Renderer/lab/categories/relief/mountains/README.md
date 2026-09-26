# Mountains

Five authored macro height/blend variants form a continuous relief surface.
Connected mountains broaden along captured adjacency; shared-edge geometry and
shadow coverage match. Grass, plains, tundra or desert climbs the lower slope
before stone takes over. Terrain decals remain later.

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
