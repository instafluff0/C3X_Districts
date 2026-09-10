# Mountains

Five authored macro height/blend variants form one continuous terrain-relief
surface. Connected mountains broaden along captured edge adjacency; shared-edge
samples and shadow coverage match. Underlying grass, plains, tundra or desert
climbs the lower slope before stone takes over. Terrain decals remain later.

The accepted material is the Civ V Environment Skin. All nine 2048×2048
base/top/snow color, height and specular files match the installed source package
byte for byte. The base/top height and specular pairs are identical in this
skin, but other packs retain their independent channels. Runtime formats remain
source-independent, with no new texture imports or asset dependencies.

Ground-to-rock color/detail/specular coverage uses final rise 0.02–0.48 world
units. Source footprint and face steepness cannot override it. Source HBLEND
remains a geometry input. The scalar coverage regression preserves equal
coverage at equal rise across footprints, normals and underlying hill heights.

## Accepted fine rock relief

The user accepted the `micro-relief` Lab proposal and requested production use
on 2026-09-09. It retains the previous proposal's white peaks and removes most
of the remaining horizontal bands by reducing broad material bump response,
while preserving finer rock grain and the existing ground transition.

The top color is patchy snow, not plain upper rock. It blends at normalized
source height 0.52–0.68; full snow blends at 0.62–0.78 with a 0.02–0.25 slope
gate. Height derivatives are computed per triplanar projection before weighting.
Broad material derivatives use amplitude 0.04; the finer 3.7× source-height
sample retains amplitude 0.12. Local crevice fill, authored color grain,
specular channels, source geometry and shadow-caster coverage remain intact.

These masks/amplitudes are accepted C3X visual choices, not recovered source
shader equations. Source ArtDef does confirm the separate lower/upper snow
material roles and its own 24/32 and 26/32 height thresholds. Controlled renders
separate white mottling from dark bands: the top mask caused the former, while
strong broad bump shading primarily caused the latter. Disabling shadows or
added crevice contrast, rotating textures, or changing texture scale did not
adequately remove those bands. The derivative correction alone was insufficient.

Detailed probes and reproduction tools are in `lab/studies/mountains/`.
Accepted Lab images and repeat hashes are under `lab/out/mountains/zebra-study/`;
production verification/staging evidence is under `lab/out/mountains/promotion/`.
The prior snow-cap control and accepted coastal zoom-224 image reproduce exactly.
Fixed references remain unchanged; they are optional comparisons, not release
gates. See `Renderer/docs/visual_fidelity_playbook.md`.

The earlier noon/dawn texture diagnosis remains useful: fixed material inputs
have different diffuse bump contrast by face/light alignment. Source color and
height survive on both faces with comparable mip selection. Wider-mip height
neighborhoods supply direction-independent crevice darkening, and local-mean
color normalization retains fine grain. These are C3X material approximations,
not geometric ambient occlusion or recovered source-engine behavior.

Production staging is complete for the accepted change. An isolated committed
baseline excluded the concurrent navigation experiments; all compiled inputs
matched Git `f2696829d9bc549ae01597cf42d25f3ddab736de`. Four accepted frames match
byte for byte, with additional gameplay evidence at widths 160 and 192. Focused
integration passed 240 tests (one skipped), and the edit witness reused 260 tiles
with zero warm/cold pixel differences. Staging and rollback identities are in
`lab/out/mountains/promotion/staging.json`; no game launch was performed.
