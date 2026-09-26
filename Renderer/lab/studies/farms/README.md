# Farms over test.biq

Run `python3 Renderer/tools/asset_compiler/build_farm_runtime.py`, then
`python3 -m Renderer.lab.studies.farms.test_biq` from the repository root with
the configured Windows VM. The study builds an isolated candidate DLL and
renders a control, noon and sunset coast views, a close coast view, inland
grassland, and a tundra coast. Its contact sheet and source hashes are written
under `Renderer/lab/out/farms/`.

`test.biq` supplies terrain and rivers. Irrigation is added only to the preview
scene by a deterministic Lab fixture; the BIQ itself is unchanged. This is a
visual study, not a fixed reference replacement or a staged game build.

The source crop texture contains several planted fields in one atlas. Mapping
the entire atlas to each small decal produced a miniature checkerboard. The
runtime farm pack selects an authored row region, preserves the source UV
orientation, and samples the decal against terrain relief. The chosen region,
tile palette mapping, and field placement are inferred for Civ III composition.
Each irrigated tile selects four row decals in a 2×2 layout. Independent,
seeded size, position, and palette choices vary the fields while a shared tile
orientation keeps their crop rows nearly parallel. Narrow gaps remain between
their transformed bounds. The fields clip against the sea shoreline and the
renderer river corridor, leaving a bank beside visible river water. Each plot
first tries modest size and position adjustments within its own quadrant so
crop rows end at a clean edge. A water-facing side is trimmed along one straight
bank line when possible; the source decal has twelve terrain samples per edge
for finer clipping where a river crosses the plot interior. Each farm
tile proposes four to six centered source trees and one centered source building.
The source's large prearranged tile cluster is omitted so inland farms do not
gain a disproportionate number of props. Raised pieces use the shared scene
lighting and shadow path. Buildings choose dry ground before trees fill the
gaps; both use shore and river clearance, and a prop is omitted only if no dry
site fits its footprint.
Building style is still shared across cultures. The current era comes from
`route_style` (the viewer's era); `territory_owner_id` is available on land
tiles, but owner culture, civilization, and era are only captured on city
tiles. A culture-aware farm kit needs explicit owner metadata and suitable
style-by-era building assets, with a neutral fallback for unowned land.
The current renderer uses base color and source geometry but does not yet shade
the source crop height/specular channels or foliage opacity. See
`Renderer/docs/visual_fidelity_playbook.md` for those fidelity requirements.
