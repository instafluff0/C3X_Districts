# Mountains

Five source macro height/blend variants, complete rock and snow materials, matching silhouette and shadow coverage. Edge-adjacent mountains use their captured topology to broaden along the range axis and compose neighboring fields into one height surface. Each influenced world patch owns one non-overlapping terrain-relief surface with identical shared-edge samples; there is no second ground mesh beneath it. The underlying grass, plains, tundra, or desert color climbs the lower slope before rock, upper rock, and snow take over, while triplanar rock height, normal, and specular detail remains on every sufficiently raised face regardless of compass direction. Terrain decals remain a later layer. Shore and shadow coverage use that same final surface.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.

The ground-to-rock blend uses only final rise above the underlying terrain:
0.02–0.48 world units. Color, height detail and specular share that coverage.
Source HBLEND remains a geometry input; it cannot override the material mask,
and face steepness cannot change the transition width. The former combination
of slope-dependent thresholds and `max(height coverage, source footprint)`
forced low left faces to rock while the opposite faces retained grass.

Upper-rock treatment uses final rise (0.08–0.62); snow follows the authored
summit mask. Fine rock relief also contributes restrained color contrast. The
local base/upper height files are byte-identical, but other packs retain their
independent channels. These are C3X material choices, not recovered source
shader equations.

The scalar production-mask regression checks equal coverage at equal rise
across source footprints, face normals and underlying hill elevations. The
previous shader fails that check. Mask-only before/after evidence is under
`lab/out/mountains/diagnosis/`; current beauty previews are under
`lab/out/mountains/coverage-correction/`. The ground-to-rock transition was
accepted; the subsequent texture-crevice and fine-grain correction is also
approved for production.
Fixed references are unchanged.

The subsequent texture probes isolate intact base color/height sampling on both
faces, comparable mip selection, and predominantly separate triplanar axes.
Sharper projection weights and disabling specular or self-shadowing do not
remove the remaining difference. At noon, sampled left/right face patches have
approximately 0.72/0.16 geometric normal–light alignment. The 06:00 view restores
strong crag shading on the left with unchanged material inputs. This points to
diffuse bump contrast as the main remaining cause; the altitude-only cavity
term does not describe texture-scale crevices. These are controlled-render
findings, not recovered source-engine behavior.
Probes, measurements and the noon/dawn comparison are under
`lab/out/mountains/texture-diagnosis/`.

The current correction compares rock height with its wider mip neighborhood at
two scales, reducing crevice fill independently of the light direction. Fine
authored rock color is normalized against its local mean for grain contrast,
with capped highlights to retain gray stone. Both responses fade with snow and
the accepted ground-to-rock coverage; normal strength is unchanged. An executable
shader regression checks neutral flat/protruding material, monotonic crevice
darkening and invariance to a constant height offset. These are C3X material
approximations, not geometric ambient occlusion or recovered source equations.
Before/after review renders are under `lab/out/mountains/crevice-correction/`.
Production verification reproduces all three approved scenes pixel for pixel
with the existing staged DLL; 185 focused tests pass. The separate, pre-existing
terrain edit-reuse replay still rebuilds all visible tiles. Its failure and the
mountain verification are recorded under `lab/out/mountains/production-check/`.
