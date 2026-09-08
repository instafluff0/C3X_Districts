# Source-fidelity Lab state of the art

This is the authoritative visual pickup for the current investigation. It
contains four retained per-system macOS Metal witnesses and the accepted
`source-fidelity-r13/inland` 100-tile natural-scene composition. It is not a
Civ III or Windows integration claim. Cities are explicitly excluded and keep
their pre-existing r2 disposition. The machine-readable authority is
[LAB_STATE_OF_ART.json](LAB_STATE_OF_ART.json).

| Study | Retained review image | Source-fidelity result |
| --- | --- | --- |
| Mountain | `beauty-mountain-civ5-r4` | Authored 256² macro relief and footprint determine silhouette; complete 2K base/top/snow channels supply triplanar material detail. |
| Grassland, plains, tundra and hills | `beauty-land-types-r1` | Distinct source material families, authored standard-hill relief and deterministic 3/2/2 source rock-patch recipes. |
| Forest | `beauty-trees-r18` | All 22 source bodies, 25 placement records and count weight 180; packed normals and `Generic_OPAC` masks. |
| Warrior | `beauty-warrior-r4` | Complete posed components, packed normals and source repeat/clamp addressing; no guessed LEAN decode. |
| Composed natural scene | `source-fidelity-r13/inland` | Exact terrain and mountain providers, source-body forests, one face/cast `ShadowL`, opacity-aware directional cast shadows and retained hydrology coexist in one 100-tile viewport. |

The [native-size composed frame](../../../terrain_lab/v2/audits/beauty/out/source-fidelity-r13/inland/h12-z1-pan00.png),
[lossless r11 comparison](../../../terrain_lab/v2/audits/beauty/out/source-fidelity-r13/inland/comparison.png),
and [matched shadow proof](../../../terrain_lab/v2/audits/beauty/out/source-fidelity-r13/inland/shadow-evidence.png)
are pinned. A separate r14 replay reproduced both raw zoom hashes exactly.

The visual-quality contract is strict:

- upstream assets and their metadata are authoritative;
- source geometry is not repainted, stretched, flattened or shortened to fit a
  later tile system;
- placement may translate, rotate and uniformly scale an intact source body;
- missing engine behavior is labeled as an inference;
- review uses scene-linear Metal output, 4x MSAA, 16x anisotropy and 2x final
  render scale with one reconstruction pass.

Integration must use the r2 composition fixture and modules, not the rejected
`source-fidelity-r1`/r4 approximation. The terrain module directly invokes
`beauty_terrain.cpp`; the mountain module directly invokes
`beauty_mountain.cpp`; the forest module uses `beauty_objects.cpp` with
caster + receiver + alpha-cutout metadata. The shared Q6 field supplies actual
source-triangle tree shadows. A final hydrology module preserves rivers and
water while suppressing the superseded low-detail hill/mountain geometry.
R13 additionally makes the Q6 `ShadowL` used to build that field authoritative
for terrain, mountain and tree face lighting, and fixes mountain projection to
the same canonical world-to-screen origin used by terrain and forest.

The older `beauty-scene.fixture.json` remains superseded because it can place
vegetation through buildings. The accepted natural witness renders no cities
and changes no city code or selection. Game Integration must apply
`ClipBuildings=true`, `ClipRiver=true` and `ClipCoastline=true` against
authoritative captured footprints before tree emission.

Known boundaries remain explicit. Exact Firaxis forest scatter and LEAN
evaluation are unknown. Hill rock composition is an inferred use of confirmed
source footprints/materials. Two source tundra snow-hill decal families remain
identified but not imported. None of these gaps authorizes a procedural
lookalike when source data is available.

Validate from the repository root:

```sh
python3 Renderer/handoffs/candidates/lab_v2_complete_r2/validate_state_of_art.py
python3 Renderer/handoffs/candidates/lab_v2_complete_r2/package.py verify --evidence --assets
```

These checks prove inputs, contracts, retained pixels and determinism. Visual
inspection remains the authority for appearance; r13 passed the agent visual
review requested before handing the work to Integration.
