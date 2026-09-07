# Isolated Lab state of the art

This is the authoritative visual pickup for the current investigation. It is a
set of four isolated macOS Metal witnesses, not a combined scene and not a Civ
III integration claim. The studies retain the user-reviewed mountain, forest
and Warrior, plus the grassland/plains/tundra/hills comparison. Cities are
explicitly excluded from this update and keep the pre-existing r2 disposition.
Their machine-readable entry point is [LAB_STATE_OF_ART.json](LAB_STATE_OF_ART.json).

| Study | Retained review image | Source-fidelity result |
| --- | --- | --- |
| Mountain | `beauty-mountain-civ5-r4` | Authored 256² macro relief and footprint determine the silhouette; 2K base/top/snow material channels add triplanar detail without changing that silhouette. |
| Grassland, plains, tundra and hills | `beauty-land-types-r1` | Distinct source material families, authored standard-hill relief and deterministic 3/2/2 source rock-patch recipes. |
| Forest | `beauty-trees-r18` | All 22 source bodies, 25 placement records and count weight 180; authored packed normals and `Generic_OPAC` masks. |
| Warrior | `beauty-warrior-r4` | Complete posed component recipe, authored packed normals and source repeat/clamp material addressing; no guessed LEAN decode. |

The visual-quality contract is deliberately strict:

- upstream assets and their metadata are authoritative;
- source geometry is not repainted, stretched, flattened or shortened to fit a
  later tile system;
- placement may translate, rotate and uniformly scale a complete source body;
- missing engine behavior is labeled as an inference and never presented as
  recovered metadata;
- review uses scene-linear Metal output, 4× MSAA, 16× anisotropy and the exact
  Civ V Environment Skin noon LUT where recorded.

The older `beauty-scene.fixture.json` is superseded as current visual evidence.
It predates the isolated fixes and can place vegetation through buildings. A
future composition pass must apply the confirmed forest metadata
`ClipBuildings=true`, `ClipRiver=true` and `ClipCoastline=true`; it must not
solve collisions by squashing buildings or trees.

The remaining honest gaps are bounded. Exact Firaxis forest scatter and LEAN
evaluation are unknown. The hill patch shader combines a confirmed source alpha
footprint with the source hill-top material as a Lab inference. The two source
tundra snow-hill decal families are identified but not yet imported. The city is
one selected medieval study, not universal era/culture/size coverage. None of
those gaps licenses a procedural lookalike when source data is available.

Validate the pickup from the repository root:

```sh
python3 Renderer/handoffs/candidates/lab_v2_complete_r2/validate_state_of_art.py
python3 Renderer/handoffs/candidates/lab_v2_complete_r2/package.py verify --evidence --assets
```

These checks prove inputs, contracts and retained pixels. Human or agent visual
inspection remains the authority for whether the images look good.

On 2026-09-07, fresh Metal quick renders reproduced all four in-scope raw hashes
and all four review-conversion hashes exactly. The corresponding `check`
matrices then passed 32/32 variants. Direct inspection retained all four
witnesses: relief and material detail remain crisp, hill rock patterns vary,
foliage silhouettes are properly masked, and the Warrior's right arm and eye no
longer carry the clamped-UV metallic smear.
