# Seasons feasibility

Seasons are feasible as material policies over the configured summer terrain.
The study now includes **executable Mac Metal scenes** for autumn, winter and
spring with both Civ VI and Civ V-style terrain packs, preserving existing
geometry and source material detail. It remains a Lab experiment, not an
implemented game feature; no Windows VM, game staging or injected build was used.

Autumn now has a [refined Civ V skin study](FALL.md) and
[direct target comparison](../../out/seasons/programmatic/fall-focus/beauty.html):
coherent golden crowns, quieter olive grass, distinct honey-straw plains,
existing forest-floor litter and improved diagnostic shadows/water. All original
terrain/tree geometry and source material detail remain. Use `--fall-focus` to
repeat the preceding refinement. The current `--fall-beauty` candidate corrects
the preview's coast/water gaps and uses stronger gold/amber foliage with quieter
bronze/olive ground. It has its own corrected Summer baseline and remains short
of the selected target; preservation checks are not visual acceptance. The
original four-season gallery retains the earlier comparisons.

Winter now has a [Civ V skin material/decal study](WINTER.md) with a separate
[before/after review](../../out/seasons/programmatic/winter-focus/index.html).
It preserves every existing tree mesh and placement. Blender supplies small
read-only exposure masks; eleven saved snow decals add relief over existing
ground. The current layered winter adds local drifts, retained summer texture
contrast, stronger canopy shelter and corrected height-based decal relief.
Its diagnostic shadows and water are improved; the four-season gallery below
retains the preceding comparisons. Use the `--winter-wonderland` commands in the
winter study to reproduce the current result.

Read the [real-game implementation playbook](PLAYBOOK.md) and inspect the
[full-scene gallery with tint comparisons](../../out/seasons/programmatic/index.html).
`seasonal_policy.hlsl`, `render.py` and `scene.mm` are the working prototypes;
`recipes.json` supplies generic atlas metadata and offline tint calibration.
The gallery explicitly identifies diagnostic water/shadow composition and the
remaining differences from the selected beauty concepts.

## Recommended simple appearance

| Season | Ground | Vegetation |
| --- | --- | --- |
| Summer | Current appearance, unchanged. | Current trees, unchanged. |
| Fall | Warm golden grass with a restrained dry-sage/olive undertone; plains remains paler honey-straw; sand barely changes. | Existing leafy trees receive varied gold, orange and occasional russet foliage; pines stay green. Matching leaf litter is inexpensive. |
| Winter | Blend existing snow over the actual ground, with different amounts of exposed grass, straw and sand. Cover upward-facing slopes; preserve rock faces and dune shapes. | Existing snowy pine materials; a procedural upward-facing snow coat for leafy trees and shrubs. Keep the current silhouette for the first version. |
| Spring | Slightly fresher/lighter grass; small clumps of soft pink, baby blue and sunny yellow wildflowers, with white/lilac accents, mostly on grassland and lightly on plains. | Fresh green leaves; optional sparse flowering shrubs later. |

Snow everywhere need not mean identical solid-white tiles. The earlier CPU swatch study used average
snow blends of 88% grassland, 77% plains and 57% desert, with exposed fractions of
roughly 7%, 19% and 41% respectively. These are authored trial values, not recovered
source settings. The current GPU policy uses stronger cool-toned coverage and
retained rock grain; see the playbook for its actual values. The visible source material and patch pattern provide identity;
subtle snow tint can supplement them. Fully replacing every surface with the
same snow material would erase the distinction. The winter swatches retain mean
RGB differences of 15, 26 and 39 on the 0–255 scale; that is a diagnostic signal,
not proof of readability at game zoom or under night lighting.

Use continuous biome weights and canonical world coordinates, not tile-shaped
snow stamps or screen-seeded random flowers. Reuse the current coastline and
forest-floor coverage. Roads should remain legible; rivers/water need not freeze.
Static flower patches are simpler than particles and need no continuous redraw.
At normal game zoom they will read as clustered color, not individually detailed
flowers; test a tiny authored flower decal if the shader marks look like confetti.

## Confirmed art and current source selection

- `TerrainNormalized/textures/snow_{base_color,height,specular}.dds` already exists,
  as do the mountain snow channels. The ground snow material is bound in
  `TerrainNormalized/materials/library/snow.json`. Source
  `Base/ArtDefs/TerrainStyle.artdef` declares `ART_DEF_TERRAIN_MATERIAL_SNOW` and
  `ART_DEF_TERRAIN_MATERIAL_MTN_SNOW`.
- Base `Clutter.artdef` declares three snowy pine bodies and two snowy pine clumps.
  Both `VegetationNormalized` and `Civ5EnvironmentVegetation` already contain their
  meshes and materials. All five base/snow pairs in each pack have equal positions,
  UVs and topology. Base-pack normals match too; selected alternate-pack normals
  differ and its snowy materials have an additional opacity channel. Preserve the
  current mesh/normals for a material-only experiment and compare masked coverage
  and shadows before promotion. Do not assume byte-identical material behavior.
- The selected natural adapter consumes `BeautyStudies/beauty_objects.bin` for
  forests. It has 22 forest bodies and 25 recipes, totaling weight 180: leafy
  bodies contribute 141, pines/shrubs 39. Its textures come from
  `Civ5EnvironmentVegetation`. Thus the current renderer already has leafy art;
  the pine-only limitation belongs to the Base Civ VI forest inventory.
- Base shared textures contain colored and white flowered-foliage atlases,
  `TEXTURE_DiffuseTint_Foliage_Bld_Flowered_{Color,White}_B_null`, each 512×512.
  Expansion 2 contains `TEXTURE_FX_Blossoms`, a 128×128 transparent blossom sprite.
  The GPU prototype isolates four complete alpha components, retains petal
  shading and uses those heads in a tiny tinted meadow atlas. Original
  model/particle bindings remain unproven; meadow use is a C3X adaptation.
  The larger foliage atlases contain carrier regions and are not repeated over
  ground. An analytic flower fallback remains available.
- The inspected Base/DLC terrain and clutter ArtDefs provide snowy biome art,
  not an established four-season terrain recipe. Fall and spring treatments here
  are C3X-authored effects; no source-engine seasonal behavior is claimed.

These licensed inputs remain local. Runtime policy should use generic seasonal
roles and pack metadata, with an authored snow material fallback, rather than
Civ VI names or package formats.

The seasonal work can now run after uninstalling Civ VI. Keep **`Renderer/packs/`
and this study's ignored `assets/` directory**: these are preserved inputs, not
disposable previews. The latter adds 5.08 MiB and contains all eleven normalized
snow decal meshes/UVs/placements and their four shared DDS channels, both flowered
foliage atlases, a blossom DDS and the tiny original blossom payload needed for
exact adaptation. The renderer prefers this integrity-checked blossom cache.
Existing ground/mountain snow and winter trees remain in the larger local packs.
This preserves the seasonal candidates selected here, not every possible future
asset from the game; new source exploration would require its installation again.
`cache_assets.py --verify` checks the saved cache without accessing Civ VI.
The [source-independence receipt](source-independence.json) records a Mac render
with the installation unavailable: all ten images across both terrain packs
match the previous renders byte for byte.

## Integration feasibility and remaining work

C3X already supplies `frame.season` and owns the saved cycle state. The port
belongs in source material composition before lighting, across terrain, relief,
surface decals, vegetation floors, foliage and the hydrology underlay. The
[playbook](PLAYBOOK.md#concrete-production-port) identifies the actual shader
branches, retained cache methods, reflection dependencies, slot conflicts and
existing game-side clock hooks.

Do not assume current geometry caches ignore seasons: the legacy
`terrain_frame_signature` folds its environment hash into its geometry identity,
and some tile keys also contain hour/season. Split those responsibilities
carefully during integration, preserving publication and shadow validity. The
legacy clock also depends on successful native seasonal PCX initialization;
custom rendering needs to use the same clock independently of that native art
path. Existing patch points suffice; this Lab changes no injected code and
identifies no new patch-table symbol.

Promotion still needs production D3D/Metal and retained-graph verification,
scroll/zoom/wrap checks, reflected seasons, road/river coherence, day/night
readability, native PCX independence, config-off behavior and return-to-Summer
parity. The executable Mac checks are evidence for the material approach and
preservation rules, not a claim that those live-game passes have happened.

## Repeat and inspect

```sh
python3 Renderer/lab/studies/seasons/render.py --all-packs
python3 Renderer/lab/studies/seasons/render.py --all-packs --case biomes --width 1920
python3 Renderer/lab/studies/seasons/compare.py
python3 Renderer/lab/studies/seasons/study.py
python3 Renderer/lab/studies/seasons/study.py --inventory-only
python3 Renderer/lab/studies/seasons/cache_assets.py --verify
python3 Renderer/renderer.py check
```

The full study selects the bundled Pillow/NumPy Python when system Python lacks
them; `C3X_RENDERER_PYTHON` can override it. Set `C3X_CIV6_ASSETS` or pass
`--assets-root` for source inventory or rebuilding the small cache from another
installed Assets tree. Reports store relative paths. The protected seasonal cache
contains persistent art inputs; there are no cloned terrain packs or staged builds.
The GPU runner reads existing packs in place, removes temporary builds/raw frames,
and refuses a run with less than 8 GiB free. Repeated runs overwrite the same
case outputs. Only reproducible PNG/HTML comparisons and small JSON receipts are
kept under `out/seasons/programmatic/`; selected concepts and source packs remain
untouched.

The earlier CPU outputs under `Renderer/lab/out/seasons/` are disposable:
`ground.png`, `trees.png`, `spring-candidates.png`, `evidence.json`. Together they
occupy about 1.2 MB. The receipt records input hashes and executable checks for
source integrity, pine/snow mesh compatibility, unchanged Summer albedo and
world-stable snow/flower sampling. The study's 32-tile periodic domain is a
diagnostic only; integration must use actual canonical map wrapping.
CPU previews omit production LEAN/PBR, shadows, biome transitions and filtering;
they are material feasibility evidence, not current D3D scene comparisons or
visual acceptance. Fixed references and production files remain untouched.

## Full-scene visual concepts

The full-scene concepts use built-in imagegen to edit the saved Summer
`test.biq` capture at
`Renderer/native/build/mountains-terrain/captures/after/grass/frame.png`
(camera 20,75). They illustrate art direction while keeping the Windows VM free;
they are not executable seasonal shader renders and may change fine asset detail.
The original capture remains unchanged. The prompt set is recorded in
`scene-concept-prompts.json`; concept PNGs and a source/output hash receipt live
under `Renderer/lab/out/seasons/scene-concepts/`.

Selected visual directions are `scene-concepts/fall.png` (the second image in the
autumn comparison) and `scene-concepts/winter-icy-blue.png`. The user explicitly
preferred these concepts. Rejected green autumn and unneeded refinement images
were removed to save disk space. These selections do not replace renderer
category reference images.

The user's direction is to make every season gorgeous, not merely identifiable.
Fall foliage was liked, but uniformly brown land was not readable enough.
Keep fall close to the original warm golden-tan palette. Grassland has only a
restrained dry-sage/olive undertone against paler honey-straw plains; avoid both
murky brown and dominant lush green. Delicate golden highlights and the liked
gold/amber/orange leaves should make fall inviting. Winter should evoke a
winter wonderland: luminous clean snow, delicate frosted canopies, gentle cool
blue shadows and clear blue water. The latest refinement shifts beige/brown ground
and rock toward silver-blue and cool gray, with pearl-white highlights. Retain
only a restrained pale sandy undertone in desert; use exposed texture, density
and landforms to distinguish biomes without a strong warm cast. Suppress dirty
warm muddy coloration beneath snow while retaining granular source detail,
rock normals and relief; avoid an airbrushed white blanket. Spring should use irregular harmonious meadow
flower clusters with fresh greenery and slightly richer pink, baby-blue and yellow
colors. Keep sparse coverage and small flower scale, avoiding evenly distributed
noisy dots or neon saturation.
These preferences guide subsequent material work; they do not approve or stage
an implementation or replace category references.
