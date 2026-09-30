# Seasons feasibility

Seasons are feasible using the current meshes, a small seasonal material policy,
and existing source snow art. This is a Lab study, not an implemented game feature.
The Mac-only probe reads installed art and current local packs, creates small CPU
material/asset previews, and never accesses the Windows VM or builds game code.

## Recommended simple appearance

| Season | Ground | Vegetation |
| --- | --- | --- |
| Summer | Current appearance, unchanged. | Current trees, unchanged. |
| Fall | Warm golden grass with a restrained dry-sage/olive undertone; plains remains paler honey-straw; sand barely changes. | Existing leafy trees receive varied gold, orange and occasional russet foliage; pines stay green. Matching leaf litter is inexpensive. |
| Winter | Blend existing snow over the actual ground, with different amounts of exposed grass, straw and sand. Cover upward-facing slopes; preserve rock faces and dune shapes. | Existing snowy pine materials; a procedural upward-facing snow coat for leafy trees and shrubs. Keep the current silhouette for the first version. |
| Spring | Slightly fresher/lighter grass; small clumps of soft pink, baby blue and sunny yellow wildflowers, with white/lilac accents, mostly on grassland and lightly on plains. | Fresh green leaves; optional sparse flowering shrubs later. |

Snow everywhere need not mean identical solid-white tiles. The study uses average
snow blends of 88% grassland, 77% plains and 57% desert, with exposed fractions of
roughly 7%, 19% and 41% respectively. These are authored trial values, not recovered
source settings. The visible source material and patch pattern provide identity;
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
  Expansion 2 contains `TEXTURE_FX_Blossoms`, a 128×128 transparent petal sprite.
  The payloads decode successfully and are shown in the candidate sheet. Exact
  model/particle bindings and suitability as meadow flowers remain unproven.
  The foliage atlases contain carrier regions: repeating their whole image over
  ground would be wrong. Procedural flowers are the simpler first choice.
- The inspected Base/DLC terrain and clutter ArtDefs provide snowy biome art,
  not an established four-season terrain recipe. Fall and spring treatments here
  are C3X-authored effects; no source-engine seasonal behavior is claimed.

These licensed inputs remain local. Runtime policy should use generic seasonal
roles and pack metadata, with an authored snow material fallback, rather than
Civ VI names or package formats. No upstream art needs to be copied for this study.

## Integration feasibility and remaining work

C3X already owns Summer/Fall/Winter/Spring, the selected cycle mode and saved
cycle state. `prepare_custom_renderer_frame` captures `frame.season`;
`terrain_frame_signature` hashes it separately from geometry. The shared
`evaluate_environment` currently gives fall/winter a modest ambient tint. Lighting
tint alone cannot produce foliage changes, snow cover or flowers.

The retained pipeline in `sandbox/fresh_pipeline.h` caches terrain/underlay
materials independently of hour/season and updates lighting separately. Add a
seasonal material identity to affected retained material caches, including
reflected terrain and static scene output. Reuse geometry, placements and world
queries. Rebuild the affected material view once on season change; ordinary
day/night updates should continue to reuse it. Continuous season fading can wait.
Ground, its surface patches, forest floors, mountain fringes and foliage must
consume the same policy so summer decals do not float over snowy ground.

There is also an existing game-side dependency to fix during implementation:
`patch_perform_interturn_in_main_loop` advances seasons only while
`day_night_cycle_img_state == IS_OK`, and suppresses its redraw if native seasonal
image reload fails. `patch_Map_Renderer_load_images` initializes that legacy art
path, which can display a missing-art error. Custom rendering should use the
existing C3X clock without requiring a complete native `DayNight/SEASON/HOUR` PCX
set. Preserve native seasonal-art handling and unrelated features with custom
rendering disabled. These are existing patch points; this investigation identifies
no new patch-table entry and changes no injected code.

Fall/spring ground and foliage have low implementation complexity. A coherent
winter pass has moderate complexity because it spans ground/decal/foliage
materials and retained cache invalidation. Snowy roofs and improvement surfaces
would be a subsequent coherence pass using generic material eligibility, not
four copies of every building or unit. Bare deciduous winter trees, snowfall,
frozen water and melting animation are unnecessary for the first version. Natural
wonders, constructed wonders and Districts keep their deferred contracts.

This design is expected to add a few material operations and occasional cache
refreshes, without extra draw calls for snow/fall or a seasonal animation loop.
That is a design expectation, not a GPU/FPS measurement. Snowy pine channel files
already occupy about 3.5 MiB in the base pack or 4.4 MiB in the alternate pack
(logical per-body totals, not unique GPU residency). Reuse them rather than creating
four complete packs. A production check must cover zoom/scroll/wrap, season
changes, reflected snow, day/night readability, opaque/masked casters, native PCX
independence, config-off behavior and return-to-Summer parity after the VM is free.

## Repeat and inspect

```sh
python3 Renderer/lab/studies/seasons/study.py
python3 Renderer/lab/studies/seasons/study.py --inventory-only
python3 Renderer/renderer.py check
```

The full study selects the bundled Pillow/NumPy Python when system Python lacks
them; `C3X_RENDERER_PYTHON` can override it. Set `C3X_CIV6_ASSETS` or pass
`--assets-root` for another installed Assets tree. Reports store relative paths.
There are no persistent extracted DDS files, cloned packs, builds or staging.

Outputs under `Renderer/lab/out/seasons/` are disposable:
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
rocky microcontrast beneath snow. Spring should use irregular harmonious meadow
flower clusters with fresh greenery and slightly richer pink, baby-blue and yellow
colors. Keep sparse coverage and small flower scale, avoiding evenly distributed
noisy dots or neon saturation.
These preferences guide subsequent material work; they do not approve or stage
an implementation or replace category references.
