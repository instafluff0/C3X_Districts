# Resource asset conversion

Status: the Base-game source inventory is complete for the 26 Civ III mappings. All static resources resolve through normalized local assets or explicit generic pack routes, and all 24 mapped-resource ambient clips have been fetched and converted to the generic `.c3anim` format. Fish and whale retain their validated skinned paths; the 24 animated clutter bodies now also compile into generic skinned resource subjects with 22 unique clips and model-aware pose caches. Native/GPU resource ownership remains separate L16/I16 work.

## Inventory and build

Inventory the art, stage every mapped embedded raw clip, extract the tile-base skins, and build the static pack on macOS:

```sh
python3 Renderer/tools/asset_compiler/resource_pack_builder.py \
  --extract-landmark-animations --extract-landmark-skins --build-static-pack
```

Convert all staged clips on the Windows conversion VM:

```bat
Renderer\tools\asset_compiler\CONVERT_RESOURCE_ANIMATIONS.bat
```

The batch reads the extraction report, applies the package-specific translation scale (`1/12` for environment clutter and `1/100` for tile bases), and rebuilds the converter once. Validate all normalized outputs on either platform with:

```sh
python3 Renderer/tools/asset_compiler/resource_animation_converter.py --validate-only
```

Compile the 24 animated clutter bodies and bind their already-normalized clips
to deterministic per-model pose caches on either platform:

```sh
python3 Renderer/tools/asset_compiler/resource_body_profile_compiler.py
```

This produces the ignored `ResourceAnimatedLab` proof pack. Runtime identifiers
and paths are generic; source entry names and pointer evidence remain only in
the ignored build report. The compiler also selects the animal model from the
two-model elephant compounds, omitting their ancillary grass model as required
by `single_primary_subject`. One shipped fox curve contains an impossible
`1.73e18` position sentinel; the 19 interpolated samples contaminated by that
one value replace only the affected position channel with the model's authored
rest channel, and the repair count is retained in source evidence.

Direct curves are useful evidence, but Fish requires Granny's model-aware binding semantics. Build minimal sampling companions from the normalized skeletons on macOS, import them offline, and bake source-independent world-matrix caches on Windows:

```sh
python3 Renderer/tools/asset_compiler/normalized_skeleton_to_cn6.py \
  Renderer/packs/ResourceNormalized/skeletons/resources/fish.json \
  Renderer/preview/out/resources/fish_sampling_companion.cn6
python3 Renderer/tools/asset_compiler/normalized_skeleton_to_cn6.py \
  Renderer/packs/ResourceNormalized/skeletons/resources/whales.json \
  Renderer/preview/out/resources/whales_sampling_companion.cn6
```

```bat
Renderer\tools\asset_compiler\IMPORT_CN6_MODEL.bat fish_sampling_companion.cn6 fish_sampling_companion.fgx
Renderer\tools\asset_compiler\CONVERT_CIV6_MODEL_POSE.bat res_fish_anim.fgx fish_sampling_companion.fgx fish_ambient.c3pose 0.01
Renderer\tools\asset_compiler\IMPORT_CN6_MODEL.bat whales_sampling_companion.cn6 whales_sampling_companion.fgx
Renderer\tools\asset_compiler\CONVERT_CIV6_MODEL_POSE.bat whale_idle.fgx whales_sampling_companion.fgx whale_idle.c3pose 0.01
```

Then build the static pack and register any converted clips that are present:

```sh
python3 Renderer/tools/asset_compiler/resource_pack_builder.py --build-static-pack
```

The inventory follows each mapped `Resources.artdef` or `Features.artdef` entry through its clutter or landmark references. It currently resolves 366 authored placements and 138 unique assets: 135 from `environment/clutter.blp`, plus fish, whales, and oasis from `landmarks/tilebases.blp`.

Large BLPs use `indexed_static_package.py`, which retains the structural reader's validation while caching reflected type resolution. On the current `tilebases.blp`, package parsing falls from roughly 79 seconds to about 2 seconds.

The generated `ResourceNormalized` pack contains:

- 105 normalized static resource bodies, with meshes, material records, DDS textures, explicit clamp/wrap addressing, and optional auxiliary maps;
- normalized fish and whale skins with direct skeleton indices, four weights per vertex, rest transforms, and inverse-bind matrices;
- source-independent placement sets for all 26 resource IDs, with all 366 clutter placements resolved locally or through explicit generic pack references;
- generic presentation contracts from `inventory/resource_presentation_profiles.json`; horses, cattle, game/deer, furs/foxes, ivory/elephants, fish, and whales select exactly one primary subject at the Civ III anchor while retaining the upstream cluster as optional source composition data;
- five uranium rock materials with the confirmed `Generic_Emissive` texture normalized as a night-activated emissive mask;
- `fish_ambient.c3anim` and `whale_idle.c3anim` direct-curve registrations plus validated `.c3pose` caches;
- 22 additional normalized clutter clips for horses, cattle, deer, foxes, elephants, banana, cocoa, and wheat, now proven against 24 normalized animated bodies in the separate `ResourceAnimatedLab` pack;
- an evidence report listing every direct, specialized-route, and rejected source entry; the current rejected count is zero.

Generated Firaxis-derived files remain ignored and local under `Renderer/packs/ResourceNormalized`. The tracked importer, formats, tests, and documentation contain no Firaxis art.

## Density and presentation

Upstream ArtDef `Count`, `MinCount`, and placement lists are source composition hints, not mandatory C3X density. The compiled resource record keeps those placements for provenance and optional rich packs, but its presentation profile is authoritative for rendering. `single_primary_subject` contains a normalized candidate list, `subject_count: 1`, the Civ III resource anchor, deterministic variant/yaw selection, an absolute-time animation phase with stable per-instance offset, and `ancillary_policy: omit`.

The renderer therefore needs no horse/cattle/deer name checks. A scenario or pack can bind any arbitrary resource ID to `single_primary_subject`, `source_authored_cluster`, or a future small-group profile. Stable instance seeds prevent animals from changing identity or angle during redraws, while phase offsets keep adjacent animated resources from moving in lockstep.

## Animation extraction

Both `environment/clutter.blp` and `landmarks/tilebases.blp` expose typed `BLP::AnimationEntry` tables. Model behavior descriptors reference clips by table index. The importer validates descriptor indices, animation name hashes, and Granny payload magic, and deduplicates shared table entries before staging raw clips.

| Landmark | Animation table index | Raw bytes | Normalized result |
| --- | ---: | ---: | --- |
| `RES_Fish` | 62 | 188,452 | 501 frames, 28 tracks |
| `RES_Whale_01` | 69 | 22,164 | 401 frames, 17 tracks |
| `FEATURE_Oasis_OB` | — | — | static; no behavior clip |

The environment/clutter sweep adds 22 unique clips across 24 animated assets: horse (1), banana (3), cattle (4), cocoa (2), deer (3), elephant (2), fox (3), and wheat (4). Together with Fish and Whale this is 24 unique clips, 6,663 normalized frames, and 373 transform tracks covering ten mapped resource IDs. Conversion and per-body cache binding are complete; L16 still owns visual selection, scale, grounding, density, and admission of animation.

The raw payloads use Granny's native magic rather than the standalone `CIVBIG` prefix. `export_civ6_animation.cs` accepts both forms and emits the same `.c3anim` contract. Tile-base translations are normalized with a `0.01` scale.

## Skin and skeleton contract

The two animated landmarks use the same proven 32-byte vertex profile: half-float position and UV, four compact palette joints, four normalized byte weights, and packed source normals. The importer resolves the source palette immediately and stores direct skeleton indices; runtime code never sees a Firaxis bone palette.

| Asset | Vertices | Triangles | Skeleton bones | Weighted bones | Rest-pose maximum vertex error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fish | 1,068 | 720 | 28 | 24 | `1.63e-7` tiles |
| Whales | 924 | 1,326 | 17 | 12 | `2.63e-8` tiles |

`normalized_skin.py` strictly validates skeleton hierarchy, finite transforms, inverse-bind shape, topology, joint ranges, and weight sums. Its dependency-free CPU proof reconstructs both bind meshes within the tolerances above. It also binds animation tracks by bone name; both assets have zero missing weighted bones and zero unknown tracks. A sampled whale pose produces a stable, plausible animated extent of roughly 0.215 by 0.203 by 0.067 tiles.

Whale's converted curves produce stable, plausible CPU-skinned poses and a deterministic five-frame contact sheet. Its model-aware cache produces the exact same contact-sheet hash, independently validating the cache reader and skinning path.

Fish exposed why the additional cache is necessary: direct per-curve composition stretches geometry early and collapses bodies later. The source package's six timeline records are zero-filled placeholders, so no missing visibility curve explains that result. Sampling the same clip against the reconstructed 28-bone model through Granny's `SampleBone` API yields a stable moving school. `fish_ambient.c3pose` stores those 501 model-aware frames in 898,508 bytes; `whale_idle.c3pose` stores 401 frames in 436,712 bytes. Both caches retain ordered bone names, normalized tile-unit world matrices, timing, and no source paths or Firaxis structures.

## Static coverage and routing

The current one-model static path accepts 105 of 135 clutter assets. It preserves all finite half-float UV coordinates and records `clamp` or `wrap` addressing instead of rejecting intentional tiling. Base color remains mandatory, while absent gloss and LEAN pairs are represented as absent optional material channels. This unlocked 36 additional distinct bodies: all 29 wrapping-UV cases and all 12 optional-map cases, with five assets belonging to both categories.

Deterministic generic previews verify representative wrapped boulder, base-color-only wheat, and no-gloss incense assets.

The 30 entries that are not valid one-model assets are now routed to specialized generic packs:

- `CompoundLandmarksNormalized` contains both elephant variants, all ten silk/snow-silk multi-model variants, and the two skeleton-carried snow-boulder decals;
- `DecalsNormalized` contains all sixteen resource surface-decal entries, including Oil's four two-descriptor compound decals.

The resource catalog records those two source-independent pack dependencies and explicit `(pack, asset)` placement references. All 135 unique clutter assets and all 366 authored placement records now resolve with zero unsupported entries. Oasis now resolves through an explicit compound-landmark route, so all 26 resources report complete static coverage. Fish and Whales have complete geometry/skeleton/material/clip binding and validated animated pose caches. These are prepared assets only: resource rendering is intentionally held for its authorized lab/integration gates.

The standalone animated-resource preview at `Renderer/preview/render_skinned_resource.py` is the acceptance test that exposed direct Fish sampling, validated the model-aware replacement, and proved deterministic Fish and Whale output. Asset conversion and storage are now ready for Terrain Lab integration when its milestone authorizes native skinning. The same skeleton plus pose-cache contract is also suitable for units and other animated models.
