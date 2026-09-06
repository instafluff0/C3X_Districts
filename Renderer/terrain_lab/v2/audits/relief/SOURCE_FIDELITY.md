# Q4 source fidelity and explicit exclusions

No Q4 beauty candidate is accepted. The user rejected invented rock faces and
asked for direct Civ VI terrain art. Earlier generated cliff faces and analytic
hill formulas remain in rejected diagnostic images only. The current generator
uses the extracted selected-skin hill height field; its coast rock surfaces consist
entirely of actual normalized source meshes.

| Component | Current source and operation | Classification / limit |
| --- | --- | --- |
| Coastal rocks | `ShoreNormalized`: four `cliff_large` and two `cliff_small` meshes, indexed triangles, original UV0, base color, geometric normals and LEAN0. Uniform scale, yaw, translation and partial burial only. | `source_adaptation`; no generated exposed rock mesh. Gloss is inventoried/bound but its final material response is pending the shared evaluator. |
| Hills | Owned `fixtures/relief/selected-source/hills_height_lod0.dds`, extracted from the selected installed skin. The normalized pack had incorrectly inherited the base-game field. Continuous source sampling at .085 UV units per tile, with topology masks and placement amplitude. | `source_adaptation`; no analytic hill/noise shape or new texture. Terrain topology masks and calibration are C3X adaptation, not source-engine equivalence. |
| Mountains | Extracted standard variants, unmodified height samples, provisional `32 / 42.399979` coordinate ratio, uniform instance transforms, source UVs. `mtn_base` material with source summit thresholds; separate grassy shoulders. | **Diagnostic reconstruction, not accepted.** Exact source height data does not prove physical proportions or canonical appearance; these renders look too narrow and spiky. Not an imported conventional mountain model. The source HBLEND footprint is used as coverage; this use is inferred, not confirmed engine behavior. |
| Volcano | Normalized ordinary-volcano height and footprint channels from `TerrainElementsNormalized`. | **Diagnostic only.** Height samples are source-derived, but the initial witness reused mountain aspect calibration. It does not establish original volcano proportions or an accepted source body. Recover physical source extent and the dormant/active material mapping before selecting it. |
| Forest/jungle | Actual alternate-skin normalized leafy broadleaf, palm and understory meshes; source UVs/base color; uniform transforms. | `source_adaptation`. Source bounds reserve full crowns against Q3/Q5 corridors and Q7 city footprints. Shared cutout caster/receiver and final material evaluation remain pending. |
| Dunes | Direct source desert-hills material-height witness, after rejecting invented macro waves. Audited five sand decals have no conventional model container. | **Diagnostic only**, not a selected macro dune solution. The material-height witness lacks broad directional crests. Neither the normalized pack nor the five audited decal entries prove global absence of source dune geometry. |
| Marsh | Existing source audit and normalized material/decal inventory read. | No new accepted marsh render. Source atlas/channel and shared wet-ground integration remain open; no replacement vegetation or pool shapes are invented. |
| Base terrain/water | Source grass/desert textures over a Q4 diagnostic receiver; real cases use Q3 signed shore/carving. Water is explicitly flat diagnostic color. | `diagnostic_proxy`; never a Q2/Q3 beauty substitute. No waves, foam or animated/frozen surf. |

All per-instance source meshes, material documents, texture files, source hashes,
scales, yaws, translations and UV-preservation flags are recorded in each owned
fixture's `provenance.json`. The `normalized_conventional_mesh_available` flag
refers only to the inspected normalized input; it does not claim source-game
absence. Historical files with the earlier flag spelling are diagnostic records.

Source evidence consumed includes `HILL_AUDIT.md`, `MOUNTAIN_AUDIT.md`,
`SHORE_ASSET_AUDIT.md`, `DUNE_SOURCE_AUDIT.md`, `L12_VOLCANO_AUDIT.md`,
`CANONICAL_REFERENCE_AUDIT.md`, `terrain_relief_builder.py`,
`generic_terrain_element_compiler.py`, and the installed Expansion2
`TerrainStyle.artdef`. The extracted mountain height/footprint and both source
sand-decal channels were also inspected directly. The latter expose very
low-amplitude detail and atlas coverage; treating the green atlas mask as an
invented macro dune would not be justified.

The chosen Civ V Environment Skin for Civ VI remains selected. No textures,
models or terrain art have been created with image generation. All source art
and derived render payloads remain local and ignored, not redistributed.

## Selected-skin correction and mountain fidelity failure

`SELECTED_SOURCE_AUDIT.json` reads typed package entries from the installed
selected skin, not names in the normalized pack. Both hill LODs differ from the
normalized files. The skin hill scalar height_scale is 10; Base is 14. Revision
14+ uses the selected payload with that relative amplitude correction; sampling
scale and topology adaptation remain provisional. Earlier hill evidence is
superseded for source-fidelity acceptance. `import_selected_relief.py` reproduces
the audit and creates only ignored Q4-owned source data.

All 30 standard mountain channels (five variants, three channels, two LODs)
match the selected package exactly. Its typed element height_scale is 25,
while its RidgelineMountain ArtDef declares MountainHeight 32 and MountainWidth
42.399979. Their source-engine relationship and coordinate extent interpretation
are not recovered. The current ratio was copied from the normalized manifest;
it is not proof of source physical proportions. The canonical reference has
broad interlocked ridges, varied rock/snow materials, and connected shoulders.
Current isolated bodies are narrow spires with stretched, weak surface detail.
The current planar material coordinates are a reconstruction choice, not source
mesh UV evidence. No mountain beauty or rigid-source-body fidelity is claimed.

Revision 15 preserves real source cliff meshes but still exposes gaps and a
straight grassy coast face. This is a visible failure, not a source-faithful
finished coast. Do not promote it or any earlier diagnostic based on hashes.
