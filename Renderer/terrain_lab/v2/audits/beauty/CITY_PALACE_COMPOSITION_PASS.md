# Single-era capital composition

The broader 47-root palace library now supplies an East Asian palace for the
Asian medieval city and an ancient-brick palace for the ancient-brick house
family. These are explicit Lab style choices, with generic asset IDs. They do
not infer capital status or production civilization mapping from source tags.
The user's single-era preference remains in force. Existing American capital
evidence is retained; this does not certify all 47 styles.

[Gameplay-size growth](out/city-palace-composition-r2/capital-growth-native.png)
shows the selected East Asian capital at 8, 16 and 24 houses (r93/r94/r92),
day above night. The palace stays fixed, and every earlier house keeps its
exact asset, scale, rotation and position. The .6-tile palace span and original
house scale are uniform in all axes. Its tiered front now belongs to the
neighborhood instead of standing across an empty gap.

[Placement diagnosis](out/city-palace-composition-r1/staged-capital-initial.png)
compares the detached small palace r88 with r93, and earlier large r86 with r92.
The matched full-light small comparison against r82 changes 5,796 daytime and
9,742 nighttime pixels above 2/255. Outside the city/shadow/glow region the
images are exact. This measured change is a gain because the visible house
groups join the palace and each other at the smallest stage, not because a
smaller bounding box or a test passed.

[Palace-off/on control](out/city-palace-composition-r2/palace-off-on-native.png)
retains all sixteen houses, their reserved palace space and camera. Adding the
palace changes 1,464 daytime pixels at `[767,451,821,495]` and 3,745 nighttime
pixels at `[727,450,820,511]` above 2/255. Its courtyard, stepped roof silhouette
and illuminated windows are readable. Outside the recorded city region pixels
are exact. The on/off images isolate visibility, not a claim that an empty
reserve is a plausible ordinary-city layout.

The canonical `Renderer/canonical/nightlights.jpg` has clearer window grids,
richer facade variation and more deliberate open-ground composition. The new
attached landmark and visible night windows move toward that reference, but
its modern architecture, zoom and mixed historical buildings are not copied
into the single-era city. The reference is not a matched-camera numerical
target. Source material/environment response and less repetitive neighborhoods
remain larger visual gaps. No new water-reflection improvement is claimed;
the earlier American capital/lake control remains the reflection witness.

## Source intake and placement changes

`qa/city_source_selection.py` allows the existing normal/material probes to
select an already converted generic palace root. It reads the verified palace
normalization strategy (768 source units per tile, versus 100 for the house
study). It does not rebuild or copy the library. Complete geometry fingerprints,
including UV0/UV1/UV2 and topology, match the existing normalized meshes.

`fixtures/beauty/city-palace-materials-r1/` preserves source packed normal and
tangent frames for eight East Asian meshes/nine primitives and seven ancient
meshes/seven primitives. Source opacity contributes two East Asian bindings
sharing one texture and one ancient binding. BC4 opacity is transported as BC3
alpha with the verified exact-coverage adapter. Existing base/AO/emissive UV
bindings, source courtyard geometry and house-ground textures are retained.
The source static-frame format is verified; full source BRDF/environment
behavior is still partial. Local facade/ground light spill is an authored
approximation with generic building-box occlusion, not recovered engine logic.

The generic growth solver now treats fixed landmarks and house connectivity
separately. A wide palace courtyard cannot bridge disconnected house groups.
Each finished growth prefix must connect on its own and share meaningful
frontage with the landmark; merely touching a courtyard corner is insufficient.
Staged search completes the earlier prefix before assigning later bodies.
Candidate ranking includes an authored projected palace-visibility cost.

Palace centers use a .35-tile envelope around the dry-land centroid of the
allowed footprint, rather than always centering on the tile anchor. On the
main coast this keeps the landmark within the usable neighborhood instead of
pushing it against the water. The final inland layout-only r100 is exactly
equal to the rendered r92 plan, so no redundant render was made. r93/r94/r92
were generated before the centroid rule and retain that historical metadata.
The frontage-only r91 was exactly equal to r86 and is not counted as a gain.

These are generic authored placement rules, not recovered Civ VI district or
road layout metadata. Connecting roads remain deferred. Buildings cannot
shrink, cover the rendered river bank, clear forests or move terrain to force a
fit. Production capital visibility still requires authoritative Civ III
`is_capital`; modder style/culture/default fallback and native capital indicators
remain as documented in `Renderer/docs/city_palace_asset_import.md`.

## Coverage and limitations

| Case | Result |
| --- | --- |
| r93/r94/r92, inland East Asian 8/16/24 + palace | Selected local composition; exact growth prefixes and fixed palace |
| r95, matched sixteen-house palace-off control | Houses, reserve and frame exact |
| r77, inland ancient-brick 16 + palace | Additional style coverage; predates final planner, independently passes prefix/frontage checks |
| r98, main coast ancient-brick 16 + palace | Selected coastal coverage; 7,667 search nodes across ten sites |
| r99, separate coastal holdout | Unresolved; six separated sites, 6,259 nodes, no fitted capital |

[Main coastal day/night](out/city-palace-composition-r2/coastal-capital-native.png).
All cases retain 100-tile `test.biq` terrain windows and 1360x800 gameplay
outputs. Cities are explicit Lab augmentation, not BIQ-captured city records.
The separate capital-untuned holdout uses `freshwater`, BIQ origin `[52,30]`,
anchor `[8,3]`. Its ordinary r73 city remains preserved. r78/r81/r96/r99 capital
failures demonstrate that success on the main coast does not generalize yet.
Neither a bounded search failure nor exhausting separated sites proves
geometric infeasibility. Do not increase budgets repeatedly without improving
the model of connected usable land.

Early capital trials r74/r75/r80/r82/r83/r84 were unsuitable at smaller stages;
r86/r88/r89 still left the palace visually detached. Those images and failed
trials remain diagnostic evidence, not selected results. Tile-anchor-centered
coastal r97 failed; dry-land-centered r98 succeeded without changing terrain,
source scale or object count. Full trial inputs remain under their revision
folders; no old augmentation metadata was rewritten.

Independent checks find no body overlap, forest-margin samples or rendered
river-bank intersection in the six selected/control scenes. Dense body-interior
shore samples remain dry at the .02 clipping boundary. Six r98 samples are
closer than the planner's nominal .05 five-point shore margin (distances
-.0367 to -.0486); this is a documented approximation, not proof of exact
continuous clearance. Source paving crossing the shore is clipped to dry
terrain by the existing ground path.

Twelve Windows/Metal comparisons pass, along with twelve terrain-preservation,
twelve light-buffer and twelve fixed-shadow-frame checks. Fourteen growth,
four exclusion and five source-frame/opacity tests pass. The six buffered
cases use 29/57/84/53/36/36 local light proxies respectively. Engineering
checks support the pixels; all human visual, native and milestone gates remain
open. This work changes no native/injected code or frozen integration handoff.

Recheck saved evidence with Python providing NumPy/Pillow:

```sh
python3 Renderer/terrain_lab/v2/qa/city_palace_composition_evidence.py
python3 -m unittest discover -s Renderer/terrain_lab/v2/qa -p test_city_growth_layout.py
python3 -m unittest discover -s Renderer/terrain_lab/v2/qa -p test_city_exclusion.py
python3 -m unittest Renderer.tools.asset_compiler.test_packed_static_frame Renderer.tools.asset_compiler.test_opacity_coverage
```

`CITY_PALACE_COMPOSITION_EVIDENCE.json` records hashes, exact growth/control
checks, packet composition, pixel bounds and preserved failed searches.
`CITY_PALACE_COMPOSITION_CLEANUP.json` records removal of completed new linear
readbacks only. Source packets, textures, BMP/PNG frames, previous bests and
rejected trials remain available. Continue broader single-era style/material
coverage and connected usable-land reasoning for the unresolved holdout;
do not mistake this checkpoint for completion of the city quality goal.
