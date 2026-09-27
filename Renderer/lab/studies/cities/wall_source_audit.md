# Civ VI wall source audit

The installed Base and DLC `Walls.artdef` files expose **three ordinary city
fortification sets**: `BUILDING_WALLS` (Ancient), `BUILDING_CASTLE` (Medieval),
and `BUILDING_STAR_FORT` (Renaissance). They are building stages, not a
wall-style matrix keyed to the city's art era or culture. The C3X Lab's
`industrial` kit names its *Civ III target era*; its source geometry is the
Renaissance Star Fort. The Modern Lab sheet currently reuses that kit. Civ VI's
Modern `TECH_STEEL` grants Urban Defenses via a gameplay modifier, with no
corresponding ordinary `WallSet` or `BUILDING_*` city-wall mesh in the installed
ArtDefs. This explains the old-fashioned industrial and modern perimeters in
the magenta sheets.

| Source set | Source gameplay use | Lab intake | City-wall interpretation |
| --- | --- | --- | --- |
| Ancient Walls | `BUILDING_WALLS` | Complete, 6 meshes | Standard |
| Castle | `BUILDING_CASTLE` | Complete, 7 meshes | Standard |
| Star Fort | `BUILDING_STAR_FORT` | Complete, 7 meshes | Standard Renaissance style |
| Tsikhe | `BUILDING_TSIKHE` | Complete, 6 meshes | Georgian unique wall; candidate |
| Pirates scenario castle | `BUILDING_CASTLE` | 2 unique meshes plus inherited standard gate/towers | Scenario variant; candidate |
| Ancient/Modern Tower Defense | Zombie Defense improvements | 5 representative Modern meshes imported, plus 5 spike-free derivatives; other source pieces inventoried | Improvement barricades, not ordinary city walls |
| Great Wall | Tile improvement | Inventoried | Linear improvement, not a city perimeter |
| Flood Barrier | Coastal flood protection | Inventoried | Coastline structure, not a city perimeter |
| Aqueducts/Baths, park/outback/hacienda fences, Moai, Civ Royale garbage barrier | Districts, improvements, scenarios | Inventoried | Different footprint or context |

The seven-kit [visual comparison](../../out/cities/wall-source-audit/wall-kit-comparison.png)
holds one modern C3X city layout fixed and swaps only the wall kit. It is a
software Lab preview on flat magenta, not an in-game `test.biq` capture. The
Modern Tower Defense source has no gate; its audition closes the ring with a
normal segment. Its towers and heavy wall look too imposing for a default
industrial/modern city style. No new kit has been selected for game use.

The [original versus spike-free comparison](../../out/cities/wall-source-audit/modern-tower-defense-spikes.png)
removes the X-shaped exterior obstacles while preserving the masonry and
towers. This is an offline mesh edit selected by a dedicated UV island in the
source atlas. The importer checks the exact removed triangle counts for all
normal and pillaged meshes, and leaves source texture/material bindings intact.
The source-backed original and the `modern_clean` derivative remain separate
Lab kits; neither is assigned to a Civ III era.

The reproducible inventory is
`python3 -m Renderer.tools.asset_compiler.wall_source_audit`; its ignored JSON
output lists every named `WallSet` and `TowerSet`, all referenced entries, all
eight installed wall BLP packages, and the entries not in the current city
intake. This installation has 8 `Walls.artdef` files, 16 named wall sets, and
22 tower sets. The local city-adjunct intake now contains 33 source meshes,
including the otherwise unreferenced `CityWalls_ANC_lg_tower` found in the Base
wall BLP. The raw unimported-entry list includes non-city structures and
source alternates; it is not a count of missing ordinary city-wall styles.

Five Modern Tower Defense tower/end variants remain unnormalized because their
source compound records reference FX attachment names absent from their
skeletons. The two imported tower variants and all wall segments suffice to
review the style. The named source entries remain in the audit report; no
missing geometry is silently substituted into a standard wall kit.

Source evidence: installed `Base/ArtDefs/Walls.artdef`,
`DLC/Expansion1/ArtDefs/Walls.artdef`, `DLC/PiratesScenario/ArtDefs/Walls.artdef`,
`DLC/Portugal/ArtDefs/Walls.artdef`, the other four installed DLC
`Walls.artdef` files, `Base/Assets/Gameplay/Data/Buildings.xml`,
`Base/Assets/Gameplay/Data/Technologies.xml`, and the BLP packages listed in
the generated audit. The city's use of these optional sets in C3X would be a
new art decision, not a claim about Civ VI's ordinary city rendering.
