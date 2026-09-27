# Civ VI city-art eras and C3X review labels

The installed Civ VI `Base/ArtDefs/Eras.artdef` maps gameplay eras to a smaller
set of art tiers. Its `LandmarkArtEra` field assigns Classical, Medieval, and
Renaissance to **`ARTERA_CLASSICAL`**. Ancient uses `ARTERA_ANCIENT`, Industrial
uses `ARTERA_INDUSTRIAL`, and Modern, Atomic, and Information use
`ARTERA_MODERN`. The Expansion 2 files add future-era art records.

| Civ VI city-art tag in installed `CityGenerators*.artdef` | Culture tags | Distinct city-component sets |
| --- | ---: | ---: |
| `ARTERA_ANCIENT` | 7 | 6 |
| `ARTERA_CLASSICAL` | 23 | 22 |
| `ARTERA_INDUSTRIAL` | 3 | 3 |
| `ARTERA_MODERN` | 2 | 2 |
| `ARTERA_FUTURE` | 3 | 1 |
| `DEFAULT` (unspecified art era) | 8 | 8 |

These counts come from the installed Base and DLC `CityGenerators*.artdef`
`GeneratorBlockList` records, grouped by `Tag_Era` and `Tag_Culture` and with
duplicate package/entry bindings removed. A culture tag is a source-art
selector, not a Civ III culture group or a guarantee of a unique model set.
For example, the `NorthAfrican` and `Nubian` `ARTERA_CLASSICAL` records name
the same city components; the `AncientEarth` and `CIVILIZATION_GAUL` ancient
records likewise match. Palaces can still differ.

## Full city-building flavor matrix

Each number is the count of distinct `(BLP package, CityBuildings entry)`
references for that exact `Tag_Culture × Tag_Era` pair in the installed
ArtDefs. A dash means no tagged city-building pool. Counts describe source
bindings, not the number of visually distinct meshes or validated imported
components. The `DEFAULT` **culture** has entries at five explicit art tiers;
the last column instead denotes literal `Tag_Era=DEFAULT` records, which have
no chronological assignment in this inventory.

| Source culture/flavor | Ancient | Classical | Industrial | Modern | Future | Era unspecified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| America | — | 31 | — | — | — | — |
| AncientEarth | 33 | — | — | — | — | — |
| AncientWood | 37 | — | — | — | — | — |
| Baltic | — | 26 | — | — | — | — |
| Brazil | — | 28 | — | — | — | — |
| Colonial | — | — | 24 | — | — | — |
| DEFAULT | 34 | 23 | 33 | 27 | 23 | — |
| EastAsian | — | 31 | — | — | — | — |
| Indonesian | — | 30 | — | — | — | — |
| Mediterranean | — | 26 | — | — | — | — |
| ModernGlass | — | — | — | 25 | — | — |
| Mughal | — | 37 | — | — | — | — |
| NorthAfrican | — | 27 | — | — | — | — |
| Nubian | — | 27 | — | — | — | — |
| RowHouse | — | — | 32 | — | — | — |
| Scottish | — | 28 | — | — | — | — |
| SouthAfrican | — | 42 | — | — | — | — |
| SouthAmerican | — | 38 | — | — | — | — |
| SoutheastAsian | — | 31 | — | — | — | — |
| Vikings | — | — | — | — | — | 37 |
| CIVILIZATION_BABYLON_STK | 12 | 27 | — | — | — | — |
| CIVILIZATION_CAHOKIA | — | — | — | — | 23 | 35 |
| CIVILIZATION_CREE | 35 | 35 | — | — | — | 20 |
| CIVILIZATION_ETHIOPIA | — | 27 | — | — | — | — |
| CIVILIZATION_GAUL | 33 | 33 | — | — | — | — |
| CIVILIZATION_KOREA | — | 33 | — | — | — | 18 |
| CIVILIZATION_MAORI | — | — | — | — | 23 | 28 |
| CIVILIZATION_MAPUCHE | 35 | 35 | — | — | — | 20 |
| CIVILIZATION_MAYA | — | 38 | — | — | — | — |
| CIVILIZATION_PORTUGAL | — | 28 | — | — | — | 15 |
| CIVILIZATION_VIETNAM | — | 33 | — | — | — | 15 |

The Classical column is Civ VI's shared visual tier for its Classical,
Medieval, and Renaissance gameplay eras. The Modern column serves Modern,
Atomic, and Information. The installed Expansion 2 `Eras.artdef` contains a
Future art tier but leaves the `ERA_FUTURE` `LandmarkArtEra` reference empty;
this inventory does not infer its exact gameplay selection rule.

The [46-pair source-art review](../../out/cities/all-era-source-auditions/review/README.md)
gives every populated cell in this table a separate pink Town/City/Metro sheet
with Base, Walls, Capital, and Both states. Its wall kits and palace fallbacks
are provisional comparison aids. All selections happen in Lab preparation;
the game renderer receives generic composition metadata.
The [wall source audit](wall_source_audit.md) records why Industrial and Modern
review rows currently share Civ VI's Renaissance Star Fort geometry.

The source graph has separate jobs:

1. `Cultures.artdef` relates civilizations to art cultures.
2. `Eras.artdef` maps gameplay eras to art tiers.
3. `CityGenerators*.artdef` chooses city-building entries by culture and art
   tier, and describes growth and optional era mixing.
4. `CityBuildings` entries in BLP packages supply the meshes and materials.
   Palaces and walls are separate source assets in the C3X study.

The Base city generator's *authored settings* for `ARTERA_CLASSICAL` place a
Classical layer toward the center and an Ancient layer farther out, each with
weight 1.0. This records source selection parameters; it does not establish
the exact source engine's placement behavior. The current C3X review follows
the user's one-era-per-city preference and composes each candidate from one
art tier rather than reproducing that mixed-era source distribution.

Our offline `city_render_strategy.json` maps Civ III Ancient to Civ VI
`ARTERA_ANCIENT`, Civ III Middle Ages to `ARTERA_CLASSICAL`, Industrial to
`ARTERA_INDUSTRIAL`, and Modern to `ARTERA_MODERN`. This is a **Civ III target
mapping**, not a claim that every `ARTERA_CLASSICAL` building is historically
medieval. The 23-family gallery should therefore be read as "Civ VI Classical
art tier, auditioned for Civ III Middle Ages." Its 23 source tags comprise 22
distinct city-component sets, with five hand-curated C3X layouts and 18
initial automatic layouts.

Source evidence: installed `Base/ArtDefs/Eras.artdef`,
`Base/ArtDefs/CityGenerators.artdef`, the relevant `DLC/*/ArtDefs/CityGenerators.artdef`
files, and the executable parsers in `Renderer/tools/asset_compiler/`.
