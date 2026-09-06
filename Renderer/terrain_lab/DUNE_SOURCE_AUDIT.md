# Desert Dune Source Audit

## Decision

L10 should reproduce the desert-wave look with connected procedural terrain geometry, not by waiting for a conventional dune mesh. The installed Civ VI base source defines a dedicated `DuneDesertHills` terrain-style layer with explicit shape controls and supplements it with five large sand decals carrying a height channel. Those are better evidence for the intended construction than a generic hill or mountain mesh.

The runtime result remains source-agnostic. These installed files are local research inputs only; no Firaxis asset or source-specific identifier becomes a C3X runtime dependency or redistributable requirement.

## Confirmed ArtDef evidence

`Base/ArtDefs/TerrainStyle.artdef` defines `DuneDesertHills/Default` and binds it to `ART_DEF_TERRAIN_MATERIAL_DESERT_HILLS` in `terrain/TerrainMaterialSet_Base`. Its authored controls are:

| Parameter | Value |
| --- | ---: |
| `DuneBase` | 0.0 |
| `DuneHeight` | 4.0 |
| `DuneWidth` | 4.0 |
| `DuneNoise` | 0.6 |
| `DuneAngle` | 0.300001 |

The same file wires that entry into `LayerSet_Default` as `DesertHillsLayer`. This confirms a separate desert-hills/dune construction path rather than ordinary `StandardHills` with a desert tint.

The normalized local terrain pack already contains both ordinary desert and desert-hills materials. Each provides base color, material height, and specular channels. The ordinary desert material also has the existing two-pixel continuous surface-detail relief. That material relief is suitable for fine sand texture, but it is too shallow and nondirectional to provide the broad dune silhouette by itself.

## Confirmed sand-decal evidence

`Base/ArtDefs/Clutter.artdef`, inside `CLUTTER_DESERT`, places these five sand entries from `environment/clutter.blp`:

| ArtDef entry | Package entry | Scale | Count | Scale variation |
| --- | --- | ---: | ---: | ---: |
| `Desert Sand_Decal01` | `TER_Desert_Decal10` | 5.5 | 3 | 0.15 |
| `Sand_Decal02` | `TER_Desert_Decal11` | 5.5 | 3 | 0.15 |
| `Desert Sand_Decal02` | `TER_Desert_Decal12` | 5.0 | 2 | 0.0 |
| `Desert Sand_Decal04` | `TER_Desert_Decal13` | 2.5 | 2 | 0.0 |
| `Desert Sand_Decal05` | `TER_Desert_Decal14` | 2.5 | 2 | 0.0 |

All five are enabled as decals, use priority 3, rotate around Z, and may overlap. A structural read of their landmark records confirms that each has exactly one `DecalDesc2`, an empty `TerrainEditDesc3` vector, and no `FGXModel::ContainerDesc::Model`, mesh, primitive-group, or material container. Therefore these are decals, not hidden static dune meshes or terrain-edit bodies.

All five decal descriptions select the same package textures:

- `TEXTURE_TER_Desert_Decal_B` — `Decal_BaseColor`
- `TEXTURE_TER_Desert_Decal_H` — `Decal_Heightmap`
- `TEXTURE_TER_Desert_Decal_FOW` — `Decal_FOWColor`

The five descriptions have different local bounds, so they provide differently framed/stretched applications of the shared sand texture set. The package evidence does not establish that the height decal changes world geometry; it establishes a height-bearing decal input used by the rendering path.

## Additional useful source

The installed Nubia scenario references `ART_DEF_TERRAIN_MATERIAL_DESERT_ROUGH`. Its base and height textures are a useful optional comparison for rough or wind-scoured sand, but it is scenario-specific and is not required for the baseline L10 design.

No standalone asset named as a dune terrain-element mesh or height/blend field was found in the audited base ArtDefs or terrain packages. The explicit `DuneDesertHills` controls are the stronger source signal.

## Confirmed versus inferred

Confirmed from package and ArtDef data:

- the dedicated dune layer name and its five numeric controls;
- the desert-hills material binding and default-layer connection;
- the five sand-decal placements and their scale/count/rotation policy;
- one height-bearing decal description per asset and no conventional model container;
- the shared base-color, height, and fog texture selections.

Inferred engine behavior:

- Civ VI most likely synthesizes the broad dune form procedurally from the five `Dune*` controls, then applies the desert-hills material and sand decals;
- the exact engine formula and the precise visual use of the decal height channel are not exposed by ArtDef/package data and must not be reported as confirmed.

## L10 implementation contract

1. Generate one viewport-continuous height field in world/tile coordinates so dune crests cross Civ III tile boundaries without restarting.
2. Treat `DuneHeight`, `DuneWidth`, `DuneNoise`, and `DuneAngle` as calibrated source evidence, not necessarily literal C3X pixel values. Preserve their relationships while tuning the final amplitude to Civ III readability.
3. Separate broad directional dune geometry from fine material-height ripples.
4. Normalize and test the sand decal base/height pair as optional local breakup. Do not use isolated decal quads as the macro dune silhouette.
5. Include flat desert, dune desert, desert hills, desert mountains, plains transition, and coast transition in the contiguous 96-tile promotion scene.
6. Provide complete, no-dunes, dunes-only, and reduced-readability views with deterministic inputs and output hashes.
7. L11 remains blocked until the user explicitly approves the L10 promotion render.

## Current lab integration

The active L10 candidate follows this contract. A single world-space height
function uses the confirmed angle, width, height, and noise relationships to
drive one connected 4x4 dune field; its phase is bent strongly at broad and
fine scales so crests form long S-curves instead of straight parallel ridges. The
normalized desert-hills base/height/specular material supplies dune-surface
response, and the shared base-color and height channels used by all five
normalized sand-decal variants provide overlapping local breakup. The macro
silhouette does not come from decal quads.

The same 96-tile scene preserves separate flat-desert and authored desert-hill
cells and adds one cell-local mountain using the normalized desert mountain
base and stripe materials. L10 also moves the promotion shoreline back toward
the x=10 Civ III ownership line while retaining medium- and fine-scale bays,
points, beach-width variation, and the accepted water/surf stack.

## Repeatable local checks

The source installation used for this audit is the macOS Civ VI bundle under `Library/Application Support/Steam/steamapps/common/Sid Meier's Civilization VI/Civ6.app/Contents/Assets`.

From the repository root, the authored dune controls can be rediscovered with:

```sh
rg -n -C 12 'DuneDesertHills|DuneBase|DuneHeight|DuneWidth|DuneNoise|DuneAngle' "$HOME/Library/Application Support/Steam/steamapps/common/Sid Meier's Civilization VI/Civ6.app/Contents/Assets/Base/ArtDefs/TerrainStyle.artdef"
```

The sand placements can be rediscovered with:

```sh
rg -n -C 12 'Desert Sand|Sand_Decal02|TER_Desert_Decal1[0-4]' "$HOME/Library/Application Support/Steam/steamapps/common/Sid Meier's Civilization VI/Civ6.app/Contents/Assets/Base/ArtDefs/Clutter.artdef"
```

Run the complete read-only inventory, including the structural BLP checks, with:

```sh
python3 Renderer/tools/asset_compiler/dune_source_probe.py --output Renderer/preview/out/dunes/source_audit.json
```

The probe uses the fail-closed `StaticPackage`, `landmark_base_model`, and texture-table decoders already exercised by the asset-compiler tests. `test_dune_source_probe.py` separately locks down the ArtDef selection logic. Any future dune importer must add fixture tests for normalized outputs before those outputs can satisfy L10.
