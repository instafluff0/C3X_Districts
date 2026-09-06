# C3X Renderer Source Adapter Contract

## Purpose

Source adapters convert native or modded art libraries into interchangeable, source-agnostic C3X packs. The first adapter targets the installed Civ VI base assets. The same contract must support alternate Civ VI visual skins, permissioned mod sources, SDK/Pantry assets, and original community art without changing the runtime renderer.

## One-Command Goal

Once an adapter and mapping profile exist, a pack author should be able to run one command. A vanilla source tree and a visual-skin overlay are separate inputs because a Civ VI environment mod may replace only part of the base asset graph:

```powershell
py Renderer\tools\import_source.py `
  --adapter civ6 `
  --source-root "Z:\...\Civ6.app\Contents\Assets" `
  --variant vanilla `
  --mapping Renderer\mappings\civ3_conquests_to_civ6.txt `
  --output Renderer\packs\Civ6Vanilla

py Renderer\tools\import_source.py `
  --adapter civ6 `
  --source-root "Z:\...\Civ6.app\Contents\Assets" `
  --overlay-root "D:\Source Art\Civ5EnvironmentSkin" `
  --variant civ5-environment-skin `
  --baseline-pack Renderer\packs\Civ6Vanilla `
  --mapping Renderer\mappings\civ3_conquests_to_civ6.txt `
  --output Renderer\packs\Civ5EnvironmentSkin
```

The command performs discovery, overlay resolution, extraction, normalization, mapping, validation, preview generation, and provenance reporting. It never modifies either source tree. `--variant` is recorded provenance and selects adapter policy; it must not introduce variant-specific branches in the runtime renderer.

The two output directories are complete runtime pack roots. C3X selects between them by replacing only a renderer definition's `#Pack path`, while preserving the same pack ID, `#Asset` aliases, and `#Rule` mappings:

```text
#Pack
id   = world
path = mod:Renderer\packs\Civ6Vanilla

[A later scenario/custom layer can replace the complete section above with:]

#Pack
id   = world
path = mod:Renderer\packs\Civ5EnvironmentSkin
```

This is a build-time choice of source roots and output directory plus a runtime choice of pack path. Neither path should be selected by inspecting source-specific metadata at runtime.

## Adapter Boundary

A source adapter may understand source-specific concepts such as ArtDefs, XLP, BLP, CIVBIG, FGX, mod dependency files, or Blender source. Its output may contain only C3X pack concepts:

- Stable logical asset IDs.
- Normalized meshes and coordinate units.
- Materials and texture roles with color-space metadata.
- Environment curves, emissive channels, analytic lights, and ambient-effect attachments normalized into generic C3X records.
- Named animations and variants.
- Bounds, anchor hints, and calibration metadata.
- Provenance, source hashes, warnings, and redistribution policy.

No source-specific parsing is allowed in the runtime pack loader, rule resolver, visible-scene contract, or renderer.

For Civ VI, the adapter's lighting discovery begins with `GameLighting.artdef`, `lighting/default_lighting.blp`, `Light.blp`, `VFX_FireFX.blp`, `SHARED_DATA` effect/emissive textures, and named attachments in landmark/model packages. The confirmed paths and unresolved typed-record work are in [civ6_lighting_findings.md](civ6_lighting_findings.md). Names alone may seed candidates but cannot establish exact light parameters, transforms, sockets, or activation behavior.

## Stable Logical IDs

Compatible visual skins should emit the same logical IDs wherever they represent the same concept:

```text
terrain/grassland/base
terrain/grassland/hills
terrain/mountains/variant_02
feature/forest/temperate
resource/horses
city/asian/medium
unit/legionary
```

This allows a scenario mapping profile to switch from one skin to another by changing a `#Pack` path instead of rewriting every `#Rule`.

If a source uses different identifiers, its adapter owns an alias table from native identifiers to logical C3X IDs. Ambiguous aliases must be reported for author selection. Missing IDs remain absent and fall back through normal C3X renderer rules.

## Base And Alternate Skin Equivalence

An alternate Civ VI environment skin is treated as a separate source root and output directory. The importer should compare it with a chosen baseline pack and emit an equivalence report:

- Logical IDs present in both packs.
- IDs replaced by the alternate skin.
- IDs inherited from a permitted baseline, if inheritance is configured.
- IDs missing from the alternate source.
- Material, mesh, animation, and texture-role differences.
- Mapping ambiguities and unsupported source records.

Pack inheritance must be explicit. An alternate pack may declare a base pack and override selected logical IDs, but output must remain deterministic and the validator must detect inheritance cycles.

Overlay precedence must also be explicit and reproducible. For each resolved logical asset, provenance records whether the winning source was the baseline tree or overlay tree, the relative source record, hashes of all candidates, and why one candidate won. The importer must reject accidental writes to an existing pack produced from a different variant unless an explicit replace operation is requested.

The Civ V Environment Skin for Civ VI is a possible M6.8 alternate-skin candidate because it exercises terrain material roles, relief/height, water, decals, vegetation, scattering, fog-of-war source art, and lighting/color-policy differences. Its published Nexus permissions currently prohibit conversion and asset reuse without the author's permission. Therefore:

- A locally downloaded Steam Workshop copy was verified on 2026-09-03 at `Z:\Library\Application Support\Steam\steamapps\workshop\content\289070\1702339134`. Workshop app ID `289070` is Civilization VI and item ID `1702339134` identifies this candidate. Its `Civ_V_Art_Skin.modinfo` names it **Environment Skin: Sid Meier's Civilization**, credits Brian Busatti, and points to ArtDefs plus Windows/macOS cooked assets, XLPs, and textures. Treat this directory only as a possible local M6.8 source root to inspect; its presence does not establish conversion or redistribution permission.
- The adapter and synthetic overlay tests may be implemented without the mod.
- Screenshots may be retained locally as non-redistributed art-direction references with provenance.
- Committing or distributing a converted pack is gated on documented permission from the rights holder. An explicitly requested private experiment may use `--local-testing-only`; its output remains ignored and carries a non-distribution provenance notice.
- No derived asset pack, mod file, or copyrighted screenshot is committed or distributed by C3X without permission.

The implemented terrain vertical slice is `tools/import_source.py`. It stages output, refuses silent variant replacement, requires an overlay baseline, and emits `provenance/equivalence_report.json` through `tools/asset_compiler/pack_equivalence.py`. A private `--local-testing-only` build records a non-distribution notice; distributable conversion remains gated on a documented grant. See [alternate_skin_integration.md](alternate_skin_integration.md) for current source coverage, commands, and remaining water/clutter/lighting work.

## Mapping Profiles

Source conversion and Civ III mapping are related but distinct:

- The adapter answers: "What normalized assets exist in this source?"
- The mapping profile answers: "Which C3X asset should represent this Civ III metadata?"

The same normalized pack may be used by multiple scenarios with different mapping profiles. The same mapping profile may be reused by multiple equivalent visual skins that expose the required logical IDs.

Import tooling may suggest mappings by ArtDef name, filename, material role, model bounds, or source metadata. Suggestions are never silently promoted when multiple candidates exist.

Natural-wonder mapping is a dedicated M9 profile, not an extension of generic terrain aliases. The Civ VI adapter inventories Base/DLC `Features.artdef`, `TerrainStyle.artdef` NaturalWonders records, Clutter, VFX, water, Landmarks, and cooked dependencies, then emits generic kit candidates and source-footprint provenance. The mapping profile binds those candidates to effective C3X natural-wonder names. Civ VI multi-tile structure may guide visual composition but cannot create C3X gameplay tiles; see [natural_wonder_rendering.md](natural_wonder_rendering.md).

## Output Layout

```text
packs/<PackName>/
  manifest.json
  models/
  materials/
  textures/
  animations/
  previews/
  provenance/
    source_report.json
    equivalence_report.json
    unresolved_mappings.json
```

Generated copyrighted art remains ignored/local unless redistribution permission is established. Manifests, schemas, synthetic fixtures, and adapter code may be tracked independently.

## Validation Gates

An adapter/skin is usable only when:

- The pack manifest and every relative path pass pack-root safety validation.
- Every material texture role has compatible dimensions, format, and color-space metadata.
- Every normalized mesh has finite bounds, supported vertex data, and a declared coordinate convention.
- Stable logical IDs are unique and deterministic across repeated imports.
- Required mapping-profile IDs either resolve or produce explicit fallback diagnostics.
- Two imports from identical inputs produce equivalent manifests and source hashes.
- Representative assets render in source-independent previews.
- An alternate-skin equivalence report accounts for every logical ID in its declared scope.

Local integration checks may consume installed or mod-provided art. Committed tests use synthetic source trees and packs.

## Licensing And Provenance

Each source root must have an explicit provenance record. The importer records source location, source hashes, adapter version, mapping version, and an author-supplied redistribution classification. The tool must not infer permission from technical accessibility.

Civ VI base art and unlicensed derivatives remain local prototypes. Third-party mod sources may be converted locally when supplied by the user; redistribution of converted output requires the relevant author's permission.
