# Resource/improvement ArtDef graph resolution

## Outcome

`Renderer/tools/asset_compiler/artdef_graph_resolver.py` resolves the installed
Civ VI ArtDef graph needed by the C3X terrain-lab resource and improvement
pipeline. It is an offline discovery/validation tool. It does not enable game
integration and it does not place source-specific identifiers in a runtime pack.

The checked source inventory currently resolves:

- 26 Civ III resource mappings (25 `Resource` targets plus the oasis `Feature`)
- all 25 Base `Improvement` targets
- 251 visited ArtDef nodes and 1,159 typed edges
- 1,602 map-visual terminals, representing 465 unique package/entry assets
- zero unresolved visual edges and zero unresolved map-visual terminals

The generated, ignored evidence report is
`Renderer/preview/out/artdef_graphs/resource_improvement_graphs.json`.

## Graph rules

Each top-level ArtDef collection item is indexed as:

```text
relative/path.artdef#RootCollection/Name
```

The resolver retains every discovered edge but classifies it as a visual
dependency, selector condition, metadata dependency, or retained non-map
dependency. Only visual dependencies are traversed into the map-visual asset
closure.

It resolves both forms used by the installed source:

- explicit `m_ElementName` / `m_RootCollectionName` / `m_ArtDefPath` references;
- implicit `XrefName` plus blank `Xref` pairs in `Clutter`,
  `ClutterVariants`, and `Landmark` child collections.

Definitions with the same target name in shared DLC or expansion ArtDefs are
included instead of silently collapsing to the Base definition. Reference
selection prefers the source node's own content directory, then Base, and fails
on a same-priority ambiguity.

## Reverse associations

Some improvement visuals are not reachable by following the improvement node
forward. The farm system is the important example: `Farms.artdef` entries point
back to `IMPROVEMENT_FARM`. The resolver therefore builds an incoming-reference
index and admits associated nodes from specialized `Farms`, `PlotSets`,
`TileSets`, and `GreatWall` roots.

In the installed graph this recovers 16 farm roots: 12 Base era/crop variants
and four Gran Colombia/Maya maize variants. Their downstream plot and tile sets
then resolve normally.

## Cooked-package inheritance

An ArtDef terminal is closed only when both its logical BLP package and entry
name are found. Package lookup follows the same-content, Base-fallback, then
other-content order. Crucially, the entry is checked before a package wins:
partial DLC packages therefore cannot hide inherited Base entries. The current
inventory exercises 16 such Base fallbacks; the other 1,586 terminals resolve
inside their defining content root.

Repeated raw entry-name strings inside a cooked package are reported through
`entry_occurrences` and `entry_name_is_unique`, but are not treated as missing.
The compiler only requires at least one occurrence at this discovery stage;
typed extraction remains the responsibility of the appropriate generic asset
importer.

## Reproduction

```bash
python3 Renderer/tools/asset_compiler/artdef_graph_resolver.py --require-closed
PYTHONPATH=. python3 -m unittest Renderer.tools.asset_compiler.test_artdef_graph_resolver -v
```

`--require-closed` refuses to write a successful result if any map-visual edge,
package, or package entry remains unresolved. Synthetic tests cover implicit
Xrefs, same-name content layering, reverse farm association, case-insensitive
Windows BLP directory names, partial-package Base fallback, and missing-entry
failure.

## Mine and farm conversion proof

`improvement_asset_importer.py` consumes the closed graph and a checked generic
strategy without enabling L18, L19, I18, or I19. The full graph exposes 18
unique mine components and 204 unique farm components. The current
representative proof compiles six mine roots and eighteen accepted farm roots;
recursive nested-component closure yields 93 normalized generic assets. It
also quarantines source resource-conditioned mine decorations until they can be
mapped explicitly to Civ III resources, rather than duplicating L16 art.

The conversion and topology policy, exact proof counts, rejection reasons, and
day/night preview command are documented in
`docs/mine_and_farm_asset_conversion.md`.
