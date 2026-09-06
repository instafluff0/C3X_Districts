# Mine and Farm Asset Conversion

This is offline preparation for Renderer Lab gates L18 and L19. It discovers
installed source art, converts a representative proof subset or the complete
discovered library into generic C3X pack records, and renders inspection
sheets. It does **not** start either lab
gate, choose final visuals, enable runtime ownership, or authorize I18/I19.

## Source inventory

The closed ArtDef graph contains 18 distinct top-level mine `TileBase`
components and 204 distinct top-level farm components. Mine variants divide
into ancient and industrial families across Base and both expansions. Farm
components divide into:

- ancient, industrial, and modern building families;
- eight tile fillers per era;
- square, angled, transition, and filler crop shapes;
- default, wheat, rice, and maize crop families.

The proof compiler intentionally starts with six Base mine roots and eighteen
accepted farm roots. Recursive component closure produces 93 normalized assets,
115 geometry parts, 177 materials, 77 deduplicated textures, 532 decal
descriptors, and 708 attachment records. Of those materials, 108 contain a
confirmed static emissive channel. The ignored build report records exact local
source hashes and all accepted/rejected dependencies.

The optional full-library pass now accepts all 18 mine and 204 farm roots with
zero rejected top-level assets. Recursive closure produces 294 normalized
components, 120 geometry parts, 185 materials, 114 emissive bindings, 93
deduplicated textures, 2,371 decal descriptors, and 2,402 attachment records.
This is fetch/conversion coverage only: final visual selection still belongs to
L18/L19.

## Why recursive composition is required

`AttachmentPointCookData` is overloaded in these packages. A 32-byte record
may be a true smoke/fire/light socket, or it may point to a six-field nested
component record. The latter supplies an asset, connection type, resource
condition, terrain-follow mode, culling policy, and animation-randomization
flag. The converter distinguishes the forms and places resolved child
components on exact normalized skeleton-bone rest transforms.

This matters especially for farms: buildings, props, and field decals are not
one monolithic model. The runtime-facing pack retains only generic component
IDs and normalized data; source package names remain in the external evidence
report.

## Mine strategy for L18

- Civ III eras 0–1 select the preindustrial family; eras 2–3 select the
  industrial family.
- Variant choice is a stable hash of world seed, tile coordinates, and era.
- Geometry follows the renderer's authoritative terrain surface rather than a
  source-specific flat tile.
- The shared C3X environment controls daylight and confirmed emissive channels.
- Source resource-conditioned ore props are excluded until an explicit Civ III
  resource mapping exists. This avoids drawing an ore pile twice when L16 owns
  the map resource.
- Worked/pillaged selection may use a component state only where the normalized
  source record proves one; there is no fabricated damage art.

## Farm/irrigation strategy for L19

Civ III's four terrain-specific irrigation atlases each encode sixteen
four-neighbor adjacency masks. The generic farm catalog preserves that exact
four-bit/16-mask input contract, but does not claim that source labels such as
`Square`, `Angled`, and `Transition` are already a one-to-one Civ III mask
mapping. L19 must establish the mapping visually in the lab.

- Civ III eras 0–1 use preindustrial pieces, era 2 industrial pieces, and era 3
  modern pieces.
- A tile composition draws an era tile base, a topology-selected building
  piece, and a crop decal rather than stamping one farm model everywhere.
- Grassland, desert, plains, and tundra remain distinct terrain families; the
  renderer clips and blends composition at irrigation ownership boundaries.
- Seasons use a shared environment material/variant policy.
- Default crops are always available. Grain, paddy, and maize are optional
  same-tile resource overrides only after an explicit Civ III mapping.
- Deterministic selection includes world seed, tile coordinates, era, and the
  authoritative adjacency mask.

L19 also carries a known Terrain Lab correction for scheduling convenience, but
it is independent of farms and irrigation. Civ III tundra (`base == 3`) must
select the normalized tundra base-color, height, and specular channels instead
of the historical catch-all grassland material path on every tundra tile,
whether irrigated or not. Farm rendering is not a prerequisite for tundra, and
tundra is not a prerequisite for irrigation on other eligible terrain.

The promotion fixture therefore has two separately testable coverage tracks.
The irrigation track must exercise every one of the sixteen four-neighbor masks
on every irrigable terrain family, with terrain-conforming farm geometry,
terrain-appropriate soil/channel treatment, and seam-free ownership clipping.
The tundra track must deliberately include irrigated and unirrigated tundra and
mixed tundra/non-tundra material boundaries. Native and reduced renders must
make tundra materially distinguishable from grassland while retaining
continuous ground blending. Automated coverage must fail when either track is
incomplete; the authoritative BIQ viewport containing tundra by chance is not
sufficient.

## Night and ambient behavior

Static emissive textures are converted as confirmed material channels. Their
night weighting is a C3X environment policy, not a claim about the source
engine's activation algorithm. Named smoke, fire, and light sockets are
preserved as unresolved evidence when their reusable typed resource cannot yet
be decoded. The preview therefore demonstrates static day/night material
response and does not fabricate particles or analytic lights.

## Deliberately unresolved inputs

- 486 resource-conditioned mine child references in the full sweep are
  quarantined pending a Civ III resource mapping.
- 57 effect resource identities in the full sweep remain unresolved and are
  not synthesized.
- One generic mine attachment transform does not match a skeleton bone and is
  retained as unresolved.
- Eighteen rice/paddy roots contain a typed but semantically undocumented
  `TerrainEditDesc3` record. Their visual decal/geometry data is converted, the
  record type/count/hash is preserved in source evidence, and terrain-edit
  application is explicitly disabled; no deformation values are guessed.
- One mine source attachment has an optional value record with no asset terminal
  and no matching bone. It is recorded as an omitted source placeholder rather
  than assigned an invented transform.
- Five optional hay-bale references lack the required base-color channel and
  are rejected without failing their parent components.
- Final adjacency recipes, close/strategic zoom readability, and the 192-tile
  promotion renders remain L18/L19 work.

## Reproduction

```bash
python3 Renderer/tools/asset_compiler/improvement_asset_importer.py
python3 Renderer/tools/asset_compiler/improvement_asset_importer.py \
  --compile-discovered-library
python3 Renderer/preview/render_improvement_sheet.py \
  --manifest Renderer/packs/ImprovementsNormalized/manifest.json \
  --output Renderer/preview/out/improvements/day_night_sheet.png \
  --report Renderer/preview/out/improvements/day_night_sheet.json
python3 -m unittest \
  Renderer.tools.asset_compiler.test_improvement_asset_importer \
  Renderer.preview.test_render_improvement_sheet
```

The derived pack and preview outputs are ignored local artifacts. C3X runtime
code has no Civ VI format or asset dependency, and no new Civ III patch symbol
or `civ_prog_objects.csv` entry is required for this preparation.
