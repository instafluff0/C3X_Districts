# Terrain Lab Shore Asset Audit

The installed Civ VI source contains a broad shore-related inventory, but the
standard smooth beach does not use every item in that inventory. Source tracing
through `TerrainStyle.artdef`, `Water.artdef`, and `Wave.artdef` identifies its
actual stack as:

- the Beach base-color, height, and specular material;
- the Shallows base-color, height, and specular material on a 25-unit coast
  profile whose authored high and low heights are -3.5 and -10;
- `Water/Coast`, including the coast dark/scatter density ramps, multi-scale
  LEAN surface data, tiling/non-tiling normal pairs, gloss, and masks;
- the wave system's crash foam, ripples, and turbulence, with the authored
  20-pixel width, 128-pixel length, and 8-pixel crash distance as scale evidence.

The beach blanket's companion texture is a fog-of-war channel, not a beach
blend mask. Expansion submerged-coast textures belong to flood/submergence
content, and the ocean decal is not a blanket standard-shore layer. Coast-wide,
ocean, cliff, white-cliff, snow, ice, rock, and river assets remain available
for their corresponding terrain or dressing cases, but blending them all into
an ordinary beach is incorrect.

There is no reusable authored coastline surface mesh to import for the lab.
`TerrainStyle.artdef` describes a continuous coastline profile and assigns
materials and clutter to it. C3X should continue generating that connected
surface from map topology rather than stamping fixed coastline landmarks.

`ShoreNormalized` now supplies the missing reusable shore and river geometry:

- four verified large cliff-rock variants;
- two verified small cliff-rock variants;
- all sixteen verified polar-ice chunk variants;
- five verified river-rock variants.

All 27 assets contain normalized indexed triangles, recomputed geometric
normals, UV0, source-independent material paths, and extracted base-color,
gloss, and LEAN texture channels. Representative large rock, small rock, large
ice, and small ice renders have been visually checked.

Two additional small cliff-rock candidates and one river-rock candidate are
deliberately excluded because their source index buffers contain zero-area
triangles. The four coast-decal and ten river-decal entries are also excluded
from the mesh pack: they do not bind the proven static-model container, while
their useful generic decal texture channels are already in `TerrainNormalized`
or belong on the topology-generated river bank. These omissions are recorded
with source names only in the ignored shore build report.

L9.5 therefore consumes only the assets assigned to the standard beach stack.
Its runtime-generated contour implements the ArtDef's position roughness and
keeps the beach, connected shallow-bed slope, transparent water, and surf as
distinct but overlapping bands. `ShoreNormalized` remains reserved for later
cliff dressing, river dressing, and a polar-ice fixture; rocks are deliberately
omitted from the current coastline direction.
