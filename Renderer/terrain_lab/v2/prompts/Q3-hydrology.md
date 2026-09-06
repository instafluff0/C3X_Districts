ROLE: Coast, shallow-water, ocean, and river owner.

Own the signed shoreline field, beach envelope, bathymetry, shallow-bed detail,
static water material stack, river corridors, banks, junctions, relief-carving
inputs, coast mouths, and wrap continuity. Publish exact deterministic crossing
anchors for the network owner; do not create road or railroad bridges.

Prioritize the static beach look: varied shoreline contours, beach width and
slope, dry-to-wet sand transitions, terrain-to-sand blending, shallow-bed detail,
and gradual water depth/color changes. Ocean waves, moving surf/foam, animated
normals/caustics, and river-flow animation are deferred. Do not implement water
animation, add time-driven hydrology redraws, or require animation for acceptance.

Use `sea_and_shore.png` and `river.png` as selective property references. A single
Civ VI screenshot may capture transient waves, foam, specular highlights, or
caustics; it cannot establish that these are static beach features. Compare
shore shape, sand, wet edges, submerged detail, and depth transitions. Classify
ambiguous source features as unresolved or deferred; do not bake a captured wave
crest or white surf ribbon into permanent shoreline geometry or texture. A good
static beach must stand on its own with wave/foam layers disabled or omitted.

Exercise coves,
points, islands, narrow channels, long straight stress runs, land-family
changes, relief-to-coast, sources, bends, junctions, mouths, and wrap. Iterate
until there is no repeated S-template, sawtooth ownership edge, water plate,
painted-on surf bands, disconnected mouth, duplicate reciprocal river, or crop-edge
pond. Shallow depth must transition monotonically while keeping authored bed
detail visible. Beach width and wetness may vary naturally while remaining
continuous with the land/water boundary.

Use cached static microfixtures and beach-only, bed-only, and composed controls
at both zooms. Day/night tests consume Q6's shared environment; they do not add
water animation. Deterministic scrolling tests move the camera over fixed water
to catch seams, aliasing, or texture swimming. Preserve river topology, banks,
mouths, and crossing-anchor obligations for Q4/Q5. Future animation is neither
a blocker nor a reason to approximate transient surf as permanent beach detail.

Hill-to-water edges are rocky shores, not sandy beaches. Select this treatment
from authoritative hill and neighboring water semantics, including water tiles
whose type is coast, sea, or ocean; do not depend solely on the water type being
named coast. Apply it only to the hill's water-facing boundary, and retain sandy
treatment for suitable adjoining lowland shores. Suppress the beach/sand ribbon
on rocky edges, publish the shoreline classification and seam/receiver data for
Q4, and join shallow water to the rock foot without a gap or floating water lip.

Inspect `civ6.rocky_hill_coast` in the reference catalog: the right-hand raised
peninsula in `sea_and_shore.png` shows exposed rock faces below grass-covered
relief and irregular rocks at the waterline. Use that static rock/land/water
relationship; the visible wave crests remain out of scope. Q4 owns the raised
rock faces and their hill connection; Q3 owns edge selection, sand suppression,
water fit, and transitions to neighboring beaches. Establish the initial Q3
contract using proxy relief, so Q3 does not acquire a reverse dependency on Q4.

Publish rendered river-water and bank exclusion envelopes for Q7 city placement,
including bends, junctions, mouths, widths, and crossing anchors, following
PLACEMENT_CLEARANCE.md. City buildings must avoid these corridors. Keep river
topology authoritative; do not move a river to fit a city's building recipe.

Add hill-versus-lowland controls, mixed rocky/sandy corners, coves, headlands,
small islands, multiple consecutive hill edges, all boundary orientations,
wrap, and both zooms. No mandatory beach beneath a hill cliff, abrupt material
cut at the shared corner, tiled rock fence, or baked surf is acceptable.
