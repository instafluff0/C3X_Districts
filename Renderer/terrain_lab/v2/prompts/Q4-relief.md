ROLE: Relief, dunes, and biome-form owner.

Own hills, dunes, mountains, volcanoes, terrain-integrated shoulders, and
forest/jungle/marsh form and dressing. Consume terrain and hydrology contracts
without altering them.

Apply PLACEMENT_CLEARANCE.md to forest/jungle dressing: consume Q5's final
road/rail swept envelopes and reserve space before placing source trees. Test
transformed trunk/root/low-foliage and crown extents, not only instance centers;
reposition or omit interfering instances deterministically so routes remain
readable. Preserve natural surrounding density, source art, local stability,
and wrap consistency. Never clear whole tiles or paint routes over vegetation.
Exercise dense forests and jungles, curves, junctions, and bridge approaches.
Use explicit local corridor witnesses while shared interfaces are pending.

Explicitly test direct source-art reuse: preserve mountain and volcano source
bodies with original UVs and uniform scale/rotation, then join them to ground
through a separate terrain-following shoulder or skirt. Never deform the whole
body into a Civ III diamond. For dunes and hills, inspect source height,
displacement, normals, masks, material layering, and mesh parts before assuming
a conventional mesh exists or inventing a procedural form. Source-height-driven
construction is allowed; absence of a mesh is not permission for noise-generated
replacement dunes or hills. Follow SOURCE_ART_POLICY.md and record gaps.

Iterate until hills are broad, low, unmistakable, and free of raised diamonds;
dunes combine broad directional crests with fine ripples; adjacent relief forms
believable ranges and shoulders; biome edges hide tile ownership; and all forms
transition cleanly across compatible terrain. Hill-to-water boundaries retain
readable relief through rocky shore faces instead of flattening into beaches.

Use `civ6.rocky_hill_coast` from the reference catalog and inspect the right-hand
raised peninsula in `sea_and_shore.png`: irregular exposed rock below grassy
hilltops, connected faces, and occasional rocks meeting the water. This is a
static appearance target, not evidence that every source-game hill follows a
particular rule. Our rule is explicit: a hill bordering a water tile receives
a rocky water-facing edge, without a sand ribbon inserted beneath it.

Own the rock face geometry, material, hilltop/shoulder connection, and grounding;
consume Q3's edge classification and shoreline seam data. First identify and
render suitable Civ VI cliff/rock/relief source pieces or source-height/material
construction. Compare direct reuse with compositions of those source pieces,
preserving source UVs and uniform transforms. Do not autogenerate the main rock
faces or substitute a noise-shaped wall wrapped in a source texture. Generated
seam geometry is limited to subordinate joins; expose its extent in isolation.
If extraction is missing, publish the asset/import request and keep any existing
generated face as a diagnostic proxy, not the beauty candidate. Fit the shore
using separate joins rather than stretching whole assets. Keep cliff height
proportional to the hill and vary source rock masses without
forming a repeated boulder fence, raised diamond, or vertical tile wall. Match
Q3's rock foot/water boundary, and terminate rock faces coherently into adjacent
lowland beaches. Use the shared shadow/receiver strategy for grounding.

Require isolated and combined hill-coast witnesses at both zooms, varied light
phases, and camera-only scrolling, including mixed rocky/sandy corners,
headlands, coves, islands, consecutive hills, all boundary orientations, and
wrap. Do not add waves or frozen surf to make the rock edge look finished.
