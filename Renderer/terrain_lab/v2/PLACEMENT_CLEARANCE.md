# Network, vegetation, and city placement clearance

Roads and rails through forest/jungle must have a readable cleared corridor:
trees go around the route, not through it. City buildings must not be placed
on river, road, or rail corridors. These are visual-placement rules, not changes
to Civ III terrain, connectivity, river topology, city anchors, or gameplay.

## Ownership and shared inputs

- Q5 publishes versioned world-space road/rail occupied and clearance envelopes
  from the same final curves, widths, junctions, shoulders/ballast, bridge decks,
  and approaches used to render the network. Include stable IDs, height ranges,
  wrap handling, and enough neighboring geometry for cropped fixtures. Do not
  approximate curved or diagonal routes by clearing only their tile centers.
- Q3 publishes river water/bank exclusion envelopes from rendered river curves
  and widths, including bends, branches, mouths, and crossing anchors.
- Q4 consumes the envelopes for forest/jungle source-instance placement. Q7
  consumes them for city buildings, walls, gates, and attached components.
- Q0 owns shared contract/composition plumbing; Q8 owns combined layout review.
  Providers first publish small owned fixtures and interface witnesses. Consumers
  may test with explicit local envelopes while plumbing is pending. No new
  serial launch gates and no cross-owner implementation edits.

## Vegetation behavior

Reserve the complete swept road/rail corridor plus a declared modest clearance
margin before placing trees. Test transformed source-instance bounds, not just
tree origins: trunks, roots, and low foliage must not intrude into pavement,
rails, sleepers, bridge decks, or usable route space. Account for crown extent
so dense overhanging foliage does not hide the route at either gameplay zoom.
Reject or deterministically reposition interfering instances into the remaining
forest/jungle area, retaining natural density and irregular edges. Do not leave
a whole tile bare, stretch source trees, cut holes in their meshes, or paint
roads on top of trees. Ordinary shadows across a route remain valid.

Clearance changes must be local and stable: adding a road must not reshuffle
every tree, scrolling must not repopulate the crop, and wrap aliases must agree.
Union intersecting road/rail envelopes and treat junctions/bridge approaches as
part of the route, not gaps in the mask. Preserve source-art provenance.

## City behavior

Reserve river water/banks and road/rail corridors before assigning building
slots. Use transformed component footprints and relevant overhang/height bounds,
not just city or building centers. Place source buildings in remaining buildable
areas with readable setbacks; preserve the authoritative city anchor, allowed
facing, source proportions, and a cohesive silhouette. Routes entering or
passing through a city retain an unobstructed passage. Walls need aligned gates
or intentional openings at route crossings, not a solid wall over the road or
track. Buildings must not cover rivers or substitute for bridges.

When a layout is tight, try deterministic alternate slots, source components,
or fewer optional decorative buildings rather than moving the river/network,
shrinking everything arbitrarily, or hiding the conflict with draw order.
Required components that cannot fit remain a reported layout failure; do not
silently discard them. Normal perspective overlap of distant objects and cast
shadows is not the same as physical intersection: inspect both world-space
clearance and actual-size composed readability.

## Verification and fast iteration

Require numeric intersection checks against the published envelopes with a
declared tolerance, plus rendered envelope/footprint overlays and normal
composed views. Cover a road and a rail through dense forest and jungle,
curves, diagonal and north-south directions, junctions, wrap/crop edges, and
bridge approaches. City witnesses cover through-routes, road/rail coexistence,
river bends/crossings, walled cities and growth, with intentionally tight cases.
Keep roads/rivers and the city anchor identical in before/after comparisons.
Check both zooms and deterministic scrolling at candidate checkpoints; reuse
small cached fixtures during edits. Synthetic tight cases supplement verified
real terrain with explicitly labeled Lab network/city augmentation layers.

Q8 rejects tree/route intersections, foliage-obscured corridors, buildings on
transport/water corridors, and blocked city gates even when isolated assets look
good. Clearance masks are placement data, not new art and not permission to
modify gameplay or generate replacement assets.
