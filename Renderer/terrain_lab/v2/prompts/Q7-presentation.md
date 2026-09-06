ROLE: City, unit, resource, and improvement presentation owner.

Own common uniform scale, tile-relative footprint, screen occupancy, grounding,
source transform and UV preservation, category-relative proportion, allowed
facing, sockets, and retained-label/HUD clearance. Do not redesign terrain,
routes, water, or lighting.

Implement PLACEMENT_CLEARANCE.md: reserve Q3 river/bank and Q5 road/rail
envelopes before assigning city building slots. Test complete transformed
component footprints and overhangs, preserve the city anchor and source shapes,
and keep through-routes unobstructed. Place wall gates/openings at route entries.
Try stable alternate slots or fewer optional decorative components when tight;
do not relocate rivers/routes, hide overlap by draw order, or silently omit
required components. Publish numeric intersection checks and footprint overlays
for city/river/road/rail, walls, and growth cases alongside composed images.
Local explicit envelopes permit independent work until shared plumbing arrives.

Build side-by-side fixtures including horse resource versus mounted unit,
infantry versus city, mine versus mountain, aquatic resources, all eight unit
facings, both zooms, and southwest/southeast building and resource witnesses.
Reject stretched aspect, arbitrary enlargement, floating or penetrating bodies,
resource scale that competes with units/cities, and static objects whose
projected forward vector is outside the approved southwest/southeast quadrants.
Unit facing remains authoritative and independent of static-object orientation.

Measure projected height, width, footprint, and visible pixel occupancy for
representative objects at both Civ III zooms. Record reference screenshot scale
and viewing differences before comparing them. Test modest uniform scale
variants with filtering and lighting fixed; compare final-size crops alongside
Q1's fixed-scale sharpness controls. Select sizes by readability, silhouette,
tile clearance, and category hierarchy, rather than enlarging everything to
match unmatched Civ VI screenshots. Horse resources must remain subordinate to
mounted units and cities; scale changes must preserve UVs, attachments, facing,
and retained-overlay clearance.

Treat cities as a dedicated composition deliverable within Q7, with a separate
city acceptance report. Own the finished assembly: component selection,
placement, spacing, skyline, orientations, walls, capital accents, material and
emissive bindings, and stable growth recipes. Read
`Renderer/docs/city_rendering_strategy.md`,
`Renderer/terrain_lab/L17_CITY_AUDIT.md`, and the L17 handoff as historical
evidence. Write new recipes and evidence inside your owned v2 object paths.

Compare the frozen golden-angle layout with deterministic arrangements based
on building footprints and projected silhouettes. Require readable rooflines,
facades, entrances, and purposeful gaps at actual display size, while preserving
the appearance of a cohesive settlement. Measure footprint collisions and
projected occlusion separately: disjoint ground footprints can still conceal
one another on screen. Avoid both an undifferentiated roof mass and excessively
isolated toy buildings. Use source geometry and uniform scales; orient each
building's mapped front southwest or southeast without distorting the shared
Civ III projection. Symmetric or ambiguous source fronts need documented mappings.

Tune component size, component count, placement, and overall city envelope as
separate variables. Inspect town/city/metropolis silhouettes, foreground versus
background height balance, wall/gate clearance, route entrances, shoreline and
river clearance, and neighboring units/resources/cities. Walls should enclose
the settlement coherently without obscuring its buildings. Preserve existing
building identities and positions across growth where practical, adding outer
slots instead of randomly rebuilding the skyline. Era or style replacements
must be deterministic and owner-color changes must not reshuffle the layout.

Bind the source-backed window/lamp emissive channels and compare exactly the
same city geometry at noon, sunset, midnight, and sunrise. Q6 supplies shared
lighting and shadows. Require legible roof and facade separation by day, visible
building-to-building and building-to-ground shadow relationships, and localized
night lights with readable unlit architecture. Log missing material channels
instead of compensating with a global city brightness boost. Emissive windows
and light cast onto neighboring surfaces are distinct features: unresolved
Light/VFX socket bindings do not authorize invented source-equivalent lights.
Animated smoke/fire and deferred wonder/district work stay at their own gates.

Start with a cached microfixture of three city sizes and nearby scale witnesses.
Compare spacing, orientation, silhouette, and night response using unsharpened
controls. On finalists, cover every effective culture/style x era x size pool,
with representative walls, capitals, owner colors, coast/river/relief neighbors,
growth transitions, all four day phases, both zooms, and scrolling. Use a
coverage ledger and targeted difficult intersections rather than multiplying
every optional state. Include component-ID/depth, emissive-only, shadows-off,
city-only, and composed controls. Report unresolved composition/lighting defects
to the correct owner and iterate; generic scale/facing tests alone cannot close
Q7 while this city gate is incomplete.

Inspect `civ3.real_gameplay_layout` (`civ3_real_example.jpg`) for representative
city-to-city spacing, object scale, and label crowding in a developed game.
Exercise several neighboring cities and nearby units/resources on real terrain
with declared Lab placements before accepting scale and clearance. Preserve
plausible density; do not tune solely against isolated, widely spaced objects.
Q8 owns the final gameplay-layout recipe and review, without blocking your
independent witnesses. The screenshot is a layout reference, not an exact map
capture or a requirement to reproduce its original sprite materials.
