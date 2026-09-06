# Coastline and Water Findings

## Retired diagnostic verdict

Failed. The implementation below established useful shared-topology and runtime
plumbing, but the rendered result does not resemble the Civ VI target closely
enough to pass M6.7. Specifically, the contour repeats scalloped lobes, the
shoreline reads as a glowing outline, water lacks convincing surface structure,
cliffs lack mass, and the provisional triangle clutter is visibly synthetic.
It is not an active rework list: the approved L9 lab handoff supersedes this
look, and M6.7 removes the provisional rock/cliff dressing from its approved
production path.

## Evidence boundary

Confirmed source data: the locally imported terrain packages expose distinct beach,
cliff, cold-cliff, coast, sea, and ocean material families. The Civ VI ArtDef
material supplied during M6.7 planning also explicitly separates convex, neutral,
and concave coastline roughness/offset controls and declares shallow-water,
cliff-clutter, foam, refraction, Fresnel, current-noise, and whitecap concepts.

Inferred engine behavior: Civ VI's exact runtime contour meshing and scatter
algorithms are not public. C3X therefore implements the demonstrated design
principles through a source-independent renderer contract rather than attempting
to reproduce undocumented Firaxis code.

## C3X implementation

- A single land/water scalar field is evaluated from the four authoritative
  neighboring terrain control points. Both land and water use that field, so
  their boundary cannot randomize independently.
- The existing continuous world-space contour warp displaces the shared field.
  Horizontal wrap canonicalization makes the result deterministic at the seam.
- A land-relative topology sign distinguishes convex tips, neutral edges, and
  concave bays and offsets the profile without changing gameplay topology.
- The normalized terrain pack stores beach, cliff, and cold-cliff layers as
  generic authored material references. Runtime code contains no Civ VI format
  or source-name branch.
- Water uses separate coast/sea/ocean weights for shallow-to-deep color, plus a
  narrow contour-following foam/haze band and a height-derived Fresnel normal.
  Periodic full-surface whitecaps and glints were rejected because they read as
  a regular dotted screen pattern at strategy-map scale.
- Rugged land/water edges generate small low-poly rock clusters. Placement is
  seeded by the canonical shared edge rather than either tile, preventing
  disagreement across tiles, scrolling, and horizontal wrapping.
- Viewport-aware tessellation and triangle-aligned upload batches preserve the
  close contour while keeping 64px, 96px, and 128px tile scales inside the
  32-bit process and D3D11 buffer budgets.

## Static shallow-detail correction

The Civ VI close reference is not primarily exposing more of the repeating
Shallows base material. It shows a subdued water surface over projected seabed
clutter: broad rocky/contour forms plus smaller crack and rock detail.

Installed-source facts confirmed on 2026-09-05:

- `Base/ArtDefs/Clutter.artdef` defines `CLUTTER_OCEAN` with density `0.52`,
  `TerrainHeight=true`, `ClipCoastline=true`, and rotation/overlap-enabled
  projected placement.
- Its nonzero weighted entries are five `TER_Ocean_Decal01..05` large decals
  at scale `3.5..5.0` and four `TER_Coast_Decal01..04` crack decals at scale
  `3.0`. The remaining small/dark variants have source count zero in this set.
- The generic decal compiler now normalizes the five nonzero ocean entries and
  four coast-crack entries into `DecalsNormalized`, alongside their exact
  ArtDef placement records. Their shared authored base/height atlas contains
  the visible rock, contour, shadow, and cracked-bed regions.
- The cooked clutter package exposes these entries as projected decal records,
  not conventional vegetation meshes. Their `DecalDesc2` projection/UV data
  must be normalized before the renderer can select the atlas regions exactly.

The L12 lab now selects only those authored atlas cells, applies the confirmed
`0.52` density through deterministic occupancy, fades every projected boundary,
and clips the crack layer to the continuous shoreline field. The footprint is
calibrated down for Civ III's projection so the source clusters read as small
submerged contours rather than dark clouds. The transparent water surface
retains its optical tint over that detail. Whole-atlas tiling, substitute rocks,
and broad transparency remain rejected because they produced seams, repetition,
or a sandy washed-out result.

The ArtDef/package bindings and numeric placement records are confirmed source
data. The packed atlas-cell rectangles and Civ VI's exact random scatter
algorithm remain inferred; the lab uses deterministic source-independent
selection while preserving the exact source pixels and confirmed density.

## Historical live-evidence note

The old standalone replay proved deterministic construction, zero fallback, and
wrap stability for the rejected prototype. It is retained for diagnosis only.
Completed M6.7 instead uses the frozen L9/L10/L11 handoffs and the automated
production matrix documented in `approved_terrain_integration.md`.
