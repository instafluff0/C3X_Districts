# Historical M6.7 Terrain Visual Direction

This document records the evaluation criteria used to reject the retired
production beauty experiments. It is not an active implementation ladder. The
approved L9/L10 lab handoffs, their audit/reference hashes, and
`approved_terrain_integration.md` are now the production source of truth.

## Target

The target is a coherent, attractive modern strategy-map scene with Civ VI's visual hierarchy: broad readable landforms, continuous boundaries, localized high-frequency detail, grounded vegetation, layered water, and controlled light/color. It is not a literal copy of Civ VI, and Civ III's square-diamond gameplay topology remains authoritative and invisible beneath the visual surface.

The three user references diagnose the current gap clearly. Native Civ III hides its grid with dense authored transition sprites and feature silhouettes. Civ VI hides its hexes with continuous terrain, broad height fields, layered coast/water, scattered geometry, and strong art direction. The current C3X preview exposes tile diamonds, repeats texture at the wrong scale, stamps relief and feature art per tile, has little contact/shape lighting, and compresses most land into a dark green value band.

## Evidence From Catlike Coding

The reviewed Hex Map project was checked out at commit `21936db1ddd2fc267d4e68be65d0ddd704163fbd` (release 5.5.0). Its important transferable ideas are architectural rather than Unity-specific:

- `HexMetrics.cs` uses coherent noise and one shared perturbation function for vertices. Boundaries stay connected because both adjacent regions use the same displaced positions.
- `EdgeVertices.cs` subdivides every cell edge, creating enough samples for curvature instead of one straight gameplay edge.
- `HexGridChunk.cs` constructs separate water and shore surfaces and explicitly stitches edge/corner topology.
- `Terrain.hlsl` blends several cell material identities with weights while sampling detail in world space, so a material is not one decal stretched over one tile.
- `Water.hlsl` layers coherent signals for waves and foam rather than relying on a flat blue color.

Catlike's numeric constants are not a style prescription for C3X. The transferable contract is: shared samples, sufficient subdivision, deterministic coherent displacement, world-space material frequency, and purpose-built transition layers.

## Material Construction Rules

The imported Civ VI base-color image is an ingredient, not a finished terrain photograph. Its deliberately soft macro variation depends on height-derived normals, directional light, roughness/specular response, geometry, and localized dressing for clarity. C3X therefore follows these rules during M6.7:

- use unbiased anisotropic sampling on the isometric plane;
- keep macro color frequency independent from gameplay-tile size and tune it from fixed-camera comparisons;
- blend height samples across the same material weights as color, reconstruct a normalized surface normal, and evaluate it against the shared environment sun direction;
- treat the specular channel as a view/light-dependent response rather than an additive grayscale brightness map;
- keep family color grading restrained and never use it to compensate for missing light/material structure;
- obtain the sharpest visual edges from relief silhouettes, coastline profiles, vegetation, rocks, and contact shading—not from excessive base-ground normal strength;
- reserve a source-independent secondary detail/clutter layer for small scale structure instead of stretching or over-sharpening the macro texture.

## Evidence From Civ VI Documentation

The community-preserved Firaxis documentation confirms several data and authoring contracts:

- Terrain materials combine base color, height information used for surface normals, and specular/gloss response rather than treating color texture as the entire surface.
- terrain styles separately bind hills, ridgelines/mountains, flat terrain, coastlines, and ocean behavior; hills and mountains are height-driven terrain elements blended into the terrain system.
- time-of-day definitions control sun direction/color, ambient or image-based lighting contribution, emission weight, and exposure curves.
- water settings centralize water material response and wave behavior; coastline haze and related boundary effects are distinct from base terrain color.
- Strategic View's documented blend-grid system stores both sides of a transition in one aligned cell, uses weighted blend masks, and cooks topology permutations for inside/outside corners and edges. This is direct confirmation for Strategic View and a strong design analogue—not proof that the regular 3D renderer uses the identical implementation.

Local `TerrainStyle.artdef` and `Water.artdef` findings remain governed by `civ6_lighting_findings.md`: named parameters and package data are confirmed evidence; an inferred runtime algorithm must continue to be labeled as inference.

## C3X Visual Construction Contract

```text
Civ III authoritative tile facts
  -> shared viewport sampling lattice
  -> continuous base height/material field
  -> one deterministic topology-aware coastline contour
  -> relief profiles blended into the common surface
  -> feature/clutter scatter with stable cross-tile seeds
  -> material response under one environment rig
  -> off-screen composite beneath retained Civ III overlays
```

Every noisy or displaced boundary must be keyed by shared world coordinates or a canonical shared edge, never independently by each tile. Horizontal-wrap aliases must evaluate to the same samples. Runtime materials and geometry consume generic C3X pack data; source-specific extraction stays offline.

This older selective-bit plan was superseded by I11's exclusive ownership rule.
With custom rendering enabled, patched `Map_Renderer_m19_Draw_Tile_by_XY_and_Flags`
captures the authoritative occurrence but skips the original `m19` wholesale.
Approved L9 hills and mountains are custom; the unapproved L12 volcano category
is absent while its underlying custom base terrain remains renderer-owned.
Configuration off remains the sole native `m19` path.

## Visual Hierarchy

Judge each fixed-camera render in this order:

1. **Silhouette:** continents, bays, hill chains, peaks, and feature masses read at thumbnail size.
2. **Value:** land, raised forms, beaches, shallows, and deep water occupy distinguishable value bands without crushed shadows or clipped highlights.
3. **Continuity:** no grid diamonds, cracks, straight per-tile edges, isolated custom-art islands, or independently randomized shared boundaries.
4. **Scale:** ground detail is smaller than landforms; trees and rocks provide mid-scale cues; peaks remain the dominant terrain features.
5. **Material:** vegetation, soil, rock, snow, wet shore, and water differ through roughness/normal/specular response as well as hue.
6. **Accent:** foam, snow, sun glints, and saturated biome color are sparse focal accents, not the base exposure.

## Renderer Lab v2 terrain-family audit backlog

Before Lab v2 promotes another combined scene, audit every visible terrain
family and verify that its Civ VI source material has a sensible Civ III
semantic destination. The primary land-material audit must cover **grassland,
plains, desert, tundra, and flood plains** individually. It must then cover
their interaction with **hills, mountains, volcanoes, forests, jungles, marshes,
coasts, sea, and ocean**, including every relevant adjacency and transition.

For each family, record the source material/texture role, Civ III terrain code,
height and relief behavior, clutter/decal contribution, shoreline behavior,
lighting response, and legal neighboring families. Use isolated, mixed-
adjacency, and no-layer controls so a shadow, decal, or relief mask cannot be
mistaken for terrain. In particular, reject any tile-sized brown or dark
smudge with a visible diamond boundary: that is evidence of a leaking overlay
or cast-shadow receiver, not a valid base terrain type. The audit should prefer
clear semantic mappings over visual approximation and must not let a Civ VI
asset's source name silently decide its Civ III role.

## Superseded Implementation Ladder

The former M6.7a-M6.7g production-art ladder is retired. Its technical findings
remain useful diagnostics, but visual changes go through Renderer Lab and a
frozen handoff. The production integration may only adapt an approved handoff
to authoritative Civ III projection, clipping, environment, ownership, cache,
and hard-failure constraints.
