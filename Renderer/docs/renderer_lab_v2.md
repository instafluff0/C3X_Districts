# Renderer Lab v2 Quality Campaign

## Purpose

Renderer Lab v1 proved the complete L9-L21 visual stack and froze its approved
handoffs. Lab v2 is a separate quality campaign that restructures the Lab for
fast macOS iteration and parallel visual refinement before the remaining Game
Integration gates resume.

Lab v2 does not overwrite an L9-L21 handoff, silently change an already
integrated I9-I18 system, or transfer new runtime ownership. It produces
versioned candidate handoffs that explicitly reference the immutable v1
baseline.

## Program steps

1. **LQ0 — Modularize Renderer Lab and establish the macOS fast path.** This is
   a zero-intended-visual-change platform step. It extracts portable scene and
   system code, defines stable render packets and composition contracts, adds a
   headless Metal backend, preserves the D3D11 backend, and establishes quick,
   system, composition, and promotion workflows.
2. **LQ1 — Parallel visual-quality campaign.** Independently owned tracks refine
   sampling, terrain transitions, hydrology, relief, routes, lighting and
   composition, object presentation, and cross-system visual QA.
3. **LQ2 — Combined convergence and v2 handoffs.** The accepted track results
   run through exhaustive focused fixtures and the 192-tile, four-phase,
   two-zoom composition gate. Frozen v2 handoffs then become eligible for
   deliberate Integration refresh gates.

The top-level project continues to expose exactly one `ready` step. Parallelism
exists inside LQ1 through per-track status files; those are not top-level
project steps and do not weaken the canonical state rule.

## Platform boundary

The portable Lab core owns scene construction, deterministic geometry,
materials, profiles, fixtures, caches, metrics, and render packets. Backend
implementations consume those packets:

```text
portable Lab core
       |
       +-- headless Metal backend on macOS (ordinary iteration)
       |
       +-- off-screen D3D11 backend on Windows (promotion parity)
```

Routine `quick`, `check`, and `compose` work must run on macOS without invoking
Parallels. Windows remains required for D3D11 parity, production renderer
builds, authoritative live Civ III capture, injected compilation, compositing,
and in-game verification.

The Metal and D3D11 backends consume the same source-independent packs,
fixtures, constants, coordinate basis, and shader behavior. Byte stability is
required within each backend. Cross-backend comparison uses bounded pixel,
silhouette, depth, seam, luminance, color, and shadow tolerances rather than
requiring identical GPU bytes.

## Stable contracts

LQ0 freezes these interfaces before visual tracks begin:

- `SurfaceSample`: position, continuous height and normal, material weights,
  terrain ownership, shore distance, water depth, and wetness.
- `Renderable`: source-independent mesh/material IDs, uniform transform, world
  bounds, layer, depth mode, caster/receiver classification, and stable ID.
- `RouteGraph`: canonical nodes and edges, connection topology, grade samples,
  and hydrology-published crossing anchors.
- `FrameEnvironment`: the existing C3X hour/season-driven sun, moon, ambient,
  exposure, water, shadow, and emissive state.
- `PresentationProfile`: tile scale, category footprint and height bands,
  grounding, allowed orientation, and both Civ III zooms.
- `FixtureManifest`: systems, inputs, camera, phase, zoom, isolation outputs,
  references, and metrics for one replayable case.
- `RenderGraph`: explicit opaque, cutout, decal, shadow, water, transparent,
  emissive, and effect relationships. Individual systems never establish final
  ordering by private depth hacks.

Only the platform/core owner edits the shared contracts. A visual track that
needs an interface change records an exact request for the coordinator instead
of editing another owner's paths.

## Quality tracks

| Track | Ownership | Primary outcomes |
| --- | --- | --- |
| `Q0-platform` | Backends, core contracts, runner, caches, build and parity | Mac-default fast loop with frozen v1 appearance |
| `Q1-sampling` | Filtering, mip/LOD, tangent basis, texel density, UV metrics | Sharp detail without stretching or shimmer |
| `Q2-terrain` | Continuous base surface and every terrain-pair transition | Intentional seamless adjacency and world-space detail |
| `Q3-hydrology` | Shore, beach, bathymetry, water, surf, rivers and mouths | Irregular coastlines, visible shallows, natural connected water |
| `Q4-relief` | Dunes, hills, mountains, volcanoes and biome forms | Broad readable relief and source-faithful rigid bodies |
| `Q5-networks` | Roads, railroads, grades, junctions and bridges | Natural all-direction connectivity over terrain |
| `Q6-lighting` | Shared lighting, shadows, depth and layer composition | Coherent day/night, grounding and occlusion |
| `Q7-presentation` | Object scale, aspect, grounding, facing and clearance | Proportionate objects with southwest/southeast presentation |
| `Q8-beauty` | References, contact sheets, defect routing and convergence | Civ VI-directed cohesion without cross-owner code edits |

The machine-readable campaign and work packages live under
`terrain_lab/v2/campaigns/Q1/`. Each package has exclusive owned paths,
read-only dependencies, forbidden paths, references, fixtures, gates, and an
independent status record.

## Iteration tiers

1. **Quick:** focused static tests plus a 4x4-8x8 microfixture, one phase, one
   zoom, and only the system plus required receivers.
2. **Check:** the complete system topology/state matrix at both zooms and the
   relevant lighting phases, followed by a deterministic repeat.
3. **Compose:** affected systems over a medium shared fixture, including
   complete, isolation, without-system, and difference outputs.
4. **Promote:** the complete 192-tile scene, four day phases, both zooms,
   isolation views, metrics, parity, and full project verification.

The 192-tile scene is a merge and promotion gate, not an ordinary edit loop.
BIQ exports, normalized packs, shaders, geometry, and shadow fields are cached
by complete input hash. Outputs are namespaced by campaign, track, and
candidate so parallel agents never overwrite one another.

## Visual evidence

Canonical Civ VI and Civ III screenshots are property-level references rather
than pixel-equality targets. The tracked catalog records logical ID, dimensions,
hash, subject, comparison mode, and rubric. Copyrighted pixels remain local
under `Renderer/canonical/` or a configured `C3X_LAB_REFERENCE_ROOT`.

Automated metrics reject structural regressions and flag visual defects. They
do not numerically define beauty. Every candidate still requires direct image
inspection, recorded observations, and the strategic approval policy in
`visual_validation_plan.md`.

## Agent operating rule

An agent owns one work package and works until its acceptance gates pass or a
genuine cross-owner blocker is proven. It repeatedly renders, inspects, revises,
and re-renders; a successful build or first plausible image is not completion.
Ordinary iteration must not ask the user for screenshots or subjective review.
The coordinator alone updates global project state, shared contracts, and final
promotion manifests.

