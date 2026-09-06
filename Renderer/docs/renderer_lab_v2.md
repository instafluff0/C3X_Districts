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

## Current ownership override — 2026-09-06

The user assigned single-agent ownership of the complete current Lab path.
The sole lead may change necessary platform, shared interfaces, terrain,
hydrology, relief, networks, lighting, objects, offline tooling and QA under
`Renderer/`. Q0-Q8 labels organize requirements and preserve historical files;
all older exclusive-owner and cross-owner start rules below are superseded.
Former tasks were checked idle before editing; no workers were restarted.
One combined scene and its short defect list drive implementation. Ordinary
renders run on the Mac. Frozen v1 handoffs, source assets, acceptance gates,
Game Integration pause and deferred M9-M11 contracts remain in force.
Current evidence and reproducible commands: `../terrain_lab/v2/audits/beauty/REVIEW.md`.

## Program steps

Art direction is source-first: all owners follow
`Renderer/terrain_lab/v2/SOURCE_ART_POLICY.md`. Reuse source meshes, materials,
and source-height data before proposing invented replacements; generated
topology and subordinate joins do not authorize generated dominant art.
Diagnostic proxies remain nonblocking test inputs, not beauty acceptance.

The user explicitly replaced the sequential start schedule with parallel work.
Q0-Q8 can all start immediately. The numbered steps below are acceptance
checkpoints: LQ0 may remain unfinished while LQ1 visual implementation proceeds.
`dependencies` now contains only start gates (none); `integration_inputs`
records candidate interfaces/results to consume and verify before convergence.
No owner waits for another track's full acceptance to begin useful work.

Q0 maintains the Mac platform and real-map registry while visual owners use
current versioned inputs, frozen data, or explicit local proxy fixtures. Q6
publishes color/alpha/lighting semantics and witnesses directly for Q0 to adopt.
Owners record precise interface requests and continue independent implementation
and rendering. Q8 builds gameplay/coverage fixtures and composes available
candidates now. Missing real-map or parity evidence stays pending; it does not
make independent work unavailable. Final acceptance still requires real inputs,
cross-system checks, and deliberate promotion.

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

Q0 versions these interfaces while visual tracks work against pinned candidates:

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
| `Q3-hydrology` | Static shore, beach materials, bathymetry, water, rivers and mouths | Irregular beaches, visible shallows, natural connected water; animation deferred |
| `Q4-relief` | Dunes, hills, mountains, volcanoes and biome forms | Broad readable relief and source-faithful rigid bodies |
| `Q5-networks` | Roads, railroads, grades, junctions and bridges | Natural all-direction connectivity over terrain |
| `Q6-lighting` | Shared lighting, shadows, depth and layer composition | Coherent day/night, grounding and occlusion |
| `Q7-presentation` | Object scale, aspect, grounding, facing and clearance | Proportionate objects with southwest/southeast presentation |
| `Q8-beauty` | References, contact sheets, defect routing and convergence | Civ VI-directed cohesion without cross-owner code edits |

The machine-readable campaign and work packages live under
`terrain_lab/v2/campaigns/Q1/`. Each package has exclusive owned paths,
read-only dependencies, forbidden paths, references, fixtures, gates, and an
independent status record.

Q3 prioritizes static beach appearance. Ocean waves, moving surf/foam, animated
water normals/caustics, and river-flow animation are deferred. Civ VI screenshot
references guide shoreline contours, beach/wet-edge materials, submerged detail,
and depth transitions; transient or ambiguous captured effects are not targets
for permanent beach art. Q1/Q8 scrolling checks use fixed hydrology with camera
motion, and do not introduce an animation prerequisite. Static river geometry
and Q4/Q5 crossing/relief contracts remain in scope; dependencies are unchanged.

Hill-to-water boundaries use rocky shores rather than beaches, guided by the
raised peninsula in canonical `sea_and_shore.png`. Q3 classifies the boundary,
suppresses sandy ribbons there, and supplies waterline/seam data; Q4 constructs
the exposed rock faces and connects them to the hill. Lowland beaches remain
distinct, with continuous mixed rocky/sandy joins. Q3 can validate its interface
with proxy relief before Q4 starts; the existing dependency order is preserved.

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

Every visual track must validate relevant named regions of the user's actual
`test.biq` before acceptance, including a neighboring or held-out region.
Use three fixture classes: immutable real terrain, real terrain with separate
deterministic Lab object/route layers, and constructed stress cases for diagnosis
or coverage gaps. Source hashes, coordinates, sufficient neighboring terrain,
and augmentation provenance are required. The current historical export named
`test_biq_l13_rivers_192.csv` is prepared from Ancient Treasures and cannot stand
in for verified `test.biq` provenance.

Q0 supplies source verification, cached Mac replay, and the initial named-region
registry before visual tracks consume them. Q8 audits final coverage; it does
not block initial registry preparation. Small cached regions stay in the fast
loop and larger region contact sheets run at candidate checkpoints. See
`../terrain_lab/v2/REAL_MAP_VALIDATION.md` for ownership, coverage, and acceptance.
The actual source identity is pending verification, not inferred from filenames.

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
Q0 publishes shared contracts after review with the semantic owner; routine
technical decisions require no additional user approval. Each owner can update
its own status file. The coordinator updates global state and final promotion
manifests. Q0's frozen baseline neutrality does not prohibit the other tracks'
versioned visual experiments.
