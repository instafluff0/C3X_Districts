ROLE: Lab v2 platform/core owner.

Maintain and complete the infrastructure while Q1-Q8 work concurrently. Preserve
the frozen baseline path; new versioned interfaces may support the visual owners'
changes without retuning that baseline. Prioritize requests that block actual
renders and the verified test.biq registry. Adopt Q6's color/alpha specification
and witnesses into shared contracts after technical review; routine interface
decisions do not need another user approval. Keep compatibility/version checks
so publishing new interfaces does not invalidate other owners' work silently.
Do not wait for Q6's complete visual gate to implement its interface proposal.

Deliver versioned render-packet, surface, route, environment, presentation,
fixture, and render-graph contracts; a manifest-driven CLI; dependency-aware
incremental C++ and shader builds; content-addressed pack, fixture, geometry,
shader, and shadow caches; namespaced outputs; and quick/check/compose/promote
selection.

Add a headless macOS Metal backend for ordinary work and preserve the off-screen
D3D11 backend for promotion parity. Prefer a shared HLSL source compatible with
the production SM5 path and the SM6 DXIL-to-Metal toolchain rather than a
separately evolving Metal-only look. Batch multiple variants through one device
and resource load. If a persistent worker is justified, serialize GPU jobs and
make cache invalidation explicit.

Acceptance requires two track owners to build and render without sharing
implementation or output paths; shader-only edits avoid unrelated C++ rebuilds;
microfixtures avoid the 192-tile path and Parallels; identical commands are
byte-stable per backend; representative Metal/D3D11 parity stays within a
versioned tolerance; corrupt caches, missing modules, and contract drift fail
clearly; and frozen v1 outputs remain unchanged.

Prepare the controls required by Q1's sharpness A/B study: explicit sampler
anisotropy and mip bias, supported sample counts and matching color/depth
resources, resolve, internal render scale versus final output size, pluggable
downsample/postprocess passes, and deterministic camera-offset sequences.
Expose color-space and alpha semantics, GPU time and memory measurements, and
machine-readable effective settings. Include these settings and backend/device
capabilities in cache identity. Unsupported requests must be reported clearly.
Batch small frames without reloading assets or launching Parallels. Keep the
baseline settings as defaults during LQ0; Q1 selects the quality policies and
owns their shaders, while Q6 owns exposure and lighting.

Implement the source verification, cached BIQ import, initial named-region
registry, and terrain-plus-augmentation replay support required by
`Renderer/terrain_lab/v2/REAL_MAP_VALIDATION.md`. Publish these inside your owned
`shared/real_map/` and `app/` paths during Q0; do not wait for Q8. Verify the actual
user `test.biq`, explicitly distinguish it from the historical Ancient Treasures
export, and record portable provenance. Use the existing parser where suitable.
Acceptance requires an unmodified real-region replay, a separate deterministic
object/route overlay with an overlay-off terrain identity check, a second region,
correct neighbor/wrap handling, and Mac replay without routine VM access.
Select initial regions from measured feature coverage and record absent cases.
Cache identity must include source, region/halo, overlay, parser, and profile
versions. Missing source identity or stale/mismatched overlays must fail clearly.
This adds evidence infrastructure, not permission to tune the frozen appearance.
