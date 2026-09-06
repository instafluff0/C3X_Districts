ROLE: Lab v2 platform/core owner.

Build the infrastructure that lets the visual owners work independently. Do
not redesign or tune any visual system.

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

