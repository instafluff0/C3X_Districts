# M6.1 Evidence

The portable `m6_1_production_terrain` gate proves this step without launching Civ III or requiring a screenshot.

- `terrain/m6_1_selector_coverage.json` reconciles 14 BIQ terrain semantics with 11 relevant M6.0 atlas contracts and gives every selector basename an explicit mapped, M7 fallback, or retained-Civ-III disposition.
- `terrain/production_terrain.py` renders a connected shared-vertex surface with deterministic topology, wrap behavior, variants, relief/depth, landmark/polar-ice state, lighting, bounded caches, and atomic terrain-item fallback.
- `samples/scenes/m6_1_terrain.fixture.json` replays two viewport sizes, scrolling, wrapping, every terrain family, representative hours/seasons, and retained M7 instances.
- `terrain/test_production_terrain.py` verifies deterministic nonblank frames, authoritative anchors, map/HUD clipping, connected seams, wrapped adjacency, depth, environment changes, missing/corrupt dependency fallback, reset, and bounded cache/streaming diagnostics.

No Firaxis-derived pixels are committed. No injected code, Civ III address, runtime presenter, or M7/retained ownership changed in this step.

