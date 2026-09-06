# M6.6 Evidence

Automated gate: `m6_6_vanilla_terrain`.

Local licensed-source and BIQ gate: `m6_6_local_biq_terrain`.

The local gate rebuilds all fourteen normalized terrain materials, extracts and validates typed Civ VI terrain height resources, parses the user-supplied 100x100 `Scenarios/test.biq` through C3X Editor, and renders two deterministic native screenshots through `C3XRenderer.dll`. It also renders a closed all-fourteen-type fixture and repeats the large preview byte-for-byte.

Generated local evidence lives under `Renderer/preview/out/` and is intentionally ignored because it contains converted Firaxis-derived pixels.

The ownership matrix is `Renderer/terrain/m6_6_runtime_coverage.json`. Retained Civ III vegetation, ice, shoreline, and landmark families remain explicit evidence of atomic fallback, not omissions or placeholder replacements.
