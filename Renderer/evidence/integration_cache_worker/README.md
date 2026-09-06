# Integration Cache And Renderer Worker Evidence

This maintenance increment preserves the approved L9-L19 production rendering
contract while removing screen anchors from reusable world-tile work.

Automated source contracts in `Renderer/native/test_native_bridge_contract.py`
prove:

- a 32-entry exact viewport LRU bounded to 128 MiB;
- canonical world-coordinate semantic records whose content keys exclude screen
  anchors and retained Civ III overlays;
- anchor-independent surface, relief, normal, and environment-specific shadow
  samples bounded to a 96 MiB LRU;
- two bounded compiled GPU regions, with 192 MiB total and a 96 MiB per-entry
  ceiling so a full expanded live viewport is never duplicated in the 32-bit
  process;
- a deep-copied frame/tile payload consumed by one renderer worker, exact
  sequence completion, worker-owned D3D/reset, UI-thread final blit, and worker
  shutdown before `FreeLibrary`.

`Renderer/native/native_smoke.cpp` proves exact recent-viewport hits, bounded
eviction, uniform anchor translation, a small compiled-region revisit, reset
equivalence, both game zooms, clipping, horizontal wrapping, retained overlays,
zero fallback, and a changed visible tile set that cannot hit the exact viewport
or whole-viewport geometry cache.

The 2026-09-06 Windows integration replay reported:

- 400 rendered tiles / 800 captured records cold: 5314.920 ms;
- uniform three-pixel/two-pixel camera translation: 26.318 ms;
- one logical tile plus companion crossing the visible boundary: 1948.219 ms;
- zero fallback and deterministic approved-scene completion.

The tile-boundary result is a measured intermediate improvement. It still
rebuilds and uploads expanded viewport geometry; future indexed/chunked GPU
buffers and staging-readback work remain valid optimizations. No stale custom
frame and no native terrain fallback is permitted while those are pending.

No new `civ_prog_objects.csv` symbol is required. The only injected lifecycle
change calls the existing renderer reset export before unloading the DLL.
