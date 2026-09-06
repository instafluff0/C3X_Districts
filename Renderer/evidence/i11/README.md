# I11 Approved Marsh Game Integration Evidence

Status: reopened after the 2026-09-05 live game produced a completely black map
plane while later native overlays remained visible. The standalone and injected
smokes below remain useful evidence, but they do not close I11. The current
diagnostic build emits short load, capture, render, ownership, and blit stages
through C3X's existing `OutputDebugStringA` path. A nonblack live custom frame
is still required.

I11 consumes the frozen, approved L9 terrain, L10 dune, and L11 marsh handoffs.
The production renderer verifies their byte-derived revisions and uses the exact
L11 `GrassMarsh` material channels plus the approved `CLUTTER_MARSH` projected-
decal composition. Until L12 is approved and I12 begins, volcano tiles keep
custom base-terrain ownership but omit the unapproved volcano relief.

The `i11_approved_marsh_integration` native gate proves both zooms, partial
clipping, pixel scrolling, duplicate horizontal-wrap occurrences, exact static
reuse, bounded cache invalidation, authoritative occurrence anchors, marsh
terrain/feature ownership, deterministic device reset, and zero production
fallback. The injected bridge gate proves custom-on `m19` never calls or replays
the native tile renderer, rejects any fallback count, retains later Civ III
overlays, and reports load/capture/render/validation/blit/reentrant failures as
visible hard failures. Configuration off still follows the original Civ III
path.

The consolidated Windows workflow builds the 32-bit DLL, runs the production
payload smoke, compiles and injects `ep.c`/`injected_code.c`, then copies and
byte-verifies the DLL, injected sources, renderer definitions, normalized packs,
and all three handoffs in the installed GOG `C3X_Districts` test folder. It does
not run `INSTALL.bat`; installation remains the user's interactive final step.

The live 800-record Civ III capture exposed a 32-bit/driver resource failure
that the original 14-record smoke did not cover. The production smoke now
includes that full capture volume. Runtime D3D textures begin at the first
authored mip no larger than 2048 pixels, retaining the source mip data and
screen-relevant samples while avoiding allocation of source-detail levels that
cannot contribute to Civ III's 128x64 maximum tile projection.

No new Civ III patch symbol or `civ_prog_objects.csv` entry is required.
