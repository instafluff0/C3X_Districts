# I14-I18 Game Integration Evidence

Status: complete through the frozen, approved L18 handoff. L19 remains a
Renderer Lab-only work item and is not present in the production DLL.

The production renderer now consumes the same approved algorithms, normalized
payloads, selector policy, placement, materials, and shared lighting used by the
L14 road, L15 railroad, L16 resource, L17 city, and L18 mine promotions. Runtime
code uses production names (`route`, `resource`, `city`, and `mine`) rather than
Lab milestone names and never reads handoff approval metadata.

Authoritative capture and ownership are proven for:

- road and railroad topology plus the visible civilization's era/style;
- visibility-conditioned resource ID, class, and name;
- city identity, owner, size band, culture group, era, capital, and walls;
- mine presence, pillage-sensitive improvement state, era family, stable
  coordinate variant, resource/terrain context, and day/night emissives.

`Renderer/native/native_smoke.cpp` renders an approved mixed fixture at both
Civ III zoom bases. It requires distinct route, railroad, resource, walled-city,
and industrial-mine ownership bits, then removes and restores each relevant
state to prove content invalidation and byte-identical restoration. The same
fixture proves exact clips, pixel scrolling, duplicate horizontal-wrap
occurrences, retained unit-overlay selectors, and deterministic device reset.
Any configured production payload failure rejects the custom frame visibly;
the bridge never replays native map-plane pixels while custom rendering is on.

The terrain cache fingerprints only custom-renderer pixels. Unit animation,
selection, native `SquareParts`, retained overlay bits, fog traversal, and exact
population counts no longer evict static terrain. Renderer-owned routes,
resources, cities, mines, rivers, environment, anchors, wrap, content revision,
ownership, and device generation still invalidate. The current implementation
retains up to 32 exact viewports within 128 MiB so ordinary unit-cycling camera
jumps can return at zero renderer ticks; a native test fills the bound, proves a
recent-view hit, and forces deterministic LRU eviction. Anchor-only scrolling
reuses retained GPU geometry, while changed visible sets reuse a bounded
canonical world-tile surface/relief/shadow sample cache.

The Windows Integration report is
`Renderer/verification/i18_cache_integration.json`. Its isolated approved
production replay reports zero fallback and deterministic L9-L18 ownership and
reset. Current post-I18 performance evidence is recorded in
`Renderer/evidence/integration_cache_worker/README.md`.

No new Civ III patch symbol or `civ_prog_objects.csv` entry was required. The
existing m19/m71 capture boundary and loaded Civ III tile, resource, city,
leader-era, and improvement structures supply the complete I14-I18 state.
