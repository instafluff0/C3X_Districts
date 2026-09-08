# Renderer integration contract

The current C3X checkout is authoritative. Historical L/I handoffs, approval
numbers and integration ledgers do not authorize current work and are not read by
the game DLL. Fixed category references are visual comparison aids only.

## Visual implementation

The production renderer consumes generic normalized/runtime packs and renderer
definitions. Current shaders and CPU geometry providers are shared under
`lab/shared`; native adapters retain D3D resource and game-scene ownership.
See [natural geometry/materials](../native/source_fidelity/README.md),
[city composition](../native/city_fidelity/CITY_FIDELITY.md) and
[unit rendering](../native/environment_refresh/UNIT_FIDELITY.md).
Source-game formats stay in offline importers, not runtime conditionals.

Preserve authored material channels, normals, transforms, stable placements and
pass ordering. Terrain identity is categorical, not a value to interpolate;
material weights and continuous height fields provide transitions. Projection
uses authoritative Civ III anchors. Shared world-coordinate sampling and
canonical wrap identity keep neighboring tiles, relief and shadows continuous.
The [environment contract](environment_lighting_and_ambient_effects.md) governs
sun/moon, ambient, exposure, water and emission together.

Preserve the distinction between canonical map coordinates and screen occurrences.
Renumbering each cropped viewport changes procedural phase while scrolling.
For diagonal relief/vegetation artifacts, check the source lattice orientation
and ground-anchor depth before altering assets: reversing the source basis or
using already-lifted screen Y for ground depth previously caused disconnected
relief and bodies clipped behind neighboring terrain.

## Capture, caching and compositing

Civ III owns map state, tile/object anchors, visibility and action timing. The
renderer produces an off-screen map image, not another presenter or game loop.
The bridge inserts it below retained fog, labels, selection, borders, HUD and UI.
The [patch ledger](civ3_patch_dependency_ledger.md) records the actual boundaries.

A dirty clip specifies where to composite; it does not authorize incomplete
scene capture or stale terrain reuse. Preserve complete visible-scene inputs,
ordered/wrapped occurrences, exact clip ownership and pixels outside the clip.
Retain the existing worker/device lifetime and serialized surface-transfer rules.

Caches must observe renderer-owned content, neighborhood/topology dependencies,
world dimensions/wrap, zoom, environment, pack/definition changes and device
reset. Native overlay activity must not turn unchanged retained terrain into
animated content. Camera translation and incremental reuse must agree with a
cold render within the applicable executable check, not an old timing report.
Keep memory and pending work bounded for the 32-bit game process.

Custom-on owns the existing map plane: rendering failure must remain visible,
not silently replay native terrain into it. Config-off preserves the native
path. Unit fallback is a separate per-body contract; do not infer map-plane
fallback from it. Resources and cities do not replace native advisor, city-screen,
Civilopedia or HUD graphics. Wonders and Districts remain deferred as documented
in the [renderer architecture](../README.md).

## Verification and delivery

Use `python3 Renderer/renderer.py integration CATEGORY` to build/reuse an
isolated candidate and run relevant native checks for the category and its
declared behavior consumers. It does not read reference images; `lab` and
`compare` remain explicit visual review actions. The command does not stage the
DLL, install C3X, launch Civ III or certify a game test. Only explicit visual
acceptance replaces a fixed reference.

Keep executable coverage for capture, ownership, invalidation, scrolling,
wrapping, both zooms, compositing/clipping, animation, reset and config-off.
Exact category screenshots do not replace those behavior checks. The known
terrain-edit reuse failure remains unresolved; see the workbench's current
limitations. Old cache timings, handoff hashes and historical passes are not
current acceptance evidence. Preserve the [visual fidelity playbook](visual_fidelity_playbook.md).
