# M6.7 Approved Terrain Integration Evidence

Status: complete through paired I11. The earlier live Windows screenshot exposed
a stale installed tree and native/custom mixing. The workflow now byte-checks
the actual VM `C3X_Districts` tree, and custom-on ownership is exclusive: no
native `m19` terrain is called or replayed on any failure path.

I9-I11 consume only the frozen, explicitly approved L9 terrain, L10 dune, and L11 marsh
handoffs. The tracked fidelity contract is
`Renderer/terrain/m6_7_handoff_fidelity.json`; the approved lab BMPs remain the
visual source of truth and are verified against their handoff SHA-256 values
when present locally.

The 32-bit production smoke runs twice. Its portable synthetic pass protects the
generic DLL ABI. When ignored normalized packs are available, the
`i11_approved_marsh_integration` pass opens the real production definition,
validates all three exact frozen handoffs, loads terrain, dune, vegetation, and
marsh material/decal payloads, and proves that dune, forest/jungle, and marsh
bodies materially change pixels before acquiring ownership.

The automated matrix covers both Civ III tile bases (128x64 and 64x32), exact
pixel anchors, partial clipping, pixel scrolling, a duplicate horizontal-wrap
occurrence, deterministic restoration after each change, device reset, static
idle reuse, bounded one-entry caching, pack/definition revisions, environment
changes, and zero production fallback. Roads, resources, cities, and units are
mutated independently and must remain cache-neutral because Civ III retains
those later planes.

The injected bridge captures authoritative real/base terrain identity,
SquareParts, visibility, canonical coordinates, and each screen occurrence. It
transfers per-tile terrain, feature, and dune ownership only after successful
render and blit, and does not call the original multiplexed `m19` renderer while
custom rendering is enabled. Load, capture, render, validation, blit, device,
and reentrant failures never replay native terrain. Device recovery is attempted
on a later normal Civ III redraw; until then the failure remains visible.

L11 marsh is consumed through its frozen `GrassMarsh` base/height/specular and
`CLUTTER_MARSH` projected-decal channels. Tundra and flood plains are part of
the approved L9 base terrain. Until L12 is approved and I12 runs, volcano tiles
render their custom base terrain without volcano relief; they never fall back
to native art.
The retired procedural color grading and shoreline rock/cliff dressing are not
part of the approved production path.

No new Civ III patch symbol or `civ_prog_objects.csv` entry was required. The
focused workflow is `python3 Renderer/tools/renderer_dev.py integration`; the
closing workflow is `python3 Renderer/tools/renderer_dev.py full`.
