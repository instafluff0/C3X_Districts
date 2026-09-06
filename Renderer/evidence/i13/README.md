# I13 River Integration Evidence

I13 consumes the frozen approved `handoffs/L13_rivers.json` record without
runtime approval logic. Production uses Civ III's captured `river_code`,
canonical map coordinates, distinct screen occurrence anchors, horizontal-wrap
state, clip rectangle, and a topology-only halo. The renderer ports the approved
shared-edge graph, deterministic curves, valleys, sources, junctions, coast
mouths, LEAN water, bank/source/clutter channels, and five normalized river-rock
bodies. It contains no road shader or geometry path.

`terrain/i13_handoff_fidelity.json` freezes the handoff/reference hashes and the
production shader revision. `native/test_native_bridge_contract.py` proves asset
identity, physical-edge mapping, valley constants, register bindings, ownership,
cache identity, atomic river omission, and no native river replay. The Windows
`native_smoke` run additionally proves authoritative mask invalidation, exact
replacement flags, both zooms, clipping, scrolling, wrap, device reset, bounded
800-record capture, and zero fallback. `renderer_dev.py integration` renders the
approved 192-tile L13 fixture through the production DLL with its 128-record
topology halo excluded from draw ownership.

The passing replay hashes are:

- near/noon: `55b04776307f0e24dcaaca55f1ef73fc1f2cbc309e8bcad79b11eb676ce8eca7`
- far/sunset: `9a9ac7a93b74d38eb6c54a67d1f08ced630024542f3878d0d55cd519aeece6c4`

No new Civ III patch symbol or `civ_prog_objects.csv` entry is required.
