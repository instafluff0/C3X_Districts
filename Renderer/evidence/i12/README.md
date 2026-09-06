# I12 Approved Volcano Integration Evidence

I12 consumes the frozen, user-approved `handoffs/L12_volcano.json` record and
the exact component mapping in `terrain/i12_handoff_fidelity.json`. Runtime code
does not parse approval records; it invokes only the integrated generic pack
paths.

The Windows production-payload smoke passed on 2026-09-05 with:

- a frozen byte-for-byte copy of the approved standalone
  `PSMain`/`PSFeature` implementation, reached through a Civ III input adapter;
  the live Lab shader is not a production include and cannot leak unapproved
  visual work into the game;
- exact base/real terrain identities, computed normals, material weights,
  signed shore distance, water depth, and distinct authored-relief and
  authoritative active-effect fields rather than inferred adapter values;
- the Lab-derived viewport-wide underlay, land, bed, water, then feature pass
  order instead of a tile-major two-pass approximation;
- a semantic production settings contract with no Lab fixture or milestone
  switches, and compile-time omission of later renderer systems;
- authored ordinary-volcano height/blend geometry and dormant/active material
  channels;
- authoritative `active_tile_effect` capture and cache invalidation;
- the approved 36/49 forest/jungle density at 0.42/0.40 scale;
- continuous shoreline relief flattening and projected land/water clutter;
- both 128x64 and 64x32 Civ III zoom bases;
- exact occurrence anchors, partial clipping, pixel scrolling, and horizontal
  wrapping;
- deterministic cache restore and device reset;
- exact terrain/feature ownership, retained overlay cache exclusion, zero
  fallback tiles, and hard failure without native m19 replay.

The corrected native bridge fixture emitted pixel hash
`7796647799848475592`; the corrected production terrain fixture emitted
`3127269838549132449`. The native smoke additionally removes and restores hill,
mountain, dune, vegetation, marsh, and volcano identities independently and
requires each approved contribution to change pixels and restore exactly. The
Windows workflow compiled the DLL and its HLSL adapter with warnings treated as
errors, then rendered all fourteen terrain identities through that DLL to the
ignored `preview/out/native_i12_current.bmp` with zero fallback. The installed Windows
`C3X_Districts` path was verified as the repository link used by interactive
`INSTALL.bat`, and the full workflow compiled the injected code successfully.
The user retains the final interactive `INSTALL.bat` step.

The 2026-09-05 corrective replay removed the remaining production-only terrain
shape approximation. Production now uses the Lab sampler's exact five standard
mountain variants, rigid transforms, connected footprints, chain maximum,
normalized height/blend sampling, hill macro/support field, topology envelopes,
dune envelope, and deterministic forest/jungle placement. The Civ III-to-Lab
lattice transform is consistently `source_x = column + row` and
`source_y = column - row`; the former reversed basis was the cause of diagonal
terrain bands, disconnected relief, and wrong feature neighborhoods. Terrain
displacement uses the approved 224 px tile and 0.82 vertical projection; feature
bodies retain the Lab's separate full 150 px basis. The preview harness now
loads the exact `C3X_BIQ_TERRAIN_WINDOW_V1/V2` Lab input and its halo. The latest
same-input production replay is
`preview/out/native_l12_same_view.bmp` (SHA-256
`590d5b2fa2ac06002318a57cd40d09245780753ef1a57facf5076451a5a3ed73`),
with zero fallback tiles. Stable production map coordinates intentionally keep
procedural texture phase fixed during scrolling rather than renumbering each
cropped viewport from zero as the standalone fixture does.

The frozen production shader is SHA-256
`2797c26c9b8d063648c853b4833b5610850207fc2285e5d7140a33f7dcfd0f08`.
Both it and the thin input adapter participate in the renderer content revision,
so shader changes invalidate cached terrain deterministically. The staged and
built Windows DLLs matched at SHA-256
`def187b8fb4b1ac34c0a6573b54d517075cfe6bdaccf73ec8641877091195433`
after final full verification.
