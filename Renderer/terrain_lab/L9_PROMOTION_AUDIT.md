# L9 Eight-Tile Promotion Audit

## Scene contract

The corrected L9 promotion candidate is an explicit 8x6 grid of 48
Civ III-proportioned 2:1 diamond cells. A terrain-conforming blue grid witness
makes the cell boundaries and count unambiguous. Terrain, authored relief,
material UVs, grass-to-sand mixing, beach and shallow-bed geometry, water
channels, shadows, and feature placement still operate in shared world space
across those edges. The native render is 2048x1280 and the readability reduction
is 512x320.

The complete scene combines only normalized Civ VI source assets already
accepted by the preceding gates: Grassland, Plains, Desert, Beach, Shallows,
Ocean, the standard
mountain variant 01 height/blend/region package, the Coast water density and
surface stack, crash-foam/ripple/turbulence effects, and real forest/jungle
bodies with their extracted placement records. No proxy vegetation, invented
beach bitmap, cliff, or shoreline-rock dressing is present.

The deterministic cell assignment uses six dense forest cells in the northwest,
six dense jungle cells in the southwest, and a full clear column before the
central relief block. Two mountain cells form a taller short range without any
vegetation composited across their silhouettes. Eight dedicated hill cells use
the real normalized `standard` Civ VI hill heightfield: four under plains and
four under desert. The eastern six-cell contour supplies coastal plains, coastal
desert, and sea adjacencies. Feature centers remain clamped to their assigned
cells, and forest and jungle never share a tile.

The authored mountain height/blend footprint and its material mask use the same
cell-local coordinates and taper before each mountain-cell edge. The exact same
8x6 grid and camera are used for complete, no-vegetation, no-water, and no-surf
witnesses.

## Verification

The Windows Direct3D 11 lab builds cleanly under `/W4 /WX`. Nine focused
vegetation and shoreline extractor tests pass. The frozen comparison hashes
remain unchanged:

- L9.3 complete: `44e86d1c71166c85325ddf38a92b9aaa52a25d39487611f2bd60bfacae766464`
- L9.5 complete: `4dbfaf0a0c249142deefbc3d2f6169616c5303ab27576328cf1f531c4b73a600`

Current promotion BMP hashes are:

- complete: `52c0238eacbff8b08128267038c4516f059b862ca749529d2d7eb25015cbb753`
- no vegetation: `2a86776fcad4c8d569f39034da5b4ed6c9f27241195f93590ea93daf4aeeb490`
- no water: `2523415756834175ed950a21fe3f97b0a28c64611ba48830e08cf792f50ff91e`
- no surf: `450528a0e08b0f8c6c9c185b3f2d7c43ef5282795fdc9c67e9aaf5fd612daa1e`
- thumbnail: `a21bf4f1d809b644d7cbff7717af79e09ee89cc334bc20ea20cfdad430b09129`

These hashes establish reproducibility. The user explicitly approved the
48-tile promotion render on 2026-09-04, completing L9 and advancing the lab to
L10 dunes.
