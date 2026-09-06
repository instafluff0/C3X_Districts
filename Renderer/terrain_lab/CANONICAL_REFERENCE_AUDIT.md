# Alternate-Skin Canonical Reference Contract

The user-supplied, locally ignored screenshots under `Renderer/canonical/` are
the qualitative art-direction target for Renderer Lab beginning with L13. They
show Civilization VI running the Civilization V Environment Skin. They are not
pixel-equality targets and are never runtime inputs.

## Preview pack policy

L13 and later lab previews use `Civ5EnvironmentSkin` for all fourteen terrain
materials and `Civ5EnvironmentVegetation` for forest/jungle bodies. Baseline
`TerrainNormalized` or `VegetationNormalized` output is not valid promotion
evidence from this point forward. The alternate pack currently inherits generic
water micro-normal, river-source, terrain-relief, and effect channels for which
the overlay compiler has no source-resolved replacement. That is explicit pack
inheritance, not permission to switch the visible terrain or vegetation skin.

The locally built alternate vegetation pack resolves the overlay's 17 added
broadleaf bodies as well as its retained bodies. Ordinary forest cells prefer
the warm broadleaf subset; the lab does not synthesize replacement tree art.

## Locked local references

| File | Pixels | SHA-256 | Primary observations |
| --- | ---: | --- | --- |
| `daynight.png` | 1608x1368 | `5e210083753c9938478c636815bf50ae6d2f5d82f4fe9cd2a45c33357db65c5a` | Panels are noon, 18:00, midnight, and 06:00 in reading order: compact noon grounding, long dawn/dusk object and vegetation shadows, warm dusk, cool readable midnight, and cooler dawn recovery. |
| `desert.png` | 1008x620 | `e270f22003589504e22de2b0f4a7365de8a109e437c44e12b5f21e079ef8b61f` | Fine ripples, broad soft boundary, no square material edge. |
| `forest.png` | 660x502 | `4c94f820423486ef056b4a7fc3e41d3f985efc022ff48f3a09dc5bf3b8dee36f` | Low, dense broadleaf crown mass with an irregular perimeter. |
| `hills.png` | 600x382 | `c0f10967a42915efef0d7b518f453439dd10c3b8cb2b46cad11430b7a1b98c1f` | Wide low relief, grass-covered shoulders, sparse exposed rock. |
| `jungle.png` | 1036x598 | `2a665669fdd56176e034f127459c4054e3a3dca8ab0073bd1607039fe9b9c2c1` | Dark layered palms mixed with brighter understory and no tile outline. |
| `marsh.png` | 398x258 | `1f1ad37936013598e74261ea868c27bd7940ffad4007d1d39be5bea0614367b3` | Shallow irregular pools embedded in muted wet ground. |
| `mountain.png` | 584x506 | `18bc5ea32b6276786a5db444a928fec153b4ada25ff21015eefa677dc05b4661` | Interlocked rocky masses; valleys and shoulders bridge adjacent peaks. |
| `river.png` | 1954x970 | `ebfbcc875f4469601b961b609188d4026afbeadac9fdb4c03ff9e0d7726a9766` | Subdued gray-blue channels, thin terrain-colored banks, smooth bends, grounded junctions and coast mouths. |
| `sea_and_shore.png` | 3430x1750 | `8a2b7561d5b1fb9a33276196cd5a520c3953373381d2083232d923f75155e3e8` | Visible shallow bed, sparse rocks/contours, narrow surf, gradual deep-water value shift. |

## L13A day-night checks

`daynight.png` supersedes ad-hoc day/night screenshot comparisons for L13A.
The Lab fixtures use its exact six-hour phases: noon `12`, evening `18`,
midnight `0`, and morning `6`. Evaluation remains continuous between those
points, but the four locked outputs must preserve the reference's ordering:

- noon is the clearest and brightest phase, with readable cast shadows;
- 18:00 is distinctly warm and carries equally readable hill, vegetation, and
  renderer-owned body shadows rotated with the light;
- midnight is substantially darker and blue-weighted while land, water,
  coastline, rivers, and relief remain legible;
- 06:00 recovers through a cooler, less orange palette than 18:00 and retains
  the same stylized shadow lengths in the opposing direction.

Forest/jungle shadows derive from actual normalized mesh height rather than
canopy width alone. Direction rotates between phases, but a fixed stylized
projection slope keeps length in the canonical image's tight visual band.
Terrain cast shadows use the same rule through a filtered multi-ray height-field
test, while hill face contrast is weighted by continuous geometric slope instead
of BIQ ownership; neither mechanism may reveal a hidden tile diamond. Every
raised class must show both normal-driven face shading and a cast footprint.
Projected body shadows track the receiver's screen-depth gradient so every
direction remains visible. Flat land does not invent raised shadows, and static
frames do not request continuous redraw.

## L13 river checks

The complete 192-cell regression viewport must retain every terrain family in
the current user-supplied canonical set plus the earlier volcano and floodplain
layers. A second 192-cell river-topology viewport concentrates on longer
channels. Together they must visibly exercise:

- shared-edge continuity through grassland, plains, hills, mountains, forest,
  jungle, and marsh;
- a source, bend, junction, coast mouth, and horizontal map wrap;
- narrow banks that inherit the neighboring terrain rather than forming a
  cream outline;
- gray-blue water with source-authored bed, lean, height, specular, clutter,
  bank-noise, and sparse rock inputs;
- continuous relief carving without a mountain/hill sprite cutting the river;
- deterministic variants and unchanged non-river output in the no-rivers view.

Every noisy boundary is evaluated from a canonical shared edge or continuous
world coordinate. A visually hard diamond, crack, detached mouth, duplicated
reciprocal edge, or raw-coordinate wrap jump fails the gate.

Static waterfall textures exist in the inherited normalized water catalog, but
the Civ III BIQ river mask has no flow direction or waterfall state. L13 does
not guess one and does not add smoke, mist, or animation. A future explicit
scene field can admit a static waterfall sheet without changing river topology.
