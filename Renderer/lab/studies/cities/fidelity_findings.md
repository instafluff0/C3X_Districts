# City fidelity diagnosis — Lab candidate

The current city design is **not visually accepted**. These observations come
from headless D3D replay of captured `test.biq` terrain with synthetic city
flags, not from an in-game screenshot. The replay uses the isolated Lab pack;
it does not stage a DLL or replace any accepted reference.

## Staged evidence

`fidelity_probe.py` draws the same AncientWood B-family house by itself at the
layout scale, then at diagnostic 1.5× and 2× uniform scales. It then draws the
palace alone, the full town, the town with walls, and the walled town on a hill.
The source mesh, UVs, material references and textures are byte-identical in
the isolated packs. Only one composition's instance count or its uniform
scale changes. The rendered comparison is at
`../../out/cities/test-biq/gallery/fidelity-probe.html`; the compact crop sheet
is `../../out/cities/test-biq/gallery/fidelity-stages-crops.png`.

At the maximum 256-pixel tile width accepted by this replay harness, the
selected house spans roughly 40 pixels at layout scale, 60 at 1.5×, and 80 at
2×. Roof edges, eaves and facade relief become substantially easier to read
as the footprint grows. The 1024×512 source color atlas visibly contains
thatch and wood detail, but that detail cannot all survive in a roughly
40-pixel-wide projection. The current four-house town plus palace is more
crowded, yet the isolated house already has the same small-scale softness.
Composition makes the problem more obvious; it is not its starting point.

`shader_probe.py` holds the same isolated house and site fixed, then compares
the normal material, unlit base color, forced top mip, a normal visualization,
and an additional −0.75 mip bias. The 1× and 2× comparisons are at
`../../out/cities/test-biq/gallery/shader-probe.html` and
`../../out/cities/test-biq/gallery/shader-probe-2x.html`. On a fixed 73×73
pixel neighborhood at layout scale, unlit sampled color versus forced mip 0
differs by 0.114 mean 8-bit channel values; an extra −0.75 bias versus the
normal material differs by 0.042. At 2×, the same extra bias produces no
pixel change in the tested 115×110 neighborhood. The normal view shows a
formed roof and facade, and the normal material has stronger local relief than
unlit color. Changing mip selection or removing lighting does not recover the
missing fine detail.

The renderer currently uses 2× scene-linear render scale, 4× MSAA, a 16×
anisotropic sampler with −1 mip bias, and a final equal-area reconstruction.
This confirms that anti-aliasing and high-resolution intermediate rendering
are active. Those settings smooth silhouette coverage but cannot invent
facade pixels in the finished Civ III-sized image. The source engine's full
city shader is not reproduced: the second LEAN/normal channel and parts of
its gloss/environment response remain unresolved, so this test does not claim
material parity with Civ VI. Those differences may affect surface character,
but the size experiment identifies a major, earlier readability limit.

## Design consequence

Do not add global sharpening or further negative mip bias to this candidate.
The next Lab composition should set a minimum **screen-space building size**
at the intended game zoom, then place intact source buildings around the
palace. Recheck the town's one-tile boundary, capital orientation, roof
clearance and wall ring after every size change. A trial using the current
greedy layout search cannot fit all four town houses even at 1.1× with the
existing target positions and palace footprint; that is a placement-search
result, not proof that no alternate arrangement exists. Cities and
metropolises have more space because their approved design may sprawl beyond
one tile. Source model selection, especially taller or more legible B-family
bodies, remains another design control. Compare each candidate at native output
size before using magnified sheets for aesthetics.

The wall ring and hill foundations require their own visual acceptance. The
current hill candidate uses small separate retaining pads under buildings;
walls stand beyond them on natural terrain. Neither should be merged into a
single masonry slab.

The current 20-case flat replay contact sheet is
`../../out/cities/test-biq/gallery/flat-contact.png`. It uses native-size
image crops with no enlargement. Several culture and era cells visibly share
building families, especially the modern row. That is a separate source
selection problem: more pixels alone will not make those cultures distinct.

## Larger layout audition

`screen_size_trial.py` keeps the canonical `layouts.json` unchanged and builds
an isolated Asian ancient pack. Two B-family houses grow uniformly by 1.5× at
all three population tiers; four houses still fit inside the town's building
boundary, and no house overlaps either the civic center or palace footprint.
The trial repositions houses only, never stretches mesh axes or rotates a
palace. Its flat and hill replay is at
`../../out/cities/test-biq/gallery/screen-size-trial.html`.

The larger roofs read more clearly, but the capital still hides much of its
palace behind the front row. On a hill, neighboring retaining pads visually
join into a broad masonry face. This remains a **diagnostic alternative**, not
a replacement layout. It demonstrates that uniform source scale gains pixels,
and it also shows that the composition and hill support must be redesigned
together before a larger building policy can be accepted.
