# Civ V Environment Skin mountain study

## Result

`beauty-mountain-civ5-r4` is a deliberately isolated, macOS Metal-only visual
study. It uses the project's compiled `Civ5EnvironmentSkin` pack and does not
use the Civ III viewport, tile dimensions, compositor, Windows VM, or injected
code. The supplied Civ VI screenshot is a property reference, not a pixel-match
target.

The final frame is
`out/beauty-mountain-civ5-r4/h12-z1-pan00.bmp`; the adjacent PNG and detail PNG
are review conversions. The deliberately simplified control is
`out/beauty-mountain-baseline-r3/h12-z1-pan00.bmp`.

Direct visual review retained r4. It has a clean terrain transition, readable
ridge hierarchy, fine rock strata, restrained snow, cool skylight in the
shadowed faces, and a soft directional cast shadow. The control uses identical
source geometry and the same shader file but intentionally falls back to one
planar albedo projection, no authored mountain height/specular response, no
snow layering, and one sample. It reproduces the washed-together appearance
despite using the same pack.

## What this isolates

The standard mountain macro height and footprint are only 256 x 256. The base,
elevated, and snow material channels are 2048 x 2048, and the grass base color
is 4096 x 4096. Increasing output resolution cannot invent a more detailed
silhouette from the macro field, but the 2K material information can survive
when it is sampled and lit as surface detail instead of stretched once across
the whole relief.

The retained pass therefore uses:

- one mesh sample per macro-height texel (256 x 256);
- triplanar base/top/snow projection on steep faces;
- all available color, height, and specular channels;
- height-derived detail normals without modifying the macro silhouette;
- source footprint blending so the relief grows out of the grass rather than
  exposing a square asset boundary;
- scene-linear lighting with warm directional light and cool sky fill;
- three-tap terrain-height shadow visibility, 4x MSAA, and 16x anisotropy.

The useful diagnosis is that the assets were not inherently low-definition.
The largest losses came from surface projection, omitted material channels,
insufficient geometric sampling of the macro field, lighting that compressed
face separation, and final edge sampling. Those are renderer/presentation
problems; they are separable from Civ III tile packing.

## Evidence and boundary

The final 1536 x 1024 Metal frame hash is
`3ac8808024a27ed6f644dd7046ca2cc9e1793e1af216d0f12ff0184517de59e1`.
Its captured GPU scope is 22.1 ms with 4x MSAA, 16x anisotropy, 14 sampled
textures, and three draws. This is a visual prototype, not an optimized runtime
budget.

Focused verification:

```text
python3 -B -m unittest Renderer/terrain_lab/v2/tests/relief/test_beauty_mountain.py
python3 Renderer/terrain_lab/v2/app/runner.py validate --fixture Renderer/terrain_lab/v2/fixtures/relief/beauty-mountain.fixture.json
python3 Renderer/terrain_lab/v2/app/runner.py quick --fixture Renderer/terrain_lab/v2/fixtures/relief/beauty-mountain.fixture.json --candidate beauty-mountain-civ5-r4 --output Renderer/terrain_lab/v2/audits/relief/out/beauty-mountain-civ5-r4
```

The shared shader and backend-neutral packet structure deliberately keep a
future Direct3D path possible, but no Windows or D3D run was made for this
user-directed experiment. This result does not close LQ0, claim backend parity,
or promote anything into Game Integration.
