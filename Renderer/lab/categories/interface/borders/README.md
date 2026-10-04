# Territory borders

## Current game integration

The selected brush treatment now runs in `Renderer/native/gpu_territory_borders.h`.
Civ III supplies native four-edge ownership, wrapping, the border visibility flag,
and the effective civilization palette color. Ownership/color changes invalidate
the retained scene. Renderer64 draws directly on its existing terrain/mountain
triangles and samples GPU scene depth to fade occluded sections. There is no CPU
composition or normal-frame depth readback in this path.

Coastal border receivers retain the whole tile grid, including triangles whose
terrain color is fully transparent on the wet side. Open-water borders reuse the
flat underlay. Each tile chooses one receiver, preventing double opacity where
natural ground and underlay coexist. The terrain/shadow coast clipping remains
unchanged; border coverage depends on native ownership rather than land alpha.

`test_city_border_capture.py` verifies native edge/palette/config-off capture;
`test_territory_borders.py` verifies ownership removal/capture, zoom, translated
occurrences and occlusion at 1/2/4 samples. `capture_city_border_examples.py`
produces standalone Renderer64 examples. Broad live-game regression is deferred
while Lab systems are integrated.

The dynamic pass culls receivers against the inverse displayed viewport, including
0.5×, 0.625×, 0.75× and 0.875×. Using the unscaled native viewport omitted entire
border receivers near the outer coast. GPU tests cover all four outward scales;
GOG capture `20261004-112852` shows the formerly broken southern coastline as a
continuous outline through the zoom/scroll sequence, with no renderer failures.

## Original Lab study

`python3 Renderer/renderer.py lab borders` draws a city and terrain through the
current renderer, then composites three border drafts around a synthetic group
of Civ III diamond tiles. The contact sheet is `Renderer/lab/out/borders/examples.png`.
The fine and brush cases compare stroke weight; the azure case demonstrates that
the same shape accepts a different civilization color. Exposed diamond-tile
edges stay straight through their centers, with small rounded turns at tile
corners. The outline is continuous, including through forest. Its opacity fades
across the stroke's width, leaving a stronger center and soft edges. Every
part of the ribbon uses the selected civ RGB color, with no white outline.
A visible semitransparent band in that same color continues into the owned
territory. Its width and opacity vary gently along the edge, then taper to
transparency; the outside has no matching band. Each inset contour samples the
renderer ground mesh and depth, so the band follows relief and fades behind
foreground geometry along with the stripe.
Adjacent owned tiles have no interior strokes.

The sample ownership, color, and city are a visual fixture. A Lab-only exporter
captures the exact ground and replacement mountain triangles built by the
production renderer for the same frame. Border points are projected onto those
triangles before the color stroke is composited. A Lab-only depth readback of
the finished city pass checks which parts of the line are behind hills,
mountains, trees, or other rendered foreground objects. Only those parts are
made more transparent; the exposed line keeps its full opacity. Small depth
differences and isolated samples do not trigger the fade, preventing spots on
clear ground. This study fixture does not change game capture. The GPU integration above
uses the same visual treatment with native ownership and palette inputs.

First capture the `test.biq` city and its matching terrain mesh through the
Windows VM (after building the isolated `city-preview` renderer):

```sh
python3 -m Renderer.lab.studies.borders.capture_test_biq
```

That capture also writes `test-biq-occlusion-examples.png`, a four-panel sheet
with the original hill crossing plus mountain, forest, and combined crossings.
The three full-size variants are `test-biq-mountain-crossing.png`,
`test-biq-forest-crossing.png`, and `test-biq-mountain-and-forest.png` in
`Renderer/lab/out/borders/`. They use the same `test.biq` city and renderer
frame, with different invented contiguous ownership groups and civ colors.
The script checks that every selected tile is land and that each variant has
an occluded border segment in the renderer depth readback.
`test-biq-inward-detail.png` enlarges a quiet grass edge of the forest variant
to inspect the subtle one-sided color fade.
`test-biq-clear-north-detail.png` and `test-biq-clear-south-detail.png` show
the two formerly spurious opacity changes after the depth-confidence check.

To try another civ color on that paired capture without rerunning the VM:

```sh
python3 -m Renderer.lab.studies.borders.preview \
  --background Renderer/lab/out/borders/test-biq-ground.bmp \
  --terrain-csv Renderer/lab/out/cities/test-biq/terrain.csv \
  --mesh-prefix Renderer/lab/out/borders/test-biq-ground \
  --site 20,64 --tile-width 256 \
  --output Renderer/lab/out/borders/custom-color.png \
  --case crimson-brush --color '#4A9D55'
```

The backdrop's terrain is captured from `test.biq`; the city placement and
territory ownership are synthetic Lab inputs. The selected territory cells are
checked against the captured map and must all be land tiles. The exported
ground triangles contain the renderer's actual mesh elevations, including the
joined mountain replacement surface. The sample territory leaves the steep southern hill tile outside its
boundary, so this example avoids a narrow crest while still following exposed
tile edges and the exact surface beneath them.
