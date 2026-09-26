# Territory borders

`python3 Renderer/renderer.py lab borders` draws a city and terrain through the
current renderer, then composites three border drafts around a synthetic group
of Civ III diamond tiles. The contact sheet is `Renderer/lab/out/borders/examples.png`.
The fine and brush cases compare stroke weight; the azure case demonstrates that
the same shape accepts a different civilization color. Exposed diamond-tile
edges stay straight through their centers, with small rounded turns at tile
corners. The outline is continuous, including through forest. Its opacity fades
across the stroke's width, leaving a stronger center and soft edges. Every
part of the ribbon uses the selected civ RGB color, with no white outline.
Adjacent owned tiles have no interior strokes.

The sample ownership, color, and city are a visual fixture. This category does
not change game capture, native border ownership, the renderer DLL, or any
approved reference. If the design is later brought into game rendering, Civ III
tile ownership and its effective civilization palette must supply the inputs.

To try another civ color on an existing city bitmap without running the VM:

```sh
python3 -m Renderer.lab.studies.borders.preview \
  --background Renderer/lab/out/cities/your-city.bmp \
  --output Renderer/lab/out/borders/custom-color.png \
  --case crimson-brush --color '#4A9D55'
```

For the map-backed `test.biq` city at tile `(20,64)`, use the existing City Lab
headless capture and its BIQ terrain CSV:

```sh
python3 -m Renderer.lab.studies.borders.preview \
  --background Renderer/lab/out/cities/test-biq/gallery/flat-asian-ancient.png \
  --terrain-csv Renderer/lab/out/cities/test-biq/terrain.csv \
  --site 20,64 --tile-width 256 --case crimson-brush \
  --output Renderer/lab/out/borders/test-biq-city.png
```

The backdrop's terrain is captured from `test.biq`; the city placement and
territory ownership are synthetic Lab inputs. The selected territory cells are
checked against the captured map and must all be land tiles. For this visual
draft, a continuous height proxy derived from the BIQ hill/mountain tile types
lifts densely sampled border points.
This is an inferred relief treatment, not a sample of the renderer's exact
terrain mesh. Production terrain attachment must query the renderer's actual
surface height and use its depth buffer for occlusion.
