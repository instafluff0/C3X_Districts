# Ocean waves

Animated coastal breakers follow the continuous shoreline. Only ordinary beaches
receive foam; hills, mountains, cliff shoulders and land are excluded. The
optional generic pack preserves all 16 crest variants, auxiliary foam and
per-row delays. See [source findings](../../../../docs/ocean_wave_findings.md).

`python3 Renderer/renderer.py lab ocean-waves` renders beach, rocky control and
mixed coasts in daylight and moonlight. `test ocean-waves` checks placement and
source recovery; `integration ocean-waves --renderer-only` checks production
playback, repeatability and cached terrain.

The production compositor overlays waves in bounded blocks over retained static
color/depth at 15 Hz using the existing ambient animation clock. Missing or
disabled wave packs leave the terrain intact. Set the normalized pack's
`enabled` flag to false and prepare to disable; `C3X_RENDERER_WAVES=0` is a
process-level diagnostic control. References have not been replaced.

The scene-linear foam pass writes premultiplied color, matching the renderer's
existing blend state. Geometry remains static across time samples, and visibility
uses the same captured topology and native overlay ownership as the water below.

Local source recovery is separate from production preparation:

```sh
python3 Renderer/tools/asset_compiler/wave_blp_extractor.py --output Renderer/packs/CoastalWavesNormalized
# Set enabled to true in the generated wave.json after reviewing the local art.
python3 -m Renderer.tools.asset_compiler.build_wave_runtime
```

The extractor defaults to disabled and never enables runtime effects by itself.
The current local normalized pack is enabled as requested. Source and derived
licensed textures remain ignored; runtime code reads only generic DDS/CWV1 data.

The current quality correction preserves faint unmarked atlas rows, uses the
embedded auxiliary foam at a visible footprint and smooths ribbon directions
across short contour edges. See the source findings for confirmed evidence
versus authored calibration. The isolated before/material/geometry comparison
lives under `Renderer/lab/out/waves-quality/`; fixed references remain unchanged.

The subsequent longer-front and shore-break experiments were not selected after
comparison with the closer Civ VI reference. The preceding feathered baseline
remains the production appearance; source findings and motion/cove tests were
retained.
