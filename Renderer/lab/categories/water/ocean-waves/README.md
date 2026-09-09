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
