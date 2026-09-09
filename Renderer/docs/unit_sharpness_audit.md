# Unit sharpness audit

Audit date: 2026-09-08. Scope: the current working checkout, local unit packs,
the supplied two screenshots, and existing production Lab captures. Review only;
no rendering behavior, staging, reference acceptance, or injected patches changed.

## Findings

1. **The supplied C3X image is approximately a 2x enlargement of a small Lab
   render.** The matching scene is `lab/out/shadows/detail/detail-h12-z128.bmp`,
   640x480. The supplied image is 1080x764 and shows a cropped part of that scene.
   A sampled RGB registration against the existing BMP fits scale 2.00125 and
   crop offset approximately (25.75, 71.75) in enlarged-image pixels. The mean
   absolute channel error, capped at 30 per channel, is 4.24/255; this is a
   geometric match, not a claim of byte-identical images or identical lighting.
   Matching displayed figure sizes therefore does not establish matching rendered
   pixel budgets. Enlargement cannot recover the original pixels' lost detail.

2. **Unit scale is intentionally small, and the showcased pose accentuates it.**
   `tools/asset_compiler/build_unit_animation_runtime.py` fits each kit's first
   idle stance to a maximum 56-pixel extent across both screen axes and all eight
   yaw directions. Weapons and accessories participate. This is an extent of
   the complete kit, not a 56-pixel body-height guarantee. The fixed uniform scale
   then applies to every action. The Lab fixture uses move, direction 3, cursor
   7 of 16, explaining the bent-forward Warrior. The 191x191 sprite is mostly
   placement and clipping space, not 191 pixels of anatomy.

   Applying the runtime palettes, fixed fit, yaw and native projection gives
   these continuous vertex envelopes at normal zoom (not raster coverage counts):

   | Subject/action | Complete visible kit, width x height | Head component |
   | --- | --- | --- |
   | Warrior move | 54.20 x 34.48 px | 7.86 x 9.49 px |
   | Warrior idle, sampled phase | 34.62 x 53.55 px | 7.83 x 11.21 px |
   | Settler move | 35.21 x 48.69 px | 6.12 x 9.02 px |

   Measurement uses the current `UnitAnimationFidelity` payloads, palette phase
   7/16 for looping clips, yaw offset plus 3*45 degrees, and nonnegative posed
   height. Projection is `(x-y)*64, (x+y)*32-z*(150*128/224)` after uniform fit.
   Thus the supplied Warrior appears about 69 pixels tall after enlargement,
   while having only about 34.5 pixels of original vertical detail.

3. **A real surface-detail channel remains unused.** The Warrior and Settler
   packs retain `normal_0` (BC5) and `normal_1` (BC4) alongside base color, AO and
   gloss. `native/unit_body_renderer.h` binds only base, AO, gloss and emission;
   `native/environment_refresh/prepare_unit_shader.py` shades with interpolated
   authored vertex normals. Surface variation from the paired LEAN inputs never
   reaches unit lighting. This is a confirmed fidelity omission, but its share
   of the screenshot difference is unmeasured. The pair's decoding is unresolved;
   earlier work found an ordinary-normal-map interpretation less faithful.
   Preserve that evidence and do not simply enable the shared shader's guessed
   mapping function. See `docs/source_art_findings.md`, Units and cities.

4. **Antialiasing does not provide a high-resolution material render.**
   `UnitBodyRenderer::render/ensure` allocate at the final sprite dimensions.
   `native/render_core/linear_target.h` uses FP16 color and 4x MSAA, then resolves
   at that same resolution. The unit pixel shader has no sample-frequency input;
   this is not four independent material samples per pixel or a 2x supersampled
   unit image. Output reconstruction defaults to scale 1. Texture minification
   at these small bodies is expected. The final native blit uses equal source
   and destination dimensions, and the cache keys include zoom and dimensions.
   Actual native custom zoom rerasterizes at the requested scale; it is distinct
   from enlarging an already rendered Lab image.

5. **Material and lighting parity with Civ VI has not been established.**
   Unit GGX, ambient fill, AO remapping, owner-color modulation and final tone
   mapping are C3X's selected equations/calibrations. They are not recovered
   Firaxis shader behavior. The output compresses radiance using
   `rgb/(1+max(rgb))` before sRGB encoding. The unit self-shadow is a 128x128
   pose-local height map with a 3x3 filter; surrounding-world shadow reception
   is unavailable to this sprite API. These can affect perceived definition and
   material separation, but this audit does not isolate their contribution.
   The supplied Civ VI capture does not establish its internal render scale,
   antialiasing settings, exact component variants, or source shader parameters.

## Checks that rule out simpler explanations

- Current Warrior body/head diffuse data is 1024x512, eight authored mip levels;
  armor/weapon diffuse data is 512x512, eight levels. The native unit loader passes
  `full_resolution=true`, retaining mip zero rather than selecting the terrain
  loader's optional resolution cap. Base color is sampled with sRGB decoding.
- Unit samplers use 16x anisotropy and zero mip bias. Material repeat/clamp
  addressing survives preparation; samplers are independent of terrain settings.
- Fidelity byte checks passed for 2,459 payload pairs, 474 unchanged DDS chains,
  and 375 authored components. Geometry, UVs, indices and animation palettes
  survive the normal-fidelity preparation. This verifies that stage, not every
  possible source LOD choice against the Civ VI screenshot.
- Independent inverse-transpose checks passed for 393 component poses; maximum
  normal-component error was approximately 2.12e-7.
- `python3 Renderer/renderer.py test units`: 103 tests passed. The additional
  `Renderer.native.environment_refresh.test_unit_fidelity` suite: two passed.
- A fresh `lab units --case detail` attempt was not confirmed: the sandbox
  blocked Parallels dispatch. VM inspection subsequently found an existing native
  preview process, so no process was killed or duplicate render dispatched.
  Existing captures are comparison evidence, not a fresh current-invocation pass.

## Recommended sequence

First make the comparison honest: render the same action and individual body
height, show images at 1:1 pixels, and record render resolution separately from
display size. Include idle and movement; the current detail fixture freezes the
same small moving figures and does not independently exercise material detail.

Next compare a true larger native projection and a same-output-size 2x unit
supersampling experiment separately. The former adds body pixels; the latter
can improve subpixel material reconstruction but cannot guarantee more apparent
sharpness. Keep clipping, cache budgets and native dirty bounds intact. Increasing
the 56-pixel fit alone affects gameplay scale and is not an isolated quality knob.

Then recover the paired LEAN semantics from source evidence and compare surface
lighting with and without the proven decode. Calibrate light/fill, gloss and tone
response only after controlling scale and pose. A blanket sharpen filter or
aggressive negative mip bias is not supported by the present evidence.

No new Civ III symbol or patch-table entry is required for this audit.
