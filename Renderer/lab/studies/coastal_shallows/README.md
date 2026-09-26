# Coastal shallows experiment

This isolated study starts from the sandbox water shader and its existing
coast/sea/ocean family weights. It first cleans repeated brown marks out of
the aquamarine shallow bed, then tests source-backed submerged rock forms and
grain. The shoreline contour, dry sand, cliffs and foam are unchanged.

The source assets and existing shore material work are documented in
`Renderer/docs/coastline_and_water_findings.md` and
`Renderer/docs/shore_river_material_findings.md`. The proposed optical strengths
are a C3X art experiment, not recovered Civ VI or Civ VII shader behavior.

Run from the repository root, using a Python with Pillow for `review`:

```sh
python3 Renderer/lab/studies/coastal_shallows/study.py build
python3 Renderer/lab/studies/coastal_shallows/study.py prepare
python3 Renderer/lab/studies/coastal_shallows/study.py baseline
python3 Renderer/lab/studies/coastal_shallows/study.py candidate
python3 Renderer/lab/studies/coastal_shallows/study.py review
python3 Renderer/lab/studies/coastal_shallows/study.py rich
python3 Renderer/lab/studies/coastal_shallows/study.py review-rich
python3 Renderer/lab/studies/coastal_shallows/study.py baseline --zoom 256
python3 Renderer/lab/studies/coastal_shallows/study.py candidate --zoom 256
python3 Renderer/lab/studies/coastal_shallows/study.py rich --zoom 256
python3 Renderer/lab/studies/coastal_shallows/study.py lagoon --zoom 256
python3 Renderer/lab/studies/coastal_shallows/study.py aquamarine --zoom 256
python3 Renderer/lab/studies/coastal_shallows/study.py aquamarine-clean-bed --zoom 256
python3 Renderer/lab/studies/coastal_shallows/study.py aquamarine-clean-bed
python3 Renderer/lab/studies/coastal_shallows/study.py reef-lit --zoom 256
python3 Renderer/lab/studies/coastal_shallows/study.py reef-lit
python3 Renderer/lab/studies/coastal_shallows/study.py review-zoom
```

`build` compiles the current sandbox source into the ignored study folder without
replacing its normal binaries. Its study client skips unrelated synthetic-unit
prewarming, disables those units and captures a two-frame water-only clip.
`prepare` freezes that client, matching renderer DLL, BIQ scene and shader
sources. Each Windows render uses that same snapshot and differs only in the
water surface and seabed shader. `review` verifies the frozen scene and binary
hashes and writes full, close, difference and compact context images. Output is
ignored under `Renderer/lab/out/coastal-shallows/`.
This is a visual Lab candidate; it does not stage a production DLL or replace
fixed references. If the sandbox water shader changes, prepare again and review
the insertion anchors before treating the new run as a comparison.

The first matched noon view retains an exactly identical far-ocean control.
`candidate` is restrained; `rich` increases coast transparency, color and the
same source-backed seabed grain to test whether the contribution reads at
gameplay scale. Neither changes sea/ocean formulas. Image comparison is the
evidence for visual judgment, not the amplified difference.

The 256-pixel tile capture is the sandbox client's highest supported tile
width. `review-zoom` writes native-resolution, matched shoreline crops, without
shrinking them into a context montage. `lagoon` tests stronger coast-family
transparency and tint, plus less absorption of the existing beach grain and
projected ocean/coast bed atlas. Sea and ocean are controls. `bed-only` is a
diagnostic shader variant that hides the coast water layer to inspect what the
authored bed contributes; it is not a visual candidate.

At this zoom, `clearwater` exposed repeated blotches. A no-clutter diagnostic
removed them, identifying projected atlas color as the cause. Jittered sparse
placement (`scattered`) and stronger rock contrast (`rockbeds`) still looked
stamped and were rejected. The authored `shallows_base_color` alpha contains a
continuous irregular pattern, while its RGB includes baked rock clusters. `continuous`
showed that pattern too strongly and looked yellow and busy. `aquamarine`
suppresses projected atlas color only in coast-family beds, uses the alpha
pattern at low contrast, and tints the shallow bed blue-green. Its water hue
and clarity worked, but a close crop still showed regularly spaced brown
shapes. The source shallows RGB contains four baked rock clusters per texture;
the submerged margin also samples a repeated gravel atlas. Removing only the
submerged gravel barely changed the stamps. `aquamarine-clean-bed` suppresses
that margin gravel and uses the source texture's highest-mip mean sand color
for coast-family beds, while keeping its independent alpha structure at low
contrast. The close crop no longer shows the repeated brown clusters. It is a
Lab visual direction, not an accepted replacement, and the current scene still
lacks the target's distinct dark submerged rock forms.
The alpha texture contents are confirmed source data; interpreting them as
seabed contrast and normal detail is a C3X inference.

Matched 256-pixel captures for `baseline`, `clearwater`, `scattered`,
`continuous`, `aquamarine`, `aquamarine-no-margin`, and
`aquamarine-clean-bed` all completed on the Windows VM. The native
`aquamarine-vs-clean-bed.png` crop isolates the remaining baked stamps beside
the cleaned bed. Both `aquamarine` and `aquamarine-clean-bed` also completed
at the 128-pixel gameplay scale. All use the
same frozen BIQ, client, DLL and noon frame; the far-open-ocean control is
pixel-identical to baseline at both zooms. No production shader, binary or
reference was replaced.

## Submerged-rock pass

The five nonzero `TER_Ocean_Decal` atlas cells supply authored rock color and
soft coverage. A separate packed channel varies across the rock interiors;
this study uses its second component as an exploratory art mask. Its physical
meaning is not confirmed, so this is not a recovered source material decode.
`reef-lit` places those source cells at deterministic, jittered world positions
with varying scale and rotation. It uses the source mask to reject each cell's
soft outer plate, adds cliff-material grain, and derives local normals from the
varying rock channel under the same scene light. The distribution, tint,
contrast and normal strength are C3X visual choices.

The native 256-pixel `clean-bed-vs-reef-lit.png` crop shows sparse, textured
submerged forms without restoring the repeating brown clusters. The 128-pixel
capture shows they remain fairly subtle at gameplay scale. The earlier
`reef-field` became a soft dark shadow; `reef-forms` admitted whole decal
footprints; `reef-ridges` and `reef-relief` improved their outline but lacked
contrast; `reef-contrast` produced a cyan spot in the wider crop. A local
surface-clarity test (`reef-window`) was visually negligible. Source-color
high-pass (`reef-detail`) revealed grain but became a patch of freckles; adding
back a restrained interior shadow (`reef-composite`) did not resolve that look.
`reef-lit` is
the best current Lab candidate, still softer and less three-dimensional than
the Civ reference. There is no claim that projected bed shading substitutes
for actual submerged rock geometry.

All close captures use the frozen scene, binaries and noon environment. The
far-open-ocean pixel control remains identical to baseline. The `reef-lit`
surface shader is byte-identical to `aquamarine-clean-bed`, so its difference
is the submerged bed alone. No production shader, binary or fixed reference
was replaced.

The local 0 A.D. `water_high.fs` at commit
`0ed48b3a1fb1b4b718a78869fa497185af55e086` blends a refracted scene with reflected
sky and objects, using depth, tint and murkiness to limit visibility and
animated normals and shoreline data for foam. The sandbox already adapts parts
of that optical structure. This study uses its depth/visibility separation as
an implementation reference; 0 A.D. does not supply the Civ-style submerged
rock/contour art. The large and small submerged shapes in the target remain a
question for C3X's existing authored bed atlas, not a reason to add synthetic
surface noise.

The first rich run exceeded the VM transport's 180-second limit after writing
three complete JPEG frames. `recover-rich` verifies that the client is no longer
running and that all three captures decode, then marks the result `capture-only`.
This supports visual review, not a full sandbox-run pass.
