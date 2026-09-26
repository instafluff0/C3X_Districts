# Coastal shallows experiment

This isolated study starts from the sandbox water shader and its existing
coast/sea/ocean family weights. It evaluates a modest coast-only color and
transparency adjustment plus a small increase in the accepted beach-alpha
seabed grain, intended to reveal more of the already-authored seabed
through shallow water. The shoreline contour, sand, cliffs, foam, normal motion,
reflections, sea and ocean formulas are unchanged.

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
continuous irregular pattern, while its RGB is much flatter. `continuous`
showed that pattern too strongly and looked yellow and busy. `aquamarine`
suppresses projected atlas color only in coast-family beds, uses the alpha
pattern at low contrast, and tints the shallow bed blue-green. This avoids the
obvious stamps, but the current scene still lacks the target's distinct dark
submerged rock forms. It is a Lab visual direction, not an accepted replacement.
The alpha texture contents are confirmed source data; interpreting them as
seabed contrast and normal detail is a C3X inference.

Matched 256-pixel captures for `baseline`, `clearwater`, `scattered`,
`continuous` and `aquamarine` all completed on the Windows VM. The native
`stamps-vs-aquamarine.png` crop shows the rejected blotches beside the cleaner
candidate. A 128-pixel `aquamarine` gameplay capture also completed. All use the
same frozen BIQ, client, DLL and noon frame; the far-open-ocean control is
pixel-identical to baseline at both zooms. No production shader, binary or
reference was replaced.

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
