# Coastal shallows experiment

This isolated Renderer Lab study tests clearer coast-family water and a more
legible submerged shelf. It belongs to the existing **shorelines** category;
sea and ocean are controls. The current checkout is the baseline. The chosen
Lab variant is `desert-ripple-broad`; it has not replaced production shaders,
binaries, or fixed reference images.

## Selected visual direction

`desert-ripple-broad` starts from the clean aquamarine bed. It transfers the
continuous source desert sand height to the shallow seabed, then adds the
source desert-hills height at a broader, rotated scale. Both affect bed normals
and restrained crest/cavity shading under the existing coast-water optics.
The dry shore, foam, water surface, sea, and ocean remain as in the control.
There are no new per-tile brown decals. The user's preferred comparison is the
256-pixel-tile close view under
`Renderer/lab/out/coastal-shallows/z256/matched-broad-v-direction.png`:
current-code control, selected sand-plus-hills, and an alternative direction
blend. The alternative made the relief flatter and more streaked, so the
original `desert-ripple-broad` remains selected. The far-ocean pixel control is
identical in all three captures.

This is **material relief**, not displaced seabed geometry. The desert and
hills height textures are confirmed source assets. Their transfer to underwater
bed normals, scale, relative strength, and color response are C3X art choices,
not recovered Civ VI or VII engine behavior. The source desert material also
uses a sparse dune decal with exact source triangles; this study deliberately
uses the continuous height instead because repeating dune/rock decals read as
stamps in coast water. The source desert category and
`Renderer/docs/visual_fidelity_playbook.md` govern that distinction between
macro form and material detail.

## Findings retained from earlier passes

- The initial sandbox water already distinguishes coast, sea, and ocean.
  Matched captures first established its noon coast, then raised coast clarity
  without changing the other water families.
- Repeated brown patches came from projected bed-atlas color, source shallow
  RGB rock clusters, and submerged margin gravel. `aquamarine-clean-bed`
  suppresses those coast-family stamps while preserving a restrained continuous
  source-alpha pattern. Its water hue and clarity form the selected base.
- `reef-lit` and `reef-stone` tested authored `TER_Ocean_Decal` rock color and
  packed-channel detail. They can produce dark forms, but the available
  placement still reads too much like repeated decals. The packed channel's
  physical meaning is unconfirmed and remains an experimental art mask.
- Stronger `shelf-relief` shading from the shallow alpha looked pebbly. A
  bed-mesh displacement probe barely read while bed and water shared a mesh;
  splitting them produced a hard exposed bed plate at the compositing boundary.
  Neither geometry probe is a candidate. True macro bathymetry remains an
  unresolved renderer-layer problem.
- A world-stable blend of orthogonal desert-height samples did vary dune
  direction, but flattened the sand-plus-hills result. It remains an optional
  comparison, not the selected variant.

The inspected 0 A.D. `water_high.fs` at commit
`0ed48b3a1fb1b4b718a78869fa497185af55e086` combines refraction,
reflection, depth, tint, murkiness, animated normals, and shoreline foam. Its
water-depth separation was useful evidence for the optics approach. It does
not provide the Civ-like sculpted seabed art. See
`Renderer/docs/coastline_and_water_findings.md` and
`Renderer/docs/shore_river_material_findings.md` for the wider source audit.

## Reproduce the selected comparison

Run from the repository root. The study writes ignored output under
`Renderer/lab/out/coastal-shallows/` and renders through the configured
Windows VM. `build-mesh-control` compiles a current-code DLL into that output;
it does not stage or install it.

```sh
python3 Renderer/lab/studies/coastal_shallows/study.py build
python3 Renderer/lab/studies/coastal_shallows/study.py prepare
python3 Renderer/lab/studies/coastal_shallows/study.py refine
python3 Renderer/lab/studies/coastal_shallows/study.py build-mesh-control
python3 Renderer/lab/studies/coastal_shallows/study.py shelf-mesh-control --zoom 256
python3 Renderer/lab/studies/coastal_shallows/study.py desert-ripple-broad --zoom 256
python3 Renderer/lab/studies/coastal_shallows/study.py shelf-mesh-control --zoom 128
python3 Renderer/lab/studies/coastal_shallows/study.py desert-ripple-broad --zoom 128
```

The 256-pixel width is the study client's highest tile zoom. Review native
resolution close crops and the 128-pixel gameplay view. A comparison is valid
only when the BIQ scene, client, DLL, sun, camera, and zoom match. Check the
open-ocean control to catch accidental changes outside coast tiles. Do not
replace a fixed reference without explicit user acceptance.
