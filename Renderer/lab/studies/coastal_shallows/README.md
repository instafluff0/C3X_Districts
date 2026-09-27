# Coastal shallows experiment

This isolated Renderer Lab study tests clearer coast-family water and a more
legible submerged shelf. It belongs to the existing **shorelines** category;
sea and ocean are controls. The current checkout is the baseline. The chosen
Lab variant is `desert-broad-mosaic`; it has not replaced production shaders,
binaries, or fixed reference images.

## Selected visual direction

`desert-ripple-broad` first transferred continuous source desert sand height
and a broader, rotated desert-hills height to the clean aquamarine shallow bed.
The user liked that relief but identified aligned stripes near the lower shore.
`desert-broad-irregular` blends in a modest amount of the source shallows
base-color alpha as a filtered height cue, reduces the sand stripe contrast,
and fades the added relief with depth. The user preferred its more broken,
fine-grained appearance.

`desert-broad-mosaic` then blends that irregular bed with the exact current-code
mesh-control bed in broad, semi-random coast regions. Two low-frequency views
of the existing river-bank noise texture make a deterministic, world-continuous
selection field with soft boundaries. No tile owns a region or decal stamp.
The user preferred this mosaic. The dry shore, foam, water surface, sea, and
ocean remain as in the control. The matched native-resolution comparison is
`Renderer/lab/out/coastal-shallows/z256/control-irregular-mosaic-close.png`;
the gameplay-scale comparison is
`Renderer/lab/out/coastal-shallows/control-irregular-mosaic-gameplay.png`.
`review-mosaic` checks matching scene, client, DLL, camera and shader receipts,
writes both sheets and a difference image, and verifies that the far-ocean
pixels are identical at both zooms.

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
  direction, but flattened the sand-plus-hills result. Full shoreline-distance
  guidance produced contour stripes. A restrained phase blend looked streaky.
  The selected mosaic preserves the irregular source-height response in some
  coastal stretches while other stretches keep the calm control bed.

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
it does not stage or install it. `review-mosaic` needs Python with Pillow.

```sh
python3 Renderer/lab/studies/coastal_shallows/study.py build
python3 Renderer/lab/studies/coastal_shallows/study.py prepare
python3 Renderer/lab/studies/coastal_shallows/study.py refine
python3 Renderer/lab/studies/coastal_shallows/study.py build-mesh-control
python3 Renderer/lab/studies/coastal_shallows/study.py shelf-mesh-control --zoom 256
python3 Renderer/lab/studies/coastal_shallows/study.py desert-broad-irregular --zoom 256
python3 Renderer/lab/studies/coastal_shallows/study.py desert-broad-mosaic --zoom 256
python3 Renderer/lab/studies/coastal_shallows/study.py shelf-mesh-control --zoom 128
python3 Renderer/lab/studies/coastal_shallows/study.py desert-broad-irregular --zoom 128
python3 Renderer/lab/studies/coastal_shallows/study.py desert-broad-mosaic --zoom 128
python3 Renderer/lab/studies/coastal_shallows/study.py review-mosaic
```

The 256-pixel width is the study client's highest tile zoom. Review native
resolution close crops and the 128-pixel gameplay view. A comparison is valid
only when the BIQ scene, client, DLL, sun, camera, and zoom match. Check the
open-ocean control to catch accidental changes outside coast tiles. Do not
replace a fixed reference without explicit user acceptance.
