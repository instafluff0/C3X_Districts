# Civ V environment-skin terrain study

## Scope

This is an isolated macOS Metal study. It deliberately excludes Civ III tile,
Windows VM, and full-scene constraints. The accepted hill-only review frame is
`out/beauty-terrain-r8/h12-z1-pan00-civ5-lut.png`; the extended grassland,
plains, and tundra comparison is
`out/beauty-land-types-r1/h12-z1-pan00-civ5-lut.png`.

## Confirmed source data

- `TerrainStyle.artdef` binds standard grass hills to
  `ART_DEF_TERRAIN_ELEMENT_HILL`, the grassland base material, and the distinct
  `ART_DEF_TERRAIN_MATERIAL_GRASSHILL_TOP` elevated material. Its authored high
  grass threshold is 10.
- The normalized standard hill element is the 512x512 authored R8 relief field
  from `TerrainElementSet_Base`; material height maps remain shading detail and
  do not replace that macro geometry.
- `CLUTTER_GRASSLAND_HILLS` has `TerrainHeight=true`, `Density=1`, and three
  active source placements: `TER_Grass_Decal_HB01` count 3, HB02 count 2, and
  HB03 count 2. All specify `RotateZ`, scale 3, and 0.1 scale variation.
- Those three entries use the source grass decal atlas imported as
  `base_color_c996c6a9d015eebe.dds`; their BC5 RG input remains available but
  its native shader semantics are not asserted.
- The final review image uses the workshop's exact
  `Civ_V_Colorkey_Noon.dds` transform.
- Plains uses its own base/elevated material pairs. Tundra uses the normalized
  `tundra_blend` base/height/specular material; StandardHills defines no tundra
  high material, while `CLUTTER_SNOW_TUNDRA_HILLS` is a separate source layer
  above source height 7 with two snow-hill decal entries at count 30 each.

## Lab adaptation

- Six broad hills use different source-field offsets, aspect ratios,
  orientations, heights, densities, and stable seeds. The source relief also
  perturbs each footprint, so the geometry is not one repeated mound.
- Per hill, the confirmed 3/2/2 HB recipe is deterministically thinned by the
  hill's rockiness. Retained patches independently randomize position,
  rotation, and the confirmed +/-10% scale range. This is stable across frames
  while remaining different from hill to hill.
- The patch footprint comes from the authored decal alpha. Within it, the
  authored hill-top material restores denser stone detail at gameplay distance.
  That combination and its contrast response are an explicit Lab inference,
  not a recovered Firaxis shader equation.
- The accepted study uses 4x MSAA, 16x anisotropy, 2x internal rendering, and a
  -1 mip bias. This is intentional evidence that source detail was previously
  lost partly during sampling rather than missing from the assets.
- The extended comparison varies continuously from grassland through plains to
  tundra along screen X. Broad source-height modulation breaks up the transition
  without introducing tile seams. Grass/plains HB patches stop before tundra;
  the distinct tundra snow-hill assets remain explicitly pending import.

## Hold point

The full scene remains intentionally blocked on user review of the isolated
terrain. When promoted, its per-hill seed must derive from a stable scene/tile
identity, and city footprints must exclude vegetation according to the already
confirmed forest `ClipBuildings=true` rule.
