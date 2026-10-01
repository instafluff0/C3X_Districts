# Winter on the Civ V skin

This study preserves existing forest bodies, weighted recipes, placements,
terrain heights, rocks, UVs and opacity. It changes material response and adds
terrain-conforming snow decals. No tree models or replacements are generated.
The Mac Lab still uses diagnostic water and shadow composition; this is not
production parity or visual acceptance.

Inspect the [before/after review](../../out/seasons/programmatic/winter-focus/index.html).
The selected icy-blue concept remains the visual direction.

## Working improvements

- **Textured snow:** normalize the configured snow albedo's coarse average to
  pearl white while retaining local contrast. The old 80% constant-color blend
  suppressed most of its texture. Two frequencies of snow height and the saved
  decal height channels add relief over surviving substrate normals. Soil
  smoothing is limited to 12%; rock retains its full substrate contribution.
- **Tree frost:** original leaf detail selects small frost clumps. Mip-relative
  luminance retains texture contrast, with darker blue-gray foliage between
  patches. Wood eligibility, normals, opacity, UVs and silhouettes remain intact.
  Existing pine bodies use saved snowy color/gloss with corrected color response.
- **Blender exposure masks:** `winter_exposure.py` reads normalized meshes into
  a BVH and tests upper-surface exposure and canopy shelter through existing
  opacity masks. Seventeen 128×128 R8 masks with five mips occupy less than
  0.4 MiB including metadata. Shared UVs store averaged exposure: this is a
  supplemental shelter mask, not an exact unique surface unwrap or a recovered
  source-engine snow algorithm. No meshes are created or changed by the bake.
- **Authored snow decals:** all eleven saved variants retain their normalized
  coordinates and actual UVs. Source base color, red height and gloss provide granular crust and
  lighting relief over existing ground. Scatter, density, scale and height-to-normal
  response are C3X adaptations. Decals add no shadow casters and cannot
  overwrite base-map validity, including partial coast/river coverage.
- **Luminous whites and blue shade:** the noon Lab uses recipe exposure 2.25,
  cooler skylight and a source-material key correction at the existing sun angle.
  Exposure is display response, not an albedo above physical white. Production
  needs a shared seasonal environment response across terrain, foliage, water
  and reflections; night/transition calibration remains open.

## Layered winter candidate

The `--winter-wonderland` profile adds local, elongated, jittered snow pillows.
An initial continuous wave train was rejected because it produced obvious map
stripes; strong diffuse drift contrast was rejected because it looked mottled.
The retained field is quiet material relief, with restrained color variation.
Snow gathers in source-height lows and sheltered faces while material highs
and rock retain thinner coverage. Ground albedo uses the already-composed
summer luminance relative to the configured biome textures' coarsest mip means,
so grit, straw, dune streaks and rock patterns survive without their warm chroma.

Foliage exposure masks now suppress snow more strongly inside sheltered
canopies. Leaf detail, original opacity and all original mapped normals remain
authoritative; a small source snow-height crust adds relief to exposed cards.
No new trees, mesh replacement, surface displacement or snow shells are used.

The saved snow decal slot is confirmed as `Decal_Heightmap`; inspection found
centered red detail and a green channel often near zero. The preceding prototype
incorrectly treated both as signed tangent slopes, biasing the whole patch. The
current shader derives C3X bump relief from centered red height. Green remains
retained and unused because its engine semantics are unresolved. This correction
is an authored material response, not recovery of Firaxis' shader.

The winter-only diagnostic shadow atlas omits terrain-conforming decal carriers
from casting; underlying terrain and all original trees/rocks still cast.
Receiver-plane derivatives and integer blocker loads with a weighted 5×5 filter
remove the earlier large triangular shadow marks. Summer uses the preceding
atlas exactly. This Lab adapter is not a new production shadow implementation.
Small black source slivers and narrow shore geometry artifacts remain visible
in both Summer and Winter and are separate harness/source issues.

Winter water now samples four cached slope textures (large, small, crossing and
river) for Fresnel/sky response and fine glints. Open river centers remain deeper
blue, with restrained frost at the banks. River/coast fields, widths and paths
are unchanged. It is a static material diagnostic inspired by the current shared
water equations; it still lacks production motion and object reflections.

`winter_layers` in `recipes.json` is `[enabled, drift_height, deposition,
frost_relief]`: currently `[1, .016, .85, .014]`. Heights are in canonical world
material units, never changes to the game terrain. Bindings 110–113 supply generic
configured grass/plains/desert/tundra colors for texture mean calibration; the
shader consumes no source-game names. The optional extra shadow atlas and water
bindings belong only to this Mac diagnostic and must not be ported wholesale.

The final noon strip measures a minimum CIE76 separation of **9.67**, up from
the preceding pass's 7.61. A brighter intermediate palette fell to 2.63 between
grassland and plains and was rejected. The current measured grassland/plains
distance is 10.85. These are flat-strip diagnostics, not a readability guarantee
at every zoom, under night lighting or for color-vision differences.

## Repeat and evidence

```sh
python3 Renderer/lab/studies/seasons/render.py --pack Renderer/packs/Civ5EnvironmentSkin --winter-focus --winter-exposure --winter-decals --winter-wonderland
python3 Renderer/lab/studies/seasons/render.py --pack Renderer/packs/Civ5EnvironmentSkin --winter-focus --winter-exposure --winter-decals --winter-wonderland --case biomes --width 1920
python3 Renderer/lab/studies/seasons/winter_review.py --wonderland
```

Set `C3X_CIV6_ASSETS` to an absent tree to repeat the source-independence check;
the source installation is unnecessary. Blender is needed only to rebuild masks
when their generator, mesh or opacity inputs change; `C3X_BLENDER` selects another
executable. Renders reuse the checked `assets/winter-exposure/` cache. Licensed
DDS inputs remain local and ignored.

The runner checks unchanged recorded inputs, exact Summer round-trip and
disabled-season parity, plus pixel equality with the preceding Summer scene.
Receipts count original scene geometry separately from decorative decal vertices.
Temporary builds, raw frames, bake inputs and opacity PNGs are removed. The review
keeps two before/after cases and focused crops; intermediate candidates are
disposable. No Windows VM, production source, game installation or reference
replacement is involved.

## Remaining gains on existing meshes

1. Extend the current **snow deposition and edges** with pack-supplied crevice
   and shelter fields where available. Compare game-scale rock and dune grain
   under grazing light rather than raising the snow brightness further.
2. Refine **drift contours** with an optional material normal/height layer and
   edge decals. The local drift field is now present; an artist-authored atlas could
   add more controlled wind-carved edges. Keep heights and anchors fixed.
3. Tune **tree frost at game zoom** with existing leaf/branch atlases and exposure
   masks. Small frost-edge decals or an overlay normal can emphasize identifiable
   boughs. Preserve darker interiors and avoid a uniformly white crown. The Civ V
   broadleaf silhouettes remain broadleaf silhouettes throughout winter.
4. Compare in the **production render graph**. The concept has richer moving water
   and object reflections than this diagnostic harness. Evaluate the material
   layers there before judging final fidelity; the main angular shadow artifacts
   are corrected here, but narrow shoreline artifacts remain. Production/VM work is outside this study.

The saved art is enough for these next material experiments. If drift or frost
detail still lacks the right pattern, create one small generic normal/height/
coverage atlas. That needs neither new trees nor another Civ VI extraction.
Blender can bake the response against existing surfaces into optional generic
material inputs. Add art to solve a measured visual gap.
