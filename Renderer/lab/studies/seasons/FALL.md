# Autumn on the Civ V skin

The current candidate uses original Summer textures and tree meshes. Inspect the
[direct target comparison](../../out/seasons/programmatic/fall-focus/beauty.html).
The selected concept remains the visual target. The preceding refinement was
visually inadequate; texture-preservation and biome-separation checks did not
establish a beauty match. The revised candidate was also rejected and still has
gaps in canopy light/shape and broad terrain richness. Next work follows the
[target matching plan](AUTUMN_MATCH_PLAN.md), including the confirmed half-height
forest conversion, crown-scale irradiance and source-derived turf patterns.
Further color-only refinement is not the next approach.

## Current material treatment

Original placements carry deterministic gold (67%), amber (26%) or russet (7%)
choices. These are authored palette probabilities, not exact visible-area shares
or source-engine behavior. The stronger multiplicative tint removes more of the
original olive cast. Leaf eligibility reads original atlas chroma before optional
owner tint, with a brown-wood guard. Leaf-only brightness and a restrained wrapped
light response make shaded cards readable. Original mapped normals, AO, gloss,
UVs, bark, opacity and geometry remain; evergreen and tropical colors are retained.
Explicit pack leaf masks remain preferable for unconventional foliage or bark.

Grassland uses quieter bronze/olive with slow world variation above the composed
source grain. Plains is paler honey-straw; floodplains stays greener. Desert,
tundra and exposed stone retain their source materials. Existing forest-floor
alpha/height decals carry restrained gold/amber litter, with water-boundary
eligibility. No autumn geometry, new trees or new extracted art is added.

`autumn_beauty` is `[profile, leaf_brightness, litter_blend, grass_brightness]`,
currently `[2, 2.65, .44, .92]`. `autumn_beauty_grass` is the current Civ V skin
calibration `[1.35, 1.02, 1.70, .65]`. The shader has no source-game branches;
these trial calibration values still require adjustment for other configured
packs. The earlier `--fall-focus` profile retains its separate recipe and images.
Production should carry palette values through generic instance appearance data,
not rebuild trees for a material change. Use the binding, cache and reflection
contracts in [the implementation playbook](PLAYBOOK.md#concrete-production-port).

## Corrected preview, separate from seasonal materials

The preceding Mac preview lacked the coast/water quality of the actual Summer
capture used for the selected concept. The new preview fixes specific gaps:

- Coast/river fields mask terrain and decal spill in all seasons within this
  preview mode. The previous underlay filled uncovered land with sand; it now
  samples the configured biome textures, leaving only a narrow beach fringe.
- Map validity blends by maximum above the opaque water underlay, preventing a
  faint ground/decal carrier from overwriting valid water with a zero-validity
  sliver. Actual foliage cutouts are validated separately to avoid a vacuous
  full-map coverage test.
- Cached large/small/crossing/river slope textures supply water facets. The
  response adapts the shared `q3` natural-water equations. Orthographic view drives
  Fresnel; a finite reflection-eye approximation localizes the glint path using
  the same shared light. Original paths and widths remain. It is static diagnostic
  water with no production motion or reflected objects.
- The corrected receiver-plane shadow atlas also serves this preview's Summer
  baseline. Ground-conforming decal carriers do not cast an extra physical shadow.

These are explicit Lab corrections, not production-parity claims or seasonal
changes to the actual game. `beauty-summer.png` contains the corrected preview
baseline; it is deliberately distinct from the earlier Summer preview. Within
that baseline, Summer round trips and disabled autumn/winter remain exact in the
linear target. The old harness mode is retained for preceding-scene regression.
Production must keep its shared coast, lighting, shadow, water and reflection
providers rather than port these diagnostic bindings wholesale.

## Evidence and remaining visual work

Both revised scene renders compile all seven modules and pass wrapped material
sampling, bark/evergreen protection, desert/stone eligibility, finite normals,
source integrity, Summer round-trip and disabled-policy checks. A dedicated
object-only GPU pass verifies identical original Summer/autumn foliage cutouts.
Scene geometry remains 10,047,726 vertices on `test.biq` and 6,205,875 in the
five-biome scene. No source installation, Windows VM or injected code is used.

Noon flat strips separate grassland/plains by CIE76 **16.30**; the smallest pair
across five biomes is **11.72**. Fine-pattern correlations to the corrected Summer
are .977 grassland, .969 plains, 1.000 desert, .983 floodplains and 1.000 tundra.
These diagnostics support readable colors and surviving grain at this view.
They do not establish visual fidelity, night readability or acceptance.

The target's fuller crowns, softer canopy light and richer grass/stone treatment
remain useful comparison criteria. Preserve original geometry and texture
channels while calibrating irradiance, tint and material contrast against the
full-resolution target crops. Its altered tree shapes are not replacement assets.
Production reflections, motion, reduced zoom, dusk/night and real-game rendering
remain unverified here. The preceding Summer is still pixel-exact in its original
harness; saved Winter remains within the recorded narrow rounding bound.

## Repeat and storage

```sh
python3 Renderer/lab/studies/seasons/render.py --pack Renderer/packs/Civ5EnvironmentSkin --fall-beauty
python3 Renderer/lab/studies/seasons/render.py --pack Renderer/packs/Civ5EnvironmentSkin --fall-beauty --case biomes --width 1920
python3 Renderer/lab/studies/seasons/fall_review.py --beauty
```

Use `--fall-focus --winter-wonderland --winter-exposure --winter-decals` in the
original harness to repeat the preceding Summer/Winter regression comparison.
The optional snow meshes are clipped out of autumn. Blender is unnecessary for
repeating this rejected candidate; the next study may use it to derive metadata
from existing tree meshes. Cached winter masks are only read for that regression.

Reruns overwrite the same bounded PNGs and small receipts. Temporary builds,
field textures and raw frames are removed automatically. Keep source packs and
`assets/`; only the generated `out/seasons/programmatic/fall-focus/` review is
disposable. The gallery can compare target, preceding refinement and the corrected
Summer baseline; its five-biome view compares actual shader scenes only.
