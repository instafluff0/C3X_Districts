# Volcano source and ownership diagnosis

Run with a Python containing Pillow and NumPy:

```sh
python3 Renderer/lab/studies/volcanoes/audit.py
python3 Renderer/renderer.py lab volcanoes
python3 Renderer/lab/studies/volcanoes/study.py
python3 Renderer/lab/studies/volcanoes/study.py --reuse-candidate --variants skin-probe --cases detail gameplay
python3 Renderer/lab/studies/volcanoes/compare.py
```

`audit.py` re-extracts the four dedicated volcano material textures and macro
height/blend/region-ID fields from installed Civ VI Expansion2 source. It compares
complete DDS bytes with the selected local packs and writes a channel contact
sheet and portable JSON evidence under `lab/out/volcanoes/`. Use `--assets` for
another installation root. Source and runtime art are never edited.

Confirmed source facts:

- Feature color and BC5 detail are 512×512 with eight mips; active color is
  256×256 with seven mips; active specular is 64×64 with five mips.
- Macro height, blend and region IDs are 256×256 at LOD0, 128×128 at LOD1.
  The package's nominal grid is 512×512; it does not contain a 512×512 LOD0 field.
  Element height scale is 25 and base height is zero.
- All seven selected LOD0/material DDS files reproduce exactly from source.
- Expansion2 `ArtDefs/TerrainStyle.artdef`, `FEATURE_VOLCANO`, binds
  `ART_DEF_TERRAIN_ELEMENT_FEATURE_VOLCANO_01` and
  `ART_DEF_TERRAIN_ASSET_FEATURE_VOLCANO_01`. Rotations are not disabled, and
  `ModelHeightFromLocalHex` is true. Exact source material equations and the
  TerrainAsset composition remain to be decoded.
- Feature color alpha and detail G have footprint-like shapes. R carries radial
  detail. These images and ranges are confirmed; their intended shader roles
  remain an inference. Do not declare a normal-map decode based on the BC5 format.

`study.py` freezes a matching current candidate and preview executable, copies
shader sources and makes read-only hardlinks to local art in an isolated root.
It generates adapters only there. No staged DLL, production source, source art or
fixed reference is modified. `--reuse-candidate` keeps that frozen DLL during
concurrent native work; it must not be represented as a fresh current-code build.

Controls:

- `current`: unchanged source, establishing that the isolated root reproduces Lab.
- `exposed`: removes the later natural surface inside the fixed synthetic volcano
  footprint, testing for an underlying volcano pass. This is an ownership
  diagnostic, not an acceptable production fix; its square mask is intentional.
- `no-bc5`: exposed surface with geometric normals, removing the guessed XY offset.
- `height-gradient`: exposed surface using R as filtered detail height, transformed
  through the actual world surface basis. Its .04 amplitude and .6 perturbation
  limit are experimental C3X choices, not recovered source values.

- `skin-probe`: routes the existing dedicated color texture (already bound at
  slot 69) onto the natural terrain and mountain shaders using the known fixture
  center and source UV mapping. This intentionally isolates color routing: it
  preserves current geometry, inherited normals and shading. Its fixed position
  and height/footprint mask are diagnostic and cannot be promoted into production.

All controls retain source textures, macro height, environment and shadows.
`--cases active gameplay coastal` adds context; `--zoom 128` checks gameplay scale.
No probe adds smoke, particle attachments, new terrain identity or wonder support.
Generated receipts retain candidate, preview, source-shader and output hashes.

## Observed results

The isolated `current` close-up exactly matches the category render at zoom 224.
`exposed` removes the cone, leaving flat ground and its cast shadow. There is no
raised legacy volcano skin underneath: the production raised-land emission gate
excludes volcanoes, while the natural ground retains their height but not their
material/activity. This is the primary demonstrated defect.

The legacy-normal probes are diagnostic controls only, not proposed appearances.
Changing an omitted raised-land shader cannot repair the missing natural-surface
material. Carry generic volcano ownership into the natural surface first; only
then use geometric-normal and R-height derivative comparisons to establish detail.

The eight category renders (four cases, two zooms) report zero fallback tiles.
The dormant/active pairs are byte-identical at both zooms. The selected category
suite passed 137 tests with one existing skip. No runtime visual implementation,
Integration promotion, fixed-reference replacement or staging was performed.

The `skin-probe` renders restore tan/gray rock and the dark crater in both detail
and mountain context without replacing any source art. They also expose the
remaining excessive steepness, narrow crater opening and coarse slopes. A generic
material owner plus a resolved macro/tessellation/join treatment is necessary;
changing color alone is insufficient. No active lava or new detail-normal behavior
is claimed by this probe. The `no-bc5` exposed control is byte-identical to the
exposed baseline, consistent with that raised-land shader being omitted.

Comparison sheets are `lab/out/volcanoes/detail-comparison.png` and
`gameplay-comparison.png`. The user accepted the skin-probe gameplay appearance; these remain Lab images.

## Accepted visual target

On 2026-09-12 the user accepted the `skin-probe` gameplay result for texture,
shape and overall appearance. Preserve the existing cone, crater, normals and
inherited mountain detail. Earlier suggestions to broaden or reshape it are
superseded by that acceptance. The fixed fixture center and material mask remain
implementation shortcuts, so acceptance selects the appearance rather than making
that diagnostic shader suitable for staging. Remaining work is recorded in the
volcano category notes: generic placement/material state, crater lava, existing
shared-shadow verification, then integration/staging. Smoke remains excluded.

## Static lava and missing cast shadow

The user requested static crater art without emissive glow or smoke, and mountains
in every volcano review preview. `gameplay` retains the exact accepted composition;
`detail`, `active` and `coastal` now also contain the adjacent mountain pair.
Historical isolated images remain useful evidence but are no longer the default
review composition. The study defaults to `gameplay`.

```sh
python3 Renderer/lab/studies/volcanoes/study.py --reuse-candidate --variants skin-probe skin-shadow-probe skin-lava-shadow-probe --cases gameplay
```

`skin-shadow-probe` changes only the caster clipping inside the known volcano
footprint. It tests whether the mountain-only coverage mask discards the raised
volcano portion of the shared surface. `skin-lava-shadow-probe` additionally blends
the existing active color's orange crater texels into albedo. Its dark background
is excluded using the existing brightness/alpha mask; there is no emission, glow,
particle attachment or animation. These fixed-coordinate controls identify the
required production ownership fields; they must not be staged as runtime fixes.

The matched gameplay render confirms the shadow-mask defect: the shadow-only
control restores the volcano's westward cast shadow while retaining the accepted
skin and geometry. The unchanged `skin-probe` still exactly reproduces accepted
image SHA256 `01a355ea0949a3154e7939e582868dfc93e9034cfa1b57282adcb625a77d3a32`.
The static lava patch is offset by (+.015, -.002) in source UVs to align its orange
center with the macro crater low point. This measured placement makes the lava
visible within the opening without changing the mesh or adding emission. The
exact source placement transform remains unresolved; this is a Lab calibration.

Run `compare.py` for `lab/out/volcanoes/lava-shadow-comparison.png`; the default
comparison now keeps the accepted mountain context on both sides. The final
shadow/lava controls reported zero fallback tiles. They remain diagnostic only;
general material and caster ownership are still required for runtime staging.

## Production follow-through

The user subsequently accepted the static-lava/shadow result and requested
production staging. The general implementation and current checks are described
in `lab/categories/relief/volcanoes/README.md`. These fixed-coordinate probes remain
preserved diagnosis evidence; they are not the staged runtime implementation.
