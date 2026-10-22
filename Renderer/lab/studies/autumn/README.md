# Autumn on Renderer64 (Lab study)

Goal: make C3X's autumn season beautiful before anything reaches Renderer64.
The target is the selected concept `Renderer/lab/out/seasons/scene-concepts/fall.png`
(an image edit of the `test.biq` view at camera 20,75). The earlier Mac Metal
study in `../seasons/` was rejected; this study renders through the **current
production renderer** instead.

## How it works

- `study.py render LABEL --look LOOK` renders the concept view (test.biq, camera
  20,75, tile 128, noon) with the current candidate DLL on the Windows VM.
  Other views: `close` (tile 256), `evening` (18:00), `plains` (60,65).
- A look is a **private shader root**: a copy of the flattened production
  shaders with this study's patches applied (`autumn.py`), selected with
  `C3X_RENDERER_SHADER_SOURCE_ROOT`. Production shaders, packs and staged
  binaries are never touched. Every patch is anchored and fails loudly if
  production drifts.
- `autumn_common.hlsl` holds the shared colour work (OKLab hue/chroma moves that
  keep the source texture's lightness detail).
- `study.py sheet A B --target [--crop l t r b]` builds labelled comparisons.
- Earlier looks (v1–v6) stay in `autumn.py` so each step can be re-rendered.
- Gameplay-context views: `village` / `village-night` add ancient cities,
  roads, irrigated farms and a hill mine to the concept's test.biq land (tile
  overlay bits on, so Civ III pine flags apply); `salamanca*` cut a 32x32
  window of the 1498 AD save with `../roads/save_window.js`.
- Before each render the harness waits until no other `prlctl exec` is running
  (peer sessions share the VM) and retries one failed dispatch.

## Current look (v10) and what each layer does

Colour targets were measured on the concept in OKLab (lightness/chroma/hue):

| Layer | Where | Change |
| --- | --- | --- |
| Ground | relief, mountain-tile and coast shaders | greens h≈111→90 (olive-gold), the greenest turf pulled further; straw h≈92→75 (honey); pale desert sand keeps its cream hue; reds, greys, blues untouched |
| Turf | relief shader | source grass/plains pattern amplified against its own coarse mean (concept has ~2× the mid-scale contrast); broad world-space dry/olive patches |
| Hills | relief shader | hill turf chroma ×1.32 (hills were the same hue but greyer) with a dry-khaki floor for near-grey turf, warmer lighter stone |
| Forest floor | relief shader | gold/amber leaf litter over the authored floor decal |
| Crowns | object shader | one colour per tree from a gold-led palette (gold, yellow, amber, orange, a little coral/russet, lingering olive); atlas light/dark structure kept |
| Crown light | object shader | ellipsoid crown normal blended in, dark crown undersides, brighter tops, softened leaf speckle |
| Leaf glow | object shader | transmission through backlit leaves; subtle at noon, visible at 18:00 when the sun is behind the crowns |
| Pines | object shader | unchanged (dark green contrast); pine-flagged forests stay evergreen |
| Farms | feature shaders | planted crops take one pale harvest-gold tint over their rows; farm ground follows the turf |
| Roads | coast/route shader | dirt roads slightly darker so they read on gold turf |
| Water | coast/water shader | deep water pulled from teal toward a clear blue, chroma ×1.9; rivers at half strength; eased back at night |

## Lab-only stand-ins that production must replace

- **Broadleaf detection** uses the leaf atlas's average colour (1×1 mip). Production
  should carry a `seasonal_foliage` flag in pack metadata (the vegetation builder
  knows leafy vs pine bodies); never key on names.
- **Season selection**: the private shader root *is* the autumn variant. Production
  needs the frame season in a constant buffer and the same functions behind it.
- **Crown ellipsoid** is generic (centre z .15, radii .2/.2/.1 in runtime body
  units). Clumps would benefit from per-body crown lobes from the pack builder.
- Jungle leafy bodies share the broadleaf atlas, so they would also turn; a
  metadata flag fixes that too.

## In-play findings (2026-10-21)

- With real tile flags most test.biq forests are Civ III pine forests and stay
  evergreen; only broadleaf stands turn. Golden "larch" pines (v8) were tried
  and rejected by the user ("looks a bit odd").
- The user asked for farms to be tinted simply; a per-plot harvest patchwork
  (v8 `autumn_field`) was dropped for the single tint (v9 mustard, v10 paler).
- Cities are unchanged (only the ground around them follows the turf).
- The Lab's unit roster fails to load in the 32-bit preview ("required unit
  source admission incomplete", an existing issue), so units are unverified
  here; units use their own shaders, which autumn does not touch.

## Coast, sea and ocean (2026-10-21)

`water_tiers.py` samples every water tile centre (exact preview anchors) and
reports OKLab tier means and separation. Coast–sea ΔE is 6–18 in Summer and
slightly larger in autumn. Sea–ocean ΔE is only 0.5–2.5 in **Summer too**:
production water depth is distance-to-shore only (`.46*(1-exp(-d/.85))`,
`render_core/world_coast.h`), levelling off ~2 tiles out, and
`q3_natural_water` never reads the tile type. Every water vertex already
carries a smooth per-type value (`surface_coordinate`: coast .34, sea .68,
ocean 1.0). The season-independent `tiers()` patch uses it to darken ocean's
final colour and hide more bed (looks `summer-tiers`, `v11-tiers`): sea–ocean
ΔE 4.3–7.6 with no tile-shaped edges. It changes the approved water look, so
it needs its own approval; it is not part of the autumn look.

## Experiments

- `v4-tall` scales every forest body to 70% of source height (pack uses 50%),
  colour and shadow casters together. This is a vegetation-wide proportion
  question, not an autumn setting; trees must not change size between seasons.
- Civ VI's base `clutter.blp` contains unused `Tree_Decid_01..03` models with a
  painterly lump atlas (`TEXTURE_DiffuseTint_Trees_Decid_B_null`) that resembles
  the concept's chunky crowns. `clutter_blp_extractor.build_feature` extracts
  them with `allow_optional_maps=True` (opaque, no gloss); that was done only
  into a scratch folder for a Blender look. A species change would apply to
  every season.

The 2026-10-21 review page (concept, Summer, v7, options) is a private
claude.ai artifact; its renders live under `Renderer/lab/out/autumn/`
(ignored, disposable).
