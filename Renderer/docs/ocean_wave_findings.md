# Ocean wave feasibility and source experiment

Adding the shoreline breakers in the supplied Civ VI reference is feasible.
The missing visual asset was the **embedded coastline crest atlas**, not another
ocean normal map. The standalone experiment draws recovered crests through the
current D3D11 water material, over the current terrain and submerged bed. It is
an approval study, not a staged feature or a live-game verification.

## Recovered source evidence

The installed `Base/ArtDefs/Wave.artdef` binds `WaveTest` in `Wave.blp` through
the `Wave` library. The cooked package contains one reflected
`CoastlineWaves::PackageEntry`, two embedded texture records and a float array.
The bounded decoder validates their typed pointers, dimensions and payload
ranges. The earlier water importer collected loose water/VFX textures but did
not extract either embedded wave texture.

| Source data | Recovered contents | Meaning / evidence boundary |
| --- | --- | --- |
| `nCrestPagesU/V`, `nCrestPagesInUse` | 8 × 2 grid, all 16 pages in use | Authored crest variants; not evidence for a sequential 16-frame flipbook |
| `pAtlasTexture` | 1024 × 1024 RGBA8 UNORM, 11 mip levels; pages 128 across × 512 along | Long irregular crest shapes with feathered trails are directly visible in RGB |
| Crest alpha | 203–255; standard deviation 2.65 | Nearly opaque; **not** the visible foam coverage. Exact shader role unresolved |
| `pAuxTexture` | 512 × 256 RGBA8 UNORM, 10 mip levels | Fine connected foam pattern in RGB; alpha is 255 throughout |
| `nDelaysPerPage`, `aWaveCrestDelays` | 512 samples per page; 8,192 little-endian floats | 4,250 active values and 3,942 `FLT_MAX` sentinels; preserves the entire table |
| Active delay values | 0.109375–0.40625 | Their spatial relationship to each crest is observable; source crash/shader evaluation remains unrecovered |

Both complete mip payloads and all delay bytes are preserved unchanged in the
generic offline export (about 6.03 MiB combined). Provenance and the disabled
generic `c3x.coastal_wave_assets.v1` manifest are separate. No source game or
BLP loader is needed by the rendering experiment.

The ArtDef explicitly supplies width 20, length 128, start/end distance 40/2,
crash distance 8, scale 0.3–1, cycle range 15–30, restart delay 0–4, fade-in
0–0.1, fade-out 0.85–1, auxiliary scroll 0.5 and U/V tiling 1/2. These are
**source values**, not proven screen pixels or seconds. The study interprets
cycle values as seconds and calibrates 64 source distance units per C3X world
unit. Its white foam is lit by the shared environment, not emitted as light.

`SplineTypes` names `CLUTTER_CLIFF` and `CLUTTER_CLIFFDOVER`. This confirms a
declared spline relationship but does not recover how Civ VI selects every
beach/cliff coastline. The study's ordinary-beach placement is a C3X choice.

## The other water art

The repeatable audit inventories every installed filename containing wave or
foam, including expansions and false positives. Its related-art channel sheet
decodes twelve representative water candidates:

- `FX_Wave_Crash_Foam`: white RGB with the actual web/streak pattern in alpha.
  It is different from the main embedded crest atlas and is not referenced by
  the decoded `WaveTest` texture pointers. It remains a possible separate crash
  particle/decal, not a substitute for the authored crests.
- `FireFX_Rock_Wake_Foam`: mottled wake detail, distinct from approaching surf.
- `FX_LakeFoam*` and `FX_Crater_Waves*`: localized water-effect patterns,
  including crest/trail and fine foam variants. These are not evidence that
  ocean surf should display lake/crater effects. Their source bindings remain
  unresolved here; natural-wonder behavior stays deferred.
- Expansion `FX_SeaFoam1_a/2_a`, Nubia `FX_WaterFoam06`, and `FXt_Wave_Flood`:
  alternative foam and flood sheets. Inventory/inspection only; no flood or
  wonder behavior was added. Shockwave assets are filename false positives.

`Water.blp` separately exposes `WaterPackageEntry::WhiteCapSettings`, primary
and secondary `LeanMapData`, small/large/river bump resources, and coast/deep/
lake/tropical density maps. Existing C3X optics already consume normalized
water surface data. Open-water ripples/whitecaps and coastline breakers are
distinct layers. This experiment leaves the existing open-water appearance
static; it does not claim animated deep-ocean swell or recovered Water shaders.

## Experiment and limits

The study copies runtime packs and shaders into an ignored private snapshot,
and freezes a matching DLL and preview executable. The only shader change is a
surf blend in the current natural-water branch. Original source RGBA and mips
remain preserved; the study packs crest intensity plus delays and auxiliary
intensity into two otherwise unused variance slots in that branch. A spare
texel carries sampled presentation time. These binding shortcuts are strictly
experimental and must be replaced with named wave resources and a frame uniform
before production work.

All 16 variants participate in deterministic selection. Width/length proportions,
scale range, cycle range and fade endpoints come from the ArtDef. Crest approach,
delay-assisted trailing foam, opacity and world-scale calibration are explicitly
**C3X reconstruction**. Unknown crest alpha is retained, not treated as opacity.

The native shader uses the existing continuous signed shoreline distance, clips
to water, and lights foam through the shared receiver lighting. Placement along
the coast currently assumes the diagnostic's raw-Y direction. This is useful
for testing source art, but is not a general spline solution: coves, islands,
changing orientation, connected-contour identity and wrapped seams still need
production implementation. The native sequence samples absolute phases by
restarting the standalone renderer; it does not prove live scheduling or FPS.

## Production recommendation

After approval, implement a generic optional coastal-wave layer with:

1. Stable, connected contour segments and accumulated arc length from the
   renderer's existing `ShoreField`; canonical identities across viewport crops
   and map wraps. Sample offshore normals and reject overlaps in tight coves.
2. Dedicated crest/auxiliary/delay bindings and the existing captured
   `presentation_time_ticks` / `presentation_frequency`. Deterministic phase
   sampling must survive skipped frames, scrolling and pauses.
3. A dynamic water/surf pass over retained terrain, with visibility-bounded dirty
   regions and redraw requests. Static complete-frame reuse currently does not
   account for wave time. Invalidating and rebuilding all terrain each frame is
   not an acceptable implementation.
4. Shared day/night, water depth and depth testing; native fog, labels, HUD,
   selection and other overlays retain their existing composition ownership.
5. Config-off, missing-wave-asset, hidden/offscreen, removal, wrap, scrolling,
   zoom and warm/cold parity checks before a strategic game checkpoint.

Existing injected capture already supplies the presentation clock and consumes
`request_continuous_redraw`. **This investigation needs no new Civ III hook or
patch-table entry.** Renderer-side animation/cache work is still necessary;
the experiment does not establish that simply changing a shader enables it.
No injected C changes, installation, game launch or reference replacement were
performed for this study.

## Reproduce and review

Use a Python environment containing NumPy and Pillow for `study.py`:

```sh
python3 Renderer/renderer.py lab shorelines --case lowland
python3 Renderer/renderer.py test shorelines
python3 Renderer/tools/asset_compiler/wave_blp_extractor.py
python3 -m unittest Renderer.lab.studies.waves.test_source
python3 Renderer/lab/studies/waves/study.py prepare
python3 Renderer/lab/studies/waves/study.py render --case gameplay --off
python3 Renderer/lab/studies/waves/study.py sequence --case gameplay
python3 Renderer/lab/studies/waves/review.py
```

Native commands use the configured Windows VM and do not stage the DLL.
Prepare after category preparation/build; freeze the DLL and preview executable
together because concurrent tasks can rebuild the shared candidate. The study
snapshot intentionally remains separate from later checkout changes.

Review artifacts live under `Renderer/lab/out/waves/`: `all-crests.png`,
`crest-channels.png`, `auxiliary-channels.png`, `related-art.png`, native controls,
motion samples, the generic source export and the evidence report. Licensed
payloads and rendered derivatives remain local/ignored. The reusable decoder,
study and findings contain no source art.

The current shoreline suite passed **144 tests**. Five source-recovery checks
passed, including full mip byte preservation, all-variant delay coverage,
truncated-file rejection, invalid texture pointer rejection and repeatability.
The final review contains **23 native images**, all with zero fallback, including
15 motion phases, lowland and rocky coasts, 64/128/256 zoom samples and
noon/evening/midnight/dawn. Repeating t=8 after the full motion sequence gives
identical pixels. Restoring the original variance textures produces an identical
wave-off control, proving that the experimental bindings have no hidden effect
on that control. In the matched rocky-coast view, waves change 1,950 pixels;
the fixed dry-ground and far-offshore control rectangles remain pixel-identical.
These rectangle checks do not claim exhaustive shoreline/fog/occlusion coverage.

See the generated `review.json` for final native image hashes and comparison
checks. Native samples are synthetic scenes rendered by D3D11, not Civ III
screenshots. The GIF uses two-second samples played at 4× speed; its loop boundary
restarts the diagnostic and is not evidence of a seamless global animation cycle.
Visual approval is pending.
