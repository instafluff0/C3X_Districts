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
The study above records the initial isolated experiment. See the production
follow-up below for the subsequently requested beach-only implementation.

## Beach-only production follow-up

The user subsequently requested ordinary beaches only, a Lab category, and
production staging. `ocean-waves` now owns beach, rocky-control and mixed coast
fixtures. `coastal_waves.h` traces connected authoritative contour segments,
seeds instances by canonical world identity and excludes hill/mountain/cliff
shoulders conservatively. It rejects ribbon sections crossing land or a second
shore. Source alpha remains preserved, rather than being mistaken for foam
opacity. All 16 source variants, auxiliary foam and all 8,192 delay floats are
available through dedicated generic textures; no LEAN texture slots are reused.

The optional `CoastalWavesRuntime` pack consumes normalized DDS and float data,
not BLP/ArtDef files. The generic compiler preserves both complete mip chains
and the exact delay values. The shader uses the recovered ArtDef ranges with
the experiment's explicitly authored time/distance calibration, not a claim of
recovered Firaxis shader equations. Other loose water effects remain documented
source evidence; wake/crater/flood artwork is not a coastline-breaker binding.

The existing ambient compositor now includes waves. Immutable static color and
depth remain cached, while only affected 128-pixel blocks receive a 15 Hz wave
pass. Wave geometry is retained for a static scene signature and bounded at
16 MiB; resource animation and static terrain keep their existing budgets.
Animation publication counts include visible wave ribbons. Missing/disabled
packs leave normal terrain intact and a view without eligible ribbons does not
request continuous wave redraw. Existing injected clock, capture, fog, overlays
and map composite boundaries remain unchanged; no new patch-table entry.

Reproduce with `python3 Renderer/renderer.py lab ocean-waves`,
`test ocean-waves`, and `integration ocean-waves --renderer-only`. Generated
current frames and wave-off/motion controls are in `Renderer/lab/out/`.
The portable/category integration suite passes 203 tests. Native lifecycle
checks exercise 1/5/9/13-second phases, repeated timestamps, time return, zoom
return, cold reconstruction, scrolling and disabling the wave pack. At 128-pixel
tiles the ordinary beach has 19 visible ribbon instances and the fully rocky
control has zero. At 64-pixel tiles the wider mixed-coast view has 26 eligible
instances; playback rebuilds/uploads zero terrain tiles/bytes.

Playback, time return, zoom return and unchanged-view cold reconstruction match
exactly. Reduced-zoom cold scrolling differed only at two inland grass pixels,
by one red-channel level each; no wave pixels differed. That is within the
existing renderer cold-scroll rounding budget. The witness keeps the other
comparisons exact and limits scroll differences to at most two channel levels
plus the existing pixel/error bounds.

Concurrent VM preview processes faulted during the expanded gallery/check run.
The isolated rocky case and reduced-zoom mixed case then passed; final delivery
verification is run serially. These tests establish correct native rendering and
cache behavior, not sustained live-game frame rate. Current output receipts under
`Renderer/lab/out/integration/` and `Renderer/lab/out/ocean-waves/` identify the
verified/staged build. Civ III has not been launched for this work.


The final serial `integration ocean-waves --renderer-only` run passed all three
production witnesses and 203 tests. The exact verified candidate was copied to
`Renderer/bin/C3XRenderer.dll`, and the hashes matched. The local generic wave
pack is enabled. Staging is the user-requested production evaluation; fixed
visual references were not replaced, and no installation or game launch ran.
The disposable `Renderer/lab/out/ocean-waves/staged/staging.json` records its
hash and preserves a copy of the previous production DLL for rollback.

## Wave quality correction

The side-by-side user reference exposed two material errors and a geometry
problem. The production material saturated the crest into a smooth opaque
strip, while mapping the entire auxiliary texture across a narrow ribbon
filtered away its connected foam veins. The new calibration modulates the
atlas with a smaller footprint of that same embedded auxiliary art, broadens
the transparent trailing wash and compresses coverage smoothly instead of
saturating it. It preserves shared lighting and premultiplied compositing.

Closer source inspection also confirms that rows with `FLT_MAX` crest markers
still contain faint RGB artwork: every page has nonzero values in these rows,
with maxima no greater than 32/255. They must not be treated as a row-opacity
mask. The shader retains that feathering; valid markers guide only the crest
highlight. The exact original marker evaluation remains unconfirmed.

Material correction alone still produced folded rectangular shapes. Computing
the ribbon direction over a 0.28-world-unit arc rather than a 0.03-unit arc
softens changes at the polygonal coast edges. The original authoritative shore
feet, water-distance rejection and beach/rock shoulder eligibility remain in
force. The result has curved, textured trails without changing water ownership
or making terrain rebuild for animation. These distances and the foam contrast,
warping, footprint and motion are C3X calibration, not recovered engine code.

`lab/studies/waves/quality.py` freezes the production DLL, preview, packs and
shader, then evaluates the current material or an explicitly supplied frozen
candidate. The local comparison under `lab/out/waves-quality/` separates the
original material, material-only correction and smoothed geometry. Native
phase controls use the same 1/5/9/13-second sequence and normal lifecycle
assertions. This improves the breakers; it does not close the separate gap in
open-water grain, depth color, shoreline sand and the overall Civ VI lighting.

The corrected category passed 119 category tests and 206 integration tests.
Serial native beach/rocky/mixed witnesses passed animation, repeat/time-return,
zoom return, config-off and warm/cold checks, with zero terrain uploads during
playback. The rocky control contains zero waves. The reduced-zoom night scroll
retains the existing two-pixel, one-channel-level rounding difference. The
isolated corrected wave-off image is pixel-identical to the original baseline.

The exact verified quality candidate is staged as the previously requested
production evaluation update. `lab/out/waves-quality/staging.json` records the
hash and previous-DLL backup; the category day/night previews use the same
candidate. Visual acceptance and fixed references remain unchanged.

## Follow-up comparison: retain the baseline

A further comparison against the user's closer Civ VI beach reference retained
the preceding production appearance. The tested shore-break, longer-front,
broader-wash and filtered/feathered variants were not visual improvements:
they tended toward thin continuous outlines or coarse foam instead of the
baseline's fine tapering streaks. A final enlargement using the baseline
material also looked coarser. The wave shader and geometry were restored;
the current renderer is rebuilt rather than replacing unrelated work with an
old study DLL. No fixed comparison images were replaced.

The isolated study under `lab/out/waves-shoaling/` preserves matched images,
wave-off controls, shader variants, frozen binaries and 30-second sequences
(120 quarter-second samples). `reference-comparison.png` uses the supplied
Civ VI image and explicitly labels its different scene/camera scale. These
are visual comparisons, not a numerical camera calibration or recovered
Civ VI animation timing.

Useful source evidence survives the rejected variants: 3,868 of 4,250 active
crest markers locate the exact maximum-intensity source column; all other
markers are within seven columns. This strongly supports a spatial crest
location interpretation, while the original shader's use remains unresolved.
The original `Wave.artdef` also explicitly sets white `WaveColor` and crash
distance 8; the tested break curves and texture footprints were authored
interpretations. Forcing crest mip level zero bypasses normal minification
filtering, but the combined filtering/material experiment did not establish
an independently better filter setting. Do not adopt that change on theory
alone.

The retained tests add curved-cove triangle checks and an optional native motion
sequence. `C3X_LAB_WAVE_SEQUENCE=N` captures up to 240 quarter-second samples,
checks that playback builds/uploads no terrain and verifies an exact return to
the initial frame. Eligibility remains ordinary beaches only, even though the
Civ VI reference also has breakers beside cliffs.

The retained baseline passed 207 current-code integration tests, serial beach/
rocky/night witnesses and a 120-frame native sequence with zero terrain rebuilds
or uploads. Current-code images at 1/5/9/13 seconds and the wave-off control are
pixel-identical to the preceding baseline. The exact verified current-code DLL
is staged; `lab/out/waves-shoaling/staging.json` records its identity and backup.
The supplied motion video contains quarter-second native samples, not a live-game
frame-rate measurement. Civ III was not launched.

## Quieter shoreline spacing

The subsequent density adjustment retains the selected baseline material,
dimensions and timing. Eligible origins compete by deterministic canonical
world-cell priority within 0.85 world units; the survivors keep their original
art selection and phase. Selection consults world topology, so viewport crops
and wrapped copies do not reshuffle the result. An origin must itself sit on an
ordinary beach, in addition to the existing per-vertex relief/land exclusions.
This distance is C3X calibration, not a recovered Civ VI spacing parameter.

The beach witness drops from 19 ribbon instances to 9; the wider mixed night
view drops from 26 to 10, and the rocky control remains zero. These are retained
wave origins, not a count of visible white crests at every animation phase.
The complete curved-coast fixture retains 33 of 66 eligible origins and checks
pairwise minimum spacing through wrapped copies. All 207 integration tests and
serial native witnesses pass. A 120-frame, quarter-second native sequence
builds/uploads no terrain and returns exactly to its starting image. The
isolated wave-off control remains pixel-identical to the baseline. Comparisons,
motion and verification receipts live under `lab/out/waves-spacing/`.

Day/night Lab previews and the staged production evaluation DLL use the exact
verified current-code candidate. `staging.json` records its identity and the
previous-DLL backup. Fixed references are unchanged; Civ III was not launched.
