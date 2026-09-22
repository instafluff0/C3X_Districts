# Frame working-set and submission results

The four authorized changes replace mechanisms inside the current renderer.
No Civ III hooks, patch addresses, ownership, quality settings or ambient clock
contracts changed. Waves, water motion and reflections remain enabled.

## Completed responsibilities

1. **One scene sample set.** Removed `LinearBackup`, its tiled MSAA attachments,
   capture/restore traversal and obsolete test. Old/new animated footprints redraw
   from retained geometry. A replacement D3D oracle checks every HDR color/depth
   sample, partial damage, resize, removal and a deliberately wrong control.
2. **Relevant reflection work.** The current resolved atlas owns exact cell keys.
   Conservative water/visibility-feather bounds reject irrelevant cells. Changed
   cells replace their samples directly; unchanged cells stay in place. Keys are
   withdrawn before mutation and allocation failure releases partial resources.
   The shared route no longer allocates or caches duplicate mirror-page textures.
3. **Shared working-set admission.** Required scene, reflection and unit scratch
   take priority within a 1 GiB logical envelope; reproducible caches share its
   remainder. Unit scratch retires after a second without actual raster use.
   Process VA pressure separately tightens caches. Geometry, source assets and
   native fronts remain separately accounted owners; logical bytes are not VRAM.
4. **Retained submission and complete-path comparison.** Static and water passes
   reuse ordered selections, shadow batches and caster inputs. The spatial index
   serves both. Frozen water participates only where damage requires a redraw;
   city lighting is gathered once and GPU-only output avoids a CPU bitmap resize.
   No new synchronization, readback, presenter or worker pool was introduced.

Compatibility rendering still uses its required regional cache. It is not a
second copy of the shared-scene reflection cache: that route assigns it zero
capacity. Current native UI/CPU surface contracts still require some synchronous
boundaries; they are not silently removed to shorten benchmark timings.

## Recorded gameplay comparison

Candidate DLL: `54aac95f5e1174aaf3c9b8f034a16c3c786fad8258a4e336c8bcce75e14558d4`.
Baseline: `46f0cfe973a73432d68f22662c585e8866c2d3c91cda93ea7da5b37d4844edea`.
Corpus: `native/build/live-captures/20260922-004213-95ba3a/inputs`.
Standard 5,000-tile map, 2240×1260, original assets/settings, all water effects.

Forensic replay executes 253,272 completed calls before event 508145 and matches
**all 518 displayed-frame fingerprints** against the pinned baseline. This is
an explicit successful prefix: two calls remain pending at the cutoff, and the
original killed session remains incomplete. Its later live device failure is
not reproduced or hidden by changing recorded return codes.

Separate unpaced timing runs remove fingerprint readbacks. Two baseline runs,
then two candidate runs, each produce the same 569 actual accepted presentations,
345 successful ambient offers and 34 map-prepare boundaries. These actual ready
outcomes differ from forensic recorded timing. No concurrent GPU test was run.

| Measurement | Baseline runs | Candidate runs |
|---|---|---|
| Replay execution envelope | 42.07 / 42.98 s | 33.24 / 34.13 s |
| Ambient offer p95 | 49.57 / 48.62 ms | 25.38 / 26.55 ms |
| Map-prepare p95, including cold/lifecycle work | 422.61 / 410.85 ms | 344.28 / 350.75 ms |
| Peak sampled process private memory | 3,139 / 3,147 MiB | 2,965 / 2,940 MiB |
| Minimum sampled free VA | 320 / 268 MiB | 643 / 581 MiB |
| Smallest sampled largest free VA region | 212 / 148 MiB | 505 / 445 MiB |

Mean execution envelope improves **20.8%**. These are production-DLL replay
measurements, not gameplay FPS: journal/input reconstruction, serial completion
order, native process heaps, thread overlap and VM scheduling differ from live
play. The sampler itself adds overhead outside the measured production calls.

Separate forensic traces explain the change: scene attachment accounting falls
from 1,171 to 652 MiB (519 MiB static backup removed); reflection cells built fall
from 2,546 to 290, with 165 current-atlas cell reuses. Reflection submission totals
fall from 1,565 to 427 ms in those instrumented traces. Median selected dynamic
records fall from 780 to 35; static redraw now selects a median 122 records instead
of restoring backup pixels. This adds real draw work, already included in the
complete replay comparison. Do not equate logical attachment savings with the
smaller measured whole-process private-memory reduction.

Evidence: `native/build/frame-working-set-final/{build-evidence,comparison}.json`,
`frame-working-set-measure-before-v2/receipt.json`,
`frame-working-set-measure-after/receipt.json`, and the corpus's
`diagnosis/frame-working-set-final/{fingerprints.jsonl,trace.log,control.log}`.

## Validation and remaining responsibility

All 18 selected portable ownership, visibility, damage, pass-order/retention,
telemetry and replay-scope tests pass. The Windows per-sample redraw oracle passes
all 18 checks. The fullscreen native fixture passes with eight mixed units and
1 GiB VA reserved: stationary/moving action continuity, blocked/unfocused UI,
water/resource animation, frozen fog, reveal with no world rebuild/upload,
tactical/grid/native UI parity, scrolling, camera adoption, cancellation,
reset/recreation, configuration-off and CPU fallback. All 1,828 direct unit
requests avoid a CPU body roundtrip. Receipt: `native/build/frame-working-set-native/`.

The 30-second native interval has idle/action/camera p95 of 22.31/26.30/277.04 ms;
minimum logged free VA is 417 MiB with the reservation in place. Separately, 120
mixed-unit visual requests average 32.22 ms and desktop completion averages
41.82 ms. These are additional coverage measurements, not a paired live-speed
comparison. They show why low idle costs must not stand in for the entire workload.

The verified DLL is staged in `Renderer/bin/C3XRenderer.dll` for evaluation. Restart
the game to load it; no injected code changed and no installer or game was launched.
`native/build/frame-working-set-final/staging.json` pins the previous DLL, retained
in the original capture for rollback. Short diagnostic capture is separately requalified for this build: the refreshed
readiness receipt and Windows preflight pass after four overhead arms, collector/
stop tests and two matching 962-frame replays. Ten-minute recording qualification
remains separate.

The next performance responsibility is the remaining long native map-preparation
and composition boundaries. Map-prepare p95 is still about 350 ms and the cold
map still takes about 7.5 seconds in replay; the 33 ms arbitrary-camera goal is
not met. Also preserve the open live recovery/fragmentation validation gap.
A fixed VA reservation tests capacity, not Civ III's exact allocation history.
No new manual recording is needed to investigate those remaining costs.

## LORE connection

The [Firaxis LORE presentation](https://www.slideserve.com/admon/firaxis-lore)
(authored slides hosted in a mirror) describes prepared rendering packets and
reduced per-draw setup. Item 4 applies that principle using existing renderer
owners. Items 1–3 address this renderer's measured memory and dependency costs.
This is not a port of Firaxis engine source, and LORE does not imply a 64-bit
process, automatic GPU saturation, or a guaranteed frame rate.

Generated fixture BMPs are archived as lossless PNGs; bulky measurement JSONL is
losslessly gzip-compressed. Intermediate candidate binaries/objects were removed,
while build receipts, the pinned baseline, final candidate and original recordings
remain. `native/build/frame-working-set-final/storage.json` records the conversions
and hashes; decompress measurements before rerunning their summarizer.
