# Resident composition and scroll-cancellation repair

## Previous repair controls

Input/HUD hooks and independent map-sampler ownership remain as implemented.
Live feedback confirms unit selection, animation activation and HUD placement.
The packed unit fixes distinguish absent optional color layers from aliases and
size scratch to the selected unit rectangle. Exact 555/565/full-color and JGL
ownership tests pass. Their staged control is `7d598ac5…`; its predecessor
`26b88ecc…` reproduces packed-unit rejection. The map-sampler lifetime repair is
`03e148cf…`, with connected Cattle receipt `42e7de0b371e4d89a755c9552d801d28`.

The 1120×1192 same-DLL delivery comparison retains 384 complete requests:
GPU/CPU means 8.94/59.43 ms idle, 39.06/149.80 ms scroll, 12.51/64.70 ms edit.
This establishes delivery-route cost, not an old/new implementation speedup.
Receipt: `native/build/gpu-composition/c423ae8e94c04d2f9ba3510eb5b38ed4`.
Final smaller-screen connected receipt: `34c3519f2d614c43932b65b044470b17`.

## Corrected live diagnosis

The latest log has successful initial map and Scout GPU draws, then loses native
map eligibility; 224 of 225 map requests use CPU delivery and no independent
visual frames are logged. The screen upload is **16-bit**: 5,340,160 bytes means
2240×1192, not 1120×1192. The earlier description of the wide fixture as unrelated
to the user's display was incorrect.

At that extent the live compositor holds 52.5 MiB before a popup requires another
20.4 MiB pair. The 64 MiB limit transfers ownership to CPU. The live Session now
has an explicit 96 MiB ceiling; generic compositor tests retain their 64 MiB
ceiling, and retained history/replay keep their existing separate limits.
Old failure: `f720e68241a844d5ab332a2da2fa2d25`.
The first 96 MiB connected run passes actual native map/unit/popup ownership and
30 independent frames. Its later camera assertion exposed a harness assumption:
2240×1192 is the maximum scene extent and cannot create a wider prepared donor.
The harness now checks exact demanded camera pixels/clock and explicit rejection
of wider preparation there; smaller views still require the prepared-donor proof.
No production extent or detail setting changed.

## Reproduced black rectangles

Canceled camera work discarded draw contributors while leaving static-guard idle
preparation eligible. The idle producer cleared regions with zero contributors,
then marked them valid. Later scrolling reused those black samples. The pixels
are already black before native overlays and do not originate in UI composition.

The existing recorder's `--scroll-coverage` mode uses 40 fine pans, both axes,
128/192-pixel tile sizes, superseded camera requests and idle preparation.
The old implementation fails at step 2 with a 60-pixel black run; a cold render
of the identical view has no black run. Trace shows `selected=0 batches=0` guard
commits immediately after cancellation. Failure receipt/images:
`native/build/gpu-composition/f5aba506d3024d8194be95f10e333bc1`.

`discard_scene_view` now retires interrupted assembly and its raster validity
without destroying compiled world content or the immutable published front.
Background guard work requires a valid draw assembly. All four abandoned-view
paths use this ownership boundary. No timers, redraw requests or native hooks.
The candidate passes all 40 steps without missing strips:
`native/build/gpu-composition/7ade508073cb474e9a533b698ec08eff`.
The executable lifecycle test also proves that discarded contributors cannot
prepare pixels and the next complete view must refill invalid coverage.

53 focused scene, scroll, input and bridge checks pass. The broader legacy
`test_frame_publication` fake-worker test does not compile against the independent
visual-frame APIs (missing timer/retained/unit stubs); the same failure is verified
against HEAD before this fix. Actual Windows worker coverage remains the connected
native fixture, rather than claiming that obsolete stub passed.

## Final verification

Measured candidate `9354abc4…` passes the 2240×1192 connected comparison:
384 complete requests, 64 per route
and workload, including map, eight units, native UI and final transfer. Capture
is outside timing. Same-DLL GPU/CPU request means (ms): stationary 19.96/74.47,
scrolling 64.65/193.20, local edit 21.11/79.83. GPU request p95 is 50.10/123.62/58.52;
there are still substantial tails. This compares delivery routes, not a guaranteed
live-game FPS improvement. Independent frames and native 555/565/full-color UI,
popup, config-off, exact camera pixels and ownership checks pass. Sampled largest
free VA at the final phase checkpoints is at least 1.05 GiB; this is not a bound
on every game/session's peak. Receipt:
`native/build/gpu-composition/25932360ba734324ad8d04be8c87bde2`.

Final review additionally invalidates internal cache hits when discarding a view;
the published front remains separately owned. The scrolling test now returns to
the exact previous camera/clock as well as advancing. The expanded 40-step cancellation test passes with no missing strips:
`native/build/gpu-composition/4139a81997e545718fa0bbc76a0e99ac`.
Final native-composition receipt `59522b30f3c5440ebfbda78382635d22` passes all
actual-resolution ownership/pixel checks and 30 independent frames (21.05 ms
request, 30.34 ms desktop means; zero native draw calls). Sampled largest free VA
is 1.18 GiB. The exact tested candidate is staged:
`61440a6b13fd8773cd02033868258aa809c38999fe7cd729006154e3f6eb587a`.
The previous `7d598ac5…` DLL is preserved under `native/build/rollback/<sha256>/`.
Ordinary `INSTALL.bat` consumes the staged DLL; live confirmation remains pending. No installation or game launch was performed.
GPU timestamps remain unused on Parallels; QPC desktop completion is not physical
scanout.
