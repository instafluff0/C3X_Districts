# Short gameplay diagnostic

This is a 60–90-second calibration capture for the bottleneck investigation.
It is not ten-minute endurance qualification, live FPS acceptance, or proof that
replay reproduces all game scheduling. The normal ten-minute capture admission
remains separate.

The tested candidate is staged and the installed game path has been verified to
resolve to it; no reinstall is needed for this update. Close Civ III, then double-click
`Renderer\CAPTURE_DIAGNOSTIC.bat` and accept the Windows capture permission.
The launcher starts the game with input recording, presentation timing, bounded
renderer logs, sampled window images and process memory measurements.

Load a game and play naturally for about a minute: linger over animated water
and resources, scroll and jump the camera, select/move units and end a turn.
Return to the capture console and press **Enter**, keeping Civ III open until
**Capture saved** appears. The recorder closes between complete transactions;
collectors stop and results are saved automatically. Then quit the game if desired
and tell the agent the capture is finished. No upload is necessary.

The capture requests an early stop below 128 MiB sampled free address space.
This preserves diagnostic evidence before the known low-headroom failure range;
sampling cannot guarantee prevention of a sudden allocation failure. The saved
metadata distinguishes this stop from a user finish. A crash may leave a verified
prefix, never a silently certified complete session.

The short admission receipt pins the DLL, replay/inspection/window tools and
launcher. It requires source-matched builds, reverse-order capture-on/off controls,
requested-stop tests, window collector tests and exact repeated replay. The
overhead checks use the fullscreen Standard fixture with all water effects on.
Their measured recorder overhead does not include every live-game scheduling
effect; observer overhead and the game's heap layout remain separately identified.
Captured FPS is diagnostic, not the unrecorded game's performance baseline.

After recording, first verify the journal and compare replayed symptoms against
the actual window samples. Use the fixed inputs for the bounded bottleneck
campaign only within the established fidelity limits. Do not claim architectural
speedups from faster API returns alone.

## Current qualification

The readiness receipt pins DLL
`54aac95f5e1174aaf3c9b8f034a16c3c786fad8258a4e336c8bcce75e14558d4`
and the current replay, inspector and window collector. Four fresh fullscreen
control arms pass at 1 GiB reserved VA with water effects enabled. Off/on p95 is
21.83/22.16 ms idle, 24.76/25.88 ms action, and 272.90/279.31 ms camera.
Two complete replays match all 962 frames from 17,226 calls. Recorder stop,
window collection, Windows `-ShortDiagnostic -CheckOnly` and installed DLL identity
also pass. Evidence: `native/build/input-recording/frame-working-set-capture-*`.
The full-duration launcher remains independently admission-gated.

## Maintaining capture readiness when staging

A renderer evaluation update must also refresh the supported short-capture
workflow before it is reported ready for recording. Use
`python3 -m Renderer.tools.qualify_short_capture` with evidence for the exact
candidate, then run `capture_game.ps1 -ShortDiagnostic -CheckOnly` on Windows and
verify the installed mod resolves to that staged DLL. Do not merely replace the
DLL hash in an older admission receipt. If capture has not yet been revalidated,
state that limitation explicitly; rendering qualification alone is insufficient.
