# Short gameplay diagnostic

This is a 60–90-second Renderer64 calibration capture for unit travel and
camera navigation. It records the actual staged 32-bit bridge, 64-bit DLL and
helper. It is not ten-minute endurance qualification or live FPS acceptance.

Close Civ III, double-click `Renderer\CAPTURE_DIAGNOSTIC.bat` on Windows, and
accept the Windows capture permission. The launcher checks the exact staged
Renderer64 binaries against `renderer64-short-capture-ready.json` before it
starts the game. It records ordered renderer inputs, helper presentation timing,
bounded logs, sampled whole-window images, and Civ III/helper memory. It does
not require `INSTALL.bat` again for the currently staged build.

Load a game and play for 60–90 seconds: idle over visible water and resources
for about ten seconds; move a selected unit twice; scroll continuously; make
one distant camera jump, including a selected-unit jump if convenient. Include
an interturn only if it fits naturally.
Return to the capture console and press **Enter**, keeping Civ III open until
**Capture saved** appears. The recorder closes between complete transactions;
collectors stop and results are saved automatically. Then quit the game if desired
and tell the agent the capture is finished. No upload is necessary.

The capture requests an early stop below 128 MiB sampled free address space.
This preserves diagnostic evidence before the known low-headroom failure range;
sampling cannot guarantee prevention of a sudden allocation failure. The saved
metadata distinguishes this stop from a user finish. A crash may leave a verified
prefix, never a silently certified complete session.

The Renderer64 readiness receipt pins all three binaries and the capture tools.
A fresh controlled direct-surface recording closed, then exact and real-time
same-route replay completed. The separate startup and launcher/window controls
passed without launching Civ III. The direct-surface fixture still disagrees
with the old x86 pixel oracle; this is a visual qualification limit, not a
claim of pixel parity. Captured FPS includes observer overhead and is not the
unrecorded game's performance baseline.

After recording, the agent runs
`python3 -m Renderer.tools.analyze_renderer64_capture SESSION_DIR` on the Mac
for bridge call time, helper present intervals, memory and coverage. Use
`Renderer.tools.inspect_window_witness` with the
session's `window` and `inspection` directories to review a contact sheet and
timestamped window samples. Compare replayed symptoms against those samples
before using the fixed inputs for candidate comparisons. The next helper present
after an input is an opportunity, not proof of a correct displayed frame.
The report's `pipeline` section separates map render waits, native CPU
readbacks, copy-admission lease reasons, and Renderer64 visual sampling and
presentation. Trace counts may be capped; missing timings mean that stage was
not captured, not that it cost zero.

## Targeted failure logs without a replay recording

`Renderer\CAPTURE_FAILURE.bat` is the earlier x86 targeted-log workflow. It does
not capture the current Renderer64 triad and should not be used to diagnose this
build. The short input-capture launcher above is the current Renderer64 path.

## Preserved earlier x86 qualification

The earlier x86 short-capture receipt pins DLL
`54aac95f5e1174aaf3c9b8f034a16c3c786fad8258a4e336c8bcce75e14558d4`
and the current replay, inspector and window collector. Four fresh fullscreen
control arms pass at 1 GiB reserved VA with water effects enabled. Off/on p95 is
21.83/22.16 ms idle, 24.76/25.88 ms action, and 272.90/279.31 ms camera.
Two complete replays match all 962 frames from 17,226 calls. Recorder stop,
window collection, Windows `-ShortDiagnostic -CheckOnly` and installed DLL identity
also pass. Evidence: `native/build/input-recording/frame-working-set-capture-*`.
The full-duration launcher remains independently admission-gated.

## Maintaining capture readiness when staging

A Renderer64 update must refresh the short-capture receipt with controlled
direct-route recording, exact/real-time replay and launcher/window evidence for
the exact staged triad. Run `python3 -m Renderer.tools.qualify_renderer64_capture`
with its three evidence paths, then run
`capture_game.ps1 -ShortDiagnostic -Renderer64 -CheckOnly` on Windows. Never
copy hashes into an older receipt to bypass qualification.
Desktop graphics replays in the Parallels VM must run in its interactive user
session (`prlctl exec --current-user`); a service-session replay cannot
qualify DirectComposition window presentation.
