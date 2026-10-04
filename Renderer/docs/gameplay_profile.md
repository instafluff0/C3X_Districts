# Profiling ordinary gameplay

Close Civ III, then double-click `Renderer\CAPTURE_GAMEPLAY.bat` in Windows.
Accept the Windows capture permission and play normally. Include the actions
that cause trouble: holding or dragging route previews, ending turns, opening
a city, and scrolling through its build choices.

After the problem occurs, return to the capture console and press **Enter**.
Wait for **Capture saved**. This works even if the game is frozen; the collector
runs separately and does not require the game to close. Closing Civ III also
ends collection. The capture stops automatically after 15 minutes, below
128 MiB of sampled game address-space headroom, or below 512 MiB free disk.
These sampled guards cannot prevent every sudden allocation failure.

Tell the agent the capture is finished and roughly what happened. Results are
saved under `Renderer/native/build/live-captures/<timestamp>/`; no upload is
needed when the agent shares this checkout. Collection writes to the VM's local
temporary directory first, then copies the evidence to the shared checkout.

Each capture includes:

- Game-window images at five samples per second and a timestamped timeline.
- Game address-space samples, plus helper CPU, private memory and working set.
- Renderer and mouse-route logs, publication queue/service timing, and helper
  presentation timing when PresentMon is available.
- The exact bridge, renderer and helper binaries and their hashes.

Only the game window and renderer debug messages are collected. Evidence stays
local. Capture files may include the visible game UI, save names and local file
paths; they are ignored build output and should not be committed.
The debug log keeps the latest 256 MiB, so a busy early turn cannot stop logging
before a later freeze. Renderer-owned trace files retain a bounded initial prefix.

This workflow does not record a replay journal. It avoids the much larger input
recording and does not require the replay qualification receipt. It still adds
observer overhead: use timings to locate stalls and compare similarly captured
runs, not as an unrecorded-game FPS benchmark. A five-Hz image sequence can miss
brief visual glitches. For a short exact-input replay investigation, use the
separately qualified `CAPTURE_DIAGNOSTIC.bat` workflow instead.

Preflight without launching anything:
`powershell -File Renderer/tools/capture_game.ps1 -GameplayProfile -CheckOnly`.
Launcher/window checks use `Renderer/tools/test_capture_launcher.ps1` with a new
output directory. Automation can request collector shutdown by creating
`stop-profile.txt` in its own local session directory; this leaves the game open.

Review with `python3 -m Renderer.tools.analyze_renderer64_capture SESSION_DIR`
for memory, presentation intervals and per-operation publication queue/service
times. The analyzer distinguishes an intentionally absent replay journal from
missing evidence. `Renderer.tools.inspect_window_witness` reviews the `window`
directory without requiring a replay inspection directory.
