# Scripted real-game renderer diagnostic

The user authorized these bounded real-game tests for this renderer integration
and asked that future agents reuse them. Run with Civ III and debug collectors
closed. Never take over an unrelated running game. Use a disposable copy of a
local save, never save the resulting gameplay, and retain the script's cleanup.
The game executable must contain the current injected diagnostic hooks.

## Repeatable workflow

1. Run focused source tests and build an isolated, unstaged renderer candidate.
2. Run the affected native pixel oracles and asynchronous GPU fixture. Record
   source signatures, the three binary hashes, and each invocation's receipt.
3. With Civ III closed, stage the matching bridge, x64 DLL and helper together.
   Verify their hashes against the qualified candidate.
4. After injected changes, run `TEST_INJECTED_CODE_COMPILE.bat`, then the console
   installer below. DLL-only changes do not require reinjection.
5. Run the bounded game scenario. Read `result.json` (including early exit and exit code), the complete renderer log
   and window samples. Command delivery and an error-free log are necessary
   evidence, but do not establish correct visible output.
6. Confirm cleanup and original-save hash before another run. Record remaining
   defects in the current renderer status; do not call fixture FPS live FPS.

The installed `Conquests/C3X_Shared_Verify` directory link points at the shared
checkout. Run compile/injection through that link so `ep.c` finds the installed
executable. `C3X_RENDERER_CIV3_CONQUESTS` overrides the default GOG install.
Candidate builds can use `Renderer.lab.platform.windows_root()` for the shared
checkout; no mapped drive or literal user home path is necessary.

## Installation without a popup

From an elevated PowerShell:

```powershell
& .\Renderer\tools\install_console.ps1
```

This compiles the existing `ep.c` installer through a small console wrapper.
Only its message boxes become console messages; installation and error exit
codes remain the existing C3X implementation. It waits for the installer to
exit, refuses to install while Civ III is running, and leaves its generated
executable under the ignored build directory. It does not stage renderer DLLs.
Run `TEST_INJECTED_CODE_COMPILE.bat` before installing injected changes.

## Bounded load and camera test

From an elevated Windows shell, including the Parallels guest command service:

```powershell
& .\Renderer\tools\run_scripted_game_test.ps1 -SaveFile 'path\to\test.SAV' -Seconds 75
```

The launcher creates a temporary task using the current interactive user's
elevated token. This satisfies the game's compatibility elevation requirement
while preserving the diagnostic child's environment. The task is removed after
the run. `scripted_game_test.ps1` can also run directly in an elevated interactive
PowerShell session.

The test copies the save, records executable/DLL/save hashes, and temporarily
sets the native menu to Load Game with that copy selected. It uses two Enter
presses to load through the normal game path. The existing post-load popup hook
recognizes the explicit `C3X_RENDERER_GAME_TEST_SAVE` environment variable only
when custom rendering is enabled, dismisses that known welcome popup, and marks
the test ready. Subsequent F24 messages reach the existing raw key-event hook
and request 32 camera moves: eight in each direction. They do not move units or
advance turns. Without the environment variable, or with custom rendering off,
the diagnostic additions leave the original behavior intact.

The test parks the cursor inside the game client to prevent accidental native
edge scrolling. It stops only its own game process, restores the original
cursor position and `conquests.ini` bytes, and verifies that the original save
is unchanged. Logs, sampled window
images and `result.json` are under `%TEMP%\C3XGameTest`. Launcher transcripts are
under `Renderer/native/build/C3XRendererGameTest-*`. These are local ignored
diagnostics and can contain machine paths; do not commit them.

`-Scenario settler -Seconds 35 -SampleHz 10` loads the disposable early-game
save and leaves its selected settler idle. It requires a completed map and no
renderer failures. Inspect the window samples for the automatic city-site
overlays when that C3X setting is enabled, then confirm the reported cleanup.

`-Scenario unit-turn -Seconds 135 -SampleHz 10 -MeasureCadence -ProfileRenderer`
uses an early save with a selected Scout, two traversable northern tiles, and
one remaining idle unit. It moves north twice, skips the remaining unit and
ends the turn, then leaves the game untouched for 26 seconds before one final
scroll. Require two accepted moves, at least one completed/prepared turn, the
final scroll and no renderer failures. Review consecutive movement samples for
body/HUD travel together and the idle interval for the selection ring before
any further input. The final scroll is a comparison, not a way to make the
idle interval pass. Original saves and configuration use the same cleanup.

`-Scenario research-turn -Seconds 200 -SampleHz 4 -MeasureCadence -ProfileRenderer`
reproduces the first-turn research dialog handoff. Use a 4000 BC save with the
Settler selected, a Worker and Scout stacked on its tile, and two traversable
northern tiles. It founds the capital, closes the city, moves the Worker north
once and Scout north twice, ends the turn and accepts Bronze Working with Enter.
No input follows for 70 seconds; the final scroll is only a comparison. Require
three moves, exactly one completed/prepared turn, all nine commands, one scroll,
and no failures. Review the idle interval for the automatically selected
Worker's ring, fidget and changing shore waves before any scroll. Presentation
counts alone cannot detect a completed image whose animation sampler retired.
Use the 64 MiB trace to verify a fresh camera adoption after world preparation.

`-Scenario turn-scroll -Seconds 205 -SampleHz 4 -MeasureCadence -ProfileRenderer`
checks exploration followed by interturn and scrolling. Use an early save with
an active two-move Scout, two traversable tiles to its north, and no blocking
production/research choices over the next two turns. It pans 24 steps, moves
north twice, skips remaining units across two turns, then pans the final eight
steps. It requires two accepted moves, at least two turns with matching
successful world preparations, all 32 camera commands, continued presentations,
and no renderer failures. Review window samples between the first move and the
second, immediately after interturn, and during the final pan for revealed
terrain, automatic selection, animated units and HUD alignment. Delivery counts
alone do not establish visible movement. Interturn preparation failures are
included in every scenario's failure scan.
`-Scenario site-toggle -Seconds 55 -SampleHz 10` opens C3X's L-key picker,
chooses Off, then reopens it and chooses the player. Inspect samples before
and after both choices; key delivery alone does not prove the displayed state.

`-Scenario debug -Seconds 55 -SampleHz 4` loads the disposable save, activates
C3X debug mode, scrolls, restores normal visibility and scrolls again. The
shortcut letters use key-up messages so individual letters do not issue unit
orders. Up selects Yes in the activation prompt. Require one `debug-reveal`,
one `debug-hide`, both scroll checkpoints and a completed first map; inspect
window samples to verify terrain and restored fog. A delivered shortcut or
closed prompt alone is insufficient.

`-Scenario debug-scroll -Seconds 190 -SampleHz 2 -MeasureCadence -ProfileRenderer`
keeps debug reveal enabled through the full 32-step camera loop.
`-Scenario turn-stress -Seconds 285 -SampleHz 2 -MeasureCadence -ProfileRenderer`
repeatedly moves the starting units east and west, skips any remaining moves,
and ends eight turns. Require eight distinct native turn-completion markers
and successful `motion-start` records across the run. Inspect the samples for
actual movement; posted keys alone do not establish gameplay progress.
Both use the same disposable-save and exact configuration/cursor cleanup.
The maximum diagnostic duration is six minutes. Cadence samples include both
processes' private/working bytes and CPU time; the window witness also records
the game's free address space and largest free region. Compare these with
completed presentations and publication latency to distinguish memory pressure
from a worker that stopped consuming commands.

`-Scenario route-city -Seconds 260 -SampleHz 4 -MeasureCadence -ProfileRenderer`
uses the preserved 3700 BC small-world witness (60 by 60, 2240 by 1260 client,
camera 2464,692; selected Warrior near tile 52,44, Scout near 52,52 and capital
55,45). It holds a route destination in the lower explored area, sends 600
pointer updates, releases the route, opens the capital and its production
chooser, then sends 50 up/down choices at 0.4-second intervals. It accepts and
closes the city before an idle tail. This is a fixture-specific destination
drag; it does not select the Scout first. Check all 609 pointer events, 52 city
commands, accepted movement, route-line traces, continued final presentations,
no renderer failures and exact save/configuration cleanup. Review the images
for the actual chooser and changed choices; command counts alone are not proof.
Compare queue delay and image service time during city cycling as well as
memory. A run that eventually recovers but retains seconds of queued UI work
does not pass the responsiveness investigation. Native folded route endpoints
must resolve to adjacent captured map anchors rather than crossing the screen.

`-Scenario city-builds -Seconds 190 -SampleHz 4 -MeasureCadence -ProfileRenderer`
uses the same capital without the preceding unit move. It opens the build list,
hovers six rows, sends two wheel events, then selects items through 20 repeated
hover/click cycles. The chooser stays open after selection; toggling the portrait
between items would close it and miss alternating choices.
Require all 96 mouse events, both closing commands, no
renderer failures and continued presentations. Verify that the production
portrait/name actually changes among Settler, Worker, Scout, Warrior, Barracks
and Granary in the window evidence; arrow-key input does not establish this.

For user-controlled play instead of a scripted fixture, use
[`CAPTURE_GAMEPLAY.bat`](../docs/gameplay_profile.md). Its console can finish
collecting evidence while the game remains frozen.
Stress scenarios stop at the first recorded renderer failure and preserve their
evidence. Detailed stress traces write to disk during the run because forced
cleanup cannot flush a stalled helper's in-memory trace buffer.

`-Scenario newgame -Seconds 55 -SampleHz 4` chooses New Game in the temporary
native menu settings and accepts the normal world/rules forms. It creates no
user save and restores the original configuration/cursor with the same cleanup.
Require a completed first map and inspect the starting-unit camera and welcome
handoff. The input save is still copied and checked by the shared harness but
is not loaded in this scenario.

Success requires all 32 camera commands and no recorded native/worker failure.
Review the sampled images and camera adoption logs as well: accepted commands
alone do not prove correct visible scrolling. Two-frame-per-second window
sampling is visual evidence, not an FPS measurement. The native GPU fixture
remains the independent controlled cadence check.

The optional `-Scenario interaction -Seconds 90` retains the native welcome
dialog, dismisses it with Enter, cycles zoom through 192/160/128, sends one
eastward unit movement key, then sends F1 and Escape. At each zoom it sends
F23 to publish two transient map messages through the ordinary native message
path. This opt-in key is gated by the child diagnostic environment, custom
rendering, a valid map and a selected unit. It operates
on the disposable save and never saves the result. The default `scroll` scenario
still performs the bounded 32-camera loop. Command delivery alone is not a
visual pass; inspect the recorded windows and renderer failures.


## From the Mac agent terminal

`prlctl exec` avoids Parallels GUI control. Compile/build commands normally use
`--current-user`. The installer and scheduled-task launcher need elevation:
omit `--current-user` to use the guest command service's elevated context. A
plain current-user installer can fail with "requires elevation".

For example, set `C3X_TEST_LINK` to the Windows path of `C3X_Shared_Verify` and
`C3X_TEST_SAVE` to an existing local test save, then run from the Mac shell:

```sh
vm="${C3X_RENDERER_VM:-Windows 11}"
prlctl exec "$vm" powershell -NoProfile -ExecutionPolicy Bypass -File \
  "$C3X_TEST_LINK\Renderer\tools\install_console.ps1"
prlctl exec "$vm" powershell -NoProfile -ExecutionPolicy Bypass -File \
  "$C3X_TEST_LINK\Renderer\tools\run_scripted_game_test.ps1" \
  -SaveFile "$C3X_TEST_SAVE" -Seconds 90 -Scenario interaction
```

The scheduled task runs in the logged-in Windows user's interactive desktop,
with the required elevated token. Running Civ III directly as the guest service
can put it in the wrong desktop; invoking a compatibility-elevated executable
with `--current-user` can instead lose the diagnostic environment at UAC.
The launcher resolves both issues without clicking menus or installer dialogs.

## Evidence and limitations

The window witness executable and DebugView CLI must already exist at the
paths checked by the script. Their absence is an explicit preflight error.
The witness writes JPEG samples and `timeline.jsonl` with capture timestamps under
`window/`; `finished.json` records coverage and dropped samples. Use those
actual timestamps, not the frame number, to align pictures with commands.
Sampling defaults to two frames per second. Pass `-SampleHz 10` for a bounded
animation diagnostic. Even that misses display frames; combine it with the
clock regression tests and do not treat it as a scanout/FPS measurement.

The interaction test currently uses an early 4000 BC save. The unit movement
must be confirmed from its changed map position. F1 may do nothing when there
are no cities; the later Escape then opens the quit confirmation. That is a
native popup witness, not proof of an advisor screen. Record the actual visible
screen. Current interaction coverage requires all ten commands and three
map-text events. Earlier runs delivered their commands but exposed disappearing
HUD pieces and stale selection graphics. The failure summary now also includes
unit-publication rejection; review images even when that summary is empty.

The shared VM volume can cache directory listings. If a finished capture looks
incomplete from macOS, copy it through guest PowerShell into the ignored
`Renderer/native/build/` directory before diagnosing missing output.

## Loading, main menu and reload

`-Scenario lifecycle -Seconds 120` loads the disposable save, uses native Ctrl+Shift+Q and confirms
the return-to-menu prompt, loads the same copy again from the menu, publishes a
map-text witness, then returns to the menu again. It never saves either game.
The result requires two successful first-map `render-done` events, two completed scene unloads,
the second map's text event, every scheduled command, and no native failures.
Escape confirms application exit; it cannot test same-process reloading. The
Ctrl+Shift+Q sequence verifies foreground ownership and releases all keys in `finally`.
Review the samples for a visible loading bar, complete maps, both menu returns
and fresh unit/UI state. A script timeout or delivered Enter key does not prove
that the native menu accepted it. Helpers may exist for menu presentation, but
the previous scene helper must close before the menu is painted.

For startup regressions, inspect `assets-prepared`, `first-map-wait`,
`first-map-ready` and `render-done`. Asset preparation blocks in scenario load;
the first native map draw waits for the actual native camera's completed map.
The renderer no longer creates or restores speculative loading views, so
`loading_maps_ready` remains zero. Debug viewer changes also use the first-map
barrier. Inspect timestamped window samples around bar dismissal and the native
welcome popup: completed requests alone do not prove a correct visible handoff.
Shader cache misses log `shader-cache-compile` begin/end and elapsed milliseconds.
A cold compilation remains part of blocking asset preparation.

### September 29 freeze reproduction

Capture `20260929-162535` reproduced the reported frozen display after 14
accepted unit moves. At 140.405 seconds the retained compositor hit its
256 MiB texture budget; the next native presentation poisoned the asynchronous
transport. The successful-presentation counter stopped at 4,776 while some
native turn logic continued. Available pagefile was about 7.4 GiB and the device
reported no removal. The original save remained unchanged. This establishes a
renderer lifetime/admission failure, rather than process address-space exhaustion.

The small GPU reproduction in `test_retained_view_lifetime.py` isolates native
HUD copies retaining expired ordinary and projected cameras. Before the fix, 80 camera
replacements retained 16,441,344 bytes and 322 nodes. The corrected version
stays at 270,336 bytes and seven nodes for projected views, or 221,184 bytes
and six nodes for ordinary views, with exact native HUD pixel comparisons.
The existing 256 MiB limit stays unchanged. `test_retained_composition.cpp` also
checks that a discarded optional history accepts valid GPU transfers until a
fresh map restores animation, while invalid tickets/images remain rejected.

### Movement evidence

For the tile-travel implementation, use the interaction scenario with
`-SampleHz 10`. Match `stage=motion-start` (unit ID and accepted source/destination)
against `fresh-unit-snapshot` positions and the window timeline. Native
`run-capture` records are diagnostic observations; they no longer drive travel.
Require several intermediate displayed positions with a moving run pose and a
coincident selection ring. Check the final idle endpoint, too. A delivered move
key, endpoint-only picture, or empty error summary does not establish smooth
movement. `Renderer/native/test_unit_motion.py` independently covers delayed
admission, early completion, recapture, consecutive steps, wrapping, zoom,
retirement, position correction and config-off delegation.

Current capture `20260927-183538` exercises movement with the same build as the
combat witness below. Both units have intermediate run positions and reach the
new tile; the selection ring follows travel and the terrain reveal is retained.
All ten commands and three map-text events completed, with no renderer failure
or early game exit and no change to the original save.

For repeatable visual review, copy the finished capture into the ignored
`Renderer/native/build/` directory and run the contact-sheet helper with a Python
environment containing Pillow:

```sh
python3 Renderer/tools/game_test_contact_sheet.py Renderer/native/build/game-test-example \
  --kind movement --crop 950 510 1410 780
python3 Renderer/tools/game_test_contact_sheet.py Renderer/native/build/game-test-example \
  --kind combat --crop 990 510 1350 770
```

These crop examples fit the current 2240 × 1260 diagnostic window; choose the
crop from the actual frame for a different save or window size, or omit it for
the whole window. The helper reads the recorded QPC frequency and writes both
the image and a frame/timestamp index. `--event` selects a later movement;
`--times` selects custom offsets in seconds. For the combat marker, the origin
is the first subsequent renderer QPC and is labeled as such. Inspect the native
action/HP log alongside the sheet. Ten sampled pictures per second cannot prove
that every rendered frame was smooth or measure live FPS.

## Combat diagnostic

`-Scenario combat -Seconds 90 -SampleHz 10` loads the same disposable save
and dismisses the welcome popup. Starting at 36 seconds, it retries the
idempotent preparation command every three seconds until the native readiness
marker appears. Four seconds after confirmation, it attacks once. An earlier
fixed schedule attempted preparation before the first map was ready and did
not exercise combat.

The fixture requires an empty, visible land tile immediately east of the
selected stack, with no city. F21 spawns a disposable owned attacker and a
barbarian defender using existing native functions. F22 sends the corresponding
native move or bombard order. Both commands require custom rendering, the
explicit save and `combat` mode environment variables, and a single-player
session. Each can run once. It never saves changes or forces a combat outcome.

`-CombatCase` selects the setup:

| Case | Native encounter |
| --- | --- |
| `melee` | Basic barbarian type against the same type |
| `victory` | Modern Armor against a defender with one remaining HP |
| `retreat` | Wounded elite Horseman against Infantry |
| `bombard` | Artillery against a land defender |
| `army` | Army with two loaded basic-unit members against a defender |
| `air` | Bomber against a defender protected by an intercepting Fighter |
| `capture` | Basic attacker against an enemy Worker |

Retreat and interception depend on native rules and RNG. A completed command
is not proof that either branch occurred. The return log records surviving
IDs, attacker location, defender owner and before/after damage. Require the
specific native outcome in that log and review its window samples. Use
`-UnitPack <simple-pack-name>` to test an isolated generated pack; its default
is `UnitAnimationFidelity`, and the override exists only in the child process.

The result requires both preparation and combat-return markers. Native unit
observations include action, queued action, HP/damage, visibility, retirement
and clock. Compare those with the window timeline and rendered action changes;
command completion alone does not prove combat animation correctness. Missing
fixture prerequisites make the test fail rather than change another tile.

Capture `20260927-174627` is the initial combat baseline: preparation and native
combat return completed, the original save was unchanged, and there were no
logged renderer failures. The attacker lost. The trace records fortify, attack
variants, five nonlethal HP changes, lethal damage, death, retirement and the
survivor's idle handoff. Window samples show the attacker incorrectly fighting
from its source tile while its native health marker advances to the combat
stance. This baseline is not a combat visual pass. `combat-animation` records
in newer candidates also include native cursor/count, frame period and accepted
target so timing and placement can be checked against the same native events.

Capture `20260927-180347` confirms the native Warrior periods: attack/death
15 frames at 0.083 seconds (1.245 seconds per cycle), fortify 10 frames at
0.066 seconds, and the attacker's accepted target 64 pixels east of its source
center. Its renderer candidate failed because normalized phases exceeded the
pose validator's 65,536-frame bound. The regression now passes each generated
phase through `prepare_native_unit_pose`; this capture is retained as a failure,
not visual acceptance.

Capture `20260927-181008` completed the melee encounter, but showed fog-edge
weapon clipping and a stale civilian from the source stack. The current
`20260927-183323` capture removes those two defects: the weapon silhouette
remains intact, the attacker approaches and fights at its accepted half-tile
stance, dies there, retires, and native civilian selection resumes. Both
preparation/return markers are present, no renderer failure was logged, and the
original save is unchanged. This is a melee-defeat witness; it does not qualify
victory, retreat, ranged or army combat, native audio alignment, or limb blending
between clips.

The rig-enabled roster and current bridge add these bounded native witnesses:

| Capture | Observed branch |
| --- | --- |
| `20260927-200320` | Attacker victory, defender death, advance into the target |
| `20260927-200759` | Wounded mounted attacker retreats; defender survives |
| `20260927-202628` | Two correctly loaded army members fight; army dies and defender survives |
| `20260927-202926` | Civilian capture, attacker arrival, stable idle after native selection changes |
| `20260927-203325` | Bombardment applies two damage without the former scratch-canvas failure |
| `20260927-203552` | Bomber is intercepted and dies; fighter returns to idle; default pack used |

All six preserve the original save and report no renderer failures. Contact
sheets qualify the observed land handoffs and army sequence. The air samples
establish interception lifecycle, not continuous flight trajectory or altitude.
Native audio alignment and the requested custom impact effects remain open.
The earlier `201515` army setup was invalid: `Unit_load_into_army` only updates
bookkeeping. Use the existing `patch_Unit_load` boundary so native container and
state fields are set too. F22 now retries until the start marker is observed;
the injected one-shot guard prevents duplicate attacks.

## Turn and held-mouse regressions

`-Scenario turn -Seconds 90` skips the two initial units and ends two turns in
the disposable starting save. It requires two `scripted-turn-end` checkpoints
and no renderer failures. Inspect the status panel as well: a completed native
turn does not establish that its new HUD was presented.

`-Scenario mouse -Seconds 55 -SampleHz 10` sends two wheel notches inward, holds the left mouse
button at the selected unit, moves right in four steps, and releases it. It
checks the foreground window before every mouse action and releases the button
in cleanup. Mouse/cursor state, native tile picks, and projected route targets
are logged. This bounded path does not prove every physical mouse/VM input case.
`mouse-events.json` records the injected input's QPC, client coordinates, flags
and wheel delta. Compare these with `map-click` QPCs and window compositor timestamps
to separate native input latency from visible renderer latency.
`compositor_100ns` is WGC's QPC-based source timestamp in 100 ns units;
`arrival_qpc` also includes delay in the observer. The contact-sheet tool uses
the compositor timestamp when present. Pass `--input-event N` to align it to
the Nth entry in `mouse-events.json`, rather than the first native drag message.

Add `-ProfileRenderer` to collect detailed renderer timings in a bounded memory
buffer, flushed by the helper at shutdown to `renderer-core.log.x64`. This avoids
one debugger round trip per rendering trace. Inspect that file as well as
`renderer.log`; a `TRACE_BUFFER dropped=0` trailer establishes complete profile
coverage. The x86 buffer can be lost when the diagnostic terminates its game
process; injected input evidence remains in `renderer.log`. The script restores
all three trace environment variables after child creation and during cleanup.
For a lower-overhead cadence comparison, use `-MeasureCadence -SampleHz 1`
without `-ProfileRenderer`. This disables detailed renderer tracing and reads
the helper's existing successful-presentation counter once per second through
a read-only IPC-header mapping. The helper channel must match this test's game
PID. `cadence.json` records QPC, helper identity, frame count and foreground
state. Compare counter differences only within one helper lifetime and the same
input phase. This measures successful Renderer64 presentations, not physical
scanout. A requested measurement with fewer than two samples fails explicitly.
Keyboard release messages include the previous-state and transition flags;
otherwise this game's JGL event dispatcher interprets a synthetic release as
another press. Earlier scripted zoom labels therefore did not reliably identify
the actual zoom; use the `zoom-key` log values.

For a manual reproduction, run `Renderer/CAPTURE_MOUSE_INPUT.bat` as administrator
with the game closed. It launches Civ III normally with input tracing enabled
only in that process. Load the desired save, reproduce the problem, then return
to the capture console and press Enter. It leaves the game open and writes the
log and binary hashes under `%TEMP%\C3XMapFailure`. It does not change game
settings, select a save, move the mouse, or save gameplay. `-CheckOnly` validates
the collector without starting a game. Input tracing records native/display
coordinates, camera/zoom, drag endpoints, cursor flags, and the destination
projection; normal launches do not enable the detailed trace.
The opt-in trace also records renderer calls over 2 ms and software-cursor
sprite draws. `game_test_contact_sheet.py --kind mouse` aligns sampled frames
to the first recorded held-drag state. The script saves witness stderr and
requires a completed, nonempty window capture; delivered commands with a failed
observer do not pass.

Capture `20260927-204831` isolated game-thread image-composition calls taking
160–250 ms while the helper continued rendering. The input-coverage graph was
repeatedly evaluating unchanged regions of full-screen keyed UI transfers.
`test_native_hit_scene.py` now exercises that pattern alongside changing sprite
uploads and retained copies. Uniform regions can be folded without reading any
GPU map pixels. Its 400-transfer host comparison took 14.93 seconds before the
change and 0.17 seconds afterward. Live `20260927-214917` confirms lower input
coverage cost but exhausted the renderer publication queue. The subsequent
`20260927-220100` and buffered `20260927-220551` runs complete without queue or
native failures after UI publication stops forcing an immediate map render.
In `220100`, the five changed cursor positions reach native input handling in
2.5–16.6 ms, while destination-marker presentation still lags. The latter is a
renderer issue, not a claim that the mouse defect is fixed. The next candidate
retains static route textures and prevents repeated UI commits from interrupting
the independent frame deadline. Later `223106`, `225000` and `230614` complete
the drag without renderer failures or save changes. Their logs separate native
input handling, transport queue delay and rendering. Use every nearby 10 Hz
sample when bounding response: sparse `+0.2/+0.6` selections overstated delay.
For example, the first changed target in `223106` is absent at +140 ms and
visible at +249 ms; `225000` is absent at +170 ms and visible at +280 ms.
These are sampling bounds, not exact input-to-photon measurements.
Those older bounds used observer arrival times. Capture `234056` gives a
first-target bound of 74–191 ms using the compositor's source timestamps
(145–261 ms using observer arrival). The route reaches the helper at +45 ms.
Keep the timestamp basis with every reported bound.

Use `-SampleHz 1` as a capture-overhead control when comparing renderer counters.
At that setting, `231750` delivered 31.66 visual submissions/sec during dragging;
`232845` delivered 44.09 after tiny background reads stopped reconstructing
full-screen textures. Median sampled composition time fell 29.976 → 17.680 ms.
Both runs completed all seven mouse commands with no native failure, early exit
or save change. This does not measure scanout FPS or establish the 52 FPS target.
Keep high-frequency capture for the separate visible-latency check.
The low-overhead `234847` run records 43.56 successful presentations/sec during
dragging and approximately 48–50/sec while stationary. It completes all seven
commands without native errors or save changes. This confirms improvement over
the old path, but does not establish the sandbox target or complete mouse UX.

The presentation-limited candidate `c704bcc8bce5401294b88494fde7b312`
retains a DXGI presentation permit across unchanged frames and polls readiness
without waiting. Comparing identical windows relative to the first mouse-down
(0–7 seconds), one-Hz runs `234847` and `20260928-000128` record 42.74 and
42.46 presentations/sec. Use the mouse-down event, not the initial wheel event,
as the phase origin. At ten-Hz sampling, `235859` bounds target changes at
4–86 ms and 62–180 ms using compositor source times. The capture settings differ
from the older profiled run, so this alone is not a controlled latency speedup.

The same candidate's `20260928-000807` scrolling run completes all 32 steps;
eight window samples retain the terrain, units and fixed HUD. Counter samples
from the first 25 seconds after cadence collection starts record 32.57/sec;
the 28–47 second interval after that origin records 46.08/sec. Collection begins
at elapsed 20 seconds; the final camera command arrives at about 47 seconds.
Wheel run `20260928-001129` records 38.54/sec in the first 14 seconds after the
first wheel event and 47.50/sec in the 15–23 second interval. Both use one-Hz
window capture, no detailed helper trace, and have no native errors, early exit
or save changes. These measurements keep the navigation shortfall explicit.

An additional opaque native-transfer GPU-copy trial passed the pixel oracles
and async fixture (`4d1970a3d1e9480d82340855b661141e`), but live
`20260928-002021` recorded 32.56/sec during the same scrolling interval and
46.41/sec afterward. This does not establish a benefit over 32.57/46.08.
The trial code was removed and the source-matched `c704` trio restored; its
three staged binary hashes were rechecked with Civ III and its helper closed.

### Wheel and transition diagnostic

`-Scenario zoom -Seconds 55 -SampleHz 10` loads the disposable save and applies
positive and negative wheel notches, both endpoint clamps, a quick reversal and
three 40-unit partial deltas. Like `mouse`, it requires the diagnostic game to
own the foreground before sending input. It records signed deltas and QPC times
in `mouse-events.json`; Windows receives the equivalent unsigned event payload.
It polls at 20 ms so the reversal and partial deltas do not become one-second
steps. Check the actual event timestamps, zoom logs, centered tile and images.
The initial `225844` witness verifies direction, clamping, accumulation and the
unchanged camera, but predates that tighter polling. `233038` delivers the later
ten-event sequence, including the 162 ms reversal and three partial deltas. Its
center pick stays at the same tile, the native camera stays fixed, and returning
to 128 restores zero translation. Neither run proves easing; that implementation
is still pending.

Capture `20260927-210000` also confirms that the go-to software cursor has no
pixels. This matches the empty second 32 × 32 cell in the installed `cursor.pcx`
and native `load_cursor_images`/`set_mode_action` behavior. The destination ring
serves as the held-drag cursor; a rejected empty sprite is not evidence of lost
GPU cursor art. Judge response from mouse-to-route timing and sampled positions.

Keep one full window sequence in the VM. Pass `--window-source` to the contact
sheet helper to read that original `window/` folder while saving only the small
sheet and frame index beside the local log. After making a contact sheet, retain
its indexed source frames and the first/last local frames, then remove redundant
local JPEG copies only after verifying their hashes against the VM originals.
Keep `result.json`, logs, timestamps, hashes and contact sheets. Reuse that
evidence before recording another large sequence.

## Smooth-zoom evidence

The zoom scenario reads the version-10 helper header every 20 ms when
`-MeasureCadence` is enabled. `cadence.json` includes `zoom_q16`: 65536 is 1×,
81920 is 1.25×, and 98304 is 1.5×. Require intermediate values after each
wheel change and compare them with window timestamps; a received target or a
changed picking value alone does not establish a zoomed frame. Header reads
are local and read-only. Other scenarios retain one-second counter sampling.
The wrapper's final exit code now comes from the explicit scenario checks,
not DebugView's last cleanup command.

`stage=world-view-boundary` records the first map/unit boundary admissions.
A native UI-only unit canvas may supply uploaded words without owning a GPU
map. Notification drawing explicitly admits its canvas when eligible so its
fixed lookup shadows can be replayed after world zoom. No map readback is
allowed to satisfy this boundary.

### Native HUD size during smooth zoom

`-Scenario hud -Seconds 115` extends the interaction case with B and Enter on
its disposable initial Settler save to found a city. It exercises both native city-screen zoom levels, closes the city screen with
Enter after its load completes, then cycles world-map zoom, shows map messages,
scrolls the founded city's label and opens/closes the advisor. Check the
`stage=city-native-zoom` records for widths 64 and 128, three map-text events,
and both scripted camera moves. Inspect the
captured city name, production label, native status icons and map-message glyphs:
only their map attachment moves; their pixel dimensions must remain unchanged.
The test never saves the founded city. Check the screenshot that follows the
name confirmation before treating this as city-HUD evidence; posted keys alone
do not prove that founding succeeded.

### Centered city view

`-Scenario city -Seconds 88 -SampleHz 2 -MeasureCadence` founds a disposable
city, holds each native zoom for 16 seconds, sends wheel input and moves the
pointer toward both screen edges, then closes the city view. The longer hold
separates initial projection preparation from the completed view. It checks
native widths 64/128 and identical `city_anchor` pixel coordinates at both
levels. Inspect the city terrain and fixed native panels in the window samples;
passing input markers alone do not establish visible correctness. The same
save-copy, process ownership and cleanup rules above apply.


### Graphics-quality comparisons

Use the same disposable save, camera sequence and capture rate. `-SceneSamples 1`
is the production native-resolution sampling; `-SceneSamples 2` is the optional
MSAA comparison. `-SceneSharpness 0` bypasses scene sharpening; `0.35` is the
selected modest amount. Omit the options to exercise production defaults.
These child-process settings are restored after the run and recorded in
`result.json`. They do not alter native HUD/text or asset packs.

For timings use `-SampleHz 1 -MeasureCadence` without `-ProfileRenderer`.
Measure completed-presentation deltas over the same timestamp interval in
`cadence.json`; report normal, changing and settled zoom separately. Window JPEGs
are sampled visual evidence, not lossless pixel or physical scanout measurements.
`test_scene_projection`, `test_scene_detail` and `test_skin_shadow` supply exact
GPU/host oracles for the quality mechanisms. See
[results and source findings](../docs/render_quality.md).
