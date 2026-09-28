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

`-Scenario lifecycle -Seconds 120` loads the disposable save, opens and confirms
the normal quit prompt, loads the same copy again from the menu, publishes a
map-text witness, then returns to the menu again. It never saves either game.
The result requires two successful first-map `render-done` events, two completed scene unloads,
the second map's text event, every scheduled command, and no native failures.
Review the samples for a visible loading bar, complete maps, both menu returns
and fresh unit/UI state. A script timeout or delivered Enter key does not prove
that the native menu accepted it. Helpers may exist for menu presentation, but
the previous scene helper must close before the menu is painted.

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
selected stack, with no city. F21 selects an owned land attacker from the stack,
or spawns the scenario's basic barbarian unit type as an owned test attacker
when the stack contains only civilians. It then creates one barbarian defender
of that type through the existing native spawn function. F22 requests an ordinary eastward native move,
which invokes Civ III combat. Both commands require custom rendering, the
explicit save and `combat` mode environment variables, and a single-player
session. Each can run once. The fixture does not force a winner or save changes.

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
