# Scripted real-game renderer diagnostic

This is an explicitly requested integration diagnostic. It does not authorize
ordinary agents to automate gameplay. Run it with Civ III and debug collectors
closed, using the shared checkout link beneath the installed `Conquests` folder.
The game executable must contain the current injected diagnostic hooks.

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
eastward unit movement key, and opens/closes the domestic advisor. It operates
on the disposable save and never saves the result. The default `scroll` scenario
still performs the bounded 32-camera loop. Command delivery alone is not a
visual pass; inspect the recorded windows and renderer failures.
