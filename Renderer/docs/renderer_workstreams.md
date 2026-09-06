# Renderer Workstreams And Promotion Contract

The custom renderer has two complementary workstreams with one-way, versioned
handoffs.  They share source-agnostic packs, renderer definitions, and visible
scene records, but they do not share responsibility for judging the same
problem.

## Renderer Lab

The **Renderer Lab** is the standalone visual and asset-development path.  Its
historical directory remains `Renderer/terrain_lab/`, but its scope includes all
map graphics: terrain, features, water, infrastructure, resources, cities,
units, lighting, animation, and effects.

The lab owns:

- source discovery, extraction, normalization, and provenance;
- logical asset mappings and reusable composition recipes;
- geometry, materials, shaders, lighting, animation, effects, and art direction;
- scale, grounding, silhouettes, readability, and deterministic variation;
- replayable standalone scenes, isolation views, contact sheets, and visual
  approval.

The lab answers: **given a complete source-independent scene, can C3X render the
right, good-looking picture?**  It never patches Civ III, suppresses native
draws, owns game state, or treats a successful build/hash as visual approval.

## Game Integration

The **Game Integration** path begins at the established off-screen bridge and
puts approved lab systems into Civ III.  It owns:

- authoritative state capture and stable scene/object identity;
- cache keys, bounded caches, invalidation, and stale-entry prevention;
- redraw scheduling, animation timestamps, pause/reset, and frame skipping;
- exact anchors, clipping, both zooms, scrolling, horizontal wrapping, and
  viewport changes;
- off-screen rendering, readback, compositing, and device-loss recovery;
- per-instance/category native ownership and suppression;
- retained fog, borders, labels, selection, status overlays, HUD, and UI;
- config-off behavior, diagnostics, performance telemetry, and visible hard
  failure without native replay.

Integration answers: **given live Civ III state, does the approved lab result
arrive in the right place at the right time, exactly once, with correct native
ownership and failure behavior?**  Integration may expose a fidelity defect, but visual
redesign returns to the lab.

## Paired Per-System Promotion

There is no global wait for the final lab scene before integration starts.  A
system becomes eligible for integration as soon as its own lab gate and required
visual approval are complete. Gates use matching suffixes: `L9` feeds `I9`,
`L10` feeds `I10`, `L11` feeds `I11`, and so on through `L21`/`I21`:

```text
lab implementation
  -> standalone technical gate
  -> explicit visual approval where required
  -> frozen handoff record and reference render
  -> same-numbered I# game integration
  -> native ownership, invalidation, compositing, and hard-failure gate
```

The handoff record must identify:

- lab gate and approval/evidence;
- pack, definition, and scene-schema versions;
- logical asset IDs and supported selector/state coverage;
- deterministic reference scene, camera, environment, inputs, and output hash;
- expected ownership boundary and complete custom-coverage unit;
- animation/update requirements and cache-relevant inputs;
- unsupported states that block integration and fail visibly if encountered in a custom-on frame.

Lab success alone never transfers ownership. Configuration off is the complete
vanilla path. Configuration on gives C3X exclusive ownership of the `m19` map
plane: Integration never places native terrain beneath a partial result and
never replays native terrain after load, capture, render, validation, blit,
device, or reentrant failure. A missing or unapproved required system therefore
fails the custom frame visibly and emits diagnostics. This deliberately keeps
ownership simple and prevents mixed native/custom terrain from masquerading as
a successful integration.

The final L21 combined scene remains a release-level visual/composition gate. It
does not hold previously approved terrain or object systems out of integration.
Natural wonders, constructed wonders, and C3X districts remain deferred to
M9, M10, and M11 respectively and cannot be pulled forward by this promotion
model.

## Verification Cadence

Routine work uses one track command from the repository root:

```text
python3 Renderer/tools/renderer_dev.py state
python3 Renderer/tools/renderer_dev.py lab
python3 Renderer/tools/renderer_dev.py integration
python3 Renderer/tools/renderer_dev.py full
```

On macOS, `lab`, `integration`, and `full` keep Python/state checks local and
dispatch their Windows-only phases automatically to the Parallels VM named
`Windows 11`, where this repository is shared from
`Y:\fun\Civilization III Complete\Conquests\C3X_Districts`.  Set
`C3X_RENDERER_VM` or `C3X_RENDERER_WINDOWS_ROOT` only when those names differ.
On Windows the same commands run directly.

The installed
`C:\Program Files (x86)\GOG Galaxy\Games\Civilization III Complete\Conquests\C3X_Districts`
path is a directory link to that same checkout through the stable
`\\Mac\Home\...` Parallels share. It is therefore the normal place to run
`git add`, `git commit`, and `git push` in Windows; no renderer or
injected-source deployment copy is required. Directly executing `./INSTALL.bat`
from Git Bash is the exception: MSYS resolves the directory link to its UNC
target before starting `cmd.exe`, so `ep.c` sees the shared checkout's parent
rather than the installed Windows `Conquests` directory and cannot find the
game executable. Launch the batch through Windows Command Prompt using the
local `C:\...\Conquests\C3X_Districts\INSTALL.bat` path, or from Git Bash first
change the subprocess working directory to `/c` and invoke `cmd.exe` with that
same local Windows path. Windows Git must trust the exact Parallels shared path,
and this repository uses `core.autocrlf=false` so Mac and Windows clients do not
rewrite one another's working files. Set
`C3X_RENDERER_WINDOWS_LIVE_TARGET` only if the UNC share name changes.

The injected smoke must execute beneath a real Windows `Conquests` directory so
`ep.c` can identify an unmodified executable. The workflow creates or reuses the
narrow directory link
`C:\Program Files (x86)\GOG Galaxy\Games\Civilization III Complete\Conquests\C3X_Shared_Verify`
pointing at the shared checkout. This compiles the current shared sources without
copying or overwriting either checkout. Set `C3X_RENDERER_CIV3_CONQUESTS` if the
GOG install moves.

- `state` is the fast pre-task contract check.
- `lab` runs focused lab/source tests, builds once, and runs only the current lab
  promotion script.
- `integration` runs focused bridge/state tests and one native build.  It runs
  `TEST_INJECTED_CODE_COMPILE.bat` only when `C3X.h` or `injected_code.c` differs
  from Git, or when `--with-injected` is supplied. After verification it checks
  that the actual installed `C3X_Districts` root remains linked to the canonical
  shared checkout and can read its Git metadata. Installation remains an
  interactive user step: launch the local `C:\...\C3X_Districts\INSTALL.bat`
  path from Windows Command Prompt and dismiss its normal success dialog. Do not
  invoke `./INSTALL.bat` directly from Git Bash while that folder is a UNC-backed
  directory link.
- `full` runs every source-independent completed gate followed by one consolidated
  Windows native build/smoke, injected compile, and the same checked live-link
  verification.
  Licensed local-asset probes
  remain targeted to the step that needs them. Use `full` when
  closing a production integration step or milestone, changing a shared
  contract, or preparing a strategic checkpoint—not before and after every
  ordinary edit.

The workflow writes ignored machine-readable reports under
`Renderer/verification/`.  Direct component commands remain useful for
diagnosis, but agents should not manually reproduce the standard command chain.
