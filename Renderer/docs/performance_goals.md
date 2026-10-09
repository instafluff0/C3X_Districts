# Renderer64 performance goals (Civ VI-comparable)

Set October 8, 2026, from the performance reviews
([October 7–8](performance_review_20261007.md), sections 11–13).

## Aim

Make Renderer64 feel comparable to Civilization VI: 60 fps, a camera that
responds within a frame and moves smoothly, and no hitches.

Two audiences shape the trade-offs:

1. **The Parallels VM on the user's Mac.** The user plays here, and the planned
   demo (starting from a new game) is judged here, so VM performance is the
   measured target.
2. **Future Windows users.** Changes must also be sound on real Windows
   hardware. No VM-only hacks that would degrade normal D3D11 behaviour.

## Constraints that do not move

- **Visual quality.** No visible quality loss. Temporary stand-ins (previews,
  resamples) must stay rare and brief.
- **Civ III stays authoritative.**
  - Vanilla edge-scroll steps and timing.
  - Native unit-movement timing.
  - Native visibility.
  - Civ III's own overlays (labels, unit status, borders) stay registered to
    the map.
- **Tests.** Every fix gets a regression test confirmed to fail on the old
  behaviour.
- **Source rules.**
  - Renderer code lives under `Renderer/`; `injected_code.c` and `C3X.h` only
    for hook points.
  - Every renderer patch delegates to vanilla when custom rendering is off.
  - `civ_prog_objects.csv` entries may be added for clean, simple, major wins
    (user permission, October 7). Record each one in the patch ledger and tell
    the user.
- **Injected code** reaches game tests only after `INSTALL.bat`. Close its
  "Success" dialog with `taskkill /IM temp.exe`, then delete `temp.exe`.
- **Capture data.** Delete it once its numbers are recorded. Disk is tight.
- **Git.** No commits unless the user asks.

## Where things stand (VM, October 8)

| | Light save | Busy save (1498 AD) | Civ VI feel |
| --- | --- | --- | --- |
| Idle | 60 fps | ~41 fps | 60 fps |
| Scroll | native step pace; each step appears ~45 ms after Civ III draws it | one step every 230–470 ms (native: 78) | continuous, 60 fps |
| Zoom transition | 27–47 fps | 16–35 fps | 60 fps |
| Far jump | 37–80 ms | 1–2 s | well under 1 s |
| Reveal after a move | ~190 ms | not measured | immediate |
| Memory (Renderer64) | not measured | up to 7.2 GB RAM, 3.0–3.9 GB GPU | about 8 GB RAM and 2 GB GPU for the whole system (Civ VI recommended) |

## Gaps, in priority order

### G2. Presentation stalls (first)

**Evidence.**
- Zoom frames wait 10–44 ms on the GPU fence or the back-buffer bind, while
  our own work is 5–9 ms of CPU and 3.4 ms of GPU.
- The fence check runs inside the renderer gate. Parallels ignores
  `DO_NOT_WAIT`, so the check blocks, and camera and UI traffic queue behind
  it.

**Hypothesis rejected (October 8).** Changing every pixel every frame did not
slow light idle or zoom (performance review, section 13), so the waits come
from work specific to zoom frames.

**Work.**
1. Find what zoom frames add between frames: native UI batches, composition
   assemblies, camera traffic.
2. Stop blocking inside the gate: wait outside it, or find a completion check
   that does not block.
3. Evaluate presentation options for both VM and Windows: dirty rectangles,
   swap effect, present interval, frames in flight.

**Done when.**
- Light zoom transitions reach ≥55 fps.
- Presentation waits are below 5 ms at p90 in idle and zoom frames.
- Nothing regresses on the busy save.

### G3. CPU cost per busy frame

**Evidence.** About 10 ms of CPU per frame:
- about 540 unit-part draws: 2.8 ms
- water and reflections: 2.7 ms
- composition: about 4 ms

**Work.**
1. Batch or instance unit parts by mesh.
2. Then water and reflections, then composition.

**Done when.** Busy 1× idle runs at ≥55 fps in the VM, with scene CPU at or
below 6 ms per frame.

### G1. Close the gap with Civ VI's practices (largest feel win)

Started as "camera decoupled from content jobs". On October 8 the user
widened it: keep the camera targets, and move toward Civ VI's practices
behind them, including a memory footprint near its requirements ("if you can
run Civ 6 you can run this"). Civ VI is a general direction, not a rulebook:
adopt a practice where it helps C3X, never where it hurts C3X performance or
fidelity.

**Evidence.**
- **Camera.** Every scroll step, zoom and jump waits for a camera job, then
  for Civ III's next 78 ms tick to draw its overlays. Busy steps take
  230–470 ms, jumps 1–2 s, and the first busy scroll step about 1 s.
- **Per-step work** (busy save, median per step; review, sections 18–19). Each
  job redoes camera-dependent caches:
  - shadow pages: 17 ms, mostly bookkeeping over all casters;
  - the static layer: 11 ms, from whole-layer re-checks, repairs and
    recentring every 2–3 steps;
  - tile uploads from the in-RAM world: 16 ms.
- **Memory** (busy save, m1). Renderer64 uses up to 7.2 GB of process memory
  and 3.0–3.9 GB of GPU memory; Civ III uses 0.3 GB.
  - Whole-world tile geometry is 2.0 GB, baked into unique meshes per tile.
  - The static layer's targets can reach about 0.5 GB.
  - Civ VI asks for about 4 GB RAM and 1 GB GPU memory as a minimum, and
    8 GB and 2 GB recommended (published requirements, approximate).

**Civ VI practices to adopt.** These are inferred from Civ VI's behaviour and
its package data, not from its engine code.
1. The camera moves every frame from resident content; background jobs only
   refine.
2. One copy of each model, drawn many times (instancing), instead of unique
   per-tile meshes.
3. Compact terrain, so the whole explored world can stay resident.
4. Small, bounded render targets instead of large pre-drawn layers per zoom
   lane.
5. A memory budget set by a hardware tier, not by however much memory is
   free.
6. Background streaming with level of detail for far zoom and jumps.

**Work.**
1. Camera design: `Renderer/docs/camera_decoupling_design.md` (written).
2. Memory census by category: GPU targets, textures, geometry, CPU caches,
   driver copies.
3. Static layer: bounded size, no whole-layer re-checks or recentring on
   camera steps.
4. Shared instanced models and compact terrain, compared in the Lab for
   fidelity, until the whole explored world fits the budget.
5. Show the requested camera at once from resident content. Keep Civ III's
   map-attached overlays aligned by shifting their layer in image space until
   Civ III redraws them. Let jobs only refine.

**Done when.**
- Busy scroll steps reach the screen within one frame of Civ III's tick.
- Zoom responds within one frame.
- A far jump shows its first image within 100 ms and full quality within
  500 ms.
- Memory moves toward Civ VI's recommended tier: on the busy save at
  1×–3×, aim for Renderer64 at about 2 GB of GPU memory and 4 GB of process
  memory, with no visible quality loss. These are directional figures, not a
  hard limit.

### G4. Heavy busy-map jobs

**Evidence.**
- Shadow-page proofs: 84–206 ms.
- Static-layer recomposition after each zoom step.
- Tile restore and upload on jumps: about 1 GB.
- Zoom-out at 16–23 fps.

**Work.** Stream and keep these resident so they never block the camera. G1
removes much of the visible cost.

**Done when.** Busy zoom-out reaches ≥45 fps and jumps reach full quality
within 500 ms.

### G5. Native UI traffic between processes

**Evidence.**
- Image batches back up behind GPU drains: up to 200–386 ms outliers, queue
  p50 233–414 ms during busy scroll.
- Synchronous adopt and begin round trips.

**Work.** Fewer synchronous round trips, smaller image batches, and keep the
gate free.

**Done when.** Image queue p90 is below 50 ms during busy scroll.

## Also open (from earlier passes)

- **Busy 2× scroll.** The view is soft while scrolling (resampled static
  stand-in).
- **First busy scroll step.** Takes about 1 s.
- **Reveal after a move.** About 190 ms.
- **Light scroll step to screen.** About 45 ms after Civ III draws it.

## Measuring

**Saves** (in `Renderer/.cache`):
- light: `compiled-composition-step/light-initial-input.SAV`
- busy: `composition-integration-step/input-1498AD.SAV`
- new game: `perf-review-20261004/newgame-4000BC.SAV`
- user: `perf-review-20261004/user-3350BC.SAV`

**Tools** (in `Renderer/tools/`):
- `near_report.py`: scroll, idle, zoom and jump segments.
- `zoom_report.py`: per-notch zoom transitions.
- `step_report.py`: camera job phases.
- `unit_timing_report.py`: unit moves and reveals.
- `game_samples_report.py`: Civ III thread samples.
- `sample_memory.ps1` (run in the VM beside a capture): process and GPU
  memory of Civ III and Renderer64 every 2 s.

**Run settings.**
- `-MeasureCadence` for fps.
- `-ProfileRenderer` with `C3X_RENDERER_TRACE_BUFFERED=0` for reliable helper
  logs.
- `C3X_RENDERER_ROUTE_WITNESS=1` for presentation times.
- `C3X_RENDERER_PROFILE=2` for per-frame GPU phases.
- `SampleHz=10` window frames for visual checks.
- `C3X_RENDERER_PROFILE=1` with `C3X_RENDERER_MEMORY_CENSUS=1` for the
  renderer's own geometry and cache memory traces.

**Noise.** Runs vary by ±30%. Repeat before concluding, and record results in
the current performance review.

## Working autonomously

Work through G2, G3, G1, G4 and G5 in that order. Start G3 while G2
experiments run in the VM, if they don't compete for it.

**Check in with the user only for:**
- a blocker
- a change that would visibly alter or degrade the picture
- a `civ_prog_objects.csv` entry that is not clean and simple
- a trade-off between the VM and real Windows that cannot be avoided
