# Shared frame preparation and delivery

This delivery extends `58e35914` with the preparation and delivery responsibility
from the performance audit. Camera preparation and independent display use the
same unit resource admission. It does not complete persistent world ownership,
local updates, contributor selection, submission organization or final adapter
cleanup. Those remain the subsequent architecture work.

## Ownership and retired work

- Generic immutable mesh/DDS jobs use the existing CPU preparation workers.
  Display ticks reuse pending keys instead of restarting jobs. Decode validates
  the resident allocation budget before resizing arrays. Catalog/device
  retirement joins workers before releasing their owners.
- Pin the whole frame's mesh/material union before eviction. CPU borrowers also
  protect mesh payloads. Retain at most 96 MiB of unit payloads and 192 MiB of
  unit GPU mesh buffers; the worker pool reserves at most 64 MiB. Admission
  fails explicitly if the required union cannot fit.
- Adopt up to four payloads and two GPU meshes per preparation turn, with a
  soft 3 ms elapsed target. A single driver allocation can exceed that target;
  it is not a hard execution deadline. Vertex packing and shadow-bound setup
  still occur during bounded GPU adoption.
- Prepare the current unit poses and render a private back texture before
  evaluating the stable native composition graph. Swap only completed frames.
  Sampling performs no unit file reads, decoding, GPU mesh creation or scene
  rendering. Pending preparation holds completed pixels and preserves its
  callback; retirement keeps completed pixels and releases the callback.
  Main, reflection and shadow consumers retain the shared pose preparation.
- Only one current frame owns preparation scratch. Archived native versions
  retain their completed images through weak samplers. Scratch is limited to
  32 MiB and included in publication working-set accounting. Ordered view,
  visibility, projection, alias and unit-incarnation ownership still apply.
- Cold preparation services independent ordered commands at safe boundaries,
  up to 64 commands or 4 ms per turn. It never enters composition recursively
  from unfinished scene callbacks. Image execution receipts now signal an
  event instead of sleeping between receipt queries. The helper yields display
  during an admitted image batch and permits periodic presentation at 4 Hz
  while the native queue has at least 512 records; normal cadence then resumes.
- Semantic admission is separate from packet admission: 65,536 semantic work
  units, 8,192 packets and 128 MiB. Reliable commands remain ordered; camera
  supersession does not remove resource/action commands. Matching bridge,
  core and helper binaries use wire version 13.

These changes remove asset work from retained sampling and receipt polling from
ordered delivery. They provide ready inputs and stable resource lifetimes for
the next persistent-world/local-update work, without tuning broad rescans or
adding another camera cache.

## Current-code verification

Evidence is local under `Renderer/.cache/frame-delivery-step/`; its final manifest
binds the source closure, matching binary trio, test logs and cleanup receipts.
`build-9/source-binaries.json` freezes 343 build inputs. The final contract run
completed 107 tests: 106 passed, one local executable byte audit was unavailable.
Five pre-existing obsolete source assertions were excluded explicitly. The
private JSON reporter failed after successful tests and its receipt was recovered
from the preserved log. The separate guarded host suite passed 26 tests.
The D3D checks include 126 exact retained-composition oracles plus pending-frame,
projected callback, UI advancement and retirement coverage.
An earlier extended preparation run reported the unchanged
`test_shared_terrain_compiler_parallel_parity` failure. It is outside these
passing suites and remains unresolved; the terrain compiler was not modified.

The final build ran the disposable 1498 AD save at 2240×1260 with normal water,
waves and reflections. Startup reached the real 76-unit, 431-part view. The early
trace adopted 112 payloads and 59 GPU meshes: the payload union peaked at
75,822,428 bytes and mesh buffers at 27,008,432 bytes. Payload adoption turns
reached 34.093 ms despite their soft target; mesh adoption reached 1.073 ms.
Later core tracing exceeded the bounded collector, so it does not establish
complete late-phase or GPU timing.

All six 20-second-spaced destinations recovered to the correct authoritative
full map and matching observed screen, with zero renderer failures:

| Destination | Full-map adoption | First observed matching screen |
| --- | ---: | ---: |
| Cold north | 14.884 s | 17.482 s |
| First east return | 12.087 s | 12.998 s |
| Cold south | 15.655 s | 17.033 s |
| Warm east | 1.162 s | 1.538 s |
| Warm north | 1.482 s | 2.516 s |
| Final warm east | 1.003 s | 2.093 s |

`recovery9-delivery.json` pairs actual input QPC with matching authoritative
handoffs; the contact sheet verifies the destinations visually. Window evidence
is arrival-sampled at a requested 2 Hz, not physical scanout. Queue high water
was 3,086 packets, 9,306 semantic work and 51,878,144 bytes; maximum observed
oldest age was 3,918.074 ms. The bounded game test stopped with 509 records still
pending, so it does not prove final queue drain.

The full-resolution native fixture, with 40 ms preparation/admission delays,
passed 32 adopted maps, native UI, selection/grid and empty unit copies.
It drained explicitly: 1,467 accepted = 1,460 executed + 7 superseded; zero
rejected, abandoned or remaining records. Poll time peaked at 2.920 ms and
map readiness at 938 ms. This eight-unit fixture is not live-game FPS evidence.
The existing full-resolution native fixture at the 64-wave cap remains unresolved.

One matched quiet run pair measured successful presentations per second:

| Phase | Accepted control | Candidate | Difference |
| --- | ---: | ---: | ---: |
| Scroll, 30–54 s | 6.633 | 6.805 | +2.6% |
| Idle, 65–89 s | 13.013 | 14.430 | +10.9% |
| Zoom/reversal, 30–44 s | 10.324 | 8.582 | −16.9% |
| Settle, 50–64 s | 16.730 | 18.663 | +11.6% |

These single pairs show mixed cadence, not a general speedup or formal statistical
qualification. Helper private peaks were 3,738,189,824 versus 3,809,988,608 bytes
for scroll and 3,647,410,176 versus 3,701,227,520 bytes for zoom. The accepted
control also ran the identical six-destination schedule, but exhausted its
8,192-packet queue during the first cold jump at 43.106 s (about 5.2 s after
input). The fatal watcher stopped that disposable run and preserved its prefix;
no later destination or full-quality latency is attributed to the control.
See `matched-cadence.json` and `control-recovery-capture/result.json`.

The remaining 12–16 second cold scene cost belongs to persistent world ownership
and local updates. Pose/self-shadow preparation and later submission costs also
remain; this step does not claim that all frame work runs on CPU workers.
GPU elapsed timings remain unqualified on the VM.

All 22,072 runtime dependencies (6,842,076,936 bytes), the original save and the
disposable input stayed unchanged. Each game run records process/task closure
and restores config, installed executable/JGL/INI, cursor and environment.
The accepted control trio is restored at handoff; the final candidate remains
available with its exact identities in `build-9/`. No reference image, injected
source or patch-table address changed; `required_user_action: []`.

Superseded generated window rasters were deleted after retaining contact samples
and bounded failure witnesses. Current candidate/control captures, logs, input
identities and binaries remain; `superseded-raster-pruning.json` records deletions.
Runtime packs, `Renderer/packs.zip` and held evidence were not modified or archived.
