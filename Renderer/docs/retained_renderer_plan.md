# Retained renderer: current implementation and priorities

## Objective and boundaries

Complete world → view → pass → publication with bounded ownership, full detail
and source animation timing. Measure work eliminated, work prepared before demand
and caller delivery separately. Civ III supplies demand; workers never advance
gameplay, notify it or request redraws. This connected implementation objective
supersedes the former single-experiment queue.

Preserve AGENTS.md, native overlays/visibility/picking, generic assets, accepted
visual contracts and deferred wonders/Districts. Tested DLL staging is authorized;
installation and launch remain the user's actions through ordinary `INSTALL.bat`,
without environment settings. The authorized Advisor lifecycle hook reuses its
existing patch-table addresses; see the patch ledger.

## Actual code responsibilities

Completion means ownership and behavior for admitted categories, not permanent
GPU residency or knowledge of uncaptured gameplay.

| Responsibility | Implemented ownership | Remaining limits |
| --- | --- | --- |
| Persistent world/instances | `CapturedScene` retains canonical appearance, revisions and three non-owning compiled projection handles. `ResidentContent` and shared assets own reusable content independently of observations and camera departure. | Captured surroundings only. Legacy city/route/improvement compilation still depends on projection; variants reuse completed work rather than removing that dependency. Units have a separate playback/pose owner. |
| Local validity | Appearance, semantic neighbors, city exclusions, coast/world samples and river-cell proofs guard compiled hits. Local changes retire known-invalid alternate views; fresh capture validates every publication. | Asset/device/world-basis changes remain global. Unobserved appearance cannot authorize output. |
| World → view selection | Current authoritative occurrences assemble passes; a spatial contributor index selects static inputs. Animated body/shadow/filter bounds are selected against the requested viewport independently of surrounding static coverage. | Static index remains view-scoped. Dynamic selection uses small scans. No complete native active-unit pass yet. |
| Compatible submission | Shared meshes, tree instancing, ordered materials, common depth and bounded shadow-page batches serve foreground and background through the same path. | Individual demanded unit misses and optional reflection profiles retain their existing execution. |
| GPU/output reuse | Stable surrounding static color/depth, selected animation, sparse restore, pending finish damage and consolidated readback. Selected updates publish only their fresh viewport; untouched donor margins keep their true old clock. | GPU allocation and full hardware MSAA resolve still cover the wide working surface. Physical targets are not narrowed. Tested retained profile has waves/reflections off. |
| Preparation/native delivery | Bounded CPU helpers and one GPU owner prepare copied nearby/zoom views and unit poses. Fresh native requests pull compatible publications. Current refresh can interrupt unfinished prospective work at safe points. | Cold/outside-coverage views still use exact blocking fallback. Submitted GPU work is not preemptible; preparation can delay ambient freshness. Native delivery is implemented for prepared coverage, not general nonblocking scrolling. |

Native selected idle/work loops preserve source frames/duration; inactive unselected
units freeze. Directed actions keep native cursors/anchors. The authorized GOG
adapter offers unit visuals at 33 ms, preserves native advancement at 66 ms and
map sampling at 15 Hz. Extra work suspends during guarded native UI operations.
No copied animator or competing presenter. See the
[patch ledger](civ3_patch_dependency_ledger.md) for capabilities and fallbacks.

## Completed world/view and zoom extension

Optional `c3x_renderer_prepare_view` copies caller-supplied alternate captures.
The native adapter supplies the existing 128/160/192 zoom levels using native
fixed-point anchors and full captured appearance. Query/admission means queued or
retained, never permission to display. Actual capture must prove exact projection,
visibility, content, ownership and lifecycle compatibility before acquisition.

Three bounded prepared views retain useful projection results and switch through
the same owner. Geographic padding is still at most 128 pixels per axis within
2240×1192; this step did not enlarge the surrounding radius. Combined committed
CPU areas are capped at 64 MiB, each at 32 MiB; one additional in-flight area is
capped at 32 MiB. Two queued prospective snapshots share a 16 MiB payload limit;
one latest current-refresh snapshot is separately bounded by API maxima (under
8 MiB). Existing GPU assets/targets, front publication and transient copies are
additional. Eviction invalidates borrowed handles without destroying world identity.

Selected refreshes preserve static output outside damage. Pending off-view damage
is bounded and finished when selected; it cannot resurrect an old pose. The wider
donor is refreshed periodically, with age used for scheduling rather than static
invalidation. Current requests take priority over unfinished future zoom builds.
This is reuse plus preparation, not fully projection-independent world compilation.

## Current regression checkpoint

The supplied game log has 2,135 map passes (median 12.05 ms, p95 173.60 ms,
maximum 4.59 s), 900 cancelled unfinished area jobs, and over 54,000 source-shadow
and unit-body detail lines. These are logged native intervals, not displayed FPS
or a matched performance comparison. Debugger overhead is not isolated.

`patch_Advisor_GUI_open` saves/sets/restores the existing modal guard across native
construction and the dialog loop. The renderer now also honors `paused_for_popup`,
which already covers the whole outer popup routine; the shorter setup-helper guard
was removed. The existing unit-command setup hook scopes the same suspension over
button reconstruction. These are function-lifetime guards, with no timeout or
change to native painting. Default tracing samples detail instead of emitting every
submission/cache hit. Full detail, pixel blending and worker scheduling stay unchanged.

Windows build, nested Advisor/cadence contract, actual x86 worker/zoom adapter,
approved injected compilation, GDI handoff and day/night unit replays pass.
Nine prepared camera images at 1440×900 match the preserved control byte for byte;
their independent redraw differences are identical to that control. Three selected
animation positions pass their redraw checks. Black bars did not reproduce there.
Immediate native CPU reads/erasure also pass with the old and new GDI paths: the
GDI batching hypothesis is unconfirmed, and no flush workaround was added.

A trial one-second delay before retrying cancelled speculative views was removed:
48 scrolling requests averaged 13.30 ms with and without it. It did not justify an
additional timing policy. Exact trial binaries, invocations and measurements remain
in `native/build/native-regressions/comparison.json`; they are not the final build.

Final whole-request mean/p95 is 13.30/16.81 → 13.31/16.29 ms for 48 scrolling calls
and 5.62/7.07 → 5.50/7.07 ms for 60 eight-body idle calls. No speedup is established.
The preserved control is source `d3acf940`, DLL `cec59b1b…820dea`; the staged final
DLL is `ccc5c1dd…d2817b`. Full hashes, `final-comparison.json` and
`final-staging-receipt.json` are in `native/build/native-regressions/`. Ordinary `INSTALL.bat`
is the user's test path. Actual Advisor/popup/button painting and the reported
black bars/shadow flicker remain the native integration checkpoint. Passive
`map-black-span` diagnostics distinguish black source pixels from later composition;
black map margins can be legitimate. No new visual acceptance or live speedup is
claimed, and no game was launched.

Prior world/view/zoom evidence remains in `native/build/world-view-zoom/`:
`comparison-v12.json`, `selection-diagnostic-v12.json` and final `v12` runs preserve
full measurements and source/binary identities. Prepared first closer zooms went
from 1.79/2.30 s to 6–7 ms after a separate five-second preparation opportunity;
nearby scrolling/idle did not improve and cold construction remained expensive.
The 20-call diagnostic retained 19 map jobs, zero geometry builds/uploads/static
submissions, and 122.35 million resolved pixels in both; reduced CPU donor copying
was not a broad GPU speedup. GPU timestamps remain unreliable.

## Next responsibility

First verify native Advisor/popup/button behavior and resolve the black-bar/shadow
source at the native composition boundary. Then make remaining
city/route/improvement content projection independent and retain
world-level spatial membership feeding selected passes. This reduces the cold
preparation itself and helps views outside finished coverage; retaining more
raster images or adding workers alone cannot remove the measured geometry cost.
Preserve the working prepared-view handoff while extending its coverage. Complete
active-unit demand collection remains separate: individual pose misses still block
native demand. Neither mechanism has established live displayed FPS. Native
zoom/camera cadence and overlay/picking acceptance remain the strategic game checkpoint, through the user's installation.

## Preserved controls and closed findings

[Checkpoint history](history/retained_renderer_checkpoints_20260914.md) preserves
prior measurements, accepted visual differences and rejected intermediate designs.
[Completed output work](history/retained_output_completed_20260913.md) records the
119.40 → 103.28 ms dense gain and no stationary gain. Do not reopen closed output
or resolve experiments without a materially different mechanism. Global 30 Hz map
sampling regressed warm requests 3.97 → 7.46 ms and remains rejected. Parallels GPU
timestamps/event-query attribution is unreliable; whole-request CPU wall time is
the performance authority, and readback includes queued GPU work.
