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
without environment settings. No new native patch address is needed for this step.

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
map sampling at 15 Hz. No copied animator or competing presenter. See the
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

## Validation and measured effects

Production Windows build, 54 focused contracts, actual worker and native zoom
adapter under MSVC/x86, injected compilation and day/night unit/underlay replays
pass. Nine prepared cameras and three selected-animation positions match independent
same-area redraws. Four topology and six appearance edits pass independent redraws;
the wider 14-step sequence passes exact revisits without fallback tiles or recovery.
All three zoom images exactly match the accepted control; full detail and existing
accepted raster behavior are preserved. No asset/reference was changed.

Control: source `02ff301b`, DLL
`4cbdfa93b44a7f41d18280db3932d216dc794ece15d08a3dfd1cfdcb3d85e8be`.
Candidate DLL:
`cec59b1b243c8fffd076430968d972a147c80196682ae86a713ffc4800820dea`.

| Whole consumer request | Control | Candidate |
| --- | --- | --- |
| First closer zooms after preparation opportunity | 1,790.81 / 2,302.34 ms | 6.21 / 6.64 ms |
| Repeated zooms, six visits, mean / maximum | 602.68 / 689.26 ms | 6.94 / 8.12 ms |
| Nearby paced scrolling, 48 calls, mean / p95 | 11.39 / 14.14 ms | 11.39 / 14.78 ms |
| Warm idle, eight bodies, 60 calls, mean / p95 | 4.85 / 6.75 ms | 5.29 / 6.63 ms |
| Wider scrolling with coverage misses, 14 steps, mean / maximum | 222.07 / 396.23 ms | 220.05 / 388.68 ms |

Endpoints include capture, render/acquisition and actual CPU output copy; idle also
includes all eight bodies. They are 1119×900 harness requests, not native displayed
FPS. Both zoom arms have a separate five-second preparation opportunity. Candidate
uses it to build future views; it is not counted as faster construction. That
harness interval has no continuous gameplay callbacks: actual readiness depends
on current demand, and a zoom requested before preparation finishes can still block.
Cold initial dense render is 7.54 → 7.33 s in the zoom pair, without a claimed gain.

Nearby/idle have no demonstrated speedup. Maximum sampled map age grows
263 → 681 ms in paced scrolling while cold alternatives compete, and 133 → 333 ms
in warm idle. Current-priority correction removed the earlier four-second hold,
but quick requests do not guarantee equally fresh animation. Native unit timing
is unchanged. General jumps remain dominated by compilation and GPU completion.

A bounded 20-call idle diagnostic records 19 map jobs in both, zero builds/uploads
and static submissions, 532 dynamic selections and 122.35 million resolved pixels
in both. Fifteen candidate updates avoid copying the entire wide CPU donor.
Finishing lanes are 1.38 → 1.46 million; background area wall intervals total
423 → 385 ms. This attributes skipped CPU publication work, not a broad draw/GPU
speedup. Final x86 runs retain at least 1.45 GiB free contiguous address space;
that is harness headroom, not a live-game guarantee.

Exact binaries, input/source receipts, invocation manifests and summaries are in
`native/build/world-view-zoom/`. Final candidate runs have suffix `v12`; matched
controls use `control-final-{zoom,scroll,idle}-v10` and `control-final-wide-v11`.
`comparison-v12.json` and `selection-diagnostic-v12.json` record the results.
The exact candidate is staged in `Renderer/bin/` for ordinary `INSTALL.bat`;
`staging-receipt.json` records source hashes and checks. No install or game launch
was performed.

## Next responsibility

Make remaining city/route/improvement content projection independent and retain
world-level spatial membership feeding selected passes. This reduces the cold
preparation itself and helps views outside finished coverage; retaining more
raster images or adding workers alone cannot remove the measured geometry cost.
Preserve the working prepared-view handoff while extending its coverage. Complete
active-unit demand collection remains separate: the latest live trace's 85 pose
misses have median 60.76 ms and p95 127.34 ms despite 95.8% hits. Neither mechanism
has established live displayed FPS. Native zoom/camera cadence and overlay/picking
acceptance remain the strategic game checkpoint, through the user's installation.

## Preserved controls and closed findings

[Checkpoint history](history/retained_renderer_checkpoints_20260914.md) preserves
prior measurements, accepted visual differences and rejected intermediate designs.
[Completed output work](history/retained_output_completed_20260913.md) records the
119.40 → 103.28 ms dense gain and no stationary gain. Do not reopen closed output
or resolve experiments without a materially different mechanism. Global 30 Hz map
sampling regressed warm requests 3.97 → 7.46 ms and remains rejected. Parallels GPU
timestamps/event-query attribution is unreliable; whole-request CPU wall time is
the performance authority, and readback includes queued GPU work.
