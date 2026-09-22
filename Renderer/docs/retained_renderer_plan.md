# Renderer roadmap and current status

## Decision and objective

Agreed September 15, 2026: retain the original six architectural responsibilities
and complete cheap GPU-ready scene rendering on top of independent visual frames.
Scrolling should usually select/draw reusable content. One world-content schedule
prioritizes nearby unfinished regions; it does not prepare destination images.
Full detail, authored animation and Civ III's gameplay/state authority remain
unchanged. The zoom-owned map-overlay extension below is a planned rendering
ownership change, not a second visibility, selection, or pathfinding system.

**Current technical checkpoint: M3.7 asynchronous retirement/recovery implemented
and automated acceptance passed.** M1 and M2 automated
acceptance remain complete. Manual pans can defer coherently; native selection,
actions, programmatic centering and projection changes retain exact behavior.
Units stay above all map geometry. Native actions, visibility, controls and unit/UI
ordering remain authoritative. Shore waves, water motion and reflections are on
in every normal performance workload. The tested renderer DLL is staged as an **evaluation build**; the current
staged working-set build loads on game restart. The new camera-consolidation
candidate removes injected speculative calls and requires reinstall after staging. Live
gameplay and performance acceptance remain pending.
[Architecture](renderer_architecture.md) owns the design;
[validation](benchmark_workflow.md) owns measurement and acceptance.

**Current implementation sequence:** the input recorder now reproduces the
successful prefix of the first live capture. The user's four measured working-set
and submission changes are implemented; final evidence and remaining limits are
in [frame working-set results](frame_working_set_results.md). This replaces the
static backup and duplicate reflection pages in the shared scene path. It does
not certify the failed live tail, complete M3.8 performance acceptance, or start
a separate-process renderer. Future [LORE work](#lore-testing-and-migration-milestones)
must follow the remaining measured costs.

The [production architecture audit](architecture_audit.md) records the remaining
cross-system consolidation: content scheduling and borrowed-input lifetimes,
prepared-image navigation, native transaction boundaries and whole-process memory
admission. M3.8 work must name and retire superseded producers/storage/callers
alongside their replacements; the four working-set changes are not a completed
architecture-wide cleanup. The implemented follow-up is described in
[camera consolidation](camera_consolidation_results.md), with its remaining
native transaction/content-lease costs and separate live validation gap.

## M3.8 current handoff — in progress

**Completed consolidation:** one world-region preparation order replaces the
neighborhood geometry queue. Exact retained-scene views replace padded/alternate
zoom image production, its queues, crop finishing and injected requests. Cache
admission accounts for concurrent publications and native composition. No new
patch symbols or rendering ownership; all normal water effects remain enabled.
Existing native barriers preserve actual CPU/UI ownership. See
[camera consolidation results](camera_consolidation_results.md) for the replacement
and retirement ledger, measured effects and independent regression checks.

**Next unfinished responsibility:** shorten the remaining exact map transaction,
content compilation/adoption and view/pass assembly on the existing recorded
workload. The cold frame is still expensive; the Standard <33 ms p95 camera goal
and the failed live tail are not accepted. Do not infer live FPS from replay API
timings or use another manual capture as a prerequisite for that investigation.

### Preserved earlier M3.8 evidence

The following records precede the camera consolidation. Binary hashes, staging
instructions and next-step pointers in this historical section are not current.

**Completed:** the saved live prefix now has reconstructed display images,
184,198 checked production lifetime decisions (zero mismatches), and 116 exact GPU
pixel checks. Strict ambient outcome validation rejects its readiness loss at
23.394 s. The captured first-stroke → public DC → revoked map-canvas sequence is
reproduced in the real DLL/JGL fixture and fixed by admitting an eligible canvas
on its first valid stroke. Genuine escaped CPU/DC aliases retain native fallback.

Native action replacement no longer invalidates the whole ambient graph: the last
published unit pose stays frozen until the native screen replaces it. Water and
resources keep their independent clock. Popup/Advisor/button setup and loss of
focus no longer pause visible ambient delivery. Injected changes are small
removals from existing hooks; the approved injection smoke passes. No new symbols,
patch-table entries or renderer ownership. Re-run `INSTALL.bat` for these wrapper
changes when testing the staged candidate.

**Current memory/submission architecture:** static animated damage redraws from
retained geometry into one scene MSAA color/depth set. The previous tiled static
backup implementation is deleted. Relevant reflection cells stay in the current
atlas with exact dependency keys; no duplicate reflection-page textures remain
in that route. Static/water submissions retain compatible inputs and frozen water
redraws only when damaged. Working attachments and reproducible caches share a
1 GiB logical envelope, independently of process VA and native-front ownership.
See [current measured effects and checks](frame_working_set_results.md).

**Historical validation below** describes the prior backup-based build. Its
correctness scenarios remain required; its memory and timing claims are not the
current implementation's results.

**Validation:** `native/build/ambient-continuity-reuse/receipt.json` passes at
2240×1260 with all water effects, eight mixed units, actual JGL/native wrappers,
12 unpublished action replacements, 24 ongoing movement steps (38 autonomous
frames, 30 map samples), blocked-UI delivery, unfocused visibility, first stroke,
fog/reveal, tactical/UI parity, navigation, reset/recreation and config-off.
120 visual requests average 31.26 ms; desktop completion averages 38.26 ms,
p95 52.88 ms. No overall frame-latency improvement over the whole-view control is
established. These are fullscreen fixture results, not gameplay FPS or scanout.
The matching 1 GiB address-pressure run also passes, averaging 47.66 ms desktop
completion with 53.18 ms p95; it is a capacity stress test, not live-game timing.

**Staged evaluation DLL:** `80c3bc111fb38433e5012aff8220605017e4681c6d040744a55adb16a11a99aa`.
Both complete native and pressure/recovery runs pass with matching source/binary
proof. `native/build/ambient-continuity-stage.json` records staging and rollback.
Normal launches keep recording disabled. No INSTALL or game launch in this turn.

**Next unfinished responsibility:** qualify the new input recorder against the
live workload before further GPU/foreground composition optimization. Scene,
ambient and native-owner inputs now re-enter production code, and visible Windows
playback is implemented; see the [concise input handoff](input_recording_handoff.md)
for the candidate and current evidence. The old 80.788-second live journal remains
a byte-limited pixel-only prefix and cannot retroactively provide those inputs.
A 1 GiB VA reservation models capacity, not Civ III heap fragmentation. Matching
pixels and targeted regression passes do not certify live performance.
Standard <33 ms p95 navigation and sustained live stability remain open. See
[findings](live_usage_findings_20260920.md) and
[validation](benchmark_workflow.md#recorded-native-composition).

Whole-world appearance enters through bounded caller-thread pages. Existing
workers prepare canonical regions and wrapped occurrences independently of the
camera route, using immutable renderer-owned inputs. A bounded 1 GiB compressed
session backing preserves compiled meshes/proofs across GPU eviction; lifecycle
retirement clears it. Foreground adoption validates proofs before publication.
Workers never read native pointers. City vertices retain their consumed channels
in 88 rather than 168 bytes; rigid infrastructure shares source buffers and
compact placements across color/reflection/shadow passes. Flat and conforming
pieces preserve their original order and bounds. Normal geometry caps remain
768 MiB for Standard and 384 MiB above 8,192 actual tiles.

**Performance-test binary:** `native/build/m38-parameter-append/C3XRenderer.dll`,
SHA-256 `7cbbfe9d65eccb7af834ad0ef8afbb56175919fa879a114eb709266876ea2247`.
This historical timing binary predates the native ownership repair staged above. Optional
off-screen guard drawing and its
blocking driver flush are retired; visible damage still uses the retained scene.
Fully invalid attachments use direct color/depth clears. Mirror cells use the
existing conservative spatial index for ordered draw traversal while complete
inputs still authorize shadows and lighting. Cache eviction transfers a compatible
reflection texture to the next cell instead of creating another allocation. The
same 256 MiB cache cap, exact dependency keys and ordered GPU copies apply.
Draw constants append into unused ranges of the same 64 KiB allocation when
supported, with DISCARD on wrap and the existing older-driver fallback.

**Corrected timing evidence:** exclude `m38-standard-unprofiled` performance because
the user was playing Civ III concurrently. Earlier `--profile` campaigns include
costly in-request address-space walks/buffer enumeration and are diagnostic, not
ordinary production latency. Subsequent runs closed the game, serialized GPU work
and disabled profiling, at 1120×1192 with full detail and all water effects on.
Each Standard campaign prepared all 5,000 tiles / 247 regions before 100 seeded
jumps to 98 distinct destinations. No world builds, restores, geometry uploads or
map readbacks occurred; all six independent cold-image comparisons passed.

| Clean Standard campaign | Request mean / p95 ms | Desktop mean / p95 / max ms |
| --- | ---: | ---: |
| Evaluation control (`m38-standard-clean-baseline`) | 144.32 / 534.02 | 171.61 / 540.56 / 644.37 |
| Mirror-selection candidate (`m38-standard-selected-mirrors`) | 70.73 / 102.63 | 168.07 / 645.51 / 932.95 |
| Same candidate, VM raised (`m38-standard-visible-vm`) | 71.36 / 131.34 | 199.34 / 814.50 / 1212.04 |
| Current buffer-reuse candidate (`m38-standard-parameter-append`) | 73.00 / 127.94 | 202.63 / 696.84 / 1111.78 |

**No overall speedup is established.** Removing optional submission and allocation
churn shortened request handling, but long waits migrated into foreground drawing
or presentation. Raising the VM did not remove them. Draw bounds tests fell from
37,994 to 3,409 per jump; 3,644 mirror textures were recycled in the timed campaign.
Those are eliminated-work measurements, not a latency win. Minimum contiguous VA
was 1,121 MiB in the mirror-selection run. Six images also match the unchanged
control byte-for-byte; full detail and visual acceptance remain intact. Buffer
reuse reduces full-buffer discards from 64.85 to 12.58 per jump (81%), but does
not establish a latency improvement. Its six saved images also match exactly;
the native D3D test passes append, wrap, fallback, depth and in-flight lifetimes.

A rejected adjacent-rigid batching experiment (`m38-standard-rigid-batches`)
saved only 0.48 draws per jump out of about 2,639. Its six images remained exact,
but desktop mean / p95 was 199.58 / 842.89 ms; the extra batching code was removed.

A viewport-sized scene experiment (`m38-standard-view-sized-scene`) reduced
attachments and improved minimum contiguous VA to 1,208 MiB, but desktop mean /
p95 was 210.00 / 760.00 ms. Three saved views differed from the control, including
a maximum channel delta of 67. The off-screen margin removal was reverted.

The existing completion probe is diagnostic only: GPU EVENT queries reported
completion within fractions of a millisecond while a subsequent one-pixel readback
waited hundreds of milliseconds. Do not infer pure GPU phase cost from those queries
on this VM. A half-coverage diagnostic reduced desktop mean from 164.12 to 98.42 ms
on the preceding texture-reuse candidate, implicating substantial pixel work; it
is not a quality change, acceptance result or usable build. VM system RAM had ample
headroom; the adapter reports 2 GiB, not a measured residency budget.

The current complete workload (`m38-append-workload`) passed idle animation,
scrolling, local edits, tactical overlays, native composition/fallback and 120
independent opportunities with 32 working units. Desktop mean (p95) ms:
idle 24.41 (34.71), scrolling 104.60 (166.67), edit 24.94 (50.55), unit frames
49.04 (68.19). The clean evaluation control (`m38-clean-control-workload`) measured
22.61 (33.57), 97.82 (165.73), 27.03 (50.02), and 50.91 (83.63), respectively.
These mixed results do not establish a broad speedup. `m38-append-recovery`
passes 16 atomic identity cases, four cancellation/config-off/reset/recreation
cases, fog freezing/reveal and tactical controls. All three current receipts
confirm unchanged source inputs. All 23 affected scene, worker, cache and native
D3D depth/parameter tests pass. No injected sources changed in this segment.

Huge capacity remains supported separately: the earlier profiling-enabled run
prepared 12,800 tiles / 520 regions, made zero combined compiler calls, uploaded
3.71 GB after eviction, and retained 903 MiB contiguous VA. Its timings must not
be compared directly with these unprofiled Standard runs. Shared sources occupied
about 3 MiB; Standard retained geometry occupied 455.6 MiB. Keep current budgets;
raising prior expanded-geometry caps consumed address space without latency benefit.

**Next unfinished responsibility:** resolve the live map-admission loss above,
prove the actual native sequence stays resident, then reduce GPU pixel/pass work and establish
reliable coherent-display latency, rather than moving waits between endpoints.
Then finish prompt native cutover for every camera trigger and remaining
edit/lifecycle/live-input evidence. **M3.8 and the Standard <33 ms p95 target remain
unpassed.** Fixture timing excludes Civ III input, capture and native overlays;
manual pans can defer, while selection/action/programmatic centering still preserves
its exact native path. Do not advertise the evaluation as faster or claim performance acceptance
based on shorter request handling alone.

## M3.7 accepted handoff (preserved)

M3.6 is committed as `c7ac7699`. Current isolated candidate:
`native/build/m37-final/C3XRenderer.dll`, SHA-256
`fc1802a8c2ccdb5402f0b046fea10e9e7b9a8631db2a246083efd3369fcdbbb7`.
The source closure matches the `/W4 /WX` build. The accepted water DLL remains
staged; this candidate has not been staged, installed or run inside Civ III.

Completed capabilities:

- Failed request copies cancel their worker ticket; allocation/import/poll failure
  cannot leave an incomplete comparison snapshot or a permanently pending camera.
  Client/adapter creation is atomic. Recoverable adoption failure returns the
  intended native camera for fresh exact rendering, granting no ready coverage.
- Cancellation, projection/lifetime retirement and viewer changes reject stale
  intent. Pending and ready-but-unvalidated views cannot commit after reset;
  old tickets cannot revive after pack/definition reload or worker recreation.
  Unfinished route capture also retires with its destination or global drain.
- Config-off settles an eligible queued destination before native image access
  or unload, including calls before Animator. Scene unload discards intent.
  The original Animator, selection centering and native camera arguments remain
  unchanged. No new native patches, state fields or CSV entries were needed.
- Reset, reload and config-off share one checked native/display handoff, including
  CPU-source presentation without a map owner. Failure retains hooks, DLL
  references and ownership; native drawing cannot consume stale CPU storage.
  Failed unload marks the renderer unavailable so a new scenario cannot use old
  definitions. Validated definition-load failure uses the same cleanup.

Verification: **321 tests passed / two existing skips**, followed by **13 passing
final affected tests** after the last retirement cleanup; approved injected
compile/injection smoke and final candidate build pass. The real JGL/GPU receipt
at `native/build/m37-identity-final/` passes four recovery/recreation cases and all
16 independent cold pixel/coverage comparisons, plus fog and tactical controls.
It preserves the displayed pixels across reset and rejects stale COMMIT. Eleven
native requests exercised 47 pending polls and ten completion hints. The initial
recovery fixture mistakenly restarted presentation while inspecting reset's GDI
handoff; its corrected oracle passes. The failed receipt remains diagnostic only.

The matched 1120×1192 workload at `native/build/m37-workload/` passes: 384 timed
CPU/GPU native requests, eight native units, 120 independent frames with 32 workers,
tactical overlays, shoreline waves, water motion and reflections on. Profiling is
off in both comparison arms. Receipts prove unchanged inputs, matching runtime
sources and complete traces.

| Complete workload, mean / p95 ms | M3.6 control | M3.7 |
| --- | ---: | ---: |
| Native GPU stationary | 12.70 / 18.36 | 13.42 / 22.25 |
| Native GPU scrolling | 93.80 / 167.15 | 98.48 / 167.77 |
| Native GPU local edit | 17.92 / 58.01 | 17.30 / 63.39 |
| Independent 32-unit visual request | 41.37 / 62.30 | 39.70 / 61.46 |

No speedup or broad latency improvement is claimed: stationary/scroll means are
5.7%/5.0% higher; local edit/visual means are 3.5%/4.0% lower. Scroll maximum is
217.22 ms. Enqueue mean/p95/max is 1.219/3.277/13.249 ms; per-request pending-poll
maximum p95/max is 0.061/0.233 ms; ready adoption plus validation mean/p95/max is
1.304/2.171/7.302 ms. First cold adoption is separately **123.093 ms**. Cold costs,
caller tails and native input-to-display latency remain material acceptance work.

All 120 visual intervals retain zero authoritative adoption, static-world
build/upload/draw, static-scene draw/readback, wave build/upload, reflection build,
water upload and caster collection; material selections are reused. Scroll work
is unchanged: 3,410 built / 56,296 reused / 5,094,936 uploaded bytes. The control
bitmap is byte-exact with M3.6. Journal peak remains 3,010,856 bytes. Minimum sampled
largest free address-space region is **647.19 MiB** (M3.6: 788.44 MiB), still above
the 512 MiB floor; no memory improvement is claimed.

Limits: tests prove controlled GPU reset/recreation and recoverable faults. A
physical device loss that makes GPU-only native surfaces irretrievable remains
fail-closed; successful lossless restoration of those surfaces is not claimed.
Idle manual pans are asynchronous; action/UI ownership, centering and projection
barriers may still wait. The fixture services completion messages, while the live
bridge retries on native Animator opportunities. Native capture/traversal and
live input latency are outside these timings.

**Next unfinished responsibility: M3.8 arbitrary-destination navigation**, with
explicit whole-world input/preparation coverage, coherent cutover for every camera
trigger and complete latency evidence as specified below. Queue responsiveness alone cannot establish
fast navigation. No live-game acceptance, nonblocking minimap/zoom/centering or
sustained frame-budget claim has been made.

## M2 automated acceptance checkpoint (preserved)

Candidate: `native/build/m26-unit-sharing-final/C3XRenderer.dll`, SHA-256
`b7d2fad4213c1b75e51576a2687521f050784469f5494235953e013d4d319bf9`.
This checkpoint was committed as `5d96597f` and pushed before M3 work.

Completed capabilities:

- Independent frames retain material-pass selections and static shadow-caster
  descriptors. Time changes no longer rediscover static submission state.
  Borrowed mesh records retire with their owning scene; scene/view/light changes
  invalidate the retained recipe.
- Direct unit revisions are collected before map execution. Missing current poses
  enter the existing bounded workers ahead of predictions; GPU execution preserves
  native order. Identical unit body contributions share the existing 192 MiB GPU
  content owner. Each occurrence still applies its own shadow, placement and
  underlay blend; hidden/retired selections cannot authorize a cached pose.
- Retained composition reuses owned result allocations and removes a duplicate
  full-map copy. Opaque city bodies remain in static color/depth. Conservative dry
  water rejection is enabled by default; mirror cells without water receivers are
  omitted. User-facing effects and diagnostic controls remain available.
- Resident reflections and shared reflected tree meshes remain enabled. The timer
  rearms after synchronous work, retaining at least 10 ms for the message pump
  after an overrun. Reassess that guard when M3 makes callbacks nonblocking.

Matched complete native workload, 1120×1192 dense coast, eight units, all effects
on, **384 requests per DLL**, 64 per route/workload. Prior `m26-final` versus this
candidate, GPU request mean / p95 (ms):

| Workload | Prior | Candidate |
| --- | --- | --- |
| Stationary native demand | 11.81 / 15.31 | 11.58 / 16.43 |
| Scrolling | 93.62 / 143.27 | 91.98 / 166.19 |
| Local edit | 16.63 / 65.28 | 21.23 / 71.49 |
| Independent 32-unit animation, 120 frames | 48.08 / 68.95 | 40.54 / 62.08 |

The **15.7% lower mean animation request** is the demonstrated gain. Desktop
completion improves 55.12 → 50.57 ms. Scrolling improvement is inconclusive;
local-edit mean and navigation tails regress in this pair. Do not describe this
as a general navigation or frame-budget win. Earlier profiled 65–72 ms visual and
177 ms scrolling numbers included expensive address-space diagnostics; they are
not comparable to these unprofiled production timings.

All 120 candidate animation intervals prove zero static-world builds/uploads,
static-scene draws/readbacks, mirror builds, wave builds/uploads and water uploads;
all reuse material selections without static-caster recollection. Of 3,840 unit
occurrences, 2,445 reuse a body contribution (64% fewer body raster passes).
Native pixel/erase-bounds, 555/565/full-color ownership, CPU fallback, timer,
config-off drain/reenable and tactical cancellation checks pass. The coastal
control image is pixel-exact to the prior DLL. Sampled contiguous address space
stays above **749.6 MiB**; this is sampled headroom, not a transient allocation bound.
Receipts and interval proofs: `native/build/m27-complete-{control,final}/`.

The same all-visible workload measures **31.36 / 37.58 / 40.54 ms mean** at
**1 / 8 / 32 units**, each over 120 frames (one selected idle unit, remaining units
working). P95 is 41.45 / 51.43 / 62.08 ms. The added 31 units cost 9.18 ms in this
repeated-worker fixture; differing unit assets/actions can have different scaling.
All intervals preserve the zero-static-work proof. Additional receipts:
`native/build/m27-scaling1/` and `native/build/m27-scaling8/`.

The separate 120-frame eight-unit mixed/visibility run passes: one selected idle,
three work loops, two frozen idle and two native-directed actions; 33.41 ms mean,
46.77 ms p95. Fog freezes map animation, hidden units emit no pixels, reveal resumes
motion, and route/selection/grid changes preserve static content.
Evidence: `native/build/m27-mixed8/`; this different workload is not a scaling
comparison against the 32-worker case.

Eight new destinations still cost **0.38–1.74 s**, with no fallback. Initial scene
preparation costs 4.61 s. Four existing world workers prepare current content;
new terrain and foreground joins remain material costs. A separate serialized
probe places 143–433 ms at even a one-pixel CPU readback after sub-millisecond
scene/finish completion queries. On this VM those queries do not establish true
physical GPU completion; the observed delay is at the readback synchronization
boundary, not proof of cheap shader execution. The earlier 4.30 s wait did not
recur, but is not proved eliminated. Evidence: `native/build/m27-distant-final/`
and `native/build/m27-distant-attribution/`. These completed-render/capture
measurements include CPU delivery, not native presentation or physical scanout.

**M2 boundary:** M3 owns coherent state publication and nonblocking camera/content
updates, including cold, evicted and invalidated views; its current substep is
recorded above. Keep the current scene valid while useful work completes; never present
mismatched camera/visibility/picking. M4 owns a demonstrated cadence budget.
Fast cold jumps, sustained 30/60 FPS and strategic live-game acceptance are not
claimed by M2's independent-animation proof. No new native hook, ABI or patch-table
entry is required.

Verification: the full dependency-selected suite passes **313 tests / two skips**.
The existing native composition, lifecycle and configuration controls pass in all
five completed 120-frame receipts (480 current-candidate frames plus 120 historical
control frames), including frozen/native action cases. Another 32 focused tests
pass, including 126 exact native retained-composition oracles and 120 clock frames. Separate native water witnesses cover beach and rocky coasts,
daylight/moonlight, 48-frame playback, fog/reveal, reset, reflection toggles,
zoom return, wrapped reduced-zoom replay and authoritative local edits. Repeat,
reset, reflection toggle, replay and edit comparisons are exact; the independent
still-water control differs at ten pixels by one channel level, within its existing
rounding contract. Disabling waves leaves water motion active; disabling both
proves a still image. Evidence: `lab/out/integration/m27-{waves-final,rocky-final,
river-lifecycle,wrapped-scroll,local-edits}/`. No renderer appearance, action/visibility
contract, injected source or executable address changed. Strategic live-game
acceptance and staging of this optimization candidate remain pending.

## Current implementation and gaps

| Original responsibility | Implemented foundation | Remaining responsibility |
| --- | --- | --- |
| Persistent world/instances | Camera-independent content, shared assets/forest meshes, revisioned units, GPU-ready ground/terrain/objects, immutable mesh ranges/materials, bounded residency | Nonblocking cold/invalidated views; broader sharing where measured |
| Local validity | Durable copied change publication, separate appearance/visibility revisions, immutable map inputs, unit revision/despawn/hidden proofs | Atomic completed-view eligibility through nonblocking native presentation |
| Spatial selection | World pass index and native/wrapped occurrences feed replacement static submissions directly | Broader spatial sharing where measured |
| Compatible passes | Compatible static layers, shared material bindings, batched occurrence parameters/uploads, forest instancing, collected poses/shared body contributions, exact native composition | Additional compatible sharing guided by measured cost |
| GPU reuse/output | Resident map color/depth, direct eligible unit geometry, incremental finishing and native composition | Reduce dynamic conversion, replay and full-map work |
| Async integration | Bounded GPU-ready ground/terrain/object preparation with shared worker capacity, selected-work urgency and independent visual delivery | Coherent general nonblocking camera/content publication |

Independent frames use the existing HWND presenter without native redraw requests.
Civ III's original gameplay timer is unchanged. Map animation still enters the
existing render orchestration. Eligible units use collected direct scene execution with shared body contributions;
oversized native canvases retain bounded GPU pose compatibility. Further GPU work,
finishing and publication costs remain for the explicit M3/M4 budgets.

## 1. Complete world → selected GPU submissions

Connect persistent content, shared mesh/material bindings, compact instances,
spatially selected occurrences and explicit compatible pass inputs. Extend current
owners and replace migrated submission paths together. Include worker ownership:
prepare dependency-ready content and selected pass inputs concurrently, separate
current-frame work from speculation, and replace redundant queues where needed.
This is part of milestone 1, with animation scheduling extended in milestone 2;
it is not another milestone. Do not deliver only a command/job abstraction or
instancing that leaves submission costs unchanged.

Start with the active dense-scrolling map path, including terrain, vegetation and
representative repeated city/infrastructure content. Choose actual conversions
from code and bounded attribution; CPU-transformed vertices already cached on warm
requests are not automatically the dominant cost. Preserve unique/deformed meshes
and exact material/depth ordering where sharing or sorting is incompatible.

**Done:** pans reuse unchanged content; local edits affect their real dependency
neighborhood; selected inputs drive the replacement production submissions.
Demonstrate whole-request effects, eliminated construction/binding work and bounded
memory on stationary animation, dense scrolling and local changes.

## 2. Make independent animation a direct scene operation

Sample eligible animation state, collect required resource/unit poses, prepare
missing content together and execute selected dynamic/shadow/finishing passes over
valid static color/depth. Feed results into the existing native compositor. Until
the explicit zoom-owned tactical-overlay cutover below, keep native overlays;
keep units above all map geometry per the final user decision in 2.3, preserve
native unit/UI ordering, and keep static UI reusable.

Shared pose buffers, GPU deformation/skinning and grouped targets are candidates
where they remove measured work. Do not force all units into one surface or alter
source playback just to simplify batching.

**Done:** idle animation does not rebuild world content or rediscover static
submission state; animated-unit scaling avoids repeated independent setup/finish
work where compatible. Verify frozen units, authored work loops and native actions.

The requested substeps are: **2.1 immutable dynamic inputs** (implemented; see
[contract](dynamic_scene_input_contract.md)); **2.2 direct unit poses** (implemented; see
[contract](direct_unit_scene_contract.md)); **2.3 shadows,
occlusion and composition**; **2.4 map effects**, shoreline waves first; **2.5 tactical
overlays**; **2.6 scheduling/reuse**; **2.7 acceptance**. The shoreline portion of
2.4 now has resident-scene lifecycle and connected native-frame coverage.

**2.3 final scope (user reversal, 2026-09-19):** units always draw above map
geometry, whether on, behind or in front of a mountain, forest or building. The
previous terrain-occlusion requirement is withdrawn, including the intermediate
same-tile exception. Keep existing pose-local shadows, native unit/UI order,
actions, visibility and controls. Remove the unused map-depth coupling and verify
exact body/native composition across overlap, movement, animation, zoom and fog.
This does not add surrounding-world shadow receivers or arbitrary-geometry unit
shadow casting. No deferred terrain-occlusion task is implied by this policy.

## 3. Complete state publication and nonblocking camera updates

Keep durable world/lifecycle updates separate from replaceable camera requests.
Publish copied authoritative changes and exact view eligibility; refine mutation
hooks only where needed. Prioritize current missing content, then nearby reusable
meshes/instances/pass inputs, then selected future views/animation pixels. Preserve
useful preparation across supersession without evicting the current working set.

| Substep | Responsibility | Status |
| --- | --- | --- |
| 3.1 | Durable copied/versioned authoritative changes and lifecycle | Implemented; committed and pushed |
| 3.2 | Replaceable exact camera requests independent of durable changes | Implemented; committed and pushed |
| 3.3 | Current-demand priority and bounded useful preparation | Implemented, measured, committed and pushed |
| 3.4 | Atomic pixels, coverage, transform, occurrences and complete identity | Implemented; identity/pixel acceptance and handoff above |
| 3.5 | Nonblocking native polling and honest pending-coverage policy | DLL/JGL transaction boundary implemented; measurements above |
| 3.6 | Coherent overlays, interactions and picking | Guarded live bridge implemented; automated acceptance passed |
| 3.7 | Cancellation, GPU reset, reload, configuration-off and failure recovery | Implemented; automated retirement/recreation and fail-closed barriers pass; physical device-loss restoration is not claimed |
| 3.8 | Arbitrary-destination readiness, coherent native navigation, edit/lifecycle and complete latency proof | In progress: whole-world input, preparation and bounded compiled backing work; paging, native cutover and full latency acceptance remain |

**Done:** initialized, unchanged supported-world navigation selects already
prepared scene content without foreground world compilation. Pixels, camera,
visibility, overlays and picking advance coherently. Cold initialization, edits,
eviction and recovery have separately measured, bounded handling. A responsive
submission queue or a fast revisit does not prove arbitrary-destination readiness.

### 3.8 arbitrary-destination navigation

**Latency acceptance:** the user accepts **<33 ms p95 on nominal 100 × 100
Standard maps (5,000 actual staggered tiles)** as the performance win, with
viewport, object density and hardware declared. Nominal 160 × 160 Huge maps
(12,800 actual tiles) remain the supported-size/capacity target; disclose their
latency separately without requiring the same 33 ms result. A 332 × 332 /
55,112-tile case remains a limit
probe, not a release requirement; do not increase residency budgets to pass it.
Whole-world copied input alone measures about 8 MiB at 12,800 tiles and 35 MiB at
55,112 tiles. Expanded geometry, worker allocations and driver address space
must be measured separately; input capacity is not render-ready capacity.

Requested September 20: moving anywhere on the map should feel immediate. Treat
this as an explicit architectural and latency requirement, not an incidental
benefit of nearby prefetch. Extend the current world/publication/content owners;
do not introduce another render world or depend on cached destination screenshots.
This applies equally to scrolling, minimap jumps, zoom/reframing, newly selected
unit centering, action following and other native programmatic camera moves.
Minimap clicks are one diagnostic route, not the scope of the requirement.
Preserving vanilla centering means preserving destination, clamping and gameplay
ordering; it does not exempt automatic camera moves from the latency objective.

Current evidence needs three qualifications:

- The 0.38–1.74 s distant-destination and 4.61 s initialization results above are
  M2 completed-render/CPU-delivery evidence, including synchronization/readback.
  They are not current native GPU minimap latency or proof that compilation alone
  consumed those intervals. M3.7 scrolling is 98.48 ms mean / 167.77 ms p95; live
  minimap latency has not been measured.
- Full-map topology/visibility observation is not full-map appearance publication.
  `capture_custom_renderer_world_topology` copies terrain/river/effect topology
  and detects visibility changes. Full object/city/route records enter through
  captured occurrences and their appearance halo. `CapturedScene` only marks full
  render/prefetch observations authoritative; a topology-only record cannot
  manufacture missing object state or certify remote appearance as current.
- The decompiled `Navigator_Data::handle_left_click` calls native
  `bring_tile_into_view(..., reason=0, false, false)`. The current centering guard
  keeps it exact. Deferred idle pans still adopt at native Animator opportunities;
  completion messages alone do not bypass that boundary. Waiting for the normal
  66 ms opportunity can itself exceed a 33 ms latency target.

| Responsibility within 3.8 | Required result |
| --- | --- |
| Baseline and attribution | Trace actual minimap, edge/drag scrolling, zoom and native selection/action centering. Separate input/capture, submission, preparation, GPU dependencies, ready wait, adoption, native composition and first coherent display. Re-run distant destinations through the current native GPU path. |
| Authoritative world coverage | Publish complete renderer-owned appearance for the renderable world independently of visiting each camera destination. Audit offscreen mutations, removals and viewer changes; retain revisions and readiness coverage. Use bounded caller-thread capture and existing change publication; workers never dereference game objects. Preserve fog, unseen coverage and frozen explored animation. |
| Render-ready representation | Combine shared source meshes/materials and compact rigid instances with compiled regional terrain/deformed/connectivity content. Prepare from world changes and initialization rather than camera misses; retain useful compiled data across GPU eviction where budgets permit. Measure CPU/GPU/disk backing tradeoffs instead of keeping all current expanded meshes resident or increasing cache caps. |
| Coherent native cutover | Give every camera trigger, including newly selected units and action following, a prompt, safe caller-thread completion boundary that advances pixels, native camera, overlays and picking together. Preserve vanilla destination/clamping and selection/action centering behavior, reentrancy guards and gameplay cadence. Do not accelerate gameplay or invoke extra action updates to consume a ready visual frame. |
| Interturn ambient presentation | Implemented and automated proof passes: owned cadence plus composition presentation on the same HWND, using copied authorized scene/visibility inputs. Actual desktop pixels change with the window thread blocked; native/GDI handoff, modal policy, reset, fog, action timing and camera adoption tests pass. Live-game acceptance remains pending. |
| Acceptance under load | Run the arbitrary-jump matrix in the validation guide, plus edits, cancellation, fog/viewer changes, reset, config-off and memory pressure. Prove prepared-world construction invariants, distinguish streaming misses and account for preparation coverage/time. |

For a destination whose complete dependencies are prepared and GPU resident,
ordinary camera changes must cause **zero static-world compilation, zero static
geometry upload/allocation, zero foreground preparation joins and zero map CPU
readback**. Normal camera/instance/animation parameter uploads and ordered GPU
execution remain necessary. The goal is no CPU wait for content construction,
readback or avoidable GPU fences, not a literal absence of GPU synchronization.
CPU-prepared/GPU-evicted destinations form a separate class: upload/paging costs
must be explicit, and geometry recompilation must not be hidden as selection.

Use 100 deterministic distributed destinations (at least 50 distinct where map
extent permits), including first visits and distant dense/coastal regions after
normal initialization. The initialization/preparation policy must be independent
of the test route. Report first-interactive time, whole-world readiness time,
prepared/resident coverage and resource peaks; a route-specific warmup or warm
subset cannot certify navigation anywhere. Preserve full detail and all water
effects, current budgets and at least 512 MiB sampled contiguous VA headroom.
No LOD, placeholder frames or hidden-content exposure is authorized by this goal.

**Latency objective:** on Standard maps under the declared viewport/density/hardware
envelope, unchanged initialized-world navigation targets **<33 ms p95 from input
or native camera decision to first correct coherent displayed frame**. Huge-map
capacity and latency are reported separately; <16.7 ms is the later
60 FPS objective. Report each camera-trigger class separately; rapid pans must
not conceal slow selected-unit jumps. Report maximums and every
>100 ms stall as well, so p95 cannot conceal a few slow regions. Both spatial
readiness and actual rendering/cutover must succeed. Prepared-resident, ordinary
post-load first visits, eviction and edits retain separate distributions; misses
in normal post-load navigation cannot be dropped to make the overall result pass.

M3.8 owns missing world/preparation and native-cutover mechanisms required by this
contract. M4 owns measured rendering/cadence costs remaining once that contract is
working. A measured latency miss stays an explicit unpassed objective with a named
cost/owner; neither milestone may call navigation instantaneous because enqueue is
cheap or the UI continues processing messages.

## 4. Raise cadence against a measured frame budget

Reduce remaining warm-view and independent-frame costs using 3.8's attribution:
spatial/pass selection, compatible submission, shadows/reflections, finishing,
native composition, GPU queueing and presentation. High tile reuse alone does
not identify which stage dominates; measure whole requests and critical-path
waits. More workers cannot remove costs that are already submission, rendering
or synchronization. Consider parallel D3D recording or less UI-thread dependence
only when measured; stay on D3D11 unless evidence justifies a change.

**Done:** demonstrate the selected cadence and the arbitrary-navigation latency
objective on representative live workloads, including navigation, animation and
UI transitions. Cold/evicted/edited destinations remain visible in the results.
A 33 ms scheduling target is not 30 FPS; 60 FPS is a later 16.7 ms objective, not a current
promise. Whole-world readiness is explicit M3.8 work, not deferred implicitly to
frame-loop tuning here.

## LORE testing and migration milestones

Requested September 21, 2026. These milestones refine M4; they do not reset M1-M3,
replace M3.8 camera/readiness acceptance, or create a second renderer project.
LORE-like means immutable draw descriptions, compact explicit frame commands,
cheap parameter payloads, redundant-state filtering and useful parallel command
preparation. D3D11 deferred contexts are optional. A new shading language,
64-bit helper process, API migration, tessellation/LOD and reduced visual quality
are outside this sequence. Natural/constructed wonders and Districts stay deferred.

Keep current assets, shading, native coordinates, pass/native order, visibility,
authored animation and all water effects. Extend existing content, draw, worker
and composition owners. Replace each migrated production route; temporary
comparison controls must not become a permanently maintained second engine.
Lab retains visual ownership; Integration owns submission, native composition,
resource lifetime and displayed-frame evidence.

| Milestone | Deliverable | Exit decision | Status |
| --- | --- | --- | --- |
| M4.0 - Establish causes and controls | Complete ten-minute renderer-input capture/replay, driver capabilities and complete cost/memory attribution | Qualify the recording, establish a matched baseline, then choose the first architectural cost to remove | In progress: timeline/capacity audit and selected-frame reconstruction pass; complete input capture pending |
| M4.1 - Prove one LORE-style pass | Immutable draw descriptions, compact commands and filtered serial execution for one production pass | Exact output and repeatable complete-workload benefit justify migration | Planned |
| M4.2 - Switch scene submission | Adopt the proven path across applicable production scene passes | Migrated paths use the executor by default; obsolete duplicate submission is removed | Planned |
| M4.3 - Parallelize useful command work | Bounded packet jobs; separately test deferred-context recording | Keep only concurrency that improves complete workloads on the target driver | Planned |
| M4.4 - Bound native composition and memory | Efficient ordered composition, explicit image lifetimes and joint resource accounting | Representative native churn remains correct and stable within process headroom | Planned |
| M4.5 - Accept the integrated candidate | Full correctness, performance, pressure and sustained replay verification | One production candidate passes the declared automated envelope | Planned |
| M4.6 - Verify live responsiveness | Batched live gameplay checkpoint with the exact tested candidate | Sustained animation and coherent navigation meet their separate objectives | Planned |

M4.0 selects priorities: M4.4 may precede or accompany scene migration if native
composition or memory dominates. M4.1 precedes broad M4.2 conversion; M4.3 requires
a working explicit command boundary, not completion of every scene category.
M4.5 joins the resulting paths, and M4.6 follows automated acceptance. Independent
work can continue while a live checkpoint is pending. Milestone numbers describe
deliverables, not permission to expand scope or a requirement to spawn agents.

### M4.0 - Establish causes and controls

Pin source, DLL, inputs, budgets, driver/VM configuration and reset state for the
control. Reuse valid existing receipts; run only missing or invalidated cases.
The newer live capture `20260921-042106-4a9c31` reports about 120 MiB available VA
and device removal after the earlier history-cap fix. Treat both as unresolved;
it does not prove memory pressure caused removal. See the
[live findings](live_usage_findings_20260920.md#latest-live-limit-and-reproducible-composition-inputs).

Query `DriverCommandLists` and `DriverConcurrentCreates`; record unsupported or
emulated behavior. Separate scene selection/parameter preparation, D3D submission,
GPU waits where measurable, native composition, presentation and camera adoption.
Count actual bindings, uploads, draws, copies, allocations and bytes, including
peak in-flight resources. Do not add overlapping CPU/worker spans or trust the
VM's uncalibrated timestamp queries. Diagnostic pass omissions identify causes;
they do not qualify as an equivalent-image candidate.

The user has superseded the symptom-by-symptom validation approach. Complete
[recorded renderer workloads](recorded_renderer_workload.md) as the M4.0
prerequisite, using existing owners and production rendering paths:

| Work package | Required exit | Current status |
| --- | --- | --- |
| Audit coverage and inspect time | Explicit missing input families, byte/time breakdown, every recorded composition display indexed; selected frame/second ranges reconstruct through strict replay | Implemented and tested on the saved live prefix; scope remains composition only |
| Capture and replay consumed inputs | Versioned initialization/assets, scene/world, action/lifecycle, camera, native-adapter and ambient/presentation events drive the production owners; short positive and missing-input controls | Native-owner protocol passes 12,930 calls and two 426-frame replays; short controls reject missing/altered inputs. Standard whole-world paging and 100 destination images also repeat exactly. Broader live coverage remains unqualified |
| Sustain 600 seconds | Bounded asynchronous segmented storage, exact payload reuse, valid crash prefix, complete real-time ten-minute native workload and measured capture overhead | Real-time 600-second full-resolution capture closes cleanly: 680 MB, 92,130 calls, 10,200 presentations, 17 MiB peak queue and a declared 512 MiB VA reservation. Two exact-DLL replays match all 10,200 display fingerprints. Capture-off also passes; one matched comparison excludes the first 180 seconds in both arms for known compiler interference. Calibrated overhead and live-pressure qualification remain |
| Establish forensic/performance controls | Deterministic logical-time frame oracles plus separately measured execution; seek agrees with full-prefix replay; pressure remains explicitly modeled | Forensic repeatability/seek and separate native-service measurements pass, including 1 GiB VA reservation. Earlier native consumption points remain constrained by recorded CPU inputs; live request-to-display equivalence is unfinished |
| Freeze a representative corpus and choose architecture | One qualified user session, repeated baseline and per-cause latency/memory attribution choose the next M4 change | Pending automated qualification; do not request another manual capture yet |

The separate development [input journal](../native/input_recording/README.md)
now re-enters production rendering from copied values, pins observed assets and
DLL identity, and checks independently captured output hashes. The short control
includes 16/32-bit CPU unit delivery and both CPU camera APIs; it has passed twice
with matching outputs. The full 2240×1260 fixture exposed repeated native-screen
payload overflow; changed-block storage produced a closed 623 MB/107-second
capture with about 16 MiB peak queue occupancy. Full replay exposed missing
presenter caller ownership and asynchronous rejection handling, then a production
translated-geometry defect: camera movement changed off-screen selection but reused
the preceding contributor set. The candidate now validates that selection before
reuse. The repaired native fixture and two full replays pass all recorded output
witnesses, with 13,128 calls and 519 identical composed presentation sources.
Frame/time selection and explicit truncated-prefix recovery also pass. See the
[coverage ledger](../native/input_recording/coverage.md). These are diagnostic
results, not complete gameplay acceptance. Whole-world callback replay now
passes too: 5,000 tiles, 198 captured page returns, 100 repeated destination images,
and six exact cold pixel comparisons. Exported visual clocks and QPC/UTC
correlation are captured; asset paths survive temporary Windows drive mappings.
A real ten-minute full-resolution native capture closes cleanly with no open
calls; both exact-DLL replays match all 10,200 display fingerprints. In the paired
420-second comparison, capture-on/off idle median is 21.59/21.88 ms and
camera-change median is 89.81/88.99 ms; other camera-phase iterations retain
roughly 260–270 ms p95 tails. This single fixture pair is not calibrated game FPS
or recorder-overhead acceptance. The CPU capture registry now shares one identity
and scratch budget across serial caller threads and rejects unsupported aliases
or concurrent capture explicitly. Eight semantic mutations reject cleanly with
their expected diagnostics. The combined native/window fixture passes all native
pixel checks and both process exits: 119 window samples align within 0.044 ms,
and two replays match all 970 display fingerprints. Its 282.2 MiB minimum available
VA remains above the earlier live 120 MiB envelope.

The input boundary now includes the actual native image/composition/navigation
owners and root reset/configuration transactions. Copied external values replace
JGL access only inside replay; production still uses the original dependencies.
The fresh 2240×1260 fixture passes 12,930 calls and two exact 426-frame replays,
including CPU-ownership output witnesses. Recorder identities survive both destroy
notifications; replay dependency data retires with the image. Safety tests reject
missing inputs before proxy dereference and preserve native leases if recording
fails. No injected source or patch-table change is needed.

Separate unpaced service measurements run actual production work, with explicit
VA capacity reservations and no display-fingerprint/export overhead. Camera
publication retains recorded consumption points; this cannot measure how a changed
native caller would react to earlier completion. The one-command capture launcher
collects/pins inputs, sampled window evidence, memory and logs and stops collectors
automatically. Its parser, shared-path quoting and requested-stop test pass without
a game launch. Capture admission remains closed pending the automated overhead
and pressure gate. After that, compare one strategic live recording with its window
evidence before accepting architecture changes as improvements to real gameplay.

[Visible Before/After playback](realtime_replay_comparison.md) now runs the same
input timeline with independent production ambient cadence. A 128-second same-DLL
control passes with 1 GiB VA reserved and visible water/unit animation; it is a
workflow check, not a performance improvement. Exact forensic playback remains
separate. Native consumption points are still recorded, so this extension does
not close the live calibration or capture-admission gate above.

The final world-page check rejects changed scope/topology and responses issued
before a newer publication, preserving owned metadata and retry position.
Thirteen portable controls pass. A fresh Standard capture passes six cold pixel
oracles and two matching 100-frame replays (1,170 calls, 201 page returns).
All three altered world range/scope/topology controls reject cleanly at the
expected production acceptance check.
The recorder/replay tooling supports the strategic capture checkpoint; the
broader full-game fidelity/performance acceptance remains contingent on that evidence.

The older v2/v3 writer remains limited to 512 MiB/180 seconds and snapshots
external map/unit pixels. Its 80.788-second prefix uses 510.745 MiB; pixel-bearing
records consume about 93.5% of storage. The linear ten-minute projection is
3.70 GiB, including startup delays; it is not a guaranteed recording size. Simply
increasing limits would preserve missing inputs and synchronous readback/file
work. `--require-input-replay` correctly rejects this journal. Its production native
lifetime decisions replay separately; it cannot regenerate adapter/ambient
producer inputs. The new input recorder is DLL-owned and does regenerate the
tested producer paths; the installed injected bridge is unchanged.
No game was launched for this work.

Current evidence: `native/build/live-captures/20260921-130328-65f6bd/`
contains `input-replay-readiness.json`, `display-timeline.jsonl` and
`replay-loss-transition.json`. The latter reconstructs every composition display
in second 23–24 while strictly checking the complete prefix. Fullscreen native
fixtures remain independent regression tests, not substitutes for complete
captured input coverage. Treat existing manual evidence as reusable; no new
manual recording is needed to implement the next input protocol.

Exit with a measured bottleneck, the first production pass to change, and matched
controls capable of accepting or rejecting that change. No broad packet rewrite
is justified solely by LORE's batch-count headline. If evidence instead identifies
pixel work or native composition as dominant, prioritize that responsibility.

### M4.1 - Prove one LORE-style pass

Choose one expensive pass from M4.0, such as dense city submission or reflection
geometry. Retain mesh/material/pipeline descriptions with their existing content
owners. Build compact occurrence/parameter commands in reusable bounded storage.
Commands specify required state; the serial executor filters redundant bindings
without changing draw order. Track resource hazards, pass transitions and state
changes outside the executor explicitly. Packet storage expires with its frame
lease; it must not become another world cache or retain native game pointers.

Keep identical geometry, shader math, effects and raster ordering. Check exact
pixels against the existing same-input control, including reflection/shadow
consumers, wrapped occurrences, invalidation and cancellation. Record command
build cost, state calls actually eliminated, allocations and total frame cost.

For broad migration, require the complete-workload improvement gate below, not
just faster submission. Permit one materially different follow-up if the first
prototype is inconclusive; after two unsuccessful designs, record the result and
redirect to the measured bottleneck rather than expanding the abstraction.

### M4.2 - Switch scene submission

Convert applicable terrain/natural, city/infrastructure, resource, shadow,
reflection and water submission incrementally through the same owners. Direct
unit passes must use the same explicit-state rules where compatible; document
specialized paths and their measured costs rather than forcing identical commands
onto unrelated work. Preserve cutout/transparent/decal ordering and native unit
composition. Batch constants and repeated state by actual compatibility, without
global material sorting that changes pixels.

Validate each affected category and shared consumers through `renderer.py`.
Keep a coverage table in the implementation receipt naming each migrated pass,
its production caller, tests, measured effect and any remaining specialized path.
Switch a validated pass to the production executor and remove its superseded
submission implementation, retaining independent pixel oracles and a preserved
control DLL. A command wrapper that still performs the old setup is not migration.
Exit with explicit coverage and no unreported old default path or memory increase.

### M4.3 - Parallelize useful command work

First compare serial and job-based packet construction using immutable leased
inputs, stable merge order, bounded storage and the existing worker allowance.
Measure critical-path time including joins and cancellation; more busy cores are
not an exit criterion. Keep small workloads serial when scheduling costs dominate.

Separately compare D3D11 deferred-context recording against the filtered serial
executor. Each context has one thread owner, explicit initial state and valid
parameter/upload lifetimes. Execute lists in the established order on the GPU
owner; preserve reset, resource retirement and native transaction barriers.
Do not queue multiple stale camera frames to manufacture throughput. Capability
flags guide the experiment but do not prove a benefit or rule out emulated trials.

Exit with each concurrency mechanism retained or rejected on complete-frame
evidence. A measured rejection of deferred contexts completes that decision;
the supported LORE-style serial executor remains the production path. No API or
process migration is implied by unfavorable Parallels results.

### M4.4 - Bound native composition and memory

Apply explicit commands and storage reuse to the existing image compositor and
retained dependency graph. Preserve source versions, native return values,
partial writes, overlapping copies, actual CPU-access barriers and animated map
dependencies. Eliminate proven redundant copies/replay work and unused versions.
Native saved images may outlive a frame; frame packet retirement must not discard
their required history or replace an animated dependency with a frozen snapshot.

Use captured composition commands for exact native-image oracles, plus production
retained-animation and admission tests for the behavior the journal cannot replay.
Exercise fullscreen HUD/save/restore churn, repeated map publications, camera
cancellation and recovery. Account jointly for world geometry, decoded assets,
live images, retained outputs, scratch and in-flight work; distinguish shared
allocations from duplicated ownership, GPU bytes from CPU VA, and caps from usage.

Exit with stable memory under repeated churn, no unexpected budget fallback or
device loss, and the existing 512 MiB sampled contiguous-VA floor in the ordinary
Standard workload. Artificial VA-reservation runs are separate pressure/recovery
tests, not proof of ordinary headroom. No automatic cap increases: any capacity
change needs measured whole-process justification and a bounded lifetime policy.
Do not claim the observed device-removal cause is fixed without reproduction or
corresponding live evidence. Remaining 32-bit capacity limits can justify a
separate helper-process proposal; they do not silently expand this milestone.

### M4.5 - Accept the integrated candidate

Run the affected full integration suite once the connected path is ready. Preserve
capture, ownership, invalidation, scrolling, wrapping, zoom, animation, exact native
composition, tactical/fog behavior, unit ordering, reset and config-off tests.
Include genuine foreign CPU-access fallback and blocked-game-thread ambient
delivery. Use existing harnesses, not a second acceptance framework.

Require at least 1,200 independent visual frames and a separate ten-minute
representative churn soak, including repeated native UI transitions, fresh maps,
navigation and retirement. An idle-only loop cannot satisfy the churn requirement.
Memory must settle within declared bounds rather than grow with action count.
Separate cold initialization, first visits, edits and eviction from warm results.
Fail on unexplained lost animation, stale camera/picking, budget collapse or
device failure. Missing coverage stays pending, even if a narrower fixture passes.

Exit with a single source/DLL receipt, pixel comparisons, matched timing results,
memory envelope and rollback candidate. Automated readiness is not staging or
live acceptance; existing visual/staging/install rules remain in force.

### M4.6 - Verify live responsiveness

After applicable staging/install/launch authorization, use one batched checkpoint:
idle ambient animation, scrolling, minimap jumps, selected-unit centering, action
following, zoom, ordinary HUD/dialog transitions, interturn and sustained play.
Prefer existing automation and captured evidence; request manual evidence only
for the remaining strategic checkpoint, once as a checklist. Keep performance
captures free of the diagnostic composition recorder's readbacks and file writes.

Require sustained fresh map animation, stable memory, and no unexplained ownership
collapse or device failure. On Standard 5,000-tile maps, retain M3.8's **<33 ms p95
input/native-camera-decision to first coherent displayed frame**, separately for
each trigger class. Adopt **at least 30 fresh animated map frames/sec with p95
display intervals <=33.4 ms** as the initial steady-animation engineering target;
measure UI response separately. Native CPU-bound interturn/input stalls must be
reported, not attributed away by a continuing renderer animation. The later
16.7 ms/60 FPS objective is not a promise of this migration.

Report all >100 ms stalls, first-use/eviction/edit results and sustained-session
failures. Huge 12,800-tile maps retain separate capacity/latency results. Missing
live evidence leaves M4.6 pending; missing camera behavior leaves M3.8 pending.
If the measured target still fails, report the remaining cost and decide between
another bounded optimization, separately approved visual tradeoffs, or a separately
scoped architecture change. Do not declare success because the migration is done.

### Comparison and decision gates

Use [validation](benchmark_workflow.md) for timing endpoints and fixture scope.
Primary resolution is the current 2240x1260 gameplay envelope; 1120x1192 is a
separate diagnostic/control result. Preserve all water effects and current assets.
Include dense legal cities, coast/water, eight mixed units, the existing 32-unit
stress case, native UI churn and distributed Standard-map navigation. Record
hardware/driver and initialization policy; do not warm only the test destinations.

At a migration decision, use at least three alternating control/candidate pairs,
at least 120 independent visual samples per applicable run and the existing 100
distributed navigation destinations. Reuse unchanged valid controls; do not rerun
this campaign on every continuation. Live p95 claims need at least 100 observations
per claimed camera-trigger class; smaller samples are provisional. Report mean,
median, p95, maximum, variation and failures, never only averaged FPS or enqueue.

The default engineering gate for expanding a prototype is **>=10% improvement in
both median and p95 complete request-to-desktop latency for its declared target
workload**, exceeding control run-to-run variation, with no repeatable >5% regression
in other representative workloads and no correctness/capacity regression. Fix the
target workload before measuring the candidate. If display quantization or noise
makes the gate inconclusive, report that outcome; shorter CPU spans alone cannot
pass it. These are planning thresholds, not already achieved results or substitutes
for M4.6's absolute targets. Revisit thresholds explicitly before another design,
not retroactively to pass a result. Safety fixes have separate correctness evidence
and do not need to pretend to be performance wins.

## Water effects: roadmap placement

Requested addition (2026-09-17): bring C3X water closer to the Civ VI target —
ocean ripples/swell, directional river flow, shoreline wave/foam activity,
reflections of nearby scene geometry, refraction, sun glint and moon/night
response, coast/lake/ocean color-depth variation, ship-wake VFX, and general
time-of-day response. Most of this is not new scope from zero; placement below
reflects what is already built versus what is genuinely gated on milestones 1–2.

**Already implemented, no milestone dependency:** sun/moon direction and color,
water Fresnel/specular response, day-night glint transition, and coast/lake/
ocean color-depth variation shipped in M6.4/I13A (`environment_lighting_and_
ambient_effects.md`) and are already driving the production water shader.
Refraction and further color/opacity tuning are the same kind of shader-only
work and can be layered in opportunistically whenever the water shader is next
touched, without waiting on any milestone below.

**Milestone 2.4 map effects:** shoreline ribbons now execute in the shared dynamic
pass over reusable static color/depth. Time-only updates build/upload no terrain
or ribbon geometry and submit no static scene draws. The existing material,
spacing, 15 Hz sampling, visibility and optional asset controls are preserved.
The earlier full-terrain-rebuild concern in `ocean_wave_findings.md` describes the
pre-retained architecture, not this path. Reflections-enabled rendering keeps its
existing compatibility path; resident reflection work remains below.

**Open water and rivers implemented:** the existing normal textures now provide
overlapping ripples and animated highlights without mesh displacement
or new texture assets. River normals follow renderer-owned immutable tangents,
derived from native connectivity toward water outlets; closed components use a
stable visual sink. Wrapped flow and remote-outlet changes participate in river
page validity. No native patch, gameplay calculation or extra input capture is
needed. Fog freezes the material sample. Time-only frames reuse geometry and
static color/depth; affected translucent forward layers retain their original
order. Existing reflection compatibility remains supported.

**Current follow-up:** see the M2.6/M2.7 handoff above for implementation and
measured limits. Water visuals are accepted and staged; the strategic live-game
check remains distinct from executable verification.

**Current visual refinement:** smaller overlapping ocean ripples use three bounded
CPU phases, with no dominant translating sheet. Sun/moon highlights form a
concentrated broken path using a material-only finite-eye approximation derived
from the authoritative anchors. Shared light direction/intensity remains the
source of that path. The map projection is unchanged, and this adds no texture,
render target, mesh or pass. The ocean optical approximation is not underwater
refraction or a claim about recovered Civ VI shader code. Rivers retain their
connected downstream motion. Motion-off camera changes expire only finished
rasters, keeping the optical path out of stale world-image caches.

Current candidate SHA-256:
`3976b380f5ce0240b16c069e7f00118c8cc6643ffe3b40c22f7a8999f1782655`.
The focused suite passes 125 tests / one skip. Integration covers 302 tests
(300 pass / two skips after updating two extracted-source fixtures for the water
fields; the repaired 26-test subset passes). Windows HDR, visibility and native
CPU handoff checks pass. Day/night, fog/reveal, river
playback, still-scroll/cold and wrapped-camera/cold witnesses pass. Repeat/fog and
camera comparisons are exact; the independent split/static finish differs by
one channel level at one pixel, within the existing rounding budget. Current
previews live under `lab/out/water-refinement/final/`. The user accepted these
visuals on September 19. The exact DLL is staged in `bin/C3XRenderer.dll`, with
matching SHA-256 and a rollback copy recorded in
`lab/out/integration/water-accepted/staging.json`. No install or game launch ran. The current full-workload receipt is
`native/build/water-refinement-workload-final/receipt.json` (pass, inputs unchanged,
no dropped trace lines). The same 1120×1192 dense coast, eight-unit native workload
has 384 requests across CPU/GPU routes, with waves on and reflections off.
This is historical evidence, not the new all-effects performance baseline. GPU request mean / p95: stationary
12.51 / 35.76 ms, scrolling 152.65 / 217.41 ms, local edits 17.73 / 44.86 ms.
The separate one-selected-unit independent workload has 30 frames: 66.76 / 79.62
ms request mean / p95, 76.16 ms desktop mean, zero static submissions and zero
content uploads in every interval. Sampled contiguous VA stays above 948.9 MiB.
The preceding run averaged 66.99 ms independent and 152.12 ms scrolling; this
similar result is not an isolated shader ablation or a frame-budget acceptance.
Capture/setup remain outside timing and desktop completion is not scanout.
`summary.json` alongside the receipt preserves distributions and interval proof.

The first attempt was stopped after its test loop starved its own deadline check
by draining continuously due timers. The witness now checks progress after each
callback, preserving the three-frame/three-second requirement. M2.6 must also
address runtime callback fairness when a visual frame exceeds the 33 ms timer
interval; fixing this witness does not improve production cadence.

**Initial water workload checkpoint (2026-09-19, before the light-path refinement):** candidate SHA-256
`f0917aef1be98b9b0c8c027b25624f1434c849e49e439a5b3e69bb6a6df5befa`.
Same DLL/assets, serial motion-off/on runs, established 100×100 coast at
1120×1192, dense cities/infrastructure, eight units, native UI and shoreline
waves enabled. Each arm includes 384 timed native requests (64 per route and
workload) plus 30 independent visual frames. GPU request mean / p95, ms:

| Workload | Water motion off | Water motion on |
| --- | ---: | ---: |
| Stationary native requests | 13.41 / 35.08 | 11.83 / 35.85 |
| Dense scrolling | 141.36 / 192.03 | 152.12 / 224.08 |
| Local changes | 17.15 / 40.97 | 15.66 / 48.71 |
| Independent visual frames | 57.20 / 64.19 | 66.99 / 73.35 |

Independent frames use one selected unit and retained UI; eight units apply to
the native-request workloads. Independent desktop-completion means are
68.77 / 76.67 ms. All 30 motion-on
intervals contain water samples, zero static scene submissions and zero content
uploads. Pixel/ownership, native fallback, independent timer and bounded unit
composition checks pass. Sampled minimum contiguous VA is 1036.3 / 1025.6 MiB,
above the 512 MiB floor; sampling does not bound transient peaks. Input identities
remain unchanged and buffered traces have no dropped lines.

This pair measures added effect cost, not an equivalent-image speedup. Small
native-request differences are inconclusive; the extra ~9.8 ms independent-frame
mean and ~10.8 ms scrolling mean are disclosed costs. Capture/setup are outside
timing, desktop completion is not scanout, and current cadence remains below
the eventual frame-budget objective. The slower optional reflection compatibility
route remains supported, not promoted as the performance path.

Receipts/counters: `native/build/water-motion-workload/comparison.json` and its
`coast-off` / `coast-on` receipts. Water Lab witnesses cover daylight, moonlight,
48-frame playback, cold/repeat/time return, fog/reveal, the independent still
control and reflection compatibility. Wrapped scrolling and authoritative local
edits match independent cold redraws exactly. Category tests: 135 passed / one skipped;
three focused water-coverage tests pass. Earlier failed diagnostic runs remain
preserved: duplicate compatibility water was fixed; the small 32×32 Lab fixture
hit an outside-map alpha oracle mismatch at the large benchmark viewport, so the
established 100×100 benchmark was used without weakening its native pixel oracle.
The water visuals are now accepted/staged; live-game evidence remains pending.

The 2.6 candidate now retains the existing planar reflections of nearby geometry
using the original `environment_refresh::Reflection` scratch and the current
scene/dependency owners. Broader reflection algorithms are not prerequisites for
optimizing this enabled production workload.

**Independent of this roadmap's milestones:** ship wake/spray is unit-attached
VFX, the same category as M7.5 attached effects (flames/smoke/steam), not
core water-shader work. It can be scheduled whenever effects work is picked
up, without waiting on milestones 1–4.

## Zoom-owned map overlays: roadmap placement

Requested addition (2026-09-17): because custom rendering owns the three-level
map projection, the renderer must eventually draw the map-plane fog/unseen
territory treatment, selected-unit highlight/cursor, and pathfinding/route
visualization rather than scaling Civ III's versions on top. This is a visual
ownership extension only. Civ III remains authoritative for tile visibility,
viewer changes, selection, hover/targeting context, pathfinding, movement costs,
route/turn semantics, picking, and gameplay; the renderer consumes copied
visibility and tactical-overlay records and never reimplements those rules.

**Milestone 1 — visibility data and static fog pass:** API 18 captures normalized
visibility/fog/unseen state from the native viewer rules as an explicit dependency
of final output. The older `visibility_mask` is a traversal mask, and `fog_status`
alone omits other native visibility fields. Renderer coverage follows terrain/objects and precedes
tactical overlays, with correct map clipping, wrapping, and all 128/160/192 tile
widths (and reduced 64). Its standalone/replay tests cover revealed, fogged,
unseen and visibility-edge tiles. The user's explicit direct-hook authorization
advances fog suppression into this checkpoint: API 18 uses the existing coherent
synchronous map publication and exclusive custom-map failure contract. The wrapper
returns in custom mode and calls native fog unchanged when off. This does not
claim milestone 3's general nonblocking publication. Units under fog are omitted;
explored resource/effect motion uses stable still samples.

**Milestone 2.5 — direct tactical-overlay pass:** implemented in the candidate for
the selected-unit highlight/cursor, the already-computed route visualization, and
the optional thin gray tile grid.
Capture semantic primitives such as the selected/hovered anchor, route segments,
turn breaks and reachable/blocked indicators from Civ III's authoritative
interaction state or its established draw inputs; do not infer a route from map
data. Keep these pass inputs separate from unit-body animation, preserve depth
and terrain occlusion where the native presentation requires it, and prove that
hover/selection/path changes do not rebuild static terrain or unit content.

**2.5 gridline support (requested 2026-09-19):** Civ III remains responsible for
its global grid setting and Ctrl+G handling. Do not add a keyboard listener,
shortcut interception, or a separate toggle state. Identify the native grid draw
function and use its authoritative enabled state or invocation within the current
map draw, whichever its call contract supports. Keep the hook minimal: in custom
mode pass the required draw/state inputs to the renderer DLL and return without
native drawing; otherwise call the original function unchanged. The DLL draws
the tile boundaries using native anchors and the current projection. Switching
on/off must add/remove only the grid, without stale lines, double drawing or
terrain rebuilds. Verify clipping, fog/unseen coverage, zoom and wrapping. This
cutover belongs to 2.5 using existing coherent publication; milestone 3 extends
it to nonblocking views. Audit the actual function before recording any concrete
patch dependency. The audited GOG grid hook and two route hooks are now recorded
in the [patch ledger](civ3_patch_dependency_ledger.md); the
[tactical contract](tactical_overlay_contract.md) records copied inputs and lifecycle.

**Milestone 3 — coherent live cutover:** publish pixels, camera/zoom transform,
visibility epoch, tactical-overlay revision and overlay inputs as one compatible
view identity. Extend the existing fog publication contract to nonblocking views
and suppress corresponding selected-unit and route draws only once matching renderer output
is ready; never show a new camera with old fog, a stale route, or duplicate
native/renderer marks. Visibility, selection, route cancellation, scroll/wrap,
zoom and device recovery retain coherent ownership. Config-off retains native
rendering; custom-on map-plane failures preserve exclusive custom-map handling.
Audit existing draw/capture seams before proposing any new patch
symbol; this roadmap entry authorizes no speculative CSV change.

**Still native unless separately extended:** unit health/activity/status and
stack HUD, civilization markers, city labels, borders, general map text, broader
selection UI, and all non-map screens. Their current transformed-native treatment
remains in place. The overlay cutover needs focused all-zoom visual comparisons
and executable ownership/invalidation tests before it is accepted; it does not
move deferred wonders or Districts forward.

## Current evidence and implementation handoff

**2.3 committed and pushed:** `9835c9a8` was committed on Mac and pushed through
Windows. Units remain above all map geometry, retaining pose-local shadows,
self-depth, authored actions, fog eligibility, native composition and controls.
Unused map-depth capture/provenance and the map-region cache are removed. Its
measurements remain in Git and `unit-composition-checkpoint/workload-comparison.json`.

**2.4 shoreline capability implemented:** retained wave cells feed the existing
shadow/foam/resource dynamic pass and damage union. The renderer restores static
samples, draws the affected ribbons and finishes that band. Waves no longer
exclude automatic shared-scene admission. No new surface owner, scheduler, API,
native hook or patch-table entry is introduced; `required_user_action: []`.
Reflections retain their configuration and existing compatibility renderer;
resident scene measurements below use reflections off. Open-water motion and
river flow are unfinished. No binaries were staged/installed, game launched or
reference images replaced.

**2.4 evidence:** the [shoreline findings](ocean_wave_findings.md#milestone-24-resident-scene-checkpoint)
preserve its 295-test/nine-replay verification, complete-request distributions,
independent-frame measurements and unchanged-asset identities.

**2.5 tactical capability implemented:** the existing composition history now owns
copied selection rings, native route segments/turn strings and gray grid edges.
The white broken ellipse has rotating inward markers; analytic antialiasing and
restrained shadows sharpen the route and label. Native selection, pathfinding,
turn arithmetic and `MapGrid_Flag` remain authoritative; no Ctrl+G listener or
second route model exists. Native erase/copy operations retire the same immutable
history. The DLL does the rendering; injected code supplies the audited seams.
Three GOG additions are recorded in the [patch ledger](civ3_patch_dependency_ledger.md#milestone-25--tactical-draws-gog)
under the user's patch-table authorization; other-build addresses remain zero.
Before composition admission/with an older DLL, native cursor/route drawing
continues. Config-off calls the original functions.

**2.5 verification:** 276 tests (**274 passed / two skipped**), approved injected
compile/injection smoke test and four production replays pass. Additional connected
native fixtures pass at 64/128/160/192 and with fog. Route cancellation/grid-off
restore the prior image exactly. Twelve independent selection samples execute
without native draw events, static terrain submissions or new unit-content builds;
the existing 128 MiB retained-history limit includes copied primitive capacity.
The preview is a synthetic scene through the real JGL/DLL path, not live gameplay.
The connected category uses the established 100×100 camera fixture: a 32×32
preview world also fails its wider prepared-camera assertion with overlays off.
No assertion was relaxed.

**Complete-workload effect:** same candidate, 1120×1192 dense modern-city coast,
reflections/waves off, 8/32 unit bodies, native UI and final transfer. Each request
contains one selected marker, one two-segment route/turn label and the viewport
grid when enabled. Six serial runs contain 2,304 timed CPU/GPU requests; the
32-unit pair was repeated in reverse order after a variable local-change tail.
GPU mean / p95 milliseconds (64 samples per 8-unit cell; 128 per 32-unit cell):

| Units / workload | Overlays off | Overlays on |
| --- | ---: | ---: |
| 8 / stationary requests | 15.01 / 39.98 | 14.02 / 41.71 |
| 8 / dense scrolling | 141.87 / 215.36 | 139.54 / 190.74 |
| 8 / local changes | 18.08 / 41.22 | 17.21 / 37.09 |
| 32 / stationary requests | 19.96 / 51.28 | 21.17 / 49.95 |
| 32 / dense scrolling | 144.77 / 214.94 | 146.55 / 205.32 |
| 32 / local changes | 23.88 / 60.79 | 27.74 / 74.01 |

This adds a measured feature cost, not a speedup: stationary/local medians rise
about 1.8–2.8 ms; the lower 8-unit means reflect tail variation. The 32-unit
local-change mean difference was 6.12 ms initially and 1.59 ms in the repeat
(3.86 ms combined). Full distributions remain in the receipts. Native route
calculation and capture setup are outside this semantic replay harness; packet
preparation, drawing and final transfer are inside. CPU/GPU images differ when
tactical visuals are enabled, so their ratio is not an equivalent-image speedup.
The 12-frame independent correctness excerpt averages 24.64 ms including its
112.89 ms cold first replay (remaining 11: 16.62 ms); it is not an FPS/cadence pass.

All six runs have complete traces and unchanged inputs; all **16,102** asset/input
files remain unchanged. Minimum sampled contiguous VA is **648.57 MiB**, above
the 512 MiB floor; sampling does not bound transient peaks. The
[combined receipts](../native/build/tactical-checkpoint/workload-comparison.json)
retain identities, original/repeated distributions and scope limits. Candidate
SHA-256: `6c3b4072c00ab51621b6e423575d2e4aa1f94b4eadf04cd83a21dedcd199fc16`.
[Context preview](../lab/out/tactical-overlays/connected-route.png),
[selection motion excerpt](../lab/out/tactical-overlays/connected-motion.mp4) and
[grid preview](../lab/out/tactical-overlays/connected-grid.png) await visual acceptance.

**Current follow-up:** the M2.6/M2.7 handoff above supersedes this historical
2.5 checkpoint. M3 owns general nonblocking publication; wonders/Districts remain
M9–M11. Tactical visual acceptance and the strategic live-game checkpoint remain
separate from automated lifecycle and performance evidence.
