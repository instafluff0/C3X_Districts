# Renderer64 scene and motion cutover

This is the architecture target for the x64 renderer migration. Civ III remains
the authority for game rules and UI. Renderer64 owns the visible map scene,
its presentation clock, and the cross-process composition surface. Work toward
one playable integration checkpoint before whole-frame FPS tuning; correctness
tests at contract boundaries are still required.

## Ownership and data flow

Civ III/C3X reads game objects only on the game thread. At load, viewer change,
and recovery it publishes a versioned scene snapshot: map dimensions and wrap,
tile appearance and visibility, city state, visible unit identities/actions,
selection/path state, camera/projection, and environment. It then publishes
ordered, bounded changes. Changes are facts copied after Civ III accepts them,
not raw pointers or a second simulation. Renderer64 acknowledges a sequence;
missing or superseded changes require a fresh scoped snapshot. The existing
tile publication journal and unit lifecycle owner are the starting points,
not duplicate worlds to retain indefinitely.

Keep Renderer-only patch functions together at the end of `injected_code.c`,
immediately before its required `main`. Existing patches that also serve other
C3X features stay with their shared logic. A hook should copy authoritative
values and forward them; sequencing, diffing, storage and visual playback belong
in `Renderer/`. Add a `civ_prog_objects.csv` entry when a concrete new hook is
needed, and record its signature, supported-build addresses, fallback and reason
in the patch dependency ledger.

The current world-page callback discovers tile/city state in batches on the
game thread. It is useful for initial population and reconciliation, but a
periodic page read is not an immediate city, visibility, or unit notification.
The first cutover stops that periodic pass after one complete map/viewer
snapshot. It sends full art only for explored tiles; never-explored tiles retain
visibility and terrain topology and are skipped by background art preparation.
Older recorded full pages are scrubbed to that same boundary when replayed.
After an accepted `Unit_move`, the existing patch sends old/new tile
coordinates; the registered game-thread callback copies only their bounded
sight neighborhoods, compares native visibility bits with the existing capture
cache, and sends only changed tile records into Renderer64's scene journal.
A post-interturn audit re-arms one paged world reconciliation for changes
without an explicit transition hook. Other-civ moves that enter visible tiles
also request an exact view capture. They still need a stable-ID accepted motion
segment before hidden-to-visible animation can be claimed. First-move reveal is an integrated display test,
not established by the synthetic publication test alone.
The existing `Leader_spawn_unit` hook now reports a scoped stable-ID birth after
the game assigns its ID. Renderer64 retires any pose from a prior use of that ID;
the next body capture supplies its art and exact screen anchor. Births, moves,
fog loss and body observations now reject older timestamps for the same ID;
a hidden move retains its timestamp so a late reveal cannot resurrect the pose.
A small copied state record carries action and HP at each native unit draw;
`Unit_despawn` sends a retirement fact before storage or ID reuse. These events
share the ordered IPC and replay sequence with birth, movement and body samples.
An explicit retirement is a tombstone: a later observation cannot revive the
same ID without a new accepted birth. The state stream does not itself grant
visibility or guess an intermediate combat strike. The x86 Renderer bridge
coalesces identical unit facts before IPC; native redraw ticks do not require
another Renderer64 roundtrip when action, HP, position and visibility are unchanged.
A rejected sparse world change schedules the bounded reconciliation on the next
native view.
Visible map capture corrects a requested tile. A page or view capture remains the
bounded recovery path when an individual transition cannot be observed safely.
Off-screen unit metadata in a tile record is not a complete unit roster;
unit instances need stable IDs and explicit retirement. Stored hidden data
never grants draw eligibility.

Renderer64 stores the most recent accepted scene generation and samples it
without reading Civ III memory. Ambient water, visible resources, eligible
selected idle units and visible working units run on its own clock. Unselected
idle units and explored-but-not-visible resources stay frozen; hidden units do
not draw. A retained front with no eligible visible animation does not start
the visual-frame cadence, even when its static pixels remain ready for native
presentation. A game-thread stall cannot pause eligible ambient animation. Native
actions, audio, combat outcomes, turn processing and path legality stay in
Civ III. The native 66 ms callback retains any gameplay/action advancement;
custom rendering suppresses only superseded native drawing.

## Accepted unit motion

Civ III validates and performs a move. C3X publishes an accepted visual segment
with stable unit and event IDs, from/to tiles and authoritative screen anchors,
direction, action/clip, start time, visual duration or native progress, camera
generation, visibility and a monotonic sequence. Renderer64 samples the segment
at each display time. It never decides that a land Scout can enter water, moves
the game unit, or invents a combat result. Native action timing can continue at
its original cadence without quantizing the displayed travel to that cadence.

The segment must have an explicit completion/correction rule. A newer native
position, interrupted action, combat, death, teleport, embark/disembark, unit
removal, fog loss, viewer change, save/load or reset supersedes it. Late messages
are sampled at current time rather than queued as delayed frames. Rapid successive
segments have bounded per-unit state and may be shortened or coalesced to avoid
visual lag behind authoritative play. Camera jumps reproject the same world-space
endpoints; map wrapping uses Civ III's chosen occurrence. Selection underlay,
path and unit body use the same sampled anchor so they do not separate. When
the required anchors or outcome are uncertain, use the latest authoritative
position rather than predict a game move.

Work and selected-idle loops need only an accepted action/visibility change and
clip start time; Renderer64 can loop their authored poses independently.
Directed combat/death clips keep Civ III's outcome and effective wait duration.
The same ordered stream must carry attack start, each accepted strike/HP
revision, interruption, death and action end. Renderer64 never predicts a hit,
damage, survival or target. A new authoritative strike can interrupt the visual
pose; presentation timestamps align impact and reaction with Civ III-owned
sound. Civ III's fight loop writes damage between animator waits. The current
`Fighter_fight` hook observes entry and exit, not each write. First establish
whether the existing unit-body capture reports every intermediate HP value at
the right time. If it does not, add one narrow accepted-strike notification at
an existing combat/animation boundary; do not poll unit memory from Renderer64
or fabricate intermediate damage.
The on-map unit health bar should use the same sampled unit anchor and copied
HP revision inside Renderer64's final image, so the cross-process surface
cannot cover a native bar or separate it from a moving body. Civ III currently
draws unit status after the body inside `Unit::tick_anim` (through
`FUN_005ba750`); this map overlay needs an explicit ownership transfer when
the direct surface becomes normal. Civ III retains
combat audio and non-map UI. This is a map-presentation transfer, not a second
combat system.
The combat boundary check must compare each visible native damage revision,
body capture and sound/animation interval in order. It must include a normal
fight, defensive and ranged bombardment, city bombardment, air strikes and
interception, retreat, death and army-member display. A bombardment can change
a unit, city or improvement without moving the attacker into the target tile;
an intercepted aircraft has another participant and a return-or-loss outcome.
The event carries those authoritative identities, target tile and result. A missing
intermediate HP revision is a capture defect; Renderer64 must never fill it in
by dividing the final damage across imagined strikes. Losing visibility
immediately removes combatants and bars from the map scene.
The body hook now copies Civ III's current and accepted target pixel positions,
draw anchor, damage, maximum HP and action into a separate observation. Renderer64 uses
successive observations to smooth visible travel between native updates, bounded
to 90 ms and corrected by each newer native pose. The first sample uses Civ III's
ordinary movement-speed estimate and the copied target; subsequent samples use
measured native progress. `Unit_move` also sends the accepted old/new tile pair,
unit identity, viewer scope and endpoint visibility. Renderer64 discards the
previous pixel prediction at that boundary; the next native observation fixes
the new segment's screen anchor. A hidden destination retires the unit visual.
This is a conservative visual refinement, not yet the full accepted segment
contract above: durations, interruption IDs,
selection/route attachment and proof that every combat HP revision is observed
at the intended presentation time still need work.

## Surface and frame lifecycle

### Current asynchronous bridge

The fresh-scene handoff now uses Civ III's captured unit center instead of the
sandbox's fixed 191-pixel canvas center. The extracted production placement and
shader math pass 8,640 size/zoom/wrap/reflection cases. The final fresh output
also runs the existing GPU visibility pass on every publication target;
resource-free scenes still refresh visibility. The GPU oracle checks 3,295,332
pixels across zoom, scrolling offsets and reset (maximum channel error one).

Candidate `34e1326f9e4b4316b1c2a1f7c8b89190` passes the asynchronous fixture at
55.89 FPS, but real run `20260927-144050` exposes a publication lifetime defect
after eight adoptions. The helper faults at RVA `0x24D9`, within tile-record
serialization. The worker's old command exclusions omitted helper presentation
commands, allowing an independent frame to free the adopted camera's records
after polling. Retirement now explicitly belongs to map/configuration commands.
The extracted completion regression fails against the old code and passes
780 borrowed-map operations with the same live records; reset/configuration
still retire them. The replacement candidate is undergoing live qualification.

The current staged candidate bounds retained input payload, in addition to
command depth. Capture `20260927-132302` first fails during a native sprite draw
at 18.221 s: changing source uploads exhaust the input description's 96 MiB
budget before any readback fault. Scripted real-game run `20260927-135838`
reproduces the same error at 22.188 s and the visible black/repeating map.
Regional compaction now considers payload cost; the cap remains 96 MiB and
these values never enter map rendering or native/GPU pixel buffers. The old
probe exhausts memory after 3,072 changing draws. The corrected regression
completes 10,400 with a copied prior view intact and 22,611,968 peak bytes.

All nine input tests and native oracle `f440524ed1e248479bcacd7c03131acc` pass.
Full GPU fixture `26f3810cd26449a9812f9b78ec6d64b1` submits 5,200 changing
sprites in 1.466 s, restores its clean map, then measures 56.76 FPS warm cadence.
It passes 32 complete scrolls in 31–125 ms, 33 adoptions/commits and independent
host/renderer progress. Complete diagnostics through teardown contain only the
intentional invalid-window test failure, with no input/texture budget fault or
unexpected map readback. The matched binaries are staged and startup passes.
Automated live run `20260927-140022` removes that input-memory failure, then
exhausts the ordered publication queue at 43.377 s after three camera moves.
Warm map rendering takes 63–81 ms but queued native image work delays adoption
by seconds. The adapter previously retained only the last sprite payload;
alternating unchanged sprites therefore caused repeated uploads and IPC. The
staged candidate retains up to 256 immutable decoded sources within 16 MiB,
compares complete content and dimensions, and retires LRU sources in command
order. Palette, animation and retained-pointer changes still create fresh
content. The GPU handle table accounts for these bounded owners; its texture
byte budget and the publication queue limits remain unchanged. Native oracle
`9a89e41d0e9f4487861d9c5213f9493d` passes exact pixels, source mutation and
config-off behavior. Full asynchronous fixture
`8bb8c078fe3e4d199771678676c3691f` passes at 55.62 FPS with 32 complete
camera updates (31–141 ms), independent process progress and no unexpected
diagnostic failures through reset. Automated live run `20260927-141651`
completes 75 seconds without either fatal error, adopts 13 maps, and reuses
69,610 sprite draws with only 22 uploads (892,624 bytes). Scrolling and visible
composition still fail acceptance. Native capture was also evaluating/drawing
legacy C3X tile highlights in all five native passes, including pending frames.
At the user's direction, custom rendering now bypasses their initialization,
worker-highlight preparation and drawing. The simpler bypass replaces the
intermediate pass restriction and leaves vanilla behavior intact. Its injected
compile and extracted hook checks pass. Live run `20260927-142322` completes
all 32 camera moves and 33 map adoptions without a renderer failure. Median
native map time is 3.959 ms, compared with about 1.1 seconds before the bypass;
median capture is 2.504 ms. Sampled terrain survives the scrolling loop.
Remaining visual work includes unit/selection alignment and visibility. The
first map still takes 6.233 seconds to render (5.357 seconds geometry).
Controlled cadence remains separate from live-game FPS.

The user explicitly requested a bounded automated real-game test for this
integration investigation. Native menu loading, the existing post-load popup
hook and raw main-form key hook now allow a copied save and 32 camera commands.
The opt-in additions guard custom rendering and the child environment variable;
ordinary and config-off behavior remain unchanged. The injected smoke test and
extracted guard/camera contract pass. See the [diagnostic guide](../tools/scripted_game_test.md)
for elevation, console installation, cleanup and evidence limits.


An earlier correction handles empty native background copies. Capture
`20260927-130716` adopts a map, then a forbidden image readback poisons
composition immediately after a unit draw. Its original diagnostic does not
identify the triggering operation; the exact live cause remains unproven.
Renderer64 unit observations return an empty native raster envelope, which
Civ III can still use for background restoration. Native JGL treats a copy with
a zero dimension as a successful no-op. The previous adapter instead entered
CPU fallback and attempted two readbacks. The adapter now handles that no-op
before admission or fallback, with no drawing, ownership change or extra state.
Readback and exception diagnostics now identify the native operation.

The native JGL oracle passes nine zero-dimension cases with no readbacks.
The expanded asynchronous fixture fails with the previously staged binaries
(`96733a37f4944e27bf1457e0437adf00`), reproducing readback, unusable session
and failed scrolling. The corrected matched trio passes
`9c572774fce745f3b1982abbf3693002`: 33 camera adoptions/commits, all 32
scroll/grid/selection/HUD updates, queued native writes, alternating clears,
empty unit copies and a fully drained consumer. At 2240×1260, warm cadence is
55.51 FPS and complete scroll updates take 31–313 ms. Host/renderer suspension
retains independent progress with no CPU map readbacks. Diagnostic capture
through reset has no unexpected operation, composition or texture-budget
failure. The trio is staged and its installed-directory startup probe passes;
no injected source or installed executable change was necessary.
Live cold startup, scrolling and the reported fog appearance remain unqualified.
Fixture cadence is not live-game FPS.

The preceding capture `20260927-124922` had no forbidden image readback or
missing-display failure. Its first completed native map redraw was
309.399 ms, compared with 7,156.676 ms in `115530`; this excludes the preceding
cold scene preparation and is not startup latency or live FPS. The first map
commits at 10.659 s and native presentation succeeds at 11.095 s. Only ticket 6
is adopted. Later redraws report `BAD_ARGUMENT` before camera begin, followed
by a retained texture-budget failure at 16.744 s.

The native composition owner rejected requests and polls whenever unpresented
native commands were queued. Civ III legitimately enters that boundary after
native drawing; the previous fixture always presented first. Renderer64 now
publishes the old-ticket batch before camera begin or adoption using its existing
asynchronous queue. The legacy exact path keeps its explicit flush contract.
An extracted regression fails on the former rejection and passes with the new
ordering, including draws queued while a camera is pending. The Windows fixture
now queues native map/UI writes before every camera request and poll. No new
game hook, address, CPU map readback, or rendering fallback is involved.

Three camera-policy and five publication tests pass. The current combined
fixture preserves this ordering coverage and adds empty unit background copies.
Absence of the earlier texture-budget failure in fixtures is not proof of its
live resolution.

Capture `20260927-115530` failed before the first displayed map. A CPU-owned
HUD form was sampled before GPU adoption, and the unscoped temporary bits lease
incorrectly revoked its lifetime evidence. Later HUD blending attempted a
forbidden readback and disabled composition. The fix scopes the verified native
sampler's temporary bits read before adoption; the GPU-owned path still answers
input locally. This supersedes the previous fixture-only startup qualification.
The reported fog appearance remains visually unqualified.

The final canvas sampler call in GOG form hit testing is at `0x006090C3`.
Its new wrapper first checks `enable_custom_rendering` and delegates directly
to the original sampler at `0x00600340` when false. CPU-owned UI also retains
that function. For GPU-owned canvases, native UI commands and copied UI source
words describe the current input surface. A query evaluates one point without
waiting for Renderer64 or accessing GPU/native map pixels. Terrain is opaque
for input regardless of its lighting; explicit transparent-key and zero UI
regions retain their native meaning. This is restricted to that form-input call,
not a general substitute for native pixel getters.

The input description uses immutable command references indexed by 64-pixel
screen regions. Source snapshots keep only the regions a command samples;
replaced history is retired locally. Deep translucent histories become bounded
input values for one region. Map words stay the semantic opaque marker. These
values never enter a GPU upload, native pixel buffer, or displayed frame.
The former whole-canvas traversal took seconds on dense map overlays and could
exceed its depth limit despite passing small HUD fixtures. The regression now
covers 20,800 map overlay draws, 20,000 partial HUD redraws, cross-region
translucent blending, lifetime reuse and both 16-bit native formats. The native
oracle compares known input values against GPU/JGL pixels.

The GOG entries were added with explicit user authorization. The injected
compile/injection check and native input recording/replay codec pass. See the
[patch ledger](civ3_patch_dependency_ledger.md#asynchronous-form-hit-testing) for
addresses, signatures, disabled behavior and exact validation receipts. Full-size validation now includes CPU form input before adoption and 2,600
transparent overlay calls, as well as 32 scrolls and independent process progress.
A diagnostic run also exposed old map callbacks retained through translucent
HUD history. Superseded fresh-map sources now freeze their completed GPU output;
static dependent recipes retire their old inputs even when pixels are unchanged.
An 80-view GPU oracle verifies exact pixels, bounded memory and current-map
animation. The full-size fixture also caught exponential input work on repeated
nonmatching destination-key transfers. Reusing the already sampled destination
removes that duplicate traversal. The final fixture times complete HUD submission
as well as camera polling, and rejects updates lasting two seconds or longer.
The preceding startup correction was installed. Its full-size fixture
`dba0b01184d345cc911e430e6ce0bd9b` passes at 55.62 FPS with all 32 complete
scroll/HUD updates in 31–125 ms and no composition errors. It verifies
independent progress while either process is paused, with no CPU map readbacks.
The native HUD oracle and installed-directory startup probe pass. Installed
disassembly confirms immediate config-off delegation and restoration of the
private CPU sampler context. General gameplay acceptance remains a strategic checkpoint;
fixture cadence is not live-game FPS.

`sandbox/async_scene_client.h` owns copied publications between the native
composition owner and the existing process transport. The game reserves image
identities locally and publishes creates, uploads, draws, unit facts and presents
in order. Pending camera captures are replaceable; replacing one moves the newest
capture behind already accepted reliable commands. Queue capacity is 128 MiB and
8,192 entries, including in-flight bytes. Exhaustion or consumer failure latches
an explicit error and preserves the completed GPU display. Automatic recovery
from that fault is not implemented yet; it needs a fresh scoped scene and native
surface reconstruction rather than replaying an incomplete queue.

Readiness inspection does not retire the displayed map. Adoption is a separate
ordered command, so all old-map draws finish before the new map identity becomes
usable by subsequent commands. The helper consumer acquires the render boundary
once for readiness/adoption; repeated try-lock polling could starve an already
completed camera behind ambient frames. The game only polls copied local state.
Camera results and uploaded arrays are owned
copies; no game pointer or borrowed surface handle survives publication. Startup,
definition loading and explicit reset still join transport outside the frame
path. CPU map readback and CPU unit rasterization are rejected in this mode.
Native UI assets may still originate in Civ III's CPU surfaces.

World capture can return `PENDING` while Civ III loads, draws, or waits for its
first displayed map. This status means no snapshot was captured. The bridge
returns non-success capture results directly without queuing an empty page or
faulting transport; the native timer can retry while map publication continues.
Successfully captured pages and deltas still enter the ordered queue as owned
copies. Async transport errors identify the failing operation in one log line.

The helper's ambient frames invoke the sandbox resident draw directly. They
refresh copied unit poses and visual time without re-entering full terrain
preparation. Geometry preparation runs on authoritative scene/camera changes.
An old retained source keeps its last complete image if a replacement has
changed the geometry generation; continuity during long cold preparation still
needs a separately owned resident scene. Displayed-view input timing, modal UI,
device/process recovery and live gameplay remain integration checkpoints.

The real-JGL fixture runs with `native.record_gpu_frame --async-bridge`, the
matching x64 helper/DLL, audited local JGL and a captured scene. It exercises
pre-display world-capture deferral, indexed UI transfers, six actual HUD pairs,
first-map adoption, scrolling and independent process progress. It pauses the
native host and renderer separately for two seconds and rejects CPU map reads.
`--window-witness-seconds` adds separately sampled compositor evidence. Present
counts measure renderer submissions, not physical scanout or gameplay FPS;
readiness measurements do not certify input-to-display latency. Preserve the
sandbox's roughly 52 FPS; further FPS tuning is deferred.

Earlier evidence remains useful: capture `090553` failed when a deferred world
capture entered the queue as an empty page (negative regression receipt
`2aaf54663a8f4505b3a5136d38f31b20`). Capture `092023` then confirmed map publication
before a HUD admission failure. Capture `094257` identified a full-screen canvas
that lost eligibility on a context-0 bits request before GPU publication; both
it and a rejected 36×30 button had no active leases at the later blend. Releasing
a lease does not prove that a caller discarded its pointer, so ownership checks
remain strict. Bounded stack candidates are checked against disassembly, not
reported as a proven backtrace; the optimized-JGL witness recovers `jgl+244d`.
The newest capture and candidate evidence are summarized above.

The Civ III bridge owns the window and creates one DirectComposition surface
per window generation. Renderer64 owns the device, swap chain, final map image
and frame scheduler. It receives the surface handle once; normal frames never
send a bitmap or per-frame request back to Civ III. Target active-display
opportunities near 16.7 ms without a catch-up queue; missed frames skip ahead
using elapsed time. Input/gameplay messages must not wait for animation frames.

The map surface must preserve Civ III UI ordering. Native operations already
represented in the final image remain there. Other writes to the same window
need a proven upper layer or an exact, rare ownership handoff; child-window
behavior alone does not prove every label, HUD or popup. Partial transfers must
retain pixels outside their rectangle. Resize, modal transitions, minimize,
config-off, helper restart and device loss retire or recreate surface generations
without displaying stale map pixels. The shared-texture presenter stays a
controlled recovery path until this full lifecycle is proven.
Final native handoffs now record the graph-window input independent of whether
the CPU snapshot or cross-process surface route consumed it. Earlier recordings
cannot gain that missing input retroactively and need a fresh capture for an
exact route comparison.

## Integration gate before performance work

Implement snapshot/change and motion contracts, the Renderer64 clock and the
cross-process surface as one vertical slice. Focused tests prove sequencing,
move interruption, fog, camera/selection alignment, partial native transfers,
resize and recovery. Then run one complete recorded native/UI workload with
the actual surface route and stage one identified build for a real game check:
Scout travel, worker action, continuous visible water/resources during movement
and interturn, combat strikes/HP bars/sound alignment, selection/path alignment,
city/UI transitions, and config-off.
Correct visual behavior and basic stability are the gate; no broad FPS tuning
or cache redesign belongs ahead of that game check. Afterward measure idle,
movement, scrolling and arbitrary jumps with all water effects enabled and
optimize the measured critical path. Remove superseded per-frame Civ III visual
requests and shared-image adoption only after the new route and recovery path
cover their callers.
