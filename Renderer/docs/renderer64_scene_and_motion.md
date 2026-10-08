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
That same move-completion notification also starts a copied viewport request,
before Civ III's next redraw. The native composition owner retains its capture
identity so a later redraw polls it rather than cancelling and restarting it.
This preserves the first step's reveal while the following step is animating;
no animation-director call, gameplay wait or invented visibility is added.
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

Looping worker animations initialize from their captured native cursor divided
by the native frame count, then advance independently at the authored clip rate.
A newly started native action at frame zero therefore begins at zero; an existing
job keeps its native starting phase. Subsequent captures and camera changes do
not reset the retained cycle. This applies to all seven work-animation slots
(fortress, road, mine, irrigation, jungle/forest clearing and planting).
The native worker setup `FUN_004068e0` already chooses `_rand() % frame_count`
when initializing an off-screen work animation; its active-unit branch queues
the animation, and `FLC_Animation::start` resets the cursor to zero. The existing
`forward_custom_unit_body` capture supplies that cursor. No additional renderer
randomization, game-state changes or patch-table entry is needed.

## Camera and animation handoff

A prepared camera retains its native request timestamp for exact duplicate
request matching. Its unit scene samples the current visual clock. Scene sampling
never moves that clock backward when an older captured view arrives.

Preparing a different camera freezes the preceding camera's completed retained image.
Native UI transfers must continue displaying that last sample even after its
animation callback retires. The original native working canvas can contain an
older unit pose. `Session::display_to` therefore uses any ready retained recipe,
including a static one. The GPU regression advances eight poses, freezes the
camera and repeats four native transfers; every transfer must preserve the
last pose. The prior code fails that pixel comparison.

Same-projection terrain preparation forks one bounded completed view. Geometry
membership shares immutable meshes, with distinct revision identities for
completed and pending branches. View dimensions, projection/depth, resource
anchors, visibility proofs and wave-buffer leases stay paired with that
membership. At cooperative preparation boundaries, the immediate-context owner
can borrow the completed view, sample actors/HUD at the current clock, and
restore the pending view on every exit. CPU compiler lanes continue using their
immutable source jobs. The pending camera draws into a separate texture; it
cannot overwrite the displayed frame.

Cancelled preparation retains the completed view until a replacement succeeds.
Map/viewer, content or device changes invalidate that lease. A single metadata
fork is capped at 32 MiB (including retained wave buffers); immutable mesh
retirement remains charged to the existing cache budget. Navigation and views
without an eligible retained front keep the existing `held` fallback and
bounded navigation clock allowance. A temporary hold never retires the sampler.
If demanded geometry cannot fit beside the retained front, the worker releases
its optional mesh pins and retries once with motion held. The displayed texture
survives this fallback; a second failure remains an error rather than a retry loop.
Externally supplied display timestamps reset the local wall-clock anchor so a
following camera call cannot count the same elapsed interval twice.

World and visibility updates at the same camera continue sampling the newly
prepared resident scene immediately. In particular, revealing terrain during a
move must not wait for Civ III to adopt another map image. Compatibility requires
the same map/viewer scope, target size, tile scale, world dimensions, wrapping
and screen projection origin. A pan, zoom or viewer change still requires the
ordered handoff. The projection and motion tests exercise these boundaries.

## Combat presentation

The existing body capture copies the accepted combat target and the native
animation's cursor, frame count and frame period. `Animation_Info::get_field_1D8`
returns that same period array in the native animation update. The new copied
record travels through the existing bounded asynchronous queue; it neither
waits for a render nor changes native action/audio timing. The older draw and
visual records keep their layouts and replay paths.

An in-tile target becomes an offset from the authoritative tile center. Renderer64
interpolates that offset, holds it through fortify/attack/death, and interpolates
a confirmed return target. Repeated native intermediate positions cannot steer
or restart this travel. Full-tile movement remains owned by its accepted move
segment. A victory target outside the current tile holds the last combat stance
until that segment arrives. The segment starts at the last displayed offset and
travels only the remaining distance; it must not reset through the source tile.
Unrelated native position corrections discard the offset. Wrapping and zoom use
the same scene projection for both kinds.

Attack clips advance on the visual clock at the copied native cycle duration.
Repeated sparse captures do not restart a phase. A native return to idle or
movement retires that directed clock, so repeating the same attack or bombard
starts at its new native cursor instead of reusing a previous encounter's phase.
Death, fortify and victory
clamp at their last pose; native action changes and retirement control their
handoff. The scene retains its last complete body while the new action's body
capture arrives, while invalidating old native selection tokens. HP, visibility
and retirement remain authoritative native events. These changes do not add
combat decisions or CPU rendering.

The capture also names the native display parent. The most recently captured
stationary group owns its tile's displayed stack; an old retained civilian does
not remain beside a newly selected combatant. An army commander and its member
share the parent's group. An admitted travelling body remains visible until
its segment finishes, even when native drawing selects the next source-stack
unit. This controls presentation only; it never deletes a gameplay unit.

Visible bodies mark stencil during their existing depth-tested GPU draw. The
final fog pass reads that coverage and preserves the body silhouette, including
weapons crossing a fog edge. Terrain, shadows, cutout holes and occluded body
fragments retain normal fog. No extra body pass or CPU map readback is needed.

### Joint and direction transitions

The GPU unit owner blends from the last displayed local joint pose into the
new action for at most 120 ms (20% of a shorter clip); run-to-idle settles over
200 ms. Destination time keeps
advancing. Local quaternions use shortest-arc interpolation; position and
scale/shear are blended before the hierarchy is rebuilt. GPU vertex skinning
and the ordinary immutable animation palettes remain the rendering path.
Interrupted transitions start from the currently mixed pose. All passes reuse
one sampled pose at the same visual timestamp.
Travel, heading and local-joint transitions share a continuous presentation
clock that excludes hidden map-preparation intervals. Adopting a prepared map
cannot consume a turn or the run-to-idle blend while the old image is frozen.

Optional generic `C3XRIG1` metadata follows the existing `C3XANM1/2` palettes:
parent indices, explicit skin-to-joint mapping, inverse binds, a SHA-256 binding
identity, and local position/quaternion/scale-shear samples. The binding covers
hierarchy, bone names, binds and component identity; equal bone counts alone
never permit a blend. Rigid equipment carries its driver's hierarchy so it
follows the blended hand. Old packs still use their original GPU palettes.
Rig data is bounded and included in asset residency accounting.

The same transition handles idle, run, fortify, attack variants, fidget,
victory, death, capture and work actions when their component binding matches.
Fidget, capture and build now use copied native timing between sparse body
captures, as attack/death already did. Native outcomes and retirement still
control visibility; blending cannot extend a dead unit's gameplay lifetime.
Fog loss, retirement, ID reuse, changed art and scene unload discard old state.

Heading turns along the shortest angle over 60–180 ms according to turn size. The live path now shares
Lab's calibrated `yaw_offset + (native_direction % 8) * 45 degrees` mapping;
its previous subtraction of one produced a 45-degree error. Accepted movement
selects the travel direction even when the preceding standing pose faced
elsewhere. This smooths model heading; it does not invent authored turning steps.

The 2,649-payload roster reconstruction check covers 85,968 used-joint samples
with maximum palette error `5.96046e-7`. The asynchronous GPU fixture
`d3c44c391c2546f39938ec917a4384b9` passes 24 action/direction changes at 59.89
frame submissions per second, 32 camera changes, and zero CPU map readbacks.
This is controlled fixture evidence, not live-game FPS. The rebuilt 78-definition,
94-key roster now supplies the normal `UnitAnimationFidelity` pack. The old
runtime manifests and preparation receipt are preserved under
`native/build/pose-transition/previous-production/`; `promotion.json` records
the evaluation staging. Reference images are unchanged. Live victory, retreat,
army-member, capture and intercepted-aircraft evidence is in the
[scripted testing guide](../tools/scripted_game_test.md#combat-diagnostic).

Native impact FLCs can use a new scratch destination with the owned map as a
separate background. The existing lifetime-checked image admission now covers
both normal and reduced lookup-over variants. This removes the attempted CPU
map borrow that previously failed bombardment and air combat. It is transport
correctness only: the requested Civ VI-derived impact particles still require
their own production pass and event ownership, as documented in the
[effects contract](bombardment_and_explosion_effects.md).

## Accepted unit motion

Civ III validates and performs a move. C3X publishes an accepted visual segment
with a stable unit ID, from/to tiles, action, visibility, map/viewer scope and
an ordered timestamp. Renderer64 owns the travel duration and run phase, and
uses the current scene's copied tile centers to place the actor. It samples the segment
at each display time. It never decides that a land Scout can enter water, moves
the game unit, or invents a combat result. Native action timing can continue at
its original cadence without quantizing the displayed travel to that cadence.

The segment must have an explicit completion/correction rule. A newer native
position, interrupted action, combat, death, teleport, embark/disembark, unit
removal, fog loss, viewer change, save/load or reset supersedes it. Late movement messages start a visual segment when admitted; renderer frames
are sampled at current time instead of queued as delayed pictures. Rapid successive
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
The accepted target hook now sends the source/destination tile pair before
native animation starts. Native body captures supply identity, art and sprite
dimensions; their screen coordinates do not determine scene travel or idle
placement. Both use the copied tile center: the native target routine adds
`(+64,+32)` at normal zoom. This avoids pairing a body captured after native
camera recentering with a previous renderer camera. No per-unit camera binding
is retained.
**Native timing (October 8, user requirement).** A move must take exactly as
long as in native Civ III, from input to arrival. The rules:
- **Start.** A step starts at its move event's QPC time, the moment Civ III
  starts it, not at the first displayed sample. The worker reports the offset
  between QPC and the scene clock (`UnitInstances::native_clock`; live clock
  only, never during replay). A late first sample (transport, camera
  preparation) begins partway into the step, skipping at most a quarter of
  its travel.
- **Turn.** The facing turn (60–180 ms, `unit_pose_transition`) runs during
  travel. It no longer delays it.
- **Duration.** Travel lasts vanilla's constant-speed time, distance over the
  art's INI `Fast Speed`, which the move event copies from native
  `Animation_Info` (225 map units/second for every stock ground unit; 225 if an
  event carries none). Civ III's animator snaps the unit onto the tile on its
  next update and confirms the move after that. The measured overhead
  (confirmation minus start, minus travel; 90–160 ms in the VM) is smoothed
  per session and stretches later steps' easing, so they arrive when Civ III
  confirms them. A path then never pauses between tiles. The first step of a
  session arrives up to that overhead early.
- **Easing.** Travel accelerates over the first 12% and decelerates over the
  final 20% of the step, so cruise runs about 19% faster than constant
  speed. The authored run cycle follows distance at native speed, including
  the slowdown and any stretch.
- **Holds.** A frozen view (camera preparation without a usable completed
  scene) still holds travel, so no unseen distance is skipped. Afterwards
  the step runs at double speed until the held time is recovered, then
  arrives on native time.
- **Diagnostics.** At trace level 1 or higher, `unit-arrival` reports each
  displayed step: `start_lag_ms`, `travel_ms`, `end_vs_commit_ms`.

The catalog reads move timing from the 32-byte generic animation header during
asset loading; older bindings record duration/frames only for ambient clips.
Without the native clock (replay), a step starts at its first visible sample.
The bounded queue retains up to eight neighboring steps and carries the same
accumulated run distance across them. A queued step starts at its native start,
never before the previous step ends.
An endpoint waiting for native confirmation holds an idle pose with its travel
heading. A late continuation pauses the run phase for that wait, then starts
at the shared endpoint. Tests use a nonzero clock origin to detect an
accidental phase reset. The later `Unit_move` event confirms game state without truncating
active travel. Once both confirmation and displayed travel finish, the segment
ends even if native stack selection never sends that mover an idle body draw.
A remaining RUN presentation settles to the catalog's idle pose at the accepted
endpoint and yields to the latest native stack owner. The capture regression
covers this missing final draw; waiting for it left the attacker running in place. Hiding, retirement, incompatible actions and unrelated position
corrections clear the route. The selection ring uses the same sampled body.

Confirmed visible movement can use the copied destination anchor while the
completed terrain view still carries older sight bits. The same rule covers the
next leg starting there; arrival must not remove the body, cursor or HUD anchors.
The exception applies only to movement newer than that native capture; an
animated map clock is not a new sight observation. A newer fog capture, native
unit hiding and retirement still revoke admission. Both
linear and indexed wrapped-occurrence paths have executable handoff coverage.

Newly copied visibility increases wait behind the committed movement segments
whose native completion timestamps precede that capture's original timestamp. Visibility is admitted
before terrain preparation, separately from display sampling: rendering an older
completed view cannot erase that admission or attach it to a later queued move. The final GPU fog pass releases each pending
reveal on visual arrival, after terrain preparation; native loss of sight
applies immediately. The gate is bounded and clears across map/viewer scopes.
No native visibility rules or gameplay movement are changed.
Camera preparation services only already-waiting overlay commands, with an
eight-command/one-millisecond batch limit. It does not wait for a producer to
submit another overlay packet while the changed terrain is unfinished.
Scene-only unit facts and sight deltas can pass queued native canvas packets
in the asynchronous bridge and merge into ordered fact batches. This prevents
the first reveal from expiring behind repeated drawing packets. Facts, scope
changes and camera adoption retain their order; canvas-bound unit draws retain
image create/use dependencies. Every reliable canvas packet still executes.
Same-view reveal preparation starts immediately. The earlier 160 ms collection
window is removed now that the completed scene can animate during preparation.
Already-superseded camera tickets are rejected before they mutate view state.
At explicit camera import, the fresh sampler updates the completed image to the
current actor clock before publication. A terrain image prepared earlier cannot
temporarily restore an older pose or fog state while native overlay packets
finish arriving. Pending assets still preserve a complete image.

The October 5 retained-view diagnostic confirms animation frames execute inside
camera preparation, rather than holding a still image for the whole job. The
current Scout witness keeps both actors through 225 pose preparations and all
41 movement window samples. Its first and second terrain reveals are separate;
the first appears before the northbound leg completes, about 0.2 seconds after
the first displayed arrival. The final unprofiled `153835` repeat keeps the
Scout in all 43 window samples, with separate reveals at 76.056 and 76.814
seconds and no sampled terrain blackout. Its first reveal remains about 0.3
seconds late. Native sight becomes available after the native travel wait;
preparing its newly visible geometry still consumes part of the arrival window.
These 10 Hz samples do not certify every displayed frame, zero reveal latency
or a frame-rate guarantee. Candidate evidence is under
`Renderer/.cache/smooth-scenes/`.

This implements ordinary visible tile travel. Hidden-to-visible admission,
combat/transport transitions, route overlays and the full action lifecycle
still need their scoped live checks. Native status text/bars and civilization
markers retain their native drawing programs and pixel size. Their scoped
captures now carry a stable unit ID; the retained compositor attaches them to
the completed map's body center. Pending preparation cannot advance the HUD
ahead of the image, and a retired camera retains its last completed anchors.
Wrapped views select the nearest captured occurrence; absent bodies suppress
their retained ink. Passing travel-clock tests alone does not certify those
overlays; the GPU compositor regression covers movement, zoom and retirement.

## Surface and frame lifecycle

### Current asynchronous bridge

The fresh-scene handoff now uses Civ III's captured unit center instead of the
sandbox's fixed 191-pixel canvas center. The extracted production placement and
shader math pass 8,640 size/zoom/wrap/reflection cases. The final fresh output
also runs the existing GPU visibility pass on every publication target;
resource-free scenes still refresh visibility. The GPU oracle checks 8,787,552
pixels across zoom, scrolling offsets, reset and visible-body stencil coverage
(maximum channel error one). It checks depth occlusion, cutout holes, stencil
clearing and both single-sample and two-sample targets.

The borrowed-publication lifetime repair is covered by 780 completion operations;
only map/configuration commands retire adopted camera records. The direct native
presentation path samples the same current visual time as autonomous frames.
Passing zero frequency there previously restored capture-time animation poses
on each native update, causing periodic rewinds.

The white selected-unit ring now carries an explicit eligibility bit in copied
body observations. It uses that unit's sampled/reprojected anchor and the shared
visual clock, before unit mesh drawing. New selection clears the prior cursor.
The old retained screen-space ring operation is consumed without duplication.
The full bridge fixture includes the cursor flag to catch boundary validators
that would otherwise reject selected bodies.

Current qualified fixture `8c7918bb0f2b4c0d927660c82c9b4b88` measures 59.51 FPS,
32 camera changes and no CPU map readbacks. See the current-status entry for
real-game evidence and remaining limitations.

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
