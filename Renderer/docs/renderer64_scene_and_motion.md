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
