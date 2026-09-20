# Dynamic scene input contract

Milestone 2.1 extends the existing capture, unit instance and native composition
owners. It adds no gameplay scheduler or second visibility authority.

`render_core/dynamic_scene_input.h` owns immutable map-effect records. Each copy
contains the authoritative ordered tile occurrences, resource IDs/classes/names,
visibility states, native anchors, world topology/revision, hour/season, captured
presentation time/frequency and map/viewer/visibility/scene epochs. No pointer into
Civ III or the caller's capture survives. Sampling returns a frame borrowing only
its retained const owner and changes presentation time in that local frame.
Unknown or hidden objects cannot become selected merely because content is cached.

`UnitInstances` remains the unit/action owner. It copies unit identity/key, action,
direction, native cursor/count, selected/captured state, appearance, environment,
clock and the occurrence's projection/anchor. Selections carry a content revision;
changing content, despawning, eviction and catalog/reset invalidation retire old
selections. Moving a projection does not rebuild unit content. Only eligible
source-authored ambient loops advance on the visual clock. Directed actions keep
their authoritative captured cursor and anchor; visual sampling drives no gameplay.

Authoritatively hidden units retire their instance selection and queued pixels,
return empty body bounds, and contribute no body, shadow or native map HUD.
Revealing a unit requires a fresh native capture; cached content grants no visibility.
Map motion is selected from current immutable visibility records, independently
of cached geometry. Unexplored animated resources/effects are omitted. Explored but
not currently visible resources use a deterministic seeded still pose; optional
shoreline waves use time zero. These samples stay exact through time advancement
and reset/reload, and do not request continuous redraw. Visible records resume
authored motion. This freezes appearance, not authoritative game state.

The existing worker/caller gate serializes admission and configuration. A new
capture does not invalidate an already displayed native front: the compositor owns
its immutable inputs until that source is replaced/released. Configuration/reset
invalidates sampling of all old map records, without forgetting their allocated
bytes before release. Epoch exhaustion rejects further capture. The native
publication owners still prove camera/view eligibility; this contract is not the
milestone-3 general nonblocking publication change.

M3.1 separates prospective authority from these displayed-frame leases.
`render_core/scene_publication.h` copies accepted native map captures into a
coalesced journal before their view requests enter the worker. Its monotonic
sequence, configuration generation, map/viewer/visibility identity, immutable
topology, environment and time survive camera replacement. Pending tile updates
are keyed by canonical world identity; a newer disjoint view cannot discard an
earlier edit. Full records can remove cities/resources. Lightweight halos update
visibility without removing omitted object fields. Retained geometry revisions
change only when normalized appearance changes; visibility has a separate revision.

The render owner adopts the journal between jobs, before selecting further work.
It is the only writer of persistent world records and their mesh handles. Older
displayed inputs can still supply their own observations but cannot reverse newer
authority or attach obsolete meshes. Existing unit capture/retirement remains the
unit lifecycle authority; map/viewer/world-basis replacement also retires unit
and dynamic-map selections. Configuration/reset advances the publication scope.

The journal has a separate **16 MiB** budget including its transactional staging
copy and conservative record/control overhead. It keeps at most the latest update
per canonical tile and shares unchanged topology metadata. Exact copied tile
inputs are also shared across unchanged captures; anchors are excluded and wrapped
coordinates canonicalized for this authority comparison. Time/identity still advance, without allocating or
adopting the same tile journal again. The last adopted batch remains available
for device-reset recovery until different tile inputs replace it; pending changes
are never cleared by that replacement. This is a bounded recent-capture lease,
not an accumulating second world or event log. Capture rejection leaves accepted updates
intact. Failed worker adoption retains the journal for retry after a later capture
and prevents output publication; it neither spins nor terminates the worker.
`scene-publication` reports sequence, configuration, adoption/change state,
unchanged-tile reuse and tracked bytes. Whole-process address-space sampling
remains the combined check.
Neither this journal nor an old displayed lease grants new view/visibility
eligibility. The existing synchronous native camera barrier is still required.

Map inputs share a **16 MiB** admission budget, including vector capacity and
record/control storage allowances. The maximum individual input is 8,192 tile
records and 1,048,576 topology cells. Existing unit metadata remains bounded to
4,096 instances; geometry, poses and composition retain their separate budgets.
Failed admission keeps the last coherent native-composition image rather than
sampling partial inputs. `dynamic-inputs` reports admitted bytes, peak, captures,
rejections and live unit count; sampled whole-process VA remains the combined
memory check. No native redraw is requested to recover this optional animation.

Clock sampling rejects invalid frequencies, negative ticks and overflow. It uses
the captured time plus elapsed renderer time from a fixed origin, avoiding drift
from repeatedly mutating the snapshot. A backward clock does not extrapolate
before the captured sample. Tests mutate/free caller arrays, change resources and
visibility, replace fronts, reset owners, exhaust budgets, and exercise authored
unit loops, directed/frozen actions, despawn and reuse.

The production retained map callback consumes these const records instead of
mutable prospective-view captures. Direct unit execution and composition are
implemented through 2.3; units retain the user's always-on-top map ordering.
Milestone 2.4 now connects shoreline ribbons to the shared dynamic scene pass:
visible waves use the captured clock, explored ribbons keep time zero, and hidden
ribbons are omitted. The immutable map owner and native publication still prove
which view can be sampled. General scheduling/reuse remains 2.6, tactical overlays
2.5, and milestone acceptance 2.7.
