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
