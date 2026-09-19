# Direct unit scene execution

Milestone 2.2 extends the existing worker, unit instances, pose preparation and
retained native composition. It does not replace Civ III's action director,
visibility authority, screen anchors or presenter.

## Production path

A published map carries an optional renderer-owned source for its raw HDR color
and depth. Native map/detail copies propagate a bounded rectangular proof of that
source and its coordinate translation. Other native drawing cuts the affected
proof. A unit is eligible only when the clipped, conservative geometry-and-shadow
coverage still belongs to one map source and coordinate translation. Empty canvas
space does not remove provenance; the native erase canvas and clipping stay intact. Aliases and partial transfers use the before-image.
Unknown provenance selects the resident compatibility route, never speculative
replacement or a native 2D body.

The GPU owner captures only these required raw map rectangles while their exact source
render generation remains valid. Captured color/depth survives subsequent camera
preparation; a stale generation cannot produce a new capture. These optional
inputs share a 64 MiB cache. Reclamation retires the raw attachment and reuses a
matching allocation without discarding any completed native-front pixels. A later
use must recapture from a valid generation or take the compatibility route.
There is one shared working color/depth attachment, capped at 96 MiB and
preserving each model's existing sample scale and MSAA. Oversized working sets
or failed optional attachment admission use the compatibility path. No second full HDR viewport is retained.

The unit pass restores those map samples at their native pixel offset and draws
the prepared geometry into the working scene attachment. A sample-preserving
extraction uses bounded conversion scratch and hardware MSAA resolve. Shader-only resolve failed exact parity on the VM and is
not the production path. The final native-boundary pass combines the existing
pose-local shadow directly into the resident native/full-color pair. It does not produce a finished CPU pose, cache a finished
GPU pose, upload a composed unit image, or submit static terrain geometry.
The map's finished full-color underlay remains authoritative for fog and native
composition. Hidden unit inputs never enter either draw route.

The current native painter-order rule is explicit: raw map depth occupies the far
band and unit self-depth the near band. This preserves the preexisting body order;
it is not shared world-space terrain/unit occlusion. Shadow receivers, overlapping
unit composition and terrain/unit occlusion are milestone 2.3. The body materials,
authored timing, owner colors, anchors and quality controls remain unchanged.

Known visible limitation: units behind mountains or other tall foreground objects
can appear on top (user report, 2026-09-19). Milestone 2.3 must replace the separate
depth bands with compatible scene-depth occlusion and test front/behind and partial
overlap through movement, animation and camera changes. Preserving 2.2's pixels
must not turn this incorrect ordering into a permanent compatibility requirement.

## Retained frames and compatibility

A retained unit operation owns copied selection and draw inputs, not game pointers.
It samples the existing revisioned unit owner once per visual frame. A changed pose
or changed underlay executes the operation; unchanged dependencies reuse completed
composition. Native opaque writes retire covered operations. Directed/frozen poses
follow captured native cursors; only eligible authored ambient loops advance.
Retired selections cannot authorize a new draw.

Intervening native drawing, overlap, unavailable raw samples or admission failure
uses the established bounded GPU pose cache and exact native composition. Cached
coverage avoids recompiling geometry merely to decide eligibility. This path remains necessary while native
surfaces and ordered overlays participate. CPU access barriers retain explicit CPU
3D delivery. Configuration-off and UI portraits retain their established behavior.
`C3X_RENDERER_UNIT_SCENE_CONTROL=1` selects the preserved finished-resident-pose
control for comparisons; it is not a required player setting.

`unit-scene` reports direct map draws, compatibility cache builds/hits, region captures,
rejections/evictions, charged region bytes, finished-pose bytes, body readbacks and
composition uploads. Complete workloads include map rendering, units, native UI
and final transfer; isolated counts do not establish a speedup. Validation results
and the next unfinished responsibility belong in the current roadmap handoff.
