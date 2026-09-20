# Direct unit scene execution

Milestone 2.2 extends the existing worker, unit instances, pose preparation and
retained native composition. It does not replace Civ III's action director,
visibility authority, screen anchors or presenter.

## Production path

Milestone 2.3 follows the user's final composition decision: units draw above all
map geometry, including a mountain or forest on a neighboring foreground tile.
Native unit-to-unit and HUD/UI ordering remain authoritative. Terrain occlusion
was explicitly withdrawn; tile coordinates do not determine unit/map ordering.

The GPU owner draws selected posed geometry into a transparent bounded attachment.
It needs no raw map color/depth, source-generation lease or provenance transported
through native image copies. Those superseded owners and the 64 MiB map-region
capture cache have been removed. The existing compositor combines the contribution
with the current native/full-color underlay in native command order.
There is one shared working color/depth attachment, capped at 96 MiB and preserving
each model's sample scale and MSAA. Oversized native canvases use the existing
resident GPU compatibility path; this includes enlarged zoom when required.

The unit pass clears and rasterizes only the selected footprint, preserving the
original native raster viewport. Moving that viewport to a cropped origin failed
exact full-color parity by one channel level on the VM; it is not used. Extraction,
hardware resolve and native composition cover the footprint rather than the erase
canvas. Scene conversion scratch grows within native bounds and is separate from
compatibility scratch, avoiding allocation churn when the routes alternate.
A sample-preserving extraction uses hardware MSAA resolve. Shader-only resolve failed exact parity on the VM and is
not the production path. The final native-boundary pass combines the existing
pose-local shadow directly into the resident native/full-color pair. It produces
no CPU pose or composed-unit upload and submits no static terrain geometry.
The map's finished full-color underlay remains authoritative for fog and native
composition. Hidden unit inputs never enter either draw route.

The existing unit pose owner also retains GPU-ready vertex buffers and exact
R32 shadow inputs under the existing pose/projection/light/color key. Its 192 MiB
allowance includes retained CPU content leases, vertices, shadows and optional
cropped body contributions. The always-on-top policy makes body pixels independent
of the map. A matching pose can reuse those pixels when their native-canvas bounds
contain the requested footprint; otherwise it rasterizes normally. Native packing,
pose-local ground shadow and underlay composition still execute separately for each
occurrence. This is attached to the existing pose-content owner, not another cache
or a retained native erase canvas. Admission cannot exceed the same 192 MiB cap;
pressure or allocation failure retains the ordinary direct draw. Reset/pack changes
clear every representation. Matching allocations recycle only after their old
identity retires, with all GPU mutation serialized on the existing owner.

Retained replay collects every reachable direct sample once before rendering the
map or joining a pose job. Missing current poses enter the existing bounded CPU
pool as required work, ahead of predictions. GPU execution then follows the original
native order. Unit IDs and visibility revisions still authorize every occurrence;
shared body pixels never authorize a hidden or retired unit.

Unit self-depth uses its normal full range, independent of map depth. Existing
pose-local self shadow and projected ground footprint retain their native
composition. Units do not receive surrounding world shadows or cast onto arbitrary
map geometry; no new receiver model is claimed. Body materials, authored timing,
owner colors, anchors, visibility rules and quality controls remain unchanged.

## Retained frames and compatibility

A retained unit operation owns copied selection and draw inputs, not game pointers.
It samples the existing revisioned unit owner once per visual frame. A changed pose
or changed underlay executes the operation; unchanged dependencies reuse completed
composition. Native opaque writes retire covered operations. Directed/frozen poses
follow captured native cursors; only eligible authored ambient loops advance.
Retired selections cannot authorize a new draw.

Oversized working attachments or direct-execution admission failure use the
established bounded GPU pose cache and exact native composition. Cached coverage
avoids recompiling geometry merely to decide eligibility. This path remains
necessary while native surfaces and ordered overlays participate. CPU access barriers retain explicit CPU
3D delivery. Configuration-off and UI portraits retain their established behavior.
`C3X_RENDERER_UNIT_SCENE_CONTROL=1` selects the preserved finished-resident-pose
control for comparisons; it is not a required player setting.

`unit-scene` reports direct draws, GPU-input builds/hits/allocation reuse, compatibility
builds/hits, working-set rejections, charged working and GPU-input bytes (region bytes are
zero; `map_draws` is the legacy trace name for direct body draws), finished-pose
bytes, body readbacks and composition uploads. The standalone
frame harness uses a bounded 32 MiB trace and reports dropped lines explicitly;
`C3X_RENDERER_TRACE_MIB` overrides diagnostic capacity within 8–64 MiB. Complete workloads include map rendering, units, native UI
and final transfer; isolated counts do not establish a speedup. Validation results
and the next unfinished responsibility belong in the current roadmap handoff.
