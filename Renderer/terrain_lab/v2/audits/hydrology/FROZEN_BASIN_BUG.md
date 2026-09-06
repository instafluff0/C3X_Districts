# Pale rectangular land patches: isolated-coast support bug

Q2 reported and Q3 directly inspected
`audits/terrain/out/Q1/Q2-terrain/composed-complete-wet-r08/h12-z1-pan00.png`
(relative to Lab v2). The large pale patch with a raised neighboring lip is
consistent with an exact support-domain defect in the frozen shoreline field.

In `shared/frozen_scene.cpp`, `biq_signed_shore_distance` loops over nearby
isolated water tiles. It computes `basin = 1 - smoothstep(distance/.46)` and
unconditionally applies `result = max(result, basin)`. **Outside the basin,
basin is zero, not negative infinity.** Negative land coverage is therefore
clamped to zero anywhere that isolated tile appears in the candidate loop.
Candidate membership changes at `floor(world_x/y)`, producing straight tile
boundaries. Zero shore distance then forces `biq_coastal_relief_envelope` to
zero, which explains the lowered plate and raised neighbor edge.

The verified q2-wet fixture has a marsh at local `(3,1)`, raw `(87,41)`, and an
isolated halo coast at `(4,1)`, raw `(88,42)`. At the marsh center, frozen corner
coordinates are `(3.5,1.5)`. Distance to the coast center `(4.5,1.5)` is 1.0,
so basin is exactly zero. The loop includes that coast and replaces negative
land coverage with zero despite being outside the basin's support. The same
mechanism can affect other land families; marsh identity is incidental.

Minimal mathematical repair is to apply the max only inside nonzero basin
support. Q3 does not edit frozen or v1 sources. Its new field already evaluates
a continuous signed-distance contour without the zero-outside-support max.
`systems/hydrology/scene_adapter.h` supplies an opt-in replacement adapter.
Request to Q0: add a null-by-default `HydrologyHooksV1` with `initialize(csv)` and
`signed_shore_distance(world_x,world_y)`, expose module `hydrology_hooks`, and
call it at the beginning of frozen `biq_signed_shore_distance`. The adapter
explicitly converts frozen corner lattice to Q3 center lattice and reverses
sign to water-positive normalized coverage. Default frozen output remains v1.
Q2 can keep its independent material weights/UV hooks active simultaneously.

This is a geometry/shore-mask defect, not an excuse to blur terrain material
boundaries or paint over the plate. Source-composed after evidence now confirms the repair: `COMPOSED_REGRESSION.json`
pins Q2's matched `wet-linear-on-r12` before and `wet-hydro-on-r13` after.
Direct inspection shows the pale rectangle and lip removed with continuous
grass/marsh and unchanged source vegetation. Broader water/shore material integration still
uses the full Q3 sample/class contract rather than this minimum compatibility
hook alone.

Q2's preserved ownership-debug view identifies the patch as LAND pass 1. This
is compatible with the cause: the frozen land shader's integrated-shore branch
uses `input.shore_distance` and `smoothstep(-.58,.10, signed_shore + noise)` to
blend beach color **inside the land pass**. A zeroed field therefore turns land
pale without changing `surface_kind`. Q2's smooth material weights likewise do
not rule out the later shoreline blend. Q0 has now published the opt-in callback;
Q2's exact matched wet regression passed direct image inspection using the
provided adapter. Bounded phase/zoom/neighbor checks remain Q2's follow-up.
