# Conservative facade light selection

The scene-light field retains the existing receiver metric `(x, -y, z *
0.648266978876)`, attenuation, orientation/diffuse tests, emission gain and ray/box
occlusion equations. No light or building blocker is capped or dropped.

`native/city_fidelity/light_spatial_index.h` builds a CPU XY grid with 0.25-unit
cells. Each cell lists all lights whose conservative finite-range XY rectangle
intersects it. Each light lists every blocker intersecting its padded influence
cube, excluding its original owner. Neighboring-city blockers remain eligible.
False positives are intentional: the original sphere and slab tests still decide
contribution. Bounds include float rounding and the slab test's safe-ray epsilon.
Negative coordinates and independent wrap occurrences need no special branch;
reflected receivers use the same world coordinates and index.

## GPU layout and reuse

The first records of the existing `t127` float4 structured buffer are unchanged:
three per light, then two per blocker. Appended cell headers and per-light blocker
headers hold scalar offsets/counts into packed index float4 records. Each index is
an exactly representable integer float. A total 2^22-record resource bound keeps
scalar offsets below 2^24. Constants at `b6` grow from 48 to 80 bytes, retaining the
original counts/envelope followed by grid origin/scale/dimensions/header bases.
Light and blocker accumulation order is unchanged.

`SceneLights` compares complete copied bytes **and both counts**. Order, colors,
intensity, position/range/direction, global owner indices and every blocker bound
participate; raw object addresses do not. Identical fields reuse their GPU data
and lists across camera/time changes and object lifetimes. Night/emission values
update independently in the small constant buffer. Changed selection/content
rebuilds the field. Reset/device retirement releases all owned data.

Overlarge/unrepresentable grids, nonfinite spatial inputs, index allocation
failure and an enlarged-buffer allocation failure select the complete original
scan. The original field resource bound remains a failure through the existing
renderer boundary; it never truncates the field. Failed uploads invalidate the
content cache so a later request cannot bind a stale index. CPU field allocation
failure returns through that same established renderer boundary.

## Shader route and diagnostics

`scene_lights.py::upgrade` adapts both legacy regional sources and previously
flattened scene-sized sources. `prepare_shader.py` calls it during generation.
The selective `prepare_renderer64_materials.py ... city-lighting` route applies
it to all ten city receiver shaders in the Renderer64 control bundle, preserving
all other materials. The FRESH relight/reflection variants inherit the same code.
A zero index mode uses the complete original shader loops.

Set `C3X_RENDERER_CITY_LIGHT_FULL_SCAN=1` before process initialization for a
quality-preserving reference. `C3X_RENDERER_CITY_LIGHT_DIAGNOSTICS=1` prints field
counts, list sizes/maxima, allocation/upload bytes, build/upload CPU/driver spans
and bounded reuse records. Diagnostics are disabled by default. List maxima and
entries are candidate-work bounds, not measured GPU ray-test execution counts.

`test_light_spatial_index.py` runs CPU candidate/illumination and D3D equation
comparisons; `test_city_lighting.py` covers capacity, ownership, upload reuse,
content/count-layout mutation, daylight/empty and full-scan fallback. Both are
included in the Cities category's dependency-selected test set.

## Bounded qualification

`tools/measure_city_light_index.py` uses the existing standalone FRESH client and
its 90-step 1×→1.25×→1× trace, preserving source clock, viewport, full effects and
`Present(1,0)`. It emits individual warmed call spans after the timed loop and
keeps preparation, priming and first transitions separately. It also compares
noon/no-city controls and a supported richer city-recipe fixture. This is a
changing-zoom witness, not the defective standalone pan path or a many-unit/live
Civ III qualification. It cannot establish the complete 60 FPS goal.

Step evidence is under `Renderer/.cache/city-light-index-step/`. An earlier ignored
build directory disappeared during verification; its cause is unconfirmed. Lost
snapshots were reconstructed against the auditor's preserved original source
hashes; its preserved original DLL is the control. The interrupted observation
is excluded. `recovery.json` records the limitation; new paired receipts use fully
fingerprinted pack inputs, binaries, shader bundles and raw logs. Fixed visual
reference images remain unchanged.
