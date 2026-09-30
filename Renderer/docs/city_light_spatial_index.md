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

## Current result

The conservative index removes most of the additional nighttime zoom cost in
this fixture. Two uninterrupted paired runs reduce warmed mean total call time
by 75.7% and 77.3%. Noon and no-city means remain around 130–136 ms. The scene
still misses the 16.67 ms target on every measured nighttime frame; this step
does not establish sustained 60 FPS or qualify live scrolling and many units.

Each row contains 87 warmed frames, excluding the first three transitions.
Values are milliseconds, including `Present` and resulting GPU backpressure:

| Workload / pair | Arm | Mean | p50 | p95 | Worst | Missed 16.67 ms |
|---|---|---:|---:|---:|---:|---:|
| Night / 1 | Original | 610.04 | 616.65 | 1283.66 | 1933.89 | 87 |
| Night / 1 | Indexed | 148.04 | 133.93 | 299.77 | 547.64 | 87 |
| Night / 3 | Original | 631.58 | 43.40 | 1407.18 | 1860.65 | 87 |
| Night / 3 | Indexed | 143.63 | 149.37 | 230.69 | 455.17 | 87 |
| Rich night / 1 | Original | 634.30 | 611.53 | 1327.55 | 2345.74 | 87 |
| Rich night / 1 | Indexed | 143.15 | 148.57 | 268.92 | 397.37 | 87 |
| Noon / 1 | Original | 133.98 | 133.33 | 151.62 | 388.10 | 87 |
| Noon / 1 | Indexed | 132.04 | 133.28 | 150.45 | 423.31 | 87 |
| Noon / 2 | Original | 135.99 | 133.31 | 249.57 | 316.45 | 87 |
| Noon / 2 | Indexed | 133.40 | 133.32 | 261.13 | 393.58 | 87 |
| No city / 1 | Original | 129.79 | 133.26 | 150.55 | 351.71 | 86 |
| No city / 1 | Indexed | 130.83 | 133.18 | 151.56 | 417.76 | 85 |
| No city / 2 | Original | 131.43 | 133.36 | 153.29 | 286.01 | 86 |
| No city / 2 | Indexed | 134.21 | 133.34 | 245.42 | 343.87 | 86 |

Original night pair 3 alternates short calls with long waits, making its median
misleading. The full warmed trace falls from 54.95 s to 12.50 s; pair 1 falls
from 53.07 s to 12.88 s. Raw draw/present samples and descriptive p99 are retained
in `timings-summary.json`; these CPU/driver spans are not GPU timestamp results
or scanout latency. Supplemental pair 2 crossed a VM suspension and is marked
separately: original mean 624.74 ms, indexed 140.28 ms. It is not needed for the
two uninterrupted pairs above.

Cold preparation and transitions remain material costs. Pair 1 original/indexed
preparation is 27.08/42.99 s and priming 34.28/37.28 s. Its first three indexed
calls are 5918.92, 432.57 and 48.27 ms. With warmed shader caches, pair 3
preparation is 26.29/19.46 s, priming 2.45/2.56 s, and first three calls are
17.89/200.04, 97.15/371.27 and 98.64/85.32 ms (original/indexed). These values
remain separate from the steady trace; the optimization does not remove cold
compilation, upload or first-zoom stalls.

The developed field retains all 620 lights and 222 blockers. Its 8,400 grid
cells contain 8,079 light entries; per-light lists contain 4,795 blocker entries,
with maxima of 49 lights per cell and 25 blockers per light. The original
36,864-byte field gains 195,824 index bytes. Main-field build spans are
1.149–2.134 ms and upload spans 0.030–0.087 ms. The larger preparation field
(1,641 lights, 591 blockers) uses 1,000,800 index bytes and leaves a 2 MiB GPU
buffer high-water allocation. Main selection uploads once after preparation and
reuses data throughout the trace. These counts bound candidate work; they do
not measure actual per-pixel light/ray invocations. The richer supported recipe
has 622 lights, 222 blockers, 8,430/5,228 entries and maxima 50/29. It exercises
a second recipe at the same six legal city sites, not a substantially denser
city-count stress case. Unit preview contains only four synthetic actors.

## Correctness, identity and staging

CPU differential coverage checks 19,465 deterministic/random cases, including
range/grid boundaries, negative and independent wrapped positions, neighboring
city blockers, owner exclusion and fallback. D3D runs the actual generated
local-light equations over 8,192 receivers at night, dusk and day: maximum
indexed/full-scan error is zero. Upload tests cover identical bytes, same-pointer
mutation, changed count layout, new object lifetime, empty/day and full-scan
fallback. Adapter tests cover all ten shader routes and idempotence.

Full-effect 2240×1260 captures cover noon/dusk/night at 1× and 1.25×, no-city and
the richer recipe. Developed and rich night comparisons differ on 74–154 pixels,
with mean channel differences below 0.000051 on the 0–255 scale. Dusk differs
on 112–128 pixels. Complete PNGs and amplified difference sheets are retained.
The first 1× noon comparison differs on 39,938 pixels; an original/original
repeat reproduces 39,918 of this scale of difference. Warmed original/indexed
noon differs on 125 pixels. Thus capture/startup variability is established,
but full-scene bit equality is not claimed. The shader irradiance comparison
remains exact. Visual inspection shows preserved city illumination and shadows.

The focused correctness/facade runs pass. The Cities dispatcher resolves to
157 passing tests after rerunning one Parallels transport failure, one optional
skip, and one unresolved historical provenance input:
`Renderer/lab/out/cities/integration/frames.json` (expected SHA-256
`cc6413f7361051c31949a10bc0744a81243b5f7cc7a075914075ece131196661`).
The runtime pack's different manifest cannot replace this missing original
source artifact. Category-definition validation passes all 28 categories.

Qualification uses Windows 11 build 26200, eight Apple Silicon virtual logical
processors, 14 GiB VM RAM, Parallels WDDM driver 20.18.2702.58673 and D3D feature
level 0xb000. Viewport is 2240×1260, scene samples 1, default reflection scale
0.375, all normal effects enabled and `Present(1,0)`. Both arms use the same
diagnostic client and source-time trace. The complete pack/definition/scene
superset remains identical before/after qualification: 47,224 files,
12,238,654,673 bytes, manifest fingerprint
`d5d272ea7ffff8249029193aa4eda32b9cdc044f80994866baa89f92077d9bdf`.
Receipts include every consumed shader source route and all binary hashes.

The measured candidate x64 DLL is
`ec7c439db5d802dbfe79eb25c19c8192567c50c4d1ae87c4e9853aa918709b9a`;
the preserved original is
`bafb4e85359286ac61957e0e8d513fdbce70ca3be4609d58c12fccb03a411821`.
After confirming Civ III closed, the candidate was staged with matching current
bridge/helper and the ten exact evaluated shader changes. The full 86-source
control shader bundle matches the evaluated candidate. Previous binaries and
shaders remain in `pre-stage/`; `stage-evidence.json` and startup-probe receipts
record hashes and health. No injected source, patch symbol, game installation
or fixed visual reference was changed by this step.
