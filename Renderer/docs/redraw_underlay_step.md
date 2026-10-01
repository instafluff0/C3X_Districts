# Terrain underlay step

Bounded assignment complete; **no production change is promoted**. A private
constant-color underlay probe saves 65–67 ms during continuous zoom, establishing
a large shading opportunity. It visibly damages coast/river composition. The one
conservative stencil/depth candidate preserves the sampled output closely but is
3.322 ms slower in its reversed primary comparison. Stop this implementation here;
the result does not rule out a different future method.

Accepted C7 runtime sources, accepted inputs and the staged game tuple remain
unchanged. The implementation is isolated in `Renderer/tools/` and private source
snapshots. No staging, injection, installation or reference replacement occurs.

## Active entry and original contract

`sandbox/fresh_pipeline.h` loads city-profile `hydrology.hlsl`, appends
`PSSandboxUnderlay`, compiles that entry with optimization level 3, installs it as
`visual.underlay`, and binds it for nonmirrored `geometry_underlay`. `draw_scene`
draws that layer before land, natural terrain, mountains and decals. Its entry
sets `surface_kind=0.5` and returns `PSMain(input).color`. The admitted geometry
uses the original VS, rasterizer, viewport/scissor and premultiplied blend state,
with depth writes enabled and `LESS_EQUAL` comparison.

The active game-renderer definitions/settings and early exits were inspected:
the underlay's normal path returns alpha one; debug/only and panel early returns
relevant to this surface also return one. Water, route, shadow and thin-surface
coverage branches require other surface kinds. `q6_scene_output` clips at
alpha minus 0.000001; this does not discard the admitted alpha-one underlay.
There is no pixel-depth output. The private entry therefore returns an opaque
magenta or green constant without changing original coverage, depth or geometry.
Shared `PSMain` and every other entry retain their original source.

Reconstruction of the exact appended/patched shader and its FNV source/entry key
matches actual compiled blobs: normal underlay is 82,036 bytes; each constant
entry is 936 bytes. Their source and blob hashes are in `shader-proof.json` and
the blobs are preserved. Both candidate coverage entries similarly match their
actual compiled blobs in `candidate-shader-proof.json`. This proves shader
identity; it does not measure GPU shader duration or invocation counts.

Separate six 90-frame noon/night diagnostic fixtures keep all 10,528 pass rows
per arm exact, including draw/triangle/acceptance/upload/target/copy/rebuild fields.
All 2,269 owner identities and layer bytes agree. Main underlay averages 629.920
draws and 383,243.770 triangles across frames 3–89. There are no water-visible
underlay rows in these fixtures; repeated underlay shading in that pass is not
claimed.

## Complete-frame measurements

Frozen full-density C7 and the same client use actual 2240×1260/60 Hz full-guest
presentation with `Present(1)`, native128 capture and unchanged anchors/features.
The fixture has 1,983 records: 759 RENDER, 752 PREFETCH, 472 topology. Camera stays
at 0,0; zoom performs two 90-frame 1→1.25→1 cycles. Source time is explicitly
29533 + frame×33 ms. Primary counters, traces, profiling, completion probes,
timestamp queries and readbacks are off; counted/captured runs are separate.

At each diagnostic hour, normal/probe then probe/normal each keep all 180 frames.
The candidate has its own noon control/candidate then candidate/control cohort.
Warmed statistics exclude only frames 0–2 of each run, leaving 354 samples per
arm. All outliers remain. CPU-observed complete-frame milliseconds:

| Cohort / hour / arm | Warmed mean | p50 | p95 | p99 | Worst | All-360-frame mean |
|---|---:|---:|---:|---:|---:|---:|
| Diagnostic / noon / normal | 141.710 | 140.475 | 147.826 | 317.727 | 440.073 | 140.315 |
| Diagnostic / noon / magenta | 77.086 | 76.563 | 82.906 | 155.430 | 165.138 | 77.577 |
| Diagnostic / night / normal | 152.212 | 150.291 | 164.621 | 345.858 | 441.208 | 150.883 |
| Diagnostic / night / magenta | 85.235 | 84.517 | 95.764 | 171.340 | 185.232 | 85.149 |
| Candidate / noon / control | 141.254 | 140.774 | 148.016 | 250.710 | 329.088 | 140.059 |
| Candidate / noon / candidate | 144.576 | 144.380 | 153.131 | 299.193 | 350.291 | 143.698 |

The diagnostic saves 64.624 ms / 45.60% noon and 66.977 ms / 44.00% night.
It replaces shading on both visible and hidden pixels: these savings are not a
prediction of recoverable hidden-only work. Even the invalid output averages
60.416 / 68.565 ms above the 16.67 ms target. Every warmed normal frame misses
that target; magenta misses 353/354 noon and 354/354 night.

The candidate increases warmed frame time by 3.322 ms / 2.35%. Its two reversed
blocks are 2.649 and 3.995 ms slower; its all-frame mean increases 3.639 ms.
All 354 warmed frames miss 16.67 ms in both arms. Control/candidate remain
124.584 / 127.906 ms above the target, about 7 FPS during continuous zoom.
These cohorts are separate from earlier assignments and are not pooled with them.

## One conservative implementation

The candidate uses the existing single-sample D24S8 target. First it reproduces
underlay depth with the original geometry and VS, no PS/color writes. Next it marks
stencil only where replacement natural terrain/mountains pass the original depth
comparison and conservative full-opacity coverage. Finally it draws the original
underlay shader where stencil is zero, then retains the original final material
and later-layer order. Coverage passes do not write depth.

Ground coverage retains the original coast alpha and texture-modulated edge
expression, additionally requiring the inland field and resulting alpha to reach
one. Other ground material families remain uncertain. Mountains use their original
coast alpha expression and require one. Uncertain/partial pixels retain original
underlay shading. The focused D3D WARP test checks 24 combinations of transparent,
partial, nearly opaque and opaque replacement alpha; nearer, equal and farther
depth; and a later coplanar overlay. Exact color and D24 agree with the original
path; direct S8 checks require the expected opaque mask and no uncertain/depth-failed
marks, preventing an inert mask from satisfying the comparison.

The path is limited to the tested city profile at one sample; other profiles/sample
modes, mirrored/cached terrain, and failed state creation retain the original path.
The mask is cleared and recomputed per applicable draw, with existing record
selection, transforms and raster state. No new retained geometry owner, texture,
target or buffer is allocated. Three device-local COM states and two pixel shaders
are added; their driver allocation cost is not a measured physical-memory bound.

Separate 23-frame early fixtures, averaging frames 3–22, include the added work:

| All-pass work / frame | Original path in candidate DLL | Candidate |
|---|---:|---:|
| Triangles | 10,625,341.45 | 13,603,467.05 |
| Draws | 5,609.40 | 6,777.05 |
| Upload bytes | 2,344,027.2 | 2,642,945.6 |
| Logical target/clear pixels | 28,734,772 | 36,356,212 |
| Logical copy pixels | 3,659,240 | 3,659,240 |
| Rebuild / reuse | 2 / 1 | 2 / 1 |

This adds 2,978,125.60 triangles, 1,167.65 draws, 298,918.4 upload bytes and
7,621,440 logical stencil-clear pixels. The source also clears stencil for the
water-visible scene call even though its admitted underlay geometry is empty.
These are submitted work/footprints, not GPU-time measurements. The experiment
does not establish whether coverage-mask effectiveness, depth/stencil execution
behavior, added geometry, or another factor dominates the lost benefit.

## Output, adoption and stopping point

Full displayed color and full padded D24 captures are compared at the same source
clock. Noon normal/private-normal differs at 101 pixels, four over eight channel
levels (mean RGB error 0.000045, maximum 17). Normal/magenta and normal/green differ
at 140,029 / 137,113 pixels; 119,297 / 110,086 exceed eight levels. Magenta/green
differ at 134,994 pixels. Night normal/magenta differs at 139,183 pixels, 121,935
over eight levels. D24 is exact across all diagnostic arms. Inspected contact sheets
show the underlay's surviving contribution concentrated at coast/river/lake edges.
The diagnostic colors are visibly invalid and are never a production candidate.

The early candidate/reference endpoint is zoom 1.122222 at source time 30259 ms.
It differs at 81 displayed pixels, one over eight levels, maximum nine and mean RGB
error 0.000031; full D24 is exact. Its contact sheet shows closely matching coast,
river and mountain composition. This is a sampled check, not general visual approval.
Final S8 being zero does not measure the earlier coverage mask: retained depth
restoration copies D24 through a shader before copying to the captured glow target.

Counted adoption keeps cache bytes 407,725,211, unique owned GPU bytes 338,461,196,
retired/selection charge 156,392, and built/reused owners 63/1,040 exact. Candidate
primary first-view means are control 3,083.620 ms and candidate 3,362.956 ms;
geometry preparation means are 2,343.315 / 2,361.166 ms. Initial/adoption receipts
are retained separately from camera-frame timings. No cold-start improvement is
established, including uncached shader compilation.

The failed complete-frame benefit stops this candidate before night/pan/fixed-zoom,
wrap/new-coast, normal-speed transition, pressure or live-game qualification.
No second design or tolerance sweep follows. Actor animation, water/reflection,
ownership and config-off behavior are not newly certified by this private result.
The existing four-actor fixture cannot certify populated-unit/live-game 60 FPS.

The local 0 A.D. source at `0ed48b3a1fb1b4b718a78869fa497185af55e086`,
`source/renderer/TerrainRenderer.cpp:374–383`, draws selected bases, blends and
decals. It grounds distinct layer responsibilities, not this stencil prepass or
the claim that every base layer is redundant.

## Evidence and reproduction

`Renderer/.cache/redraw-underlay-step/` retains 2,160 raw primary camera-frame rows,
complete/warmed distributions and draw/Present phases, all 20 run receipts, separate
counts/owner CSVs, 16 lossless color/depth archives, inspected contact sheets,
source snapshots, compiled shader evidence, binaries, successful build/test logs
and hashes. A receipt-only correction records actual candidate shader-root hashes;
the inherited navigation receipt had listed the accepted root despite the supplied
candidate root. Runtime inputs and measured results did not change.

Verification matches all 47,224 accepted input files (12,238,666,273 bytes), 193
accepted runtime sources, 194 private and 195 candidate snapshot files, actual
shader roots, four binaries and the staged tuple. All 36 new image archives passed
compressed/decompressed hash verification. Only this step's 20 unused automatic
Legacy initial images and 12 reproducible compiler objects were discarded afterward,
with hashes/reasons retained. Prior protected evidence and concurrent seasons work
are untouched. The VM confirms no fixture/compiler/Civ III process remains.

Reproduction entry point: `python3 -m Renderer.tools.measure_redraw_underlay`, with
`prepare`, `build`, `run --label ... --arm normal|private-normal|magenta|green`,
`prepare-candidate`, `build-candidate`, and `run --arm candidate|candidate-reference`.
Use `--hour 12|1 --frames 180`; `--counts --captures` selects separate diagnostics.
Existing evidence/run directories are preserved. Primary timing must be sequential.
Focused test: `python3 -m unittest Renderer.native.test_underlay_occlusion -v` on
the approved Windows native-test path.

## Independent review and bounded follow-up

Independent review reproduces all 2,160 raw primary frame rows, complete/warmed
distributions, display/source-clock inputs and timing options. All 823 evidence
hashes, 193 runtime, 194 private and 195 candidate source identities, four binaries,
five public review files and both actual shader roots match. Six independently
computed full color/D24 comparisons reproduce the report. All 2,269 CSV rows are
compared with every field and duplicate multiplicity preserved; the six diagnostic
runs' 10,528 pass rows match their controls. Candidate extra-work totals reproduce.
The D3D WARP test passes on rerun, including all 24 cases and direct mask checks.
Diagnostic and candidate contact sheets were inspected. The receipt is
`.cache/redraw-underlay-step/auditor-review.json`.

The large underlay cost and the candidate regression are accepted measurements.
The explanation of the regression remains open: the actual full-scene S8 mask
immediately before shaded underlay submission was never captured, and reduced
pixel-shader execution was never established. Synthetic WARP correctness does
not establish useful mask coverage on the Parallels hardware path. Final output
differences also do not measure underlay overdraw or shader invocations. Keep the
candidate unpromoted, but close this specific gap before abandoning a 65–67 ms
shading opportunity.

**Next assignment:** inspect the existing candidate's real mask at the point of
use in a separate diagnostic; quantify marked pixels overlapping admitted
underlay, check stage bindings, and run bounded positive controls for rejection.
Do not replace the design or relax opacity guards speculatively. If broad valid
coverage exists but expensive work is not rejected early, test one correction
to the dedicated opaque underlay entry. HLSL's
[`earlydepthstencil`](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-attributes-earlydepthstencil)
requests depth/stencil testing before pixel shading. It is a candidate to verify
on the actual adapter, not an established diagnosis or speedup. Preserve the
proven alpha-one/no-discard/no-depth-output contract. Never force early stencil
updates on the coverage shaders: their `clip` operations decide whether a pixel
may be marked.

Optional [pipeline statistics](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ns-d3d11-d3d11_query_data_pipeline_statistics)
can report pixel-shader invocation counts, but must first pass a small rejection
control on this adapter and stay outside primary timings. Failed/unavailable
queries should not become another profiling project. Captures and causal timing
controls remain useful, with their limits explicit. Skip the extra coverage work
when underlay records are empty. One corrective candidate is the limit for this
follow-up; require substantial repeated complete-frame savings before broader
quality/navigation qualification. Stop if the mechanism remains ineffective.
