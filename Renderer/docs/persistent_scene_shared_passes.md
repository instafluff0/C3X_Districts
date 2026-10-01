# Persistent scene ownership and shared passes

The production Renderer64 FRESH path borrows immutable scene generations for
camera and pass selection. `SceneMembership` publishes the existing content
leases and occurrence records together; mutation detaches only a borrowed
generation. Old views retain their meshes through the existing retirement
ledger. Camera transforms and wrapped occurrences are applied during traversal
rather than copying another assembled scene. Authoritative tile content,
visibility, scene membership, projection, lighting and retained pixels have
separate validity proofs.

Retained pixel proofs own resource-free dependency metadata, including world
semantics, appearance, coast/river inputs, contributor generations and native
visibility. They do not validate world content against a different preparation
camera's anchors. Selected geometry still validates its native anchor closure;
normalized occurrence keys reject changed bindings and contributor generations.
New contributors within covered pixels force a redraw, while contributors
outside the retained region do not invalidate its existing pixels. Both
canonical and projected static raster owners remain bounded and independent.

Main, reflected-water and shadow selection uses the existing spatial indexes
before submission. Wrapped occurrences and original layer/alpha order remain
explicit. Reflection eligibility includes the water receiver region, distortion
and filter reach; source shadows retain offscreen caster reach. Selection cache
identity includes projection zoom: reversing zoom at an unchanged camera must
reopen contributors near the viewport edges. Compatible vegetation submissions
borrow immutable instance buffers, with the original streaming path as a bounded
fallback. The immediate context remains owned by the existing renderer worker.

Unit preparation samples the union of eligible main and reflection occurrences
once per frame. Both consumers borrow the same pose, ground, facing, material,
palette and transition result. Pose-local self-shadow sharing uses exact mesh,
cutout, sampled palette, scale, facing and light dependencies. All needed entries
are pinned before either pass; overflow uses the original working shadow path.
Native action clocks, incarnation handling, hidden visibility, blends and cursor
placement continue through their existing inputs.

New bounded ownership includes a 16 MiB palette pool, 64 MiB pose-shadow cache,
4 MiB maximum pose-shadow key metadata, 64 MiB immutable instance plans with
8 MiB maximum key metadata, and 16 MiB contributor proofs per raster owner.
These are application resource/payload bounds, not physical GPU memory figures.
Water motion, coastal waves, reflections and day/night remain enabled. No art
reference was replaced, and deferred renderer ownership was not expanded.

## Evidence and limits

Private evidence is under `Renderer/.cache/persistent-scene-step/`. The final
matching build is identified by `build-10/source-binaries.json`: 337 frozen
source/shader inputs, unchanged throughout the build. The executable production
cache regression reproduces zoom reversal without any world, camera or native
visibility edit. Focused scene ownership, invalidation, wrapping, unit action,
shadow, composition and config-off contracts are preserved. Five obsolete
source-text assertions fail unchanged at the accepted checkout; their baseline
reproduction is recorded separately rather than altering them to obtain a pass.

Independent retained-versus-full raster witnesses cover native 128-pixel anchors,
pan, zoom, wrapping, deep camera return, local semantic/visibility edits and
night lighting. The predecessor build's independent full output also matches
the accepted baseline closely; tiny shadow-edge RGB differences are measured
separately from depth correctness. Full game window captures provide an
additional check of map edges, units, labels, selection, minimap and HUD.

The final 18-view witness at matched 1.25x projection has exact retained/full
depth agreement; RGB differences reach 8,785 pixels / maximum channel delta 46 /
mean 0.00305. Independent final-versus-baseline full output also has exact depth,
with up to 319,105 changed RGB pixels / maximum delta 107 / mean 0.66125. These
larger RGB differences concentrate around foliage/shadow pixels after projection
selection changes and are retained for review; this is not a claim of bit-identical color.
Full-frame comparisons and enlarged difference views preserve the geometry,
water, cities and coverage. An earlier comparison with mismatched base zoom is
explicitly excluded from quality claims.

The final native 128 fixture adopts 32 camera views. Thirty resident transitions
build/upload nothing; two transitions admit 22 and 25 genuinely missing bindings
(408,236 and 11,623,104 upload bytes). All dependency/context mismatch counters
remain zero. These first-use admissions are not redundant unchanged geometry.
A bounded native 64 fixture also covers 24 real unit
pose changes. Fullscreen native 64 fixtures exceed existing coastal-wave
allocation limits in both accepted and candidate paths: no wave truncation or
budget increase was used to obtain a pass. Parallels rejected every sampled GPU
timestamp query, so phase evidence reports CPU intervals and makes no causal
GPU timing claim.

Matched game measurements use the same disposable 1498 AD save at 2240x1260,
normal effects, one scene sample, verbose object tracing disabled and a read-only
successful-presentation counter. These are presentation cadence measurements,
not physical scanout. A separate traced phase run is used for attribution.
The final trials report:

| State | Accepted baseline | Candidate |
| --- | ---: | ---: |
| Scroll, fixed 30–54 second window | 6.176/sec | 6.483/sec |
| Stationary after scrolling, 65–89 seconds | 13.151/sec | 16.208/sec |
| Zoom/reversal, 30–44 seconds | 9.549/sec | 10.526/sec |
| Settled 1.25x zoom, 50–64 seconds | 16.778/sec | 18.754/sec |

These are single-trial observations. Native deferral coalesced camera requests:
the scrolling endpoints differ by one native tile in Y, so these values are not
an isolated causal speedup ratio. Of 32 requested scroll commands, 24 baseline
and 23 candidate destinations have a subsequent exact-anchor handoff. Matched
handoff medians are 0.597 and 0.624 seconds, with maxima 4.508 and 3.032 seconds;
coalesced destinations have no measured first-correct frame. Handoffs prove
native anchors, not every full-quality pixel or physical scanout. The final zoom
run's sampled map edges are complete, including reversals; the transient black
edges found in an earlier candidate are covered by the new executable regression.
The 60 FPS idle/scroll and 40–50 FPS zoom targets remain unmet.

In the final stationary phase trace, 82 main and 56 reflected unit occurrences
share 82 pose samples / 443 parts. Median shadow work is 11 samples / 127 reuses,
with no pool overflow. CPU phase medians are preparation 7.177, reflection 1.466,
static 0.001, water 8.631, units 3.457 and reconstruction 0.359 ms. Medians are not
additive; preparation now includes work previously repeated in both unit passes.
The direct visual interval still includes nested scene drawing: its 30.479 ms
median cannot be attributed entirely to native HUD replay. That interval retains
2,294 operations, 217 copies / 34,857,886 copied pixels and 44 assemblies /
22,619,610 assembly pixels. Water, shared dynamic preparation and composition
remain substantial work. The invalid GPU queries do not separate their GPU costs.

Helper private-memory peaks are 3,741,626,368 bytes during the quiet scroll run,
3,700,617,216 during zoom, and 3,841,298,432 during the failed jump run. These are
process private bytes, not physical GPU memory or a resource census.

Cold minimap jumps expose a reliable publication queue exhaustion in both the
accepted baseline and candidate. After this error, presentation counters freeze;
post-failure intervals must not be reported as successful jump performance or
quality recovery. The final candidate fails at 43.137 seconds with 8,192 queued packets /
21,183,140 charged bytes; the baseline fails at 50.231 seconds with the same
packet ceiling. No first-correct cold destination or warm-return recovery is
qualified. The candidate is therefore **unqualified for production integration**.
The accepted matching trio is restored as the installed baseline, with both the
candidate and its evidence preserved for auditor review.

The installed executable, JGL, INI, cursor, environment, original/disposable save
and temporary processes/tasks are checked after each bounded run. No injected
source or patch-table change is needed; `required_user_action: []`.


The candidate binary identity is:

| Binary | SHA-256 |
| --- | --- |
| `C3XRenderer.dll` | `e2b9f2950f04075e3da8971a120c06cb85ca53515db5156f1eb6388c02388ae8` |
| `C3XRenderer_x64.dll` | `c151f9665b5e0c70321d21f36847fc95ded7b6b6725c1ba507bf0d3a544277b1` |
| `C3XRendererHelper64.exe` | `ea99906bbd2e91a968b5513e86678ea9389dd36853bacaac6fddd2264466b623` |

The restored accepted hashes remain those recorded in
[bounded native HUD composition](bounded_hud_composition.md). Build, native,
full-frame comparison, game capture, dependency identity, cleanup and restoration
receipts are indexed by the private completion handoff. The held material
worktree, ignored evidence and recovered I2 capture remain preserved.
