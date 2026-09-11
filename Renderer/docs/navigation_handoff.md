# Navigation renderer handoff — 2026-09-09

> Historical evidence and activation record. For current work, use the
> [benchmark workflow](benchmark_workflow.md) and current status at the top of
> [the retained plan](retained_renderer_plan.md). Embedded "next" instructions
> below are superseded; historical hashes and timings require matching receipts
> before being used as current evidence. Do not rerun these workloads merely
> to start the tooling transition.

## Current continuation

Read [the destination reproduction and ongoing optimization evidence](navigation_continuation.md)
before using the historical activation or timing statements below. At the user's
explicit request, the verified navigation DLL is now staged: SHA-256
`c2b3a2e8cf5ebfc8c32c185ea411cb68062a2683fbacead9660ad592d18b8af9`.
The existing cache switch enables the verified larger-memory, wave/backdrop and
receiver-index defaults. All 21 animated verification images match the explicit
settings at 128/160/192. No installer or game launch was performed; restart the
game to load the DLL. Rollback is
`Renderer/lab/out/navigation/promotion/previous-C3XRenderer.dll` (the prior
accepted mountain DLL). This is not a live-game performance pass.
Waves-on, zoom, distant preparation and native presentation have separate evidence.
Timestamped [usage logging](live_usage_logging.md) is included.
The [busy continuous session](busy_navigation_session.md) is implemented with
queued discrete input and independent post-session images. Read its final evidence
before claiming a workload pass. The user's [actual game log analysis](live_usage_findings_20260909.md)
shows cold geometry and unit-pose stalls, limited idle cadence, and a final debugger
exception whose disposition is unknown. A [ready-to-paste next-agent prompt](navigation_next_agent_prompt.md)
contains priorities, reproduction, storage and compiler-environment instructions.
The original plan and historical records below remain intact.

## Historical staged correction: animation background corruption

The staged DLL is now `navigation-linear-backdrop-fix-20260909`, SHA-256
`71e969c7d2321ad90b084c5da622546eee896589dd9bd08a99dd6a6d21dc057f`.
This supersedes the e25 DLL described in the historical measurements below.
The previous DLL is preserved in that candidate directory. The old 100-view
performance distribution has not been rerun against this correction.

A live screenshot exposed repeated 128-pixel terrain blocks behind resources.
The animation-backdrop producer called `submit_geometry` expecting current MSAA
linear color and depth, but a completed-region cache hit restored only BGRA
output. The next animation pass accumulated over another region's stale scratch
buffers, potentially including an earlier animated body. Cache keys alone could
not fix this missing-output contract. `require_linear_backdrop` now disallows
bitmap-only cache hits for this caller; main-map region reuse is unchanged.

The old DLL reproduced corrupted squares in the animation fixture. The fixed
DLL's initial image and six animation images all match same-build independent
rendering byte-for-byte; the old images differ at about 160,000 pixels each.
The old DLL passed its own zoom/scroll/cold parity checks despite the corruption:
consistent wrong output is not proof of correctness. Preserve independent-path
comparisons. All 13 region-cache/animation-retention tests passed, and the new
adapter regression verifies this caller cannot use the BGRA-only cache.

Evidence: `Renderer/native/build/animation-linear-backdrop-{old,cache,independent}`,
and the candidate's `animation-comparison.json`. Runs used 1024x768, width 128,
waves/reflections off, ring-four cache settings, fixed resource animation clocks;
each has unchanged input and binary receipts. `record_navigation_evidence` now
accepts `--scenario animation` to exercise zoom return, stationary animation,
scrolling and removal rather than the static navigation workload. No new
live-game pass or continuous frame-rate claim is made. The renderer-only fix
needs a game restart to load the staged DLL, not another injected-code change.

## Follow-up: C3X configuration flags

The user subsequently requested ordinary C3X flags for reflections and waves.
`enable_custom_rendering_reflections` and `enable_custom_rendering_waves` now
default to true; either can be set false independently. The added
`enable_custom_rendering_cache` flag activates the measured ring-four settings
without the evaluation launcher (distributed default false, local custom value
true). Local effects remain true so the user can choose when to disable them.
See [configuration details](renderer_config_spec.md). These injected settings
override the corresponding launcher environment values before DLL loading.
They require installation of the updated injected code and a restart; no
installation was performed here. The staged DLL remains unchanged.

Verification: 43 effect-adapter/native-bridge/custom-zoom tests ran successfully
(39 passed, four skipped). The actual extracted adapter ran all eight flag
combinations and setter-failure cases. `TEST_INJECTED_CODE_COMPILE.bat` passed;
it compiles/injects into a suspended verification process and terminates it
without entering gameplay. No new performance or live-game pass is claimed.

The strongest verified static navigation candidate is staged in `Renderer/bin/C3XRenderer.dll` at the user's explicit request. This is evaluation staging under the Lab rules, not visual acceptance or a live-game performance pass. Neither INSTALL.bat nor Civ III was run during this handoff.

Read this document, [the current continuation](navigation_continuation.md),
[the retained-renderer plan](retained_renderer_plan.md), and the [native
presentation audit](native_async_presentation_audit.md). Consult
[zoom_performance.md](zoom_performance.md) only for receipts referenced by the
current plan. The supported custom widths are **128, 160 and 192**; config-off
preserves native behavior.

## Artifact and activation

The evaluation launcher explicitly selects `city-fidelity` (the DLL's normal
empty-environment profile), so an inherited older visual-profile override cannot
silently bypass world-region caching. This is a launcher setting only; the
measured and staged DLL is unchanged.

- Source baseline: Git `f8ac91decb6242a54370c61a82e983b99d20f184` (More cache optimization work), plus the handoff working-tree changes. Transfer those changes too; this handoff did not create a Git commit.
- Candidate: `Renderer/native/build/navigation-region-inputs-20260909/C3XRenderer.dll`.
- Staged SHA-256: `e25e139c85265de233e9e7d0da5057466f2c321ed8677f184e5c038a36d5c406`.
- Preview SHA-256: `87821a3bae3aa14da6a4c72c2942d814acd286a2be9d632dc7c4ee13d8bb54b8`.
- Rollback DLL: `Renderer/native/build/navigation-handoff-20260909/previous-C3XRenderer.dll`, SHA-256 `9bc3a029be873acc6dad991cee64b091e3c532172d7bade8abc8dec8af5cb01f`.
- `Renderer/run_navigation_evaluation.cmd` starts the existing installed game with scoped static-cache settings. Run it yourself when ready for game evaluation; `--check` only prints settings. It does not install C3X. Ordinary game launches retain the DLL's default-off experimental settings. The profile keeps waves and reflections enabled, so its game performance is **not** the waves-off number below.
- The custom three-zoom restriction lives in injected source. Staging a renderer DLL cannot update an already-installed injected bridge. Verify/reinstall the current C3X source only when the user authorizes installation on the destination machine.

The abandoned `RegionProofCache` prototype was removed completely before staging. Current renderer C++ matches the Git baseline logically; restoring it normalized line endings, so its raw source hash differs from the historical build receipt. The staged DLL is the original measured binary, not an untested rebuild. Do not claim a newly built DLL has that identity.

## What the experiments establish

At 2240×1192, 128-width zoom, waves disabled, 100 distinct resident camera changes, the clean ring-four run measured:

| Metric | Median | p95 | p99 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Render plus capture completion | 60.130 ms | 137.580 ms | 176.845 ms | 226.245 ms |
| CPU draw/submission interval | 37.363 ms | 51.142 ms | 63.812 ms | 81.042 ms |
| Readback including pending GPU work | 5.653 ms | 46.967 ms | 78.733 ms | 131.601 ms |
| Geometry assembly | 2.346 ms | 36.695 ms | 38.609 ms | 68.216 ms |

There were 18,487 completed-region hits, 35 misses, no rejection or eviction, and zero unchanged static mesh builds/uploads. Region image memory peaked at 62,368,512 bytes and key metadata at 22,834,772 bytes. All 100 outputs matched the earlier ring-two sequence exactly by BMP hash. That comparison crosses DLL builds; it is not a new same-build independent-render pair.

An earlier same-build receiver-shadow cache/control pair matched all 100 images: cache median/p95 56.978/197.096 ms versus independent rendering 938.716/1021.405 ms. See performance documentation for identities and differences between experiments. Do not calculate a controlled speedup by mixing those timings with the later run. The separate `region-input-ring4-100` timing overlapped storage investigation; only `region-input-ring4-clean-100` is the clean latest timing run.

The approach is useful for **moving through already prepared nearby content**, not merely repeating one viewport. Raster reuse now works across camera translations. It has not established fast first-time distant jumps, all-three-zoom transitions, animated scenes, or native input-to-display latency. Earlier cold preparation still took seconds per view (roughly 8 seconds median), so whole-map preparation/compiled storage remains a major architectural requirement.

| Plan target | Current status |
| --- | --- |
| Warm correct response ≤50 ms p95; final ≤100 ms p95 | Not achieved; latest standalone final p95 137.580 ms, no separate first-response/native measurement |
| 30 delivered FPS; p95 ≤33.4 ms, p99 ≤50 ms over 1,000 presented frames | Not demonstrated |
| Unseen/evicted destination after map preparation ≤100 ms p95 | Not implemented/demonstrated |
| Game-thread bookkeeping ≤2 ms p95 | Standalone camera begin previously 1.175 ms p95; injected bridge remains synchronous |
| Exact quality and bounded ownership | Focused parity and cache tests pass; full lifecycle, animation, pressure and live-game acceptance remain pending |

## Current execution priority

The next implementation target is the dense, no-water resident scroll path.
Translate the immutable static front, render only newly exposed strips, and
independently invalidate city, improvement and resource bounds. Require exact
pixels and ownership, zero fallback/recovery, no completed-map reuse, and first
less than 100 ms, then less than 33 ms. Do not add another cache tier until this
gate is met.

After that, complete latest-exact asynchronous publication at the existing Civ
III map boundary, then build compact complete-appearance regional preparation
for distant jumps. Keep cold pose preparation and water/reflection throughput as
separate measurements; neither should displace the resident static/object path
unless fresh evidence proves it is the dominant cost.

For the detailed operating rules, use
[the autonomous renderer execution contract](autonomous_renderer_execution.md).

Explain each experiment's expected game benefit and stop condition to the user. Distinguish stationary views, nearby pans, zoom changes and distant jumps. Stop low-impact polishing when measurements point to another architecture layer.

## Memory and verification

Normal GPU geometry budget is now 768 MiB; CPU natural/ground 96 MiB, viewport 32 MiB, backdrop 128 MiB. Completed static regions separately cap GPU images at 256 MiB, metadata at 96 MiB, and entries at 4,096. These are category ceilings, not a total process bound. Publication owners are each capped at 32 MiB. Audit transient overlap and deferred GPU release. The installed local Civ3Conquests.exe has the x86 LAA flag, while the unmodded backup does not; neither proves destination live-game headroom.

At handoff, 14 focused tests passed across `test_render_region_cache`, `test_frame_telemetry`, and `test_custom_zoom`, plus the storage-preflight test in `test_zoom_mesh_cache` (15 total). The launch profile passed `--check` without launching. Earlier injected compilation passed after the three-zoom change. No fresh full category integration or live-game pass is claimed. An earlier broad shadows integration had failures/errors involving platform/compiler/locks/provenance; inspect and resolve on the destination rather than assuming they are harmless.

Reproduce from the repository root with the available Python 3 interpreter:

```sh
python -m Renderer.native.record_renderer_build --out Renderer/native/build/navigation-destination-build
python -m Renderer.native.record_navigation_evidence --binaries Renderer/native/build/navigation-destination-build --out Renderer/native/build/navigation-destination-clean --resident --resident-steps 100 --tier normal --waves 0 --block-clip 1 --world-grid --raster-control --region-size 128 --world-regions --region-metadata-mib 96 --region-receiver-shadows --tight-natural-bounds --region-input-ring 4
python -m unittest Renderer.native.test_render_region_cache Renderer.native.test_frame_telemetry Renderer.native.test_custom_zoom
```

Use fresh output paths. For independent static redraw add `--world-regions-control`; hold everything else fixed and compare images. Then extend the workload with waves enabled, `--tile-width 160`/`192`, eviction, local edits, visibility changes and mixed unit activity. Existing category verification through `Renderer/renderer.py` remains required.

## Transfer and disk

`Renderer/native/build/navigation-handoff-20260909.zip` contains a repository-relative overlay: handoff working files, staged DLL, preview, rollback DLL, build/input/timing receipts and a checksum manifest. It is **not a full repository or asset distribution**. Start with the Git baseline above, overlay these files, and privately transfer/recreate required local packs and verification inputs. The original plan is included in the docs, so the attachment path is no longer required. Read local AGENTS and workflow files before continuing. Check the destination dispatcher/toolchain configuration instead of copying machine-specific paths.

Licensed source/derived art must remain local and must not be redistributed with C3X. The receipt's `inputs.json` lists the exact local inputs; the package omits those assets. Preserve needed ignored inputs such as `Renderer/lab/.local/verification/world.csv` and runtime packs when privately moving this checkout.

Historical benchmark images were pruned after roughly 40 GiB accumulated. Logs and summaries remain, but old exact comparisons cannot all be recomputed without regenerating images. The handoff contains compact receipts, not image sequences. The evidence runner now checks estimated output plus an 8 GiB reserve before creating a run; verify free space on both filesystems if dispatching remotely. Cleanup scripts default to preview and target only obsolete generated BMPs. Do not resume recursive filesystem compression as a storage workaround.
