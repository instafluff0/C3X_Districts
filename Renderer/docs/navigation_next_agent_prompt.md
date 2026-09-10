# Prompt for the next agent

Continue the Civ III renderer navigation optimization from the current checkout.
Read `AGENTS.md`, `Renderer/README.md`, `Renderer/lab/README.md`, the affected
catalog/category recipes, and `Renderer/docs/navigation_handoff.md`. Follow its
links to `navigation_continuation.md`, `navigation_implementation_plan.md`,
`zoom_performance.md`, and `native_async_presentation_audit.md`.
Then read `busy_navigation_session.md`, `live_usage_findings_20260909.md`, and
`live_usage_logging.md`. Verify the exact evidence filename through the handoff
if a historical link has changed. Preserve all existing and uncommitted work.

Follow `Renderer/docs/autonomous_renderer_execution.md` as the operating
contract. Do not pause for approval between ordinary renderer edits, focused
measurements, isolated builds and evidence updates. Pause only for staging,
installation, launching Civ III, injected hook/patch changes, visual acceptance,
reference replacement or another materially external action.

The user asked this agent to wrap up after finishing the continuous busy fixture
and analyzing their actual game log. Resume optimization, not another handoff.
Explain every experiment's expected in-game benefit and why it is the highest-value
next step. Support custom widths 128/160/192 and higher bounded memory. Preserve
visual quality, native camera/capture, visibility, overlays, picking, presentation,
unit movement/direction/combat/interruption, and independently phased idle units.
Do not synchronize units just to improve cache hit rates. Keep deferred wonders
and Districts outside current ownership.

Treat the short no-water resident-scroll test as a diagnostic gate only. After
each meaningful change, report the broader convergence position and exercise the
next realistic rung: dense resident movement with 24/64 independently acting
units, then the full busy session with effects, zooms, distant jumps, reversals,
combat, visibility changes and queued-input accounting. Do not claim progress
toward the real game workload from a narrow cache-hit or worker-completion test.
Every update must state the current dominant bottleneck, what remains unmet, and
the next gate that would change the overall viability assessment.

The best verified performance defaults and timestamped OutputDebugStringA usage
logging are already staged at the user's explicit request. Current DLL SHA-256:
`c2b3a2e8cf5ebfc8c32c185ea411cb68062a2683fbacead9660ad592d18b8af9`.
Rollback and verification limits are in the handoff and promotion receipt. Do not
replace it with a benchmark variant or assume that staging proves native performance.
No installer/game launch was performed by the finishing agent. Read current local
preferences: waves/reflections were off for the supplied real-game log, but on for
the busy fixtures. Do not silently change those preferences.

The supplied log's final debugger exception (`0x0000087A`, parameters
`0x887A0001, 0x00000053`) is a parallel diagnostic, not a blocker for renderer
work. All 109 render calls had completed successfully, but clean shutdown is
not established; do not invent a cause or call it a confirmed crash without
more evidence.

Prioritize the measured remaining stalls. The active first target is the dense,
no-water resident scroll: translate the immutable static front, render only the
newly exposed strip, and independently invalidate city/improvement/resource
bounds. Require exact pixels and ownership, zero fallback/recovery, no
completed-map reuse, and first <100 ms then <33 ms before adding another cache
tier or returning to water effects. Keep cold pose preparation as a separate
measurement and preserve native phase/action semantics; cached unit composition
is already a secondary cost after warm-up. Then complete latest-exact native
publication, followed by compact complete-appearance regional preparation for
distant jumps. Topology alone is insufficient for cities/resources/forest
exclusions or visibility-dependent appearance. Version and invalidate any
compiled regional cache by complete immutable inputs, world edits, pack/compiler
identity and wrap state.

Use the completed busy fixture as a later regression workload: continuous renderer,
waves and reflections on, cold startup, no unit warm-up, independently moving/acting
units, idle → scroll/reverse → all three zooms → two distant jumps/local scrolling →
return and idle. First use the short dense no-water resident-scroll gate so the
fixture does not hide the static-object bottleneck behind water or cold-start cost.
The busy fixture retains discrete zoom/minimap clicks while coalescing continuous
scroll, records queued action delays, and independently replays bounded snapshots
after timing. Use 24 and 64 units per zone and report actual visibility. Keep
startup, first/repeated zoom, nearby scroll, distant preparation, final idle,
unit-body cost, input backlog and native presentation separate.

Native asynchronous completion remains unfinished. The current native timer/caller
is synchronous; standalone queue call timings are not native integration evidence.
Never display an old-camera bitmap beneath current-camera overlays or picking. Preserve
native fallback and exact publication identity. Record required_user_action in the patch
ledger; do not edit the address CSV or add speculative hooks.

Report against the original targets: first correct response ≤50 ms p95; warm final and
prepared unseen/evicted regions ≤100 ms p95; game-thread submit/poll ≤2 ms p95; native
30 presented FPS with p95 interval ≤33.4 ms and p99 ≤50 ms over 1,000 frames; live VA
headroom ≥512 MiB or twice the largest transient, with adequate contiguous allocation.
The old 60/138 ms standalone resident, waves-off result is not an all-workload pass.
Neither is the supplied 109-call game log: only width128, limited density, no physical
presentation/input timestamps and no waves-on coverage.

## Operating notes

- Use Python 3.12 and run repository modules from the root (`python -m Renderer...`).
  The bundled runtime can be located through Codex's workspace dependency tool; system
  Python 3.9 is insufficient. `PYTHONPATH=.` is needed for private helper scripts.
- Native D3D runs in the Windows 11 Parallels VM through the existing dispatcher and
  shared checkout. GPU benchmarks must run serially, after builds/storage work finish.
  Confirm child completion/process absence before retrying a transport failure.
- Visual Studio updated during wrap-up. `vswhere` normal discovery returned no complete
  installation because reboot is pending. The existing installed compiler successfully
  built the preview when its verified installation path was supplied through
  `C3X_VS_PATH`; the build receipt records this override. No VM restart was performed.
  Recheck discovery rather than assuming the environment is unchanged.
- Preserve ignored assets: they cannot be recovered from Git. Older generated benchmark
  BMPs were losslessly archived as `.bmp.gz`, with decompressed SHA verification before
  removing originals. Identical archives share APFS extents. The manifest is
  `Renderer/native/build/navigation-compressed-evidence.json`; the navigation analyzer
  reads compressed BMPs directly. Decompress only needed images for other legacy tools.
  Keep the 8 GiB evidence reserve; do not restore all archives at once.
- Do not commit raw debugger output, local paths or machine identifiers. Local raw log
  and analysis are under `Renderer/lab/out/navigation/live-usage-20260909/`; the findings
  document is portable. Review concurrent Git changes and scan touched files before
  finishing. No injected code changed during wrap-up, so no injected compile was needed.
