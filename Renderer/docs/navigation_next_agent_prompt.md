# Prompt for the next agent

Continue renderer benchmark engineering from the current checkout, preserving all
existing and uncommitted work. Read AGENTS.md, Renderer/README.md,
Renderer/lab/README.md and the relevant catalog/category, then:

- `Renderer/docs/benchmark_workflow.md` for the one active implementation sequence;
- `Renderer/docs/retained_renderer_plan.md` for the short current capability status;
- `Renderer/docs/autonomous_renderer_execution.md` for scope and reporting rules.

The user requested groundwork, not restarting tests. The groundwork has specified
the workflow; persistent sessions, corrected timing endpoints and automatic batch
decisions are not yet implemented. If asked only to review/continue groundwork,
keep the work read-only or documentary. When asked to implement, start with the
next unfinished tooling deliverable, not another baseline/oracle or busy-session
run. Validate changed tooling narrowly; do not confuse this with rendering gains.

Extend the existing preview, evidence runner, build scripts and analyzer. First
account for setup/playback and fix the short scroll capture timing. Then amortize
setup with explicit reset/residency and immutable session inputs. Next add one
controlled route/object diagnostic batch with matched repetitions and automatic
retain/reject/inconclusive decisions. Finally prove the retained route/object
change on a tiny boundary/depth fixture before the fixed dense navigation sequence.
Do not implement a second simulator, change production rendering defaults, stage
binaries or modify injected hooks as part of this tooling sequence.

Keep the current immutable front, cached unit-pose composition, camera queue,
geometry/backdrop retention, cancellation safeguards and useful preparation.
The selected rendering bottleneck is route/object composition and synchronous
completion during dense navigation; the below-100 ms gate is still unmet.
Stationary standalone gains do not prove native gameplay or 30 Hz. Preserve
128/160/192, action-director phases, visibility/ownership, off-screen presentation,
config-off and separate unit fallback. Wonders and Districts remain deferred.

Use older handoffs only to retrieve needed evidence and operating details.
Their embedded "next" instructions are superseded. Do not repeat abandoned
projection/transparent-wave/index experiments without a new causal reason.
Preserve staged/rollback artifacts and all ignored licensed assets. Current binary
identity and toolchain availability must be checked when relevant; old hashes,
compiler environment and approvals do not automatically apply to a new candidate.

Report the deliverable, validation actually performed, dominant remaining cost,
strongest validated workload and one next action. Update current status in place.
No renderer performance gain is implied by a successful tooling change.

The notes below preserve environment/storage findings; they are historical and
must not override the active workflow or current tool discovery.

## Preserved operating notes from the earlier handoff

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
