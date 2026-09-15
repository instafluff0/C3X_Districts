# Retained renderer: implementation and next review

## Objective and authority

Complete world → view → pass → publication with full detail, source animation
speed and bounded ownership. Renderer-owned visual frames are authorized. Civ III
retains gameplay, visibility, camera anchors, directed actions and UI decisions.
The renderer never requests a Civ III redraw or creates another presenter/window.

Preserve AGENTS.md, generic asset contracts, accepted visual differences and
M9/M10/M11 wonders/Districts deferral. Tested DLL staging is authorized; the user
installs and launches through ordinary `INSTALL.bat`, without environment setup.

## Current frame-ownership implementation

[Visual frame ownership](visual_frame_ownership.md) describes the connected path:
copied current map selection and retained unit instances feed dynamic GPU leaves;
versioned native composition retains exact underlays, overlays and UI operations.
Opaque coverage removes obsolete dependencies. Completed native transfers publish
only their actual rectangles. Independent samples use selected scratch rectangles
and shared immutable textures, leaving native working canvases untouched.

An ordinary renderer-owned 33 ms timer uses the existing presenter thread and GPU
worker. Native captures share its visual clock. Extra native `Animator_update`
calls and renderer rearming of Civ III's timer are removed; original gameplay
advancement remains. Authored idle/work sampling also feeds existing bounded
preparation workers. Frozen units and unchanged samples skip visual submission.

The tested DLL is staged for ordinary `INSTALL.bat`. Exact GPU replay, independent
resource/unit frames, actual timer transport, UI pause, config-off and native
compile checks pass. Live game coverage remains a separate checkpoint.

## Six architectural responsibilities

| Responsibility | Actual ownership and remaining work |
| --- | --- |
| Persistent world/instances | Retained camera-independent tile content, shared catalogs and revisioned unit instances exist. GPU residency is bounded. Current map selection owns one copied scene input; old composition versions retain GPU results. |
| Local validity | Full captured tile content, dependency revisions, unit content revisions and explicit despawn invalidate retained use. Fresh native capture still supplies authoritative game changes; this is not yet a comprehensive game mutation event stream. |
| Spatial selection | World pass membership and native affine anchors select terrain occurrences. Reachable native composition selects visible unit/overlay occurrences, including clipping and wrapping. No retained asset grants visibility. |
| Compatible pass submission | Terrain/object groups, GPU unit shadows and exact native composition passes are implemented. Bodies still finish as separate pose textures; scene-wide compatible unit batching/instancing remains a major opportunity. |
| GPU reuse/output | Admitted map, bodies, shadows and composition stay resident through the existing presenter, with fuller map color and native-word compatibility. CPU access barriers remain explicit. Per-pose surfaces and remaining full-map conversion work need assessment before further optimization. |
| Asynchronous integration | Existing bounded preparation plus independent visual scheduling are integrated. Native capture/publication still owns camera/content changes. Cold or outside-coverage camera requests can block, and presentation still needs the UI message pump. |

The next architecture review should use these owners and measured costs to choose
a connected change, rather than another nearby output helper. Do not assume an
extra worker or a higher timer frequency reduces total work. Directed movement/
combat anchors and cursors remain authoritative native samples; intermediate
visual interpolation is not implemented by this frame scheduler.

## Validation and controls

Checkpoint: `native/build/gpu-composition/visual-frame-checkpoint/`, staged DLL
SHA256 `3a024a15dd79843a8da7cccc8657474674a31ac26a9fa403165e5c67ae88bf8a`.
126 exact GPU replay comparisons and 120 independent samples pass; units suite
passes 131 tests with one existing skip; 43 focused native/unit contracts and
`TEST_INJECTED_CODE_COMPILE.bat` pass. The connected 1120×1192 fixture verifies
30 independent resource/unit frames with zero native drawing calls, three actual
timer frames, exact retained UI, modal pause, partial transfer and config-off.
It retains 27 nodes / 27.0 MiB; the 32-bit process still has a largest free region
of at least 1.37 GiB. No game was installed or launched.

Same-harness GPU whole-request comparison (64 samples/workload, ms mean / p95):

| Workload | Preserved `95dab634` control | Independent-frame implementation |
| --- | --- | --- |
| Idle | 8.95 / 15.52 | 9.52 / 19.37 |
| Dense scrolling | 47.58 / 59.62 | 51.19 / 84.35 |
| Local change | 10.73 / 41.03 | 11.45 / 36.52 |

This is a scheduling capability gain, not a demonstrated foreground speedup.
Native composition/presentation/preparation adds about 0.7–1.5 ms mean in this
run; scrolling remains dominated by map work (44.1 of 51.2 ms). The focused
architectural correction replaced full-canvas replay scratch with selected
rectangles and immutable source binding; its intermediate control is preserved.
A separate independent-frame run averages 17.60 ms whole request / 29.04 ms
including desktop completion. Desktop completion is not physical scanout;
neither this fixture nor a 33 ms target establishes sustained in-game FPS.
Receipts `a898871e…`, `19bec32e…` and `bf75163b…` preserve control, final workloads
and final connected visual/UI evidence. Live checkpoint: idle/work clips,
scroll/zoom, movement/combat, popup/Advisor and command-button return, picking,
focus changes and config-off. Passing automated fixtures does not establish
coverage of every live screen.

Preserved controls and expensive findings:

- `native/build/gpu-composition/unit-instance-checkpoint/`: source `95dab634`, DLL
  `1e9ac58a…81a0fd6`; retained units/3D-only baseline. GPU whole-request means were
  8.64 ms idle, 50.88 ms scrolling and 10.62 ms local change; no speedup claim.
- `native/build/gpu-composition/visual-before-cropped-pass/`: exact intermediate
  independent-frame sources/DLL. Full-size replay scratch was superseded by
  selected destination/underlay scratch and direct immutable-source binding.
- Accepted JGL owner repair is preserved in `ui-owner-fix-checkpoint/`; the
  misnamed factory reset and 8-bit-default corruption must not be reintroduced.
  Startup, resident-map, ready-content and unit-shadow checkpoints remain under
  `native/build/gpu-composition/` with source, binary and test identities.
- `native/build/world-content/`: canonical world content removed standard-zoom
  recompilation (first closer zooms 1,282/1,697 → 193/294 ms), but dense scrolling
  remained about 170 ms. An intermediate speculative zoom job caused two-second
  freshness stalls; current demand now supersedes unstarted alternate-zoom work.
  Existing small precision differences and their approval status remain recorded.
- `native/build/world-view-zoom/`: prepared zooms reached 6–7 ms after a separate
  five-second opportunity; cold construction and idle/scrolling did not improve.
- [Historical checkpoints](history/retained_renderer_checkpoints_20260914.md) and
  [completed output work](history/retained_output_completed_20260913.md) preserve
  earlier controls and rejected approaches. Global 30 Hz map sampling regressed
  warm requests 3.97 → 7.46 ms; this change preserves existing map sample buckets.
  Parallels GPU timestamps remain unreliable. Whole-request wall time, actual
  output age and work eliminated are the performance authority.

Detailed earlier status remains in Git at `95dab634` and in the existing
checkpoint directories. Do not delete ignored findings, assets or controls
without establishing that their necessary inputs can be recovered.
