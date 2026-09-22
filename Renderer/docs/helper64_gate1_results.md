# M3.H Gate 1 — cross-process graphics boundary

**Decision:** The isolated graphics boundary is feasible on the Windows 11
ARM64 Parallels VM. The x64 producer and x86 consumer are both emulated there.
This advances the helper evaluation to the Gate 2 decision; it does not adopt a
separate production renderer or establish a game-speed improvement.

The standalone probe in `Renderer/native/helper_trial/` uses two keyed,
single-sample BGRA D3D11 textures. The producer duplicates NT shared handles
into the consumer, which verifies the adapter LUID, opens the textures, and
copies a ready frame into the existing `NativePresenter` DirectComposition path.
The x86 process still creates the window target. No framebuffer travels through
CPU memory during ordinary frames. One-pixel staging reads at checkpoints are
validation only. The presenter keeps working while its HWND thread is blocked.

The final **2240 × 1260 visible** run passed all native scenarios: shared
import, exact sampled pixels and frame identities, native child UI above the
map, a partial 16-bit native transfer preserving the untouched map, bounded
two-slot backpressure, producer termination/restart, resize, and GDI restoration.
All **119/119** steady desktop samples showed both the current map color and the
native child UI color. The producer dropped one deliberate command when both
slots were occupied. A separate **500-frame hidden** stress pass also passed.
Both runs are small synthetic tests; neither launches Civ III.

| Full-resolution visible measurement | Mean | p95 |
| --- | ---: | ---: |
| 64-byte x86↔x64 shared-memory/event round trip, 1,000 samples | 0.013 ms | 0.016 ms |
| Keyed acquisition on the x86 worker | 0.080 ms | 0.097 ms |
| Shared frame → native retained display → swap buffer GPU-copy submission | 0.209 ms | 0.352 ms |
| Independent `Present` call | 0.357 ms | 0.638 ms |
| `DwmFlush` after present | 15.51 ms | 20.77 ms |
| Synthetic request → desktop completion with the probe's synchronous text pipe | 34.85 ms | 42.69 ms |

The graphics self-test deliberately uses a synchronous text-pipe command/reply
to make failures easy to diagnose. Its round trip alone costs 18.44 ms mean /
25.66 ms p95. The separate shared-memory/event result shows that cost is a
property of the diagnostic control path, not a lower bound imposed by process
separation. The event microbenchmark does not include scene capture, rendering,
native composition, or display, so these timings cannot be combined into a
claimed <33 ms gameplay result. GPU-copy and producer-submit timings are CPU
submission intervals; the desktop witness is the completed visual check.

The 500-frame producer's sampled private bytes changed from 10.91 MiB to
10.86 MiB, and its driver-reported GPU usage stayed at 22.04 MiB. These
synthetic allocations are much smaller than the game's measured scene
attachments and do not prove that moving the renderer resolves device removal
or the 32-bit capacity floor. The driver-reported budget/usage are diagnostic,
not physical VRAM accounting.

Evidence is retained in ignored
`Renderer/native/build/helper_trial/6a82dcbc925b458b8b8f485d915e32ba/`
(visible) and
`Renderer/native/build/helper_trial/e81923db9641422f917a71d64390d644/`
(500-frame stress). Each has a child completion record, source/binary hashes,
PE machine checks, native report, and aggregated receipt. The runner's four
integrity tests pass. Generated output is small; no production DLL was built,
staged, installed, or run in Civ III.

**Next decision:** Gate 2 would move the existing render core behind a
versioned, value-only scene protocol and compare the same recorded workload
against the current x86 path. That is necessary to decide whether moving the
large world and scene targets out of Civ III actually improves capacity,
reliability, and complete input-to-display latency without losing native
ordering or ambient animation. Gate 1 alone does not justify migration.
