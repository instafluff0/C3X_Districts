# M3.H Gate 2 — real renderer scene and capacity trial

**Decision: adopt a separate 64-bit renderer as the target architecture; do not
cut over the installed renderer until the integrated path passes.** The current render core builds and renders in x64 on the Windows
11 ARM64 Parallels VM. An x86 driver sends bounded, versioned, copied scene
values through shared memory and events. The x64 worker renders the same 51
recorded scenes as an x86 control, converts its 14 GPU map images to shared BGRA
textures, and an x86 D3D device imports and copies them. No Civ III pointers or
CPU framebuffers cross the process boundary. Diagnostic readback confirms exact
pixels for all 37 CPU map outputs and all 14 GPU map outputs on both paths.

The isolated graphics/presenter boundary was already proven by
[Gate 1](helper64_gate1_results.md). Gate 2 now shows the **actual renderer core**
can live in x64 and send real map images to x86. The x86 trial driver stays near
30 MiB private memory while the x64 renderer holds about 1.9 GiB. The x86
control holds about 1.9 GiB in its own process. Under a declared 1,920 MiB x86
virtual-address reservation, the x86 control fails with `bad allocation` after
8 of 54 operations; the x64-helper path completes all 54. Both paths complete
with 1,024, 1,536 and 1,792 MiB reservations; at 2,048 MiB the reservation
itself fails before either workload starts. This is a controlled capacity test,
not a live Civ III memory measurement.

The normal unpaced scene-only runs favor x64 by roughly 10–15% in renderer
service time on this VM. The final source-matched paired run measured 51 scenes
at 12.79 s x86 versus 10.92 s x64, including the helper's real map export. The
copied scene inputs totaled 20.04 MB. For 54 ordered operations, event/shared
memory round-trip overhead beyond helper service totaled 6.09 ms. The 14
x86 GPU imports/copies totaled 11.37 ms. These are CPU submission
intervals; they are **not** input-to-desktop latency or a gameplay FPS result.
Both binaries are emulated on this ARM64 VM.

A mid-workload fixture deliberately terminates the x64 helper after 27
operations, starts a fresh helper, restores the last copied definition inputs,
and continues. All 51 subsequent scene results retain exact CPU map hashes;
with diagnostic readback, all 14 GPU map hashes also match the uninterrupted
x86 control. The protocol allows one outstanding operation, so this fixture
cannot hide a growing queue. The x64-only export is guarded by
`C3X_HELPER_TRIAL`; ordinary x86 candidate compilation passes and the installed
DLL is unchanged. No injected C, patch table, installer, or game process is
involved.

**Limit of the evidence:** the scene-only selector omits native UI and unit
operations, ambient ticks, and camera/native composition interleaving. A full
x86 replay of this same 106.5-second recording passed 15,503 calls and 805
accepted presentations, taking 93.73 seconds unpaced in the replay process.
Its scene calls alone consumed much more time when interleaved with native
work than in this 51-scene isolated trial. The monolithic x64 replay diverges
at a CPU unit pixel witness after 3,940 calls; that x86-specific work is
intended to remain with the native side, so the monolithic run is not a valid
split-process control. Gate 1's native presenter was exercised with synthetic
frames; this Gate 2 driver imports real map frames but does not yet present
them under a live Civ III UI. A full game-speed or <33 ms navigation claim would
therefore be premature.

Gate 2 closes the **architecture and capacity decision**, not the gameplay
performance acceptance. The next migration gate puts an x86 native composition
owner and the x64 renderer behind the **same complete recorded call stream**.
It must preserve ordered native partial transfers, unit/action/selection state,
autonomous water and resource time, camera identity, fog and picking while x86
alone owns the game window. Compare whole-workload input-to-correct-frame latency
and memory in both processes, with no hidden queue; include normal, pressure,
reset and helper-crash cases. Cut over only after this integrated candidate
passes and a game-window checkpoint confirms normal interactions. Keep the
working x86 renderer available for fallback during migration. The concrete
ownership and retirement sequence is in the
[64-bit migration plan](helper64_migration_plan.md).

The source-matched normal receipt is the ignored local file
`Renderer/native/build/helper_trial/gate2/runs/145a63ea125b497988be8e83147a818c/receipt.json`.
The source-matched restart and exact GPU-pixel diagnostic is
`Renderer/native/build/helper_trial/gate2/runs/c4a980fd4aaf48f18c27ab0b1ec408a6/receipt.json`;
its readback and import times are excluded from the normal-path timing above.

Run the reproducible trial against a retained local journal with:

```sh
python3 Renderer/native/helper_trial/run_gate2.py --capture Renderer/native/build/input-recording/<capture>/inputs
```

Add `--verify-pixels` for diagnostic GPU readback or `--crash-after 27` for
mid-workload recovery. The runner records source/binary/journal hashes, child
completion, parity and timing in ignored `Renderer/native/build/helper_trial/gate2/runs/`.
Candidate builds and replay do not stage C3X or launch Civ III.

## Full native-interleaved shadow checkpoint

The next probe now replays all 15,503 recorded calls in the x86 production
owner while sending copied scene, GPU-image, GPU-unit, native visual-policy,
presentation and ambient-frame values to an x64 sidecar. x64 composes final
BGRA frames into a reusable shared D3D texture; x86 imports and hashes them.
The source-matched receipt is the ignored local file
`Renderer/native/build/helper_trial/interleaved/runs/de3e73cbab3e45138b7fb5671046df23/receipt.json`.

In the GPU-owned interval before the recorded reset, every executed result
matched. Of 235 imported final frames, 230 were byte-exact. Four mismatches
were one pixel by one color level; the fifth followed a unit pose and affected
385 pixels in a 72-by-53 region, with maximum channel difference 34. All 200
GPU unit bounds matched. Eight of 2,507 diagnostic GPU readback hashes still
differ after four unit poses. The raw `R32_UINT` shared-map trial also retains
exact scene/GPU pixel parity (`gate2/runs/4210bef201ba40398b5be1336f4f5218/receipt.json`).

After the reset, the recording uses CPU/native scene presentation. The x64
sidecar lacks that owner, so its later visual-policy/results are **not** a
valid x64 gameplay comparison. The x86 replay peaked at 2,435 MiB private;
the x64 sidecar peaked at 2,424 MiB. These processes both render in this
diagnostic, so their combined memory and timing cannot predict the split
production path. The 36.7-second x64 service total is unpaced workload time,
not FPS or input-to-display latency. No installed binary was changed.
