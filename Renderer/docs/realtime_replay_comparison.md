# Watching a renderer change Before and After

`Renderer\COMPARE_REPLAY.bat` runs a recorded workload twice on Windows: **Before**,
then **After**. Choose the input session and the candidate DLL. Before defaults to
that session's frozen DLL. Both DLLs must support real-time replay; older captures
can still supply inputs, but an older DLL may need a diagnostic build with the
same rendering implementation and the new playback exports.

The launcher verifies the journal, freezes both DLLs and the player, copies the
journal to the VM's local disk before timing, and runs each build in a fresh
process. It uses the same recorded input times, configuration, assets and optional
address-space reservation for both. It stops on failure or Escape. Results and
binary hashes are saved under the session's `comparisons/` directory. Assets stay
in their original local installation; they are verified, not redistributed.

For a prepared comparison, explicit paths avoid dialogs:

```powershell
.\Renderer\COMPARE_REPLAY.bat -SessionDirectory RECORDING -BaselineDll BEFORE_DLL -CandidateDll AFTER_DLL -ReserveMiB 1024
```

`-OutputDirectory NEW_DIRECTORY` selects the output location. `-CheckOnly` validates
files without playing. A same-DLL pair is reported as a repeatability control.
Runs are sequential so the renderers do not compete for the GPU. They remain
ordered experiments: OS caches and unrelated machine activity can differ. Repeat
or reverse the pair before attributing a small timing difference to a code change.

## What changes with a faster renderer

External game inputs retain their recorded timeline and order. The real-time mode
replaces recorded ambient offers with the production scheduler and its ordinary
wall clock. Water, resources and eligible unit loops can therefore produce new
intervening frames. Faster frame production improves smoothness without speeding
up the animation clock or native action/input sequence. Current production cadence
targets 33 ms, with a 10 ms pause after slow frames; this tool does not silently
replace it with a faster test-only scheduler.

A slow candidate can miss input deadlines. The player preserves the inputs and
reports lateness; it does not drop game changes or hide delays by rescaling time.
The JSONL includes native service time, dispatch time, input lateness, process
memory samples and independently accepted ambient-presentation timestamps. These
are presentation acceptance measurements, not physical monitor scanout. Pending
ambient offers and telemetry overflow/errors are explicit. No per-frame readback
or image export is added to normal comparison playback.

## Keep the correctness control

`PLAY_REPLAY.bat` / `--watch` remains exact forensic playback. Its recorded logical
clocks and output witnesses establish correctness separately. The new mode is
`replay_inputs --development DLL INPUTS --compare-candidate --realtime NEW_JSONL`.
It allows time-dependent pixel/pose samples to differ, retains structural/lifecycle
checks, and disables forensic image export/fingerprinting. Both comparison arms
use this same real-time mode; comparing forensic Before to real-time After would
confound the player change with a renderer improvement.

Native camera/CPU handoffs still use recorded consumption points because the
external CPU inputs needed earlier are not available. The feeder also preserves
completion order; it does not simulate different future decisions by Civ III.
Game heap fragmentation, driver scheduling and live GPU contention remain subject
to calibration. A smooth replay is useful visible evidence, but its baseline must
match the live symptoms before its FPS predicts live performance. This feature
does not change the admission gate for another long gameplay recording.
