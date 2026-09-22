# Durable content and native transaction consolidation

This follows [camera consolidation](camera_consolidation_results.md) within
M3.8. It uses the same recorded live prefix and the existing production owners.
Waves, reflections and water motion remain enabled. No injected changes or new
patch symbols are required.

## Ownership and retirement

| Responsibility | Production change | Preserved boundary |
| --- | --- | --- |
| World compilation | Combined jobs own current observation records and share immutable topology/coast snapshots. `ContentPreparation::schedule` replaces pending demand while retaining active jobs and ready results. The frame-local `compile_world` callback, `WorldJoin` and combined-queue `finish_lease` are removed. | Stable asset/device ownership ends at the existing reset/configuration barriers. Every result is checked against current local dependency proofs, including results completed during a newer view. |
| Camera cancellation | Waiting for a demanded result can return when its camera becomes obsolete; the producer retains its input and useful work. | Cold demanded content can still require a keyed wait. No incomplete frame is published. |
| Native composition | Draws preceding create/upload/destroy/readback execute as a prelude in the same worker packet. The former separate flush/handoff at those boundaries is removed. | Draw ordering, admission refusal, terminal failure, explicit CPU readback and frame/session boundaries remain authoritative. A refused create never replays its already executed prelude. |
| View submission | Adjacent compatible rigid meshes share an instanced draw. Each instance supplies projection, placement, depth and material. | No sorting across intervening draws, alpha ordering or pass boundaries; deformed geometry retains its own representation. |
| Memory admission | Geometry growth and preparation capacity use measured free process VA, including existing assets, native composition and driver mappings. The policy reserves process and compiler headroom; immutable input snapshots have a separate 64 MiB bound and shared topology ownership. | Logical GPU allocation sizes are not physical VRAM or VA. Required frame attachments and protected visible geometry still constrain the achievable headroom. Existing pressure handling suspends optional regional work. |

Compatibility ground/terrain/object compilers remain for explicit worker-off,
unsupported combined-projection and oversized-result cases. Their borrowed leases
are still necessary until those callers move to owned inputs; they are not used
as a second producer alongside combined jobs. CPU-native access barriers and
cold-content waits have not been renamed away or declared eliminated.

## Validation and measurement

The final candidate reproduces all **518** recorded display fingerprints on the
successful prefix: **253,272 calls**, ending at event **508,144**. This does not
certify the killed capture's missing tail. New executable contracts cover active
job replacement without joins or duplication, independent observation lifetime,
reset cancellation, resource preludes, admission refusal and terminal failure.

The original native fixture had 2,014 draw batches and 6,402 worker calls. The
new transport executes those batches with 4,475 calls (30.1% fewer handoffs).
This is a boundary count, not a claim of equivalent FPS improvement.

Two paired runs replay the same successful prefix with all water effects on.
These are unpaced native API service measurements, **not live FPS**. Each run has
569 actual presentations, 345 successful ambient offers and 34 map boundaries.

| Measurement | Before (two runs) | Current (two runs) |
| --- | --- | --- |
| Service envelope | 34.97 / 34.02 s | 35.04 / 29.50 s |
| Map boundary median | 12.20 / 14.54 ms | 12.36 / 11.17 ms |
| Map boundary p95 | 342.35 / 336.50 ms | 332.69 / 378.64 ms |
| Cold maximum | 7.73 / 7.64 s | 7.25 / 6.93 s |
| Ambient p95 | 34.49 / 27.95 ms | 37.02 / 24.90 ms |
| Peak private memory | 2968 / 2914 MiB | 2926 / 2893 MiB |
| Minimum free VA | 562 / 678 MiB | 636 / 650 MiB |
| Minimum contiguous VA | 422 / 523 MiB | 491 / 507 MiB |

Cold preparation improves about 6–10% in these pairs. Warm tails and the service
envelope remain variable; there is no demonstrated stable overall FPS gain.
Contiguous headroom still falls below 512 MiB. The Standard <33 ms p95 camera
acceptance, complete live overlap/heap fragmentation, and failed live tail remain
unfinished. The next responsibility is shortening the exact map boundary through
view/pass assembly, GPU execution and native composition, using this same workload.

An intermediate pressure policy reduced ready-result capacity to 32 MiB. With
16 MiB reserved per active job, one completed result could effectively serialize
four requested workers. Its map p95 regressed to 541 / 411 ms; that candidate was
rejected. The current budget includes all requested lanes plus a ready slot, and
an executable pressure/concurrency test prevents that failure from returning.

Evidence is under `native/build/`: `durable-content-release/comparison.json`,
`durable-content-measure-before/receipt.json`,
`durable-content-measure-release/receipt.json`, and the release `replay/` directory.
The release DLL SHA-256 is
`9e76737a26cf4a1c282e512c97d2b2c6b9f7e979a6b3565215a3e7eca0c3941a`.
Its MSVC x86 build passes `/W4 /WX`; 45 distinct selected executable contracts
pass. The final four-arm off/on/on/off capture campaign passes all median/p95
overhead thresholds at 2240×1260 with 1 GiB VA reservation. Fullscreen window
capture, requested stop, native recovery and two exact 960-presentation replays
pass. The DLL is staged with a matching `short-capture-ready.json` receipt;
no game or INSTALL was launched. Ten-minute and live-FPS qualification remain off.
Capture evidence is in `native/build/input-recording/durable-content-capture-*`.
The short launcher preflight and installed DLL identity pass; the unqualified
long-capture path still refuses to launch. Re-run `INSTALL.bat` if the preceding
camera-consolidation bridge changes have not yet been installed.

Generated images and measurement JSONL are losslessly gzip-compressed, with
SHA-256 verification before removing their uncompressed copies. Disposable
intermediate objects are removed; final incremental objects, DLLs, receipts,
fingerprints, original inputs and rollback remain. This recovers 1.83 GiB; the
file-level ledger is `native/build/durable-content-release/storage.json`.
