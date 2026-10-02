# Compiled composition and static validation

Current qualification: 2026-10-02, against renderer commit `ef5f612a`.
This step improves stationary delivery but regresses ordinary camera redraws;
the accepted installed runtime has been restored. The candidate is retained as
source and a locally qualified build, pending the next performance step.

## Implementation

Compatible native image/HUD operations execute as ordered per-pixel spatial
programs. Immutable operands, source atlases and compiled preparation survive
equivalent map publications; each publication retains independent before-images
and rebinds its current output pair. Unsupported dependent reads and capacity
refusals preserve ordered interpreter execution. Exact source identity and
revision checks govern reuse, including lexical HUD redraws whose wrapper
serials change while their immutable patch contents remain identical.

Static raster consumers watch exact producer keys, including missing inputs,
through a resource-free 4096-entry change window. Unchanged proofs avoid repeated
content, visibility and membership scans; changed contexts, reset/overrun and
ownership changes require complete validation. River flow and coast producers
participate. Ordered receiver grids reuse unchanged contributors, and coverage
memoization weakly references completed generations.

Denied presentation permits return BUSY for the existing bounded retry; unchanged
static fronts remain PENDING. Once-per-second helper counters distinguish attempt
outcomes and report batch queue/execution/retirement wall spans. The renderer
keeps one immediate-context owner, no new production readback or GPU wait, the
256 MiB native and 128 MiB replay budgets, 512 public image handles and eight
recipe-history entries. No injected source, patch symbol or export changed.
`required_user_action: []`.

## Matched results and limits

Both arms used the same disposable save, route, common diagnostic injected
executable, 2240×1260 client and normal water/waves/reflections. Busy idle had
77 main/shadow/reflected units and 438 part samples; the fixed initial-4000BC
fixture had one unit and four part samples. Actual source/workload witnesses
matched throughout idle and at all 29 busy / 12 early-game noncanceled endpoints.
The intentionally canceled outward request is excluded.

| Measurement | Control | Candidate |
| --- | ---: | ---: |
| Busy idle successful presentations/s | 14.61 | 21.12 (+44.5%) |
| Busy idle mean / p95 Present interval, ms | 68.54 / 103.51 | 47.27 / 78.39 |
| Early-game idle successful presentations/s | 45.08 | 54.63 (+21.2%) |
| Early-game idle mean / p95 Present interval, ms | 22.13 / 38.93 | 18.31 / 34.37 |
| Busy ordinary-scroll mean correct-view latency, ms | 1117.94 | 1799.94 |
| Busy zoom mean correct-view latency, ms | 743.62 | 1594.38 |
| Early-game scroll / zoom mean correct-view latency, ms | 543.97 / 517.98 | 701.80 / 556.13 |

Correct-view latency runs from native acceptance to successful Present of an
explicitly identified map source at the requested projection. Most additional
busy-scroll latency precedes camera adoption; zoom adoption is immediate but
delivery regresses afterward. The parallel audit identifies expensive dependency
registration during redraws as the next investigation. This step does not claim
a confirmed single cause or improved interactive responsiveness.

Busy idle preparation overlap fell from 29.36 to 20.81 ms, residual composition
from 25.23 to 16.28 ms, and the unclassified cadence/ownership gap from 13.83 to
10.05 ms. Warm counters show no additional source binds, atlas copies or spatial
plan builds; the compatible live HUD block has approximately 2125 commands per
spatial dispatch. Whole-route intervals over 100 ms fell from 338/1371 to
223/1826, but the maximum interval rose from 881 to 1050 ms. Cold/warm/far-return
endpoint results are mixed; no eviction is inferred from camera distance.

Quiet-counter cross-checks were 14.57→20.63/s busy and 50.19→53.98/s early-game.
They omit source-qualified successful-Present witnesses and cannot certify
same-frame correctness or causal speedup. The smaller quiet early-game gain
shows sensitivity to detailed tracing. Present intervals and helper spans are
CPU/API wall measurements, not GPU duration or physical scanout. BUSY combines
permit denial and call-gate contention; no OS timer-quantum attribution is made.

## Verification and local evidence

The frozen candidate passed 556 exact Windows GPU oracles and 120 independent
clock frames, covering native 555/565/full color, clipping, aliasing, mutable
writes, paired operations, 900+ immutable sources, publication/pair rebinding,
lexical redraw reuse, weak retirement, refusal and unchanged caps. Host groups
passed 34 static/dependency tests, 19 cadence/protocol tests and 22 lifecycle,
camera, scroll, preparation and config-off tests. These groups can overlap.

Bounded gameplay exercised zoom/text and visibly opened/closed the advisor.
The movement command left the unit on the same sampled tile, so actual movement
is unverified. Disposable saves and game configuration remained unchanged;
all owned processes/tasks closed. The exact accepted executable/trio, directory
link, cursor and renderer environment were restored and hash-verified.

Private evidence is under `Renderer/.cache/compiled-composition-step/`:
`current-build-r4-source.json`, `current-build-r4-build.json`,
`gpu-oracles-r5.json`, `full-busy-r4-paired-analysis.json`,
`light-r4-paired-analysis.json`, both `quiet-*-r4-paired-analysis.json` files,
capture manifests/cleanup receipts and `accepted-runtime-final-verification.json`.
All 802 frozen inputs remain identical except the subsequently corrected
lifecycle test fixture, whose current version passed the 22-test host group.
Failed trials and original evidence remain local. Only verified rebuildable
compiler intermediates and hash-identical VM capture duplicates were removed;
source assets and archived captures were preserved. No reference image changed.
