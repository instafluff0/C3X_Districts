# Civ III adapter and shared frame preparation

This implements performance-audit block 6 against `33986a93`. It consolidates
existing renderer consumers and preserves their current ownership. It adds no
renderer ownership for deferred wonders or Districts and changes no patch-table
entry. The patch dependency ledger lists the existing hook contracts and has
`required_user_action: []`.

## Prepared data and removed work

| Owner | Exact dependencies and lifetime | Removed work and remaining work |
| --- | --- | --- |
| Dynamic map input | Normalized copied frame fields, tile records, topology, identity; weak reference to the current owner | Identical admission shares the immutable input. There is no history cache. Changed input gets its own owner. |
| Prepared map frame | Shared copied dynamic input plus the existing epoch, projection, device and serial checks | Removes another full frame/tile copy. Back texture and unit poses remain independently charged. |
| Body requirements | Source generation, exact layer/member identity, live placement source, projection/translation/depth and material version | One deduplicated requirement set serves coverage, proof and pass packing. Renderer64 no longer also prepares the superseded native body ranges. Legacy/oracle paths keep their own consumers. |
| Replacement instance union | Same live resident slot and generation, source/material/count and raster dependency proof | Validated unchanged ranges are copied between GPU buffers. Only changed records are packed and uploaded by the CPU. The old buffer is immutable and remains charged while readers pin it. |
| Unit asset selection | Catalogue/device/mesh/texture generations, exact selected unit/action set and resource liveness | A completed exact union skips repeated asset discovery and scheduling. Frame leases are begun every time; incomplete work still retries. Liveness probes remain. |
| Published front | Actual retained/canonical/projected/front source generation and existing view guards | Shared fronts retain source provenance. Publication never substitutes a local image ticket for a GPU source serial. |
| Repeated native recipe | Complete scalar command and clipping, six operand pictures, ordered immutable patches and operand alias relationships | An eight-entry weak index shares an eligible full-canvas keyed native transfer or opaque expansion. Changed inputs or later destination writes retain independent versions. |

Body requirements and retained generations share a 32 MiB owner budget. The
requirement key is 24 words; sorted indices and entry storage are charged. Asset
proofs are bounded to 4,096 unit/action keys, 8,192 resource IDs and 96 KiB of
metadata; the existing 96 MiB asset payload budget remains. Replacement buffers
and old-reader pins are charged together. A temporary copy-source pin is released
after submission; independent reader leases remain valid. The implementation
adds no CPU map of production instance buffers or synchronous GPU wait.

## Native text

The real native scientific-leader message reproduces a 106-byte `TextOutA`
operation using Lucida Sans, baseline alignment, transparent background and a
simple map-sized clip. Its response raster is 543 by 42 pixels. The previous
16,384-pixel admission rejects that rectangle and requests unsupported CPU
ownership of an owned map image.

The adapter now shapes the whole native string on its DC-owning thread for each
bounded vertical response strip. It packs one immutable GPU response and submits
it in the existing map/detail order. It preserves the captured font, native DC
state, alignment, clipping, background and ordering. Strings are never split for
shaping, and destination map pixels are never read back. Unsupported state still
refuses explicitly.

Scratch admission remains 16,384 pixels; the packed response is bounded to
32,768 pixels/128 KiB. Curve admission remains 1,024 with at most 68 KiB of curve
storage; text is bounded to 1,024 bytes and the response cache to 8 MiB. Placement
uses widened checked arithmetic. The native qualification also exposed and
corrected odd-width `TA_CENTER` rounding: a 515-pixel advance needs native ceil
half-width placement.

The original interturn failure's exact historical message remains unproved.
The current actual-game message reproduces the same size/ownership refusal and
then succeeds through the GPU path. The private opt-in F23 fixture emits the
loaded player's existing official message; it changes no research, unit, RNG,
turn or save. Its absence leaves ordinary F23 testing unchanged.

## Gameplay capture

Movement invalidation uses the native configured maximum sight range, bounded
from 0 through 7. Raw isometric offsets `(u,v)` become `(u-v,u+v)` and retain the
native parity lattice. The old/new union is wrapped or clipped using actual map
facts and paged through existing 128-record capture calls. Older DLLs retain the
vanilla export and use paged fallback for larger ranges. Local appearance
invalidation keeps its existing radius.

Existing selection/member/action/camera hooks request a bounded producer refresh
after native attacker acceptance. Army member state is observed before the body
clock is selected. New renderer patches immediately delegate with unchanged
arguments when rendering is disabled; additions to shared patches remain gated
without disabling other C3X features. Executable tests cover these contracts.

## Measurement contract

Timing uses the approved 1498 AD save at 2240 by 1260 with normal water, waves and
reflections. Loading is separate. A native accepted camera command, actual
copied transport ticket binding, source publication and completed `Present`
witness must agree before destination latency is admitted. Local camera/image
tickets and remote camera/map tickets have distinct namespaces. Canceled
commands, missing or mixed sources, contradictory generations and unmatched
representative workloads remain unqualified.

Seven disjoint CPU/API subspans partition preparation: setup, capture, raster
proof, body requirements, city shadows, unit selection and unit poses. They do
not add to the enclosing preparation interval a second time. Preparation nested
inside composition is subtracted when reporting residual composition. `Present`
QPC is sampled immediately after the call. These are CPU/API and successful
presentation witnesses, not GPU durations or physical scanout.

The immediate-baseline control receives only the common route/ticket witnesses;
both arms use the same diagnostic injected executable. This is an instrumented
renderer comparison to `33986a93`, not a pristine full-system binary comparison.
The earlier `ea43e511` evidence retains its differing routes/contributors and
cannot establish a speedup. One pair cannot establish a causal performance gain.

## Qualification and remaining limits

Final candidate compilation, injection smoke, native GDI text qualification and
the actual GPU buffer-copy fixture pass. The captured text suite checks 136 GDI
samples and 16 GPU order cases; the existing response-interpolation edge bound
remains two channel levels. A real DEFAULT-buffer fixture verifies unchanged
GPU copies, changed CPU uploads, immutable pinned old fronts and complete budget
retirement. Cropped fixture readbacks are test-only.

The actual candidate emits the previously refused scientific-leader message,
completes two distinct interturns, and passes reveal/hide with subsequent native
handoffs. Bombard and army selection were also observed before any movement in
bounded disposable-save games; those captures use the earlier candidate binary
before the final observational ticket and text changes. Accepted retained
membership and lossless visual captures support those checks; they are not an
exact pose-to-Present proof.

A complete local NTFS input journal verifies 4,313,442 events across fourteen
segments and contains ten accepted city native-metadata changes and sixteen
native-overlay changes. Examples include population changes and new railroad
flags. These are authoritative paired-call facts, not a pixel or presentation
comparison; no city body-class transition was observed. A separate run using the
same candidate completed both requested turns. The earlier UNC recording stopped
at its existing queue limit and remains an incomplete startup prefix.

The heavier complete-journal run failed during its second turn when a retained
output allocation would exceed the existing 256 MiB texture limit. Its graph
contained repeated full-canvas native operations. The reuse implementation
checks exact live recipes before sharing their operation; a diagnostic dump alone
cannot establish equality. The limit, pinned ownership and failure receipts are
preserved. The final candidate completed both turns without native errors or a
texture-limit refusal, but its journal did not close within the existing cleanup
bound. Its 4,798,039-event prefix verifies and contains seventeen city metadata
and twenty-seven native-overlay changes; the missing footer prevents complete
recording qualification. The raw harness result remains failed. Reload reseeds
the game's RNG, so this does not prove reproduction of the earlier failing DAG.
A separate read-only inspector uses the production segment verifier
and field codecs, respects partial city and native-overlay permissions, and
bounds its output. Missing facts are never inferred from acknowledgements.

Actual configured-range-7 seam movement and offscreen attacker cycling remain
unobserved in the game. Production-extracted executable tests cover the raw
range-0–7 footprint, wrapped edges, hidden-unit exclusion, selection, local
invalidation, reset, navigation and config-off delegation. Those tests do not
replace an unperformed native game scenario.

Actual configuration-off evaluation displays the native map, units, labels and
HUD with no renderer helper/module in the sampled observations. Three lossless
captures were checked; the original runtime link, save, INI, cursor and
environment were restored exactly. Absence between polls is unproved.

Final route measurements, runtime restoration and exact evidence bindings are
recorded with the completed step receipt. No visual reference was replaced and
no gameplay change was saved.

## Integrated measurements

Both final captures complete the same requested route. Idle has identical 77
unit IDs and main/shadow/reflection membership. Twenty-eight of thirty steps
have matching actual routes and representative workloads: control scroll 12
lacks a unique source-publication witness, and outward reversal 20 is superseded.
These exclusions retain the original guards.

| Qualified observation | Instrumented `33986a93` | Candidate |
| --- | ---: | ---: |
| Idle successful Present gap, mean | 79.327 ms | 69.048 ms |
| Idle successful Present gap, p95 | 138.267 ms | 107.049 ms |
| Idle preparation/render overlap, mean | 36.888 ms | 29.451 ms |
| Idle residual composition, mean | 28.283 ms | 25.986 ms |
| Idle Present call, mean | 0.128 ms | 0.128 ms |
| Idle unclassified cadence/queue/ownership gap, mean | 14.028 ms | 13.483 ms |
| Fifteen admitted scroll destinations, mean acceptance-to-Present | 1,178.017 ms | 1,023.459 ms |
| First visit / warm return | 7,061.749 / 780.505 ms | 6,899.475 / 677.782 ms |
| Pressure return, eviction unproved | 522.127 ms | 403.139 ms |

The idle means partition the actual Present interval; the enclosing composition
span is not added again. The candidate still performs about 2,152 retained
operations, 204 copies and 32.7 million copied pixels per idle frame. Its work is
still far above a 16.7 ms budget. Preparation/render and residual composition
dominate; the gap is unclassified, not automatically attributed to sleeping.

There are regressions: zoom 160 takes 875.891 ms versus 784.503 ms, and the
candidate's residual-composition median is 30.097 ms versus 25.519 ms. Whole-route
Present p95/max are 212.952/607.541 ms versus 212.181/529.184 ms. Those whole-route
distributions have differing view weights and are not workload-normalized
comparisons. The earlier R3 pair had an idle regression. The final pair's lower
idle and mean scrolling costs therefore do not establish a stable causal or
all-workload speedup.

Final work counters show reused body requirements with zero new visits/builds,
2,937 unique coverage probes and 3,145 duplicate requirements eliminated. Across
the candidate run, validated union replacement copies 15,087,680 bytes on the
GPU, uploads 1,380,992 host bytes, and allocates 16,468,672 cumulative bytes.
Asset union builds/schedules total 55 with 984 reuses and 81,672 live-resource
probes; these probes remain real work. The native recipe index observes 1,100
eligible operations, 1,050 probes and 33 exact reuses. The GPU fixture checks 352
exact outputs and keeps thirty-two matching saved pairs at 24 MiB without
raising the 256 MiB cap. Fixture readbacks remain test-only.

Helper private-byte peaks are 4,631.004 MiB in control and 4,680.543 MiB in the
candidate; final samples are 4,587.090 and 4,635.469 MiB. Game peaks are 381.133
and 390.980 MiB. The short run ends near its helper peak; it does not certify a
long-session plateau. Loading is separate at 32.338/32.135 seconds.

Private evidence is bound by
`.cache/adapter-ownership-step/delivery-receipt.json`, with measurements in
`matched-route-final-analysis.json` and `matched-route-final-summary.json`.
The final source and frozen build agree on all 795 compile inputs. The approved
injected smoke uses the unchanged injected sources. Accepted executable, trio,
configuration, verification link, save, cursor and environment are restored;
owned game/helper/collector/tasks are closed. Private config-off runtimes were
unlinked without traversing their source links, and their raw evidence remains
preserved. The root audit and unrelated live scene are excluded from this change.
