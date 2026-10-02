# Prepared resident submission

The renderer retains generic source meshes and one immutable union of 64-byte
placement records. Main, reflection and shadow passes borrow that union through
four-byte selected indices; viewport, light/page and wrapped shadow offsets stay
in small pass constants. Exact source generation, version, material, projection
and anchor facts authorize reuse. Traversal order remains separate from placement
identity, so repeated occurrences still draw in their original order. Opaque
grouping requires compatible bindings and independent coverage in both main and
reflection passes; overlap, depth ties, alpha, decals and native ordering remain
barriers.

Static raster translation must preserve both whole-pixel sampling and the
two-by-two pixel groups used by terrain height derivatives. An odd projected
pixel displacement redraws the static region from resident geometry; MSAA sample
count does not change this pixel-group requirement.

Known permitted world regions prepare through the same production compiler during
loading, then continue on bounded worker turns. Canonical full-detail content uses
a fixed 128 by 64 tile basis and keys actual authority, assets and quality rather
than target dimensions or camera coordinates. Legacy projection/detail remains a
separate identity. Missing authority stays unavailable until native capture supplies
it. The initial loading opportunity is bounded to 60 seconds; a ready displayed
map alone does not certify that every world region has completed preparation.
Tile fallback reuse validates the complete compiler context and captured
dependencies before renewing a publication revision. A matching lookup signature
alone cannot relabel legacy content as canonical or change its detail policy.

Viewport assembly reuse also validates its selected resident owners before
borrowing their mesh ranges. Native camera facts alone cannot prove unchanged
cross-tile world appearance. This check covers current compile quality, source
scope/assets and complete captured dependencies, including shared natural sources;
the existing observation and anchor memo keeps unchanged owners cheap to validate.
It visits the current capture's selected handles and does not scan the whole world.

Prepared compiler output has a delete-on-close backing owner capped at 1 GiB of
physical file extent and 16 MiB per record. Freed ranges coalesce, trailing space
trims, and bounded compaction prevents generation replacements from consuming an
ever-growing file. Evicted geometry can recover from this backing without terrain
compilation. The bounded readiness fixture requires cumulative compiler checkpoints,
actual eviction/restoration, six independent cold oracles and repeated-route memory
plateau; absent trace events do not count as zero work.

Bounded indexed hill preparation emits rock and canopy-floor triangles through
one first-reference index table, using all 168 bytes of the original vertex
identity. The table and old/new vector allocation overlap count against the
existing 8 MiB transient gate; its scratch retires before mesh packing. Packed
vertices, indices, order and dependency proofs match expanded compilation.
Unbounded recovery and the legacy expanded compiler retain their original paths.

Unit contribution bounds use at most 8 MiB of compact CPU metadata. Main, reflected
and ground-shadow candidates form one conservative required union before expensive
asset/pose work. Unknown bounds remain eligible. Captured types prepare on demand;
raw payloads are released after decoding, and GPU/payload eviction preserves bounds.
Native visibility, wrapped anchors and authored/interrupted pose bounds remain part
of selection.

Only the current placement union is retained for reuse. Camera reassembly or
lighting changes can borrow it when all required ranges already exist. A genuinely
new range requires a replacement immutable union, including a copy of unchanged
records. Required ranges have first claim; a replacement can then carry valid
resident world ranges within an 8 MiB limit for the complete optional union.
Packed CPU values, weak source identities and temporary source-handle metadata
share the 32 MiB joint allowance. The existing resident slot, immutable generation
and dependency proof reject expired or changed sources. Placement retention does
not pin an earlier view's draw meshes; active pass leases own the sources they
actually bind. Selected/inflight consumers keep retired source and placement
allocations charged until their final leases release. This implementation does
not preload every unit/action or add a camera image cache.

Retained source proofs index the unique first placement in the immutable union;
the full dependency key remains in its range table. Carry traversal still follows
that table's original order. Completed old selection plans retire before a
replacement union is admitted, while active readers keep their leases and charges.

| Owner | Bound and accounting |
| --- | --- |
| Shared placement submission | 32 MiB joint CPU/GPU limit, including pinned generations, replacement staging, exact-key metadata, selected-index plans and stream/scratch storage. At most 65,536 placement records and 16,384 exact-key entries. |
| Rigid source buffers | 32 MiB GPU limit and 64 MiB joint staging/retained CPU/GPU limit, including retired pinned sources. |
| Natural instance source buffers | Independent 32 MiB GPU and 64 MiB joint CPU/GPU limits with source deduplication staging and retired pins charged. |
| Shared selected-index stream | 256 KiB GPU stream plus 512 KiB reserved CPU scratch, charged within the shared 32 MiB limit. Warm immutable selection plans also share that allowance. |
| World geometry | 2 GiB x64 ceiling; 768 MiB x86 ceiling. Measured process and adapter headroom can reduce admission. Selected and retired readers remain charged. |
| Compiler recovery | Ready pool: at least two, at most seven 16 MiB slots. Future headroom reserve: 512 MiB plus the ready pool and up to six 48 MiB worker lanes. |
| Prepared backing | 1 GiB physical session-file ceiling; 16 MiB raw/packed record ceiling. |
| Fresh shadow field | 25 fixed 1024 by 1024 R32 pages: 100 MiB, plus the actual still-held production field (normally 128 MiB) and an 80-byte constant buffer. |

Shadow sampling derives from the current receiver extent and exact light/wrap
facts before reuse. Global light-plane cells drive derivatives and all nine PCF
taps; canonical page projection stays identical across physical slice changes.
The current cold float span caps density, including its boundary rounding case.
The fixed field reuses its physical allocation across camera rebases and quality
changes. The host oracle verifies indexing and coverage; native pixel parity is
qualified separately.

These are owner limits, not a total renderer memory budget. Prepared terrain,
textures, scene targets, units and shadow fields have separate accounting. The
`memory-resident-owners` trace reports `shared_joint`, `shared_cpu`, `shared_gpu`,
`shared_peak`, `rigid_joint`, `rigid_gpu`, `natural_joint` and `natural_gpu`.
`fresh_shadow_working_bytes` accounts for the current 100 MiB fixed fresh shadow
field and its still-owned 128 MiB production field when both exist.

The helper creates one master NT handle for its shared display texture and
duplicates it for each delivery, following the
[CreateSharedHandle contract](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_2/nf-dxgi1_2-idxgiresource1-createsharedhandle).
The native consumer retains one imported texture, keyed mutex and private handle.
It reuses them only when kernel-object identity, consumer device and descriptor
match; numeric handle equality cannot authorize reuse. The logical frame alias is
bounded by checked width times height times four, at most 16 MiB, and refers to the
producer's allocation. Unsupported identity comparison or failed private handle
duplication uses the existing uncached route. Replacement, reset and failures
retire that owner; a pending presentation retains it. Keyed-mutex ordering and the
existing two copies and Flush remain unchanged. This accounting does not measure
physical driver allocations or prove when queued driver references retire.

The x64 DLL compiles shaders from `C3X_RENDERER_SHADER_SOURCE_ROOT`, or from
`Renderer/packs/Renderer64CutoverControl` when that override is unset. A resident
submission DLL requires new vertex entries in six shader files. Preserve the exact
pinned baseline and prepare a separate candidate from the repository root:

```sh
python3 Renderer/tools/prepare_resident_submission_shaders.py \
  --baseline Renderer/packs/Renderer64CutoverControl \
  --out Renderer/.cache/resident-evaluation/shaders
```

The tool copies the complete baseline, appends only generic resident input/decode
and vertex adapters, verifies every inherited prefix and unrelated file, and
publishes a new output directory. Existing output, input changes, missing inherited
entries and resident namespace collisions are rejected. Its
`resident-submission-overlay.json` pins the baseline inventory, candidate shader
hashes, adapter sources and tool hash. `--baseline`, `--out` and `--source-root`
are configurable; no pinned pack is modified.

The added entries are `VSResidentInstance` for natural bodies and canonical
casters, `VSResidentReflectionInstance` for natural reflections,
`VSResidentPlacedInstance` for wrapped casters, and the corresponding
`VSResidentSharedFeature`, `VSResidentSharedFeatureReflection`,
`VSResidentSharedCaster` and `VSResidentPlacedCaster` rigid entries. The source
vertex stream remains 32 bytes; selected `uint` indices use `TEXCOORD1` in the
second stream and address `C3XResidentPlacements` at vertex SRV slot `t15`.
Inherited material and projection functions remain unchanged.

Compile-only verification remains available through
`Renderer/native/BUILD_RENDERER64.bat no-stage` (optional second argument
`oracle`). Staging checks all required resident entries and input bindings before
copying any binaries. An old pinned runtime therefore refuses ordinary staging
when the override is unset. The startup probe checks bootstrap health; rendering
and native shader compilation still need their executable qualification fixtures.

For explicit evaluation, close Civ III and use a Windows command prompt at the
repository root. Select the absolute candidate path before building/staging:

```bat
set "C3X_RENDERER_SHADER_SOURCE_ROOT=%CD%\Renderer\.cache\resident-evaluation\shaders"
call Renderer\native\BUILD_RENDERER64.bat
```

Keep the matching `C3XRenderer.dll`, `C3XRenderer_x64.dll` and
`C3XRendererHelper64.exe` under `Renderer/bin/renderer64`; preserve the exact trio
and runtime receipt/hashes together. After qualification, run the ordinary
`INSTALL.bat` for the matching injected bridge. The evaluation game process must
inherit the same absolute `C3X_RENDERER_SHADER_SOURCE_ROOT` from its launch command
prompt; staging's local environment does not configure a separately launched game.
Installation does not build shaders or prepare assets. Staging for evaluation
does not accept visual changes or replace reference images. This document records
the implementation contract; measured results and native qualification belong to
the delivery evidence.

### Qualification and measurements

The final oracle build and production build use the same 929 input hashes.
Both native capacity cases execute at the full 2240 by 1260 guest desktop and
pass their original guards. In the asynchronous case, all 32 camera steps build
zero geometry; 24 upload zero geometry and eight upload a total 18,769,920 bytes.
Ready adoption has median 312 ms, nearest-rank p95 516 ms and maximum 531 ms;
maximum client poll is 3.057 ms. These are CPU/API adoption measurements. The
shared submission's charged peak is 30,064,832 bytes against its 33,554,432-byte
joint cap. They do not measure total process memory, physical VRAM or scanout.

The prepared-world fixture completes first visits, forced eviction/recovery,
reversals and repeats. Cumulative compiler checkpoints remain unchanged across
all 16 timed views; 3,267 contributor restore calls recover from prepared backing; this is not a
count of unique contributors. Repeated
views build and upload zero geometry. Request median/p95 are 72.368/93.560 ms for
seven repeated views; outer desktop-completion median/p95 are 111.524/245.435 ms.
First-view request median/p95 are 147.658/326.889 ms; prepared recovery request
median/p95 are 505.349/732.505 ms. Zero compiler work does not eliminate those
measured adoption stalls.
The first repeat sweep grows resident geometry by zero bytes; reported virtual-address
consumption growth is 4 KiB and then zero on the next sweep. This is a bounded
memory plateau, not proof of physical driver allocation reclamation.

The derivative-phase guard resolves the material dense shift-return mismatch:
74,060 changed pixels previously become zero in all 11 collected dense color
checks. A separate unchanged strict dense fixture still stops on one green
channel difference of one value. Plain comparisons retain nine strict failures
(maximum RGB difference seven); rebase and unseen cases are exact. Six world
cold comparisons retain strict color failures; the largest changes 0.5698% of
pixels. Root visual review of actual-size samples judged the remaining foliage and
terrain shading differences immaterial under the user's allowance. This is a
bounded visual judgment; the worst world comparison still has 16,083 changed
pixels and maximum RGB difference 94.
All original failures remain recorded. Pixel-perfect, semantic-alpha and direct
depth-buffer parity remain unqualified; no reference or tolerance was changed.

Validated unchanged work now reuses canonical compiler output, source buffers,
material/rig/self-shadow preparation and the same shared allocation's imported
resource/master handle. Indexed wave/hill storage removes expanded corner
duplication, and conservative pass selection avoids preparing unselected units.
A covered placement union reuses its immutable data; replacement unions still
copy/upload unchanged records. The measured live scroll upload totals are higher,
so this is not a claim that all repeated uploads were removed. Required
transparent/decal/native ordering, changing poses/lighting, two texture copies,
Flush and Present remain. Adapter consolidation is the following assignment.


Four real-save captures use 2240 by 1260, normal water/waves/reflections, a
certified ready map and the same five-second settle/input schedule. Values below
are descriptive successful presentation-counter increments per second; the
input column includes the fixed one-second recovery window.

| Run | Launcher ready, baseline → candidate (s) | Input/recovery rate | Settled tail rate |
| --- | ---: | ---: | ---: |
| Scroll 1 | 46.661 → 51.074 | 6.822 → 6.016 | 13.385 → 11.947 |
| Scroll 2 | 50.732 → 50.002 | 7.462 → 6.040 | 14.220 → 11.024 |
| Zoom 1 | 33.879 → 33.590 | 9.017 → 7.327 | 18.165 → 15.743 |
| Zoom 2 | 31.496 → 32.510 | 8.652 → 7.575 | 18.954 → 15.895 |

Ready render-span medians also increase: 22.128 → 34.155 ms and 22.501 →
33.9575 ms for the scroll pairs; 52.4845 → 69.9005 ms and 21.858 → 62.1555 ms
for the zoom pairs. These are CPU/API spans, not GPU duration.

These captures show no presentation-rate gain. Posted inputs match, but actual
native camera targets differ in both scroll pairs, and zoom 2 has seven baseline
versus eight candidate width transitions. Only zoom 1 passes strict request
metadata eligibility; even there the reported contributor workloads differ.
All initial captures copy 77 records for 76 representatives. Ready main/pose/part
counts remain 76/76/431, while reflections increase from 72 to 76; settled zoom
main/reflection counts change from 76/48–49 to 74/73. These are literal prepared
contributors, not a certified list of displayed unit IDs.

Sparse input-phase frame events report 472 → 399 builds and 66,060,664 →
90,495,312 upload bytes in scroll 1; scroll 2 reports 468 → 435 and
66,003,160 → 94,158,576 bytes. They do not cover all background/init compiler
work. Zoom has no such frame events. Exact correct-view latency, complete live
compiler coverage, GPU timestamp duration and physical scanout remain unavailable.
The preserved `0469689f`/`ea43e511` startup-overlapping measurements remain
historical context, not a matched control for these ready-map windows. The
end-to-end performance target remains unmet; 60 FPS is not demonstrated.

Recorded host checks include the affected 130-test run (one skip), a later
41-test ownership/submission run, and three final derivative/raster checks.
These are separate recorded runs, not an aggregate unique-test count.
The oracle and production builds share all 929 frozen inputs. A final test-only
whitespace cleanup leaves compiled/runtime inputs unchanged. Runtime comparison
finds 22,059 unchanged dependencies; thirteen intentional owned source/generated
shader changes and two new owned shader files account for all differences.
No other asset, configuration or save dependency changed.

The required two completed interturns remain **failed and open**. The final
bounded production run begins native turn 504 and completes none. At 58.420 s,
a native text fallback requests CPU ownership of the full surface; the existing
asynchronous image client rejects readback with `BAD_ARGUMENT` (action 5,
result 2), and the session correctly stops publishing. This is an unsupported
adapter ownership transition, not evidence of a device or keyed-mutex failure.
The available trace does not identify which text-preparation refusal triggered
the fallback, so a safe corrective change is not established. The readback
prohibition and failure guards remain intact; adapter consolidation is deferred
to the following assignment. Earlier guarded attempts and their failures remain
recorded, including one missing-copy receipt whose zero launcher exit is invalid
as a pass. The final original/disposable save hashes and exact INI restoration
pass. Native capacity and navigation results do not qualify this interturn route.

The evaluation is not installed as the accepted runtime. The prior accepted
executable and matching Renderer64 trio are restored and hash-verified, the
startup probe passes, and the owned game/helper/collector tasks are closed.
Cursor, environment, configuration and save inputs are restored. No new Civ III
patch symbol or user patch-table action is required.
