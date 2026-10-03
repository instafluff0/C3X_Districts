# Prepared resident submission

The renderer retains generic source meshes and immutable 64-byte placement
records. Main and reflection share their active union; shadows retain a separate
active placement lease within the same 32 MiB joint allowance. Passes borrow
records through four-byte selected indices; viewport, light/page and wrapped shadow offsets stay
in small pass constants. Exact source generation, version, material, projection
and anchor facts authorize reuse. Traversal order remains separate from placement
identity, so repeated occurrences still draw in their original order. Opaque
grouping requires compatible bindings and independent coverage in both main and
reflection passes; overlap, depth ties, alpha, decals and native ordering remain
barriers.

Owned visibility facts retain their exact ordered tile anchors, target, wrapping,
map/viewer scope and visibility/content/device generations. Clock-only updates
check those facts without rebuilding coverage maps or neighbor lookups. A protected
membership observation retains water/river occurrence visibility until those
inputs change. Animated poses keep advancing; explored fog freezes their time.
The pose-only path skips unused legacy backdrop allocation and dirty tracking.

Static raster translation must preserve both whole-pixel sampling and the
two-by-two pixel groups used by terrain height derivatives. An odd projected
pixel displacement redraws the static region from resident geometry; MSAA sample
count does not change this pixel-group requirement.

Known permitted world regions prepare through the same production compiler during
loading, then continue on bounded worker turns. The existing valid-world loading
boundary first completes camera-free RAM recipes, then admits their GPU buffers
through the foreground canonical keys and allocation owners. Measured physical
and adapter headroom reserve future targets, composition, units, shadows and
publication overlap. Admission never evicts a surviving loading generation;
capacity-deferred content keeps its RAM recipe. The byte receipt distinguishes a
completed admission sweep from a fully resident world. Unchanged generations
remain resident through changed-world preparation. Canonical full-detail content uses
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

Only the active body/reflection and shadow placement leases are retained for
reuse. Camera reassembly or lighting changes can borrow existing required ranges.
A genuinely new range replaces its affected immutable union, including a copy
of unchanged records. Required ranges have first claim; a replacement can then carry valid
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
| World geometry | Existing 8 GiB x64 safety ceiling; 768 MiB x86 ceiling. Measured process and adapter headroom determine actual admission; missing adapter measurements freeze optional loading growth. Selected and retired readers remain charged. |
| Compiler recovery | Ready pool: at least two, at most seven 16 MiB slots. Future headroom reserve: 512 MiB plus the ready pool and up to six 48 MiB worker lanes. |
| Prepared backing | 1 GiB physical session-file ceiling; 16 MiB raw/packed record ceiling. |
| Fresh shadow field | 25 fixed 1024 by 1024 R32 pages: 100 MiB, plus the actual still-held production field (normally 128 MiB) and an 80-byte constant buffer. |

Shadow producer registrations retain exact dependencies and shared immutable
input owners across camera changes. The existing 4096-event journal selects
revalidation; barriers, owner changes and overruns require the complete proof.
Projected caster bounds retain generation, occurrence and light identity.
Entering/leaving contributors update affected page vectors; unchanged pages do
not rebuild or sort them. Proof and page metadata share the existing charged
placement allowance, with a 16 MiB combined ceiling and direct draw fallback.
Distinct immutable source owners remain pinned and independently validated; exactly
equal normalized dependency lists share one charged allocation. Hashes select
candidates, full equality authorizes sharing, and the last owner releases the list.
Page vectors hold compact occurrence IDs; the retained occurrence map owns full
exact caster keys. IDs never recycle while any completed page can refer to them,
and vector replacement charges both old and new allocations during overlap.
The current caster set renews captured visibility only with exact content and
visibility revalidation. Optional page failure keeps validated dependencies when
the minimum charged owner still fits and redraws the complete field; unproved
or unchargeable dependencies clear.
Unchanged frames retain the journal fast path. Up to sixteen changed entries use
selective validation; broader journals use full exact content and visibility
validation, avoiding repeated source-map searches. Barriers and overruns also
require full validation.

Shadow sampling derives from the current receiver extent and exact light/wrap
facts before reuse. Global light-plane cells drive derivatives and all nine PCF
taps; canonical page projection stays identical across physical slice changes.
The current cold float span caps density, including its boundary rounding case.
The fixed field retains physical slices through a logical-to-physical page table.
Exact ordered contributor proofs include current off-screen casters, additions and
removals; light, wrapping, quality, asset configuration, scope and device changes
invalidate affected contents. Canonical caster batches and shadow placement groups
reuse unchanged sources independently of body/reflection placement. Optional merged
buffers, metadata and staging share the existing charged allowance. Failed page
updates cannot authorize a complete atlas. The native oracle compares retained
and forced rebuild depth and receiver PCF bits.

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

### Current qualification — October 3

The reviewed navigation/composition/submission refactor passed 110 host contracts
and seven native fixtures, including 657 GPU pixel oracles and rotating display
targets. The original PROFILE=0/census-off gameplay routes completed 30/30 busy
and 13/13 light requests with normal water effects. Exact paired workloads admit
15 busy and 12 light endpoints; busy idle remains unmatched.

Light scrolling improved median native-input-to-correct-Present latency from
243 to 130 ms; source presentation cadence rose from 14.57 to 26.38/s. Light idle
was essentially flat (55.2 to 55.7/s), and zoom remained about 447 ms. Busy scroll
latency improved from 3093 to 2868 ms while cadence fell from 6.77 to 6.27/s;
matched jump/return latency regressed from 1541 to 2680 ms. These mixed results
establish no broad FPS gain. Cadence includes prior/mixed sources; continuous
distinct displayed motion, physical scanout and GPU durations remain unavailable.

Evidence is sealed under `Renderer/.cache/integrated-refactor-validation/`.
The accepted runtime/settings were restored exactly. Compiler objects and surplus
screenshots were deleted without archives; raw logs, source/trios, endpoint images
and receipts remain. The comparison used the earlier worker hash phase; commit
`69a75ace` preserves native initial worker phases and awaits its subsequent native
build/visual check. Ordinary water still uses per-tile buffers; cross-tile spatial
water batching is not implemented by contiguous same-buffer range joining.
