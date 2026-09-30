# Persistent world generations: bounded FRESH ownership slice

The existing tile cache now owns immutable chunk generations independently of
camera occurrences. The measured benefit is avoiding occurrence-specific wrapped
reconstruction. Ordinary resident returns were already retained by the control.
The 16.67 ms busy-scene target remains unmet.

## Ownership and scope

`CachedMeshGeneration` owns the existing layer vectors and their resource
references. Cache metadata binds it through the existing `ResidentContent` weak
registry. A bounded `ResidentSelection` retains each generation once, independent
of draw/occurrence count; FRESH's retained records share that selection. Bitmap
history remains weak. Clearing metadata invalidates its handle immediately;
selected chunks survive until the final lease ends. Cache slot serials survive
reset, preventing stale-handle revival. The geometry epoch guard remains.

The cache budget includes active cache bytes, selection allocation estimates and
retired payloads. Evicting old selected metadata retires its chunk payload under
the same cap; it can release dependency proofs without releasing borrowed buffers.
The pending epoch still prevents eviction of content being assembled. There is
one cache, no COM reference per draw, no per-occurrence mesh copy and no budget
increase. Retired payload accounting covers chunk arrays, logical chunk buffer
bytes and retained city lighting. It is an existing logical geometry budget,
not a complete physical VRAM meter; shared asset allocations, targets and worker
queues retain their separate existing accounting.

The resource owner applies to existing cached layers. Canonical wrap construction
is restricted to the active FRESH retained/shared world path at native tile width
at least 96 (the measured route is 128). It covers shared ground, water surfaces,
natural terrain/relief/forest/cliff geometry, rivers, static infrastructure and
local/city objects already in these generations. Dynamic wave/resource/unit poses,
projected bitmap caches, global asset ownership and scene targets keep their
existing contracts. Native 64 and Legacy keep occurrence-specific representation.
The conservative world preparation key still includes native size/target extent;
compile contexts, dependency proofs and grid distinctions remain intact. No
injected code, live suppression scope, wonders or District ownership changed.

## Capture and coordinates

Both arms use one corrected standalone client. The former twelve-full-tile-width
RENDER margin is preserved as stress evidence, not native workload qualification.
`capture_model.h` models native anchor rectangles, a twelve-coordinate topology
ring and the explored 4/8-coordinate appearance ring from
`capture_custom_renderer_topology`. RENDER, PREFETCH and TOPOLOGY_HALO remain
separate. Native 64 rescales anchors around the same target center; projection
zoom does not change native capture size. The existing foreground contributor
selection is unchanged. This is source-grounded synthetic capture, not an injected
native gameplay witness.

Canonical world identity and deterministic seeds are separate from original
occurrence coordinates and authoritative captured anchors. Canonical compile
windows also select canonical river dependencies. River receiving banks preserve
the requested continuous neighbor coordinates at the cut; modulo of every vertex
or material coordinate would be incorrect. Mesh projection uses the canonical
owner basis; placement, visibility and wrap stay in occurrence records. Water
phase retains its existing periodic camera rule.

A seam-crossing viewport needs a continuous shadow query basis. Its per-frame
constant uses the captured occurrence window center and folds only shadow queries
across the canonical cut into that nearby continuous occurrence. The CPU and GPU
use the same numerator-first half-period rule; the hardware witness caught and
fixed a division/rounding disagreement at the boundary. Receiver bounds use the same basis; the existing replicated casters
and 4096-square shadow target remain. This prevents an entire world-width field
and lower sampling density. Material, water and local-light coordinates are not
folded by this helper. Ordinary homogeneous views retain their field.

## Evidence and reproduction

Preserved evidence is in ignored `Renderer/.cache/world-ownership-step/`.
`before/` captures the accepted starting source; `candidate-source/` and its
manifest freeze the measured candidate. `control-alias/` contains the accepted
previous candidate DLL. `control-inspected/` adds only an untimed standalone
owner receipt to the frozen control. Early `control/` and corrected-control
preview observations used the older copy-reference DLL and are excluded from
primary attribution. All preliminary and failed receipts remain preserved.

`measure_world_ownership.py` runs the same production camera
begin/readiness/adoption API with the current full guest display, normal effects,
fixed source shader inputs and the developed coastal scene. Primary navigation
uses two reverse-order pairs per hour and twelve frames per view; primary zoom
uses two reverse-order pairs and ninety frames. Detailed traces/counters are off
for primary runs. Short ownership/cancellation/edit/pressure diagnostics are
separate. Receipts expose submission, readiness wait, adoption, first complete
render and subsequent draw/present samples; initial reference preparation is
also retained. A bounded stationary on/off pair checks instrumentation overhead.

The fixture has four synthetic actors. It does not qualify many-unit native play,
long-run p99 tails, or sustained 60 FPS. The seam/return captures advance an
identical 33 ms presentation clock, but wait for full-quality adoption between
inputs; they qualify sampled appearance continuity, not live 30/60 Hz navigation
latency. No progressive detail reduction hides first-view delay. Simultaneous
visible copies of a whole world are outside this viewport; the executable lease
witness exercises 8,192 duplicate occurrences without resource duplication.

## Results

At 2240 × 1260 / 60 Hz, vsync-one borderless presentation, the two reversed
pairs per hour give these mean preparation waits and successful full-quality
draw/present submission times (milliseconds). These are CPU-observed timings,
not GPU timestamp or physical scanout measurements.

| Hour / transition | Control wait | Candidate wait | Control first view | Candidate first view | Upload bytes, control → candidate |
|---|---:|---:|---:|---:|---|
| 12 / origin | 722.7 | 20661.4 | 913.6 | 21424.3 | 0 → 65,846,590 |
| 12 / diagonal | 2413.5 | 175.1 | 2515.2 | 297.4 | 68,266,810 → 1,351,344 |
| 12 / wrap | 5921.3 | 97.2 | 5947.7 | 121.2 | 301,760,740 → 0 |
| 12 / negative_wrap | 6077.9 | 95.8 | 6358.7 | 119.6 | 301,492,514 → 0 |
| 12 / jump | 2373.7 | 1724.4 | 2461.7 | 1746.6 | 89,156,876 → 81,630,698 |
| 12 / resident_wrap | 91.4 | 85.1 | 116.5 | 108.3 | 0 → 0 |
| 1 / origin | 1047.2 | 2999.7 | 1161.7 | 3112.7 | 0 → 65,846,590 |
| 1 / diagonal | 2400.6 | 187.0 | 2501.4 | 259.7 | 68,266,810 → 1,351,344 |
| 1 / wrap | 7097.2 | 90.1 | 7139.6 | 114.9 | 301,760,740 → 0 |
| 1 / negative_wrap | 7579.8 | 89.7 | 7880.8 | 112.8 | 301,492,514 → 0 |
| 1 / jump | 2525.9 | 1769.9 | 2644.6 | 1791.9 | 89,156,876 → 81,630,698 |
| 1 / resident_wrap | 92.6 | 92.2 | 117.8 | 116.3 | 0 → 0 |

The noon candidate origin includes a cold first FRESH draw: 38,411.8 ms wait,
39,799.7 ms first view, of which geometry is 2,341.9 ms and the draw/setup path
36,042.4 ms. The second pair is 2,911.1 / 3,048.9 ms. Neither observation is
removed. Night candidate origin waits are about three seconds. Both arms also
pay their separately logged initial Legacy/reference preparation (roughly
6–8 seconds for this capture). Canonicalization rebuilds 63 wrapped-halo owners,
65,846,590 bytes, on entering FRESH; its cost moves work ahead of the first pan.
It does not establish a native startup improvement. Cold wrapped-first checks
perform genuine construction after their own matching reference capture.

Corrected diagnostics distinguish new content from camera misses. Diagonal,
strip, equivalent and jump build respectively 94, 139, 2 and 385 newly required
static owners; their raw uploads are 1,351,344 / 1,998,264 / 28,752 / 81,630,698
bytes. Unchanged overlaps preserve generations. Once resident, positive/negative
wraps, origin/strip/wrap/jump returns and 1.25 in-place zoom build zero unchanged
static owners and upload zero bytes. The control already achieves most ordinary
resident returns; its first two equivalent wraps construct 1,103 owners each,
about 302 MB per occurrence.

`verified-generation-attribution.json` covers both shared mesh owners and their
empty-chunk ground metadata owners. The latter are not another set of GPU ground
meshes. Ground draw layers live in the shared generations. Per-cause logical layer
bytes are kept separate from raw upload bytes and unique cache buffer allocation
bytes: prepared slab allocations can include unconsumed jobs/shared indices,
so summing owner rows is not a physical upload-byte oracle. The local edit changes
one owner and an 80-owner dependency closure; 1,022 static owners remain identical.
The new closure has 40,073,116 logical layer bytes; raw upload is 36,714,854 bytes.
Visibility changes retain all 1,103 static owners with zero upload. Synthetic
viewer/world identity changes still rebuild 63 owners under the conservative
journal/representation boundary. These tests do not qualify a real player switch.

Normal traversal cache peak is 513,962,415 bytes (control: 1,263,005,746); unique
cache GPU allocations peak at 425,690,030 bytes. Selected generation-map storage
peaks around 323 KB, with 2,206–2,300 generation leases including empty ground
owners. The 768 MiB pressure variant completes far-east, far-west and return:
432 / 959 / 449 built owners, 0 / 1,926 / 786 evictions, and 230,446,752 /
191,648,746 / 112,544,050 uploaded bytes. Maximum reported cache bytes are
804,258,334 under the existing 805,306,368-byte cap. Retired/selection ledger peak
is 8,958,939 bytes; admission includes it. The old pinning attempt failed and is
preserved. Reconstructed return depth matches its original depth exactly.
Shadow caster records keep their existing independently owned COM references and
batch allocations; this slice does not provide a complete physical VRAM budget
for that owner or for global assets/targets.

Continuous full-detail zoom remains expensive:

| Hour | Control mean | Candidate mean | Candidate p50 | Candidate p95 | Warm samples per arm | Misses of 16.67 ms, candidate |
|---|---:|---:|---:|---:|---:|---:|
| 12 | 137.31 | 137.49 | 138.10 | 147.42 | 174 | 174 |
| 1 | 147.29 | 147.70 | 148.75 | 237.84 | 174 | 174 |

These short samples show no material zoom gain and do not qualify p99 tails.
The stationary overhead pair has medians 16.35 ms off / 16.19 ms on; means
17.10 / 24.73 ms include an on-run 232.43 ms outlier. It establishes neither a
speedup from instrumentation nor a precise overhead bound. Primary conclusions
use counters/traces off. Parallels transport failures are preserved; retries
use new labels only after process absence is confirmed.

## Appearance and limits

Cold-first ordinary, positive-wrap and negative-wrap endpoints have identical
D24 depth to the ordinary unwrapped control. RGB differences are 120 / 176 / 248
pixels of 2,822,400, with only 3 / 4 / 3 pixels above eight channel levels. Ordinary
warm views/edit/visibility checks are similarly close, with occasional one-pixel
depth differences; they are not claimed byte-identical. Initial and first-wrap
warm shading have several thousand small RGB differences, preserved in the raw
comparison. Actual pan depth changes after a 96/48 relocation match the baseline
exactly, including the -48 depth-translation change. The hardware depth witness
also proves common occlusion and coplanar ordering across origin changes,
nonzero immutable buffer ranges, in-flight releases and reset.

The seam changes the control's occurrence-dependent hill/tree arrangement to
canonical world seeds. At its midpoint, all 59,112 displayed changed-depth pixels
are on the wrapped bank, x=0–851, before the x=1,136 cut. The far positive bank
has zero depth changes; it has 6,081 noon / 2,627 night RGB pixels above eight
levels, consistent with shadow-field/reflection response to the corrected geometry
and field bounds (an inference from unchanged positive-bank geometry and the
limited shader-query change). Substantial RGB/negative-bank geometry differences are an
explicit correctness change, not blanket process noise or reference acceptance.
Homogeneous equivalent-wrap parity, receiving-bank coordinates, canonical
neighbor proofs and the continuous shadow-query hardware checks support that
attribution. The sampled return changes 2,404 depth pixels through animation in
both arms, with exactly identical depth deltas; no generation/variant return pop
is observed. Advancing-clock noon/night captures retain reflection, water,
lighting and all normal effects. The field remains 4096-square; its coordinates
are folded, not its resolution reduced. The 33 ms sampled presentation clock is
appearance evidence only; full-quality adoption still stalls between inputs.

Native-64 full-guest probes return renderer ERROR in both arms after approximately
24 seconds, without reaching the 120-second deadline. That path remains an
unqualified existing boundary, not a demonstrated ownership regression. Device
and content-reset proofs cover generation lifetimes and existing GPU
publication/composition failures; this is not an injected driver-removal or
native gameplay qualification. Vertical-wrap algebra is exercised on hardware;
vertical-wrap map appearance and simultaneous whole-world occurrences are not
qualified by this scene. No visual references were replaced.

## Checks and identities

The final focused witnesses execute 28 passing tests across immutable ownership,
corrected capture, shared depth/projection, scene/dependency ownership, cancellation,
retirement, publication and composition. Category dispatch executes 126 tests:
125 pass and one optional asset oracle skips. Two invalid module names in an
initial test command were corrected; their failed command receipts remain.
No injected compilation is required because injected sources are untouched.

All 47,224 frozen input hashes (12,238,666,273 bytes) match. The 200-file measured
source snapshot and final binaries are independently rechecked. The accepted
previous redraw evidence and installed bridge/DLL/helper tuple are unchanged.
The isolated step diff, hashes, manifests and raw receipts are preserved in the
step evidence directory. Concurrent seasons work is outside this diff.

| Artifact | SHA-256 |
|---|---|
| verified-candidate | `f082005c03bb48ee7b1f01c4e60253c4f5b063242416f75b833568603e438530` |
| verified-candidate-pressure | `64d723f90de28c1e6d324431052543505b87ff7eb3351488bc7fa47d1aa3f8f6` |
| control-alias | `6b68d398331a8552e1b376e5a3d9767aec3b62a73995c9e418a23a16a8f6efd1` |
| control-inspected | `3c7f7f61f90042a39d37c63feb558d3f2c2ebeaf86b0f9c3e5ad3c466d85872f` |
| control-inspected-pressure | `c790cd2a3650b94060f1e47b250091b813f77f0f386d665b521b0d42d9b241c7` |
| client | `51346303f96e22264daeb35a18e37c41bc9d15c5c720f931b4089232808ebadf` |

The inputs fingerprint is `4402e2630fb7ae145252cea6cd8e555336969e58af1a15c4fab6cecb8e23ac90`.
No new Civ III patch symbol or user patch-table action is required.
