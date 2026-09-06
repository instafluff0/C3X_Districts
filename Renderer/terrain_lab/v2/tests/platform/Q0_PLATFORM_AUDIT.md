# Q0 platform convergence audit

Disposition: active final verification; candidate not accepted or promoted into
Game Integration. The earlier Q6 attachment and verified-map blockers are
resolved. Global milestone closure remains coordinator-owned. This audit
supersedes the earlier blocked report without changing any immutable v1 record.

## Contextual source evidence first

`app/out/Q1/Q0-platform/source-linear-04/h12-z1-pan00.bmp` is the directly inspected
4x4 actual-test.biq coastal view with a separate deterministic city/road layer.
The coastline, source mountains, forest/jungle, city and routes remain present.
The direct source-linear-02/03/04 comparisons have zero changed channels: adding
world/provenance plumbing does not retune the scene. The former transparent
crop fringe was corrected by allowing only opaque geometry to establish validity.
The city is an inherited oversized source presentation; legacy projected shadow
approximations and relief seams remain historical visual-system limitations,
not Q0 beauty acceptance. Q6 now replaces those casts in its owned world pipeline.

Mixed and neighboring holdout matrices were directly inspected at four phases
and two zooms. The held-out dense canopy still obscures portions of the river;
Q4/Q7 clearance integration owns that visual defect. Wrap-region replay remains
continuous water, without inventing a crop shoreline. Every source artifact is
labeled as actual terrain plus optional Lab augmentation; no added object is
represented as captured game state.

## Contract, ownership and source identity

The complete generated contract is `app/TASK_CONTRACT.txt`. The current contract
includes contextual gameplay, source-art reuse, source-shaped cast shadows and
full footprint/crown/overhang clearance. All Q0 edits are in its six owned
subtrees plus its explicitly declared status file. Other owners' source, v1
handoffs/audits/images, project status and injected code were not edited.
`shared/frozen_guard.json` pins the actual original files, preserving the older
handoff hashes' documented pre-territory scope.

Actual test.biq was located and copied read-only through the approved Windows
dispatcher, then checked before/after parsing. SHA256:
`a6a88d7fffcc567c3500bbd5aa947398dd48170d4f412aa1e518bb45ffe8453e`.
It is 17,173 bytes, 100x100 with 5,000 diamond tiles, wrapX true/wrapY false;
CITY, UNIT, CLNY and SLOC records are empty. Dataset payload SHA256:
`4e5fa91a03586b9b36da1ec3d8ee8249d19e803fe710ecd6659a174697cee837`.
The existing parser closure is pinned in `shared/real_map/registry_v1.json`.
Historical complete.csv derives from Ancient Treasures, not this test.biq.

Measured coverage includes desert/plains/grass/tundra/floodplain, hills/mountains,
forest/jungle/marsh and coast/sea/ocean. Volcano is absent and remains an explicit
coverage gap. Named regions include mixed/relief, neighboring holdouts, wrap,
and requested Q2 dry/cold/wet and Q3 mouth regions. The Q2 large evaluation region
is 16x8 at raw58,56; its user-evaluation role does not imply acceptance or invent
a holdout. Default halo6 covers the inherited 5.76-tile shadow-query extent.
Source coordinates, wrap, terrain/river records, parser/profile and layer hashes
are validated before replay. Stale source or mismatched overlays fail explicitly.

## Delivered platform

- Headless Metal ordinary rendering and off-screen D3D11 parity use the same
  versioned packets and HLSL. Mac compilation is pinned glslang/SPIRV-Cross and
  Apple's runtime Metal compiler because the preferred DXIL converter is not
  installed. No separately evolving Metal art path was created.
- Manifest CLI quick/check/compose/promote, ownership validation, dependency
  ordering, independent C++/shader closure builds, content-addressed pack,
  fixture, geometry/shadow and shader caches, namespaced outputs, and shared
  device/resource batches. Cached payloads, external blobs and auxiliary source
  metadata have integrity checks and atomic publication.
- Frozen wire2 remains default; readers accept1–6. Opt-in wire3 transports Q6
  linear-premultiplied RGBA16F plus R8 validity, explicit depth/blend and final
  exposure. Wire4 supports independent per-draw shader namespaces. Wire5 adds
  shared b1; wire6 adds explicit world/normal/UV/alpha/caster/receiver metadata.
  Exact fields and runnable examples are in `contracts/source_adapters_v1.md`.
- Q1 controls include capability-checked anisotropy/mip bias, matching MSAA
  attachments/resolve, internal scale, contract2 reconstruction, final-pixel
  camera sequences, effective settings and device fingerprints. GPU cost
  excludes uploads, CPU readback and shared CPU output conversion; allocation
  high-water is sampled rather than an undocumented transient-driver peak.
- Q1 independently detected invalid colors being averaged beneath zero R8
  validity. The shared box reference now contributes zero for those samples;
  Q1's two previously failing source controls pass its independent reference.
  No tolerance was widened.
- Cached actual-map import/export/registration, separate legal deterministic
  overlays and exact overlay-off identity. Ordinary replay does not start the VM.
- Exact frozen surface queries, Q2 material hooks, Q3 shoreline/sample hooks,
  checked world/corridor sidecars and Q4 vegetation placement hooks. These
  transport owner policies; Q0 does not invent source relief or collision rules.
- Owned module and final-composition packet postprocessors allow Q6 to build a
  field from final actual geometry. Source world attributes come from actual
  transformations, never clip-depth reconstruction. Independent layouts declare
  wire6 metadata. Cast-shape correctness remains Q6/Q7/Q8 visual acceptance.
- Source metadata records normalized bundle/DDS identities, mesh/index/UV/normal
  hashes, exact instance scale/yaw/anchor and explicit legacy vertical calibration.
  Tangents absent in the source runtime format are declared absent. The full
  source paths remain portable. Q1 independently matched34mesh identities,
  208DDS records and175instances in its source fixture, with identical pixels.

## Render/inspect/correct history and evidence

All names below are relative to `app/out/Q1/Q0-platform/` unless stated otherwise.
Large images, raw attachments and source packets remain ignored local evidence.

| Witness | Verified result |
| --- | --- |
| iteration-01 before/after, compact-matrix-02 | Fixed micro camera half-tile clipping; directly inspected four phases/two zooms with stable repeats |
| tests/platform/owner_isolation_evidence.json | Two disjoint synthetic owners built/rendered concurrently; actual Q1/Q2 now independently consume the same platform |
| real-terrain-01 and real-overlay-off-01 | Byte-identical terrain image SHA e77b6ac5c94ab67dcd9b3f35adc0f48e5e5bb7774f8c84e5f27314df5aa7fde0 |
| real-augmented-03 | Directly inspected legal source city/road overlay after removing unintended inherited territory in the explicit sparse fixture |
| real-mixed-matrix-02, real-holdout-matrix-02 | Eight frames each, stable repeats; direct phase/zoom contact review |
| real-wrap-02 | Two-zoom actual wrapped water region, directly inspected |
| linear-01, linear-post-01 | Opaque/cutout/translucent/HDR/depth diagnostic views; linear-post eight Metal/native pairs pass with MSAA4/scale2/Q1post2 and stable repeats |
| frame-01 | Shared b1/postprocessor directly inspected; Metal/native byte-identical with native repeat |
| source-linear-02/03/04 | Actual coastal context, metadata before/after zero changed channels; crop fringe absent |
| placement-hook-01 | Actual transformed forest/jungle vertices and fixture context delivered; accept-all hook passes |
| full-01 | Original192/fourphase/twozoom Metal/native matrix, all repeats stable; worst RGB MAE0.000003704,p99zero |
| full-02 | Correctly failed the bounded blob allocation limit after optional vertex fields inflated the frozen packet; no image accepted |
| full-03/full-04 | Collateral ENOSPC during shared test copytree expansion; neither failed run accepted |
| full-05 | Resource-safe final baseline rerun pending |

The optional fields originally increased a large frozen vertex buffer beyond
512MiB. Serialization now includes only requested fields while retaining exact
old offsets/stride when disabled. The remapped source texture-index bug exposed
by Q1 unit/Q2 rocky-shore fixtures was fixed by preserving the source-local
identity during bundle load, rather than interpreting shader slots as source
indices. Their same fixtures then rendered successfully without source omission.

The later disk failures were traced to coordinator-owned test_lab_v2._copy_campaign
copying the entire v2 tree, expanding ignored cache hardlinks into temporary full
copies. Three tests repeatedly exhausted25GiB. This corrected the initial
working-set/swap hypothesis. The coordinator fixed the test to copy only its
explicit manifest/policy graph, added a regression, and verified29 related tests
in0.113seconds. Q0's first broad lab run completed128/131 tests with those three
copy errors; all temporary trees were automatically cleaned. No artifact deletion
or other owner process termination was used as a workaround.

Source/native parity is a platform regression measure. It does not judge source
material completeness or approve legacy masks. Synthetic rectangles in linear,
b1 and independent-shader witnesses are explicitly diagnostic proxies only.

## Reproduction and scope

```sh
python3 Renderer/tools/lab_v2.py prompt Q0-platform
python3 Renderer/tools/renderer_dev.py state
python3 -B -m unittest discover -s Renderer/terrain_lab/v2/tests/platform -p 'test_*.py' -v
python3 Renderer/tools/lab_v2.py validate
python3 Renderer/terrain_lab/v2/app/real_map.py import
python3 Renderer/terrain_lab/v2/app/real_map.py export mixed --owner Q0-platform --output Renderer/terrain_lab/v2/tests/platform/out/real-mixed-augmented --augment
python3 Renderer/terrain_lab/v2/app/runner.py quick --fixture Renderer/terrain_lab/v2/tests/platform/frame.fixture.json --candidate frame-01
python3 Renderer/terrain_lab/v2/app/parity.py Renderer/terrain_lab/v2/app/out/Q1/Q0-platform/frame-01/report.json
python3 Renderer/terrain_lab/v2/app/runner.py check --fixture Renderer/terrain_lab/v2/tests/platform/linear.fixture.json --settings Renderer/terrain_lab/v2/tests/platform/linear-post.settings.json --candidate linear-post-01
python3 Renderer/terrain_lab/v2/app/runner.py compose --fixture Renderer/terrain_lab/v2/tests/platform/final-compose.fixture.json --candidate final-compose-01
python3 Renderer/terrain_lab/v2/app/runner.py promote --fixture Renderer/terrain_lab/v2/tests/platform/complete.fixture.json --candidate full-03
python3 Renderer/tools/renderer_dev.py lab --report Renderer/terrain_lab/v2/tests/platform/out/lab-verification.json
```

The parser/runtime import and first source acquisition are setup operations;
ordinary region replays use the cached dataset. Q0's additional real fixtures and
source-linear controls are under tests/platform/out and named in the machine
handoff. Effective settings, source/module/tool hashes and exact packet paths
are in every run report.

After a VM restart, Parallels' Y:/Z: drive mappings were unavailable to remote
commands. The existing installed checkout directory link remained accessible;
a per-process C3X_RENDERER_WINDOWS_ROOT override restored dispatch. Packet and
shader readers now support Windows long paths through UTF8-to-wide Win32 APIs.
No VM reset, installed game patch, shared-drive change or source-copy workaround
was made. Native parity failures are retained; no hash failure is waived.

The coordinator explicitly authorized the scoped Q0 baseline matrix and portable
checks and forbade general renderer_dev full/injected compilation for this
standalone work. That workflow unconditionally writes forbidden native outputs
and compiles injected code. Global verification/closure remains the coordinator's
job; this is a scope decision, not an unresolved platform launch blocker.

Final immutable-file/privacy checks and final matrix observations are recorded
in Q0_PLATFORM_CANDIDATE.json at delivery. No new Civ VI engine behavior is
claimed from imported asset evidence; no source art is redistributed here.
