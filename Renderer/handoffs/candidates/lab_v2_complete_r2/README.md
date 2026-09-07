# Consolidated Renderer Lab pickup r2

This is the single entry point for selected Lab work. Its authoritative current
visual layer is the [isolated Lab state of the art](LAB_STATE_OF_ART.md): four
source-faithful macOS Metal studies for mountain, grassland/plains/tundra/hills,
forest and Warrior. Cities are explicitly unchanged. The older 24-system/27-scene catalog remains
available as implementation history, but it is not the current beauty target.
The package copies no licensed art, packet archives, native code or caches.

**Mac Lab state of the art prepared; not promoted into Integration.** A combined
scene is deliberately not required by this pickup. The obsolete beauty-scene
fixture is specifically non-authoritative because it predates the isolated
fixes and can place trees through city buildings. Existing Integration gates
and M9/M10/M11 remain unchanged.

Read [IMPLEMENTATION.md](IMPLEMENTATION.md) for the pickup order and interface
map, [SHADOWS.md](SHADOWS.md) for the confirmed unit/resource gaps and exact
coordinate conversion, and [CHECKPOINT.md](CHECKPOINT.md) for validation and
remaining acceptance work. [manifest.json](manifest.json) is the versioned
machine-readable catalog; its historical selections remain pinned while the
user-directed state-of-art layer is refreshed explicitly. [catalog.py](catalog.py) makes its selection policy
readable without parsing thousands of hashes.

From the repository root:

```sh
python3 Renderer/handoffs/candidates/lab_v2_complete_r2/package.py list
python3 Renderer/handoffs/candidates/lab_v2_complete_r2/package.py state
python3 Renderer/handoffs/candidates/lab_v2_complete_r2/validate_state_of_art.py
python3 Renderer/handoffs/candidates/lab_v2_complete_r2/package.py verify --evidence --assets --packets
python3 Renderer/handoffs/candidates/lab_v2_complete_r2/package.py shadows
```

`validate_state_of_art.py` checks the four current witnesses, their source entry
points, render settings, retained image hashes and cross-system invariants.
`verify` additionally checks the historical package pins. Visual quality still
requires direct inspection. Native hashes are advisory: integration
may continue improving native code, and this package must never restore an old
native implementation over those changes. Common Lab/tool-library files are
pinned for dependency completeness; only the explicit system/case selections
are candidates to enable. Do not compile or activate all revision folders.
The manifest records a Git baseline and hashes in the existing checkout;
preserve that Git revision when moving to another worktree. Local licensed packs
and generated packets remain local prerequisites. Asset pins cover terrain
loaded channels, legacy runtime bundles and pack manifests, not every art blob
in every source-intake pack; normal pack/compiler validation remains required.

An optional exact-input Metal replay uses a new, bounded output directory:

```sh
python3 Renderer/handoffs/candidates/lab_v2_complete_r2/package.py replay --case capital-inland --output Renderer/terrain_lab/v2/audits/beauty/out/consolidated-r1-capital-replay
```

Use the established Python runtime with NumPy/Pillow for render/visual tools.
`verify`, `list`, and `shadows` need only Python’s standard library and, for the
probe, a C++17 compiler. Replay preserves old images and enforces the 8 GiB
free-space floor. The per-case batch is portable even though historical local
reports can contain machine-local cache paths. No raw historical batch file is
copied into this package.

Importer library pins use committed Git blobs from the stated baseline because
other tasks are actively extending importers. Their working-copy differences
are advisory, not selected changes. Retrieve an exact pinned file with
`package.py extract-source --source Renderer/tools/asset_compiler/unit_family_asset_importer.py --output Renderer/verification/consolidation-source/unit_family_asset_importer.py`.
This never overwrites active source. The initial r1 preparation catalog is
superseded by r2's committed importer pins; use r2 for pickup.

The historical natural-scene stack is shadow-receiver-r1 → canopy-variation-r1 /
river-corridor-r3 → water-natural-r6 + GPU water-reflection-r5. Current American
capital selections are r111 inland and r112 freshcanopy, with corrected source
materials, source-hull paving, facade-plane night lights, central placement and
orthogonal footprints. The coastal fallback is preserved but **does not meet
the new central-palace preference**. Asian/ancient and ordinary modern cases are
explicitly conditional; they are not proof of a universal city recipe.

Goody huts, colonies, fortifications, airfields, outposts, radar, victory sites,
pollution, craters and territorial borders are all accounted for. Barbarian
camps and source-only effects are separately marked preparation-only. Unit and
resource animation uses integration’s newer runtime, not an older frozen Lab
pose implementation. Nothing in this catalog authorizes new roads or early
wonders/Districts work.
