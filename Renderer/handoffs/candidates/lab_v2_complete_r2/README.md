# Consolidated Renderer Lab pickup r2

This is the single entry point for the selected Lab work and the current
production baseline. It replaces the **terrain-only package as a navigation
entry point**, not its immutable historical record. It supplies 24 system
dispositions, 27 selected conditional scene witnesses, source hashes, portable
replay jobs, the implementation order and concrete unresolved work. It copies
no licensed art, packet archives, native code or cache directories.

**Prepared for implementation; not promoted or visually accepted.** There is
not yet one production image combining every newer Lab system. Existing
LQ0/LQ1/LQ2, pending formal Integration gates, and M9/M10/M11 remain unchanged.
Historical L9–L21 approvals apply to their recorded versions only. The user’s
requested consolidation does not retroactively approve experimental revisions.

Read [IMPLEMENTATION.md](IMPLEMENTATION.md) for the pickup order and interface
map, [SHADOWS.md](SHADOWS.md) for the confirmed unit/resource gaps and exact
coordinate conversion, and [CHECKPOINT.md](CHECKPOINT.md) for validation and
remaining acceptance work. [manifest.json](manifest.json) is the immutable
machine-readable catalog. [catalog.py](catalog.py) makes its selection policy
readable without parsing thousands of hashes.

From the repository root:

```sh
python3 Renderer/handoffs/candidates/lab_v2_complete_r2/package.py list
python3 Renderer/handoffs/candidates/lab_v2_complete_r2/package.py verify --evidence --assets --packets
python3 Renderer/handoffs/candidates/lab_v2_complete_r2/package.py shadows
```

`verify` checks pins, not visual quality. Native hashes are advisory: integration
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

The selected natural-scene stack is shadow-receiver-r1 → canopy-variation-r1 /
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
