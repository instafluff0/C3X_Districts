# Reproduce Q1 candidate evidence

Run from the repository root. The Mac GPU runner uses system `python3` and Q0's
cached tools. Numeric/PNG inspection needs a Python with NumPy and Pillow; set
`C3X_Q1_PYTHON` to that executable. The desktop dependency runtime is suitable.
Commands below assume `C3X_Q1_PYTHON` is set. No source packs are downloaded or
redistributed. Q0's registered local BIQ/cache and normalized pack mounts must
already exist. Reuse caches; do not run all checkpoints during a small edit.

```sh
python3 Renderer/tools/lab_v2.py prompt Q1-sampling
python3 Renderer/terrain_lab/v2/systems/sampling/prepare_contexts.py
"$C3X_Q1_PYTHON" -B -m unittest discover -s Renderer/terrain_lab/v2/tests/sampling -p 'test_*.py'
"$C3X_Q1_PYTHON" Renderer/terrain_lab/v2/systems/sampling/audit_source_meshes.py
```

The context recipe asserts the exact source BIQ SHA, calls Q0's registry export
with halo 2 and recreates two legal city/tank/road/rail layers, off controls and
four existing attack poses. Exported terrain is ignored. No captured gameplay
state is fabricated. Older historical crowded-crop runs are optional diagnostics.

Small A/B, sharper filter and pan experiments (one cached device load per call):

```sh
python3 Renderer/terrain_lab/v2/systems/sampling/render_study.py --candidate c006mixedab --mode ab --post linear_box
python3 Renderer/terrain_lab/v2/systems/sampling/render_study.py --fixture Renderer/terrain_lab/v2/fixtures/sampling/frozen_crop.fixture.json --candidate c003post --mode post
python3 Renderer/terrain_lab/v2/systems/sampling/render_study.py --candidate c007pan --mode pan --post linear_box
python3 Renderer/terrain_lab/v2/systems/sampling/render_study.py --fixture Renderer/terrain_lab/v2/fixtures/sampling/real-mixed-q1/fixture.json --candidate c019real-pan --mode pan --post linear_box
```

The phase and held-out checkpoint:

```sh
python3 Renderer/terrain_lab/v2/systems/sampling/render_study.py --fixture Renderer/terrain_lab/v2/fixtures/sampling/real-mixed-q1/fixture.json --candidate c011real-final --mode matrix --post linear_box
python3 Renderer/terrain_lab/v2/systems/sampling/render_study.py --fixture Renderer/terrain_lab/v2/fixtures/sampling/real-holdout-q1/fixture.json --candidate c012holdout-final --mode matrix --post linear_box
python3 Renderer/terrain_lab/v2/systems/sampling/render_study.py --fixture Renderer/terrain_lab/v2/fixtures/sampling/real-mixed-off/fixture.json --candidate c013off-mixed --mode pair --post linear_box
python3 Renderer/terrain_lab/v2/systems/sampling/render_study.py --fixture Renderer/terrain_lab/v2/fixtures/sampling/real-holdout-off/fixture.json --candidate c014off-holdout --mode pair --post linear_box
python3 Renderer/terrain_lab/v2/systems/sampling/render_study.py --fixture Renderer/terrain_lab/v2/fixtures/sampling/real-animation-0/fixture.json --candidate c015animation0 --mode pair --post linear_box
python3 Renderer/terrain_lab/v2/systems/sampling/render_study.py --fixture Renderer/terrain_lab/v2/fixtures/sampling/real-animation-1/fixture.json --candidate c016animation1 --mode pair --post linear_box
python3 Renderer/terrain_lab/v2/systems/sampling/render_study.py --fixture Renderer/terrain_lab/v2/fixtures/sampling/real-animation-2/fixture.json --candidate c017animation2 --mode pair --post linear_box
python3 Renderer/terrain_lab/v2/systems/sampling/render_study.py --fixture Renderer/terrain_lab/v2/fixtures/sampling/real-animation-3/fixture.json --candidate c018animation3 --mode pair --post linear_box
```

These are explicitly legacy source-map controls. The scene-linear source adapter
will use a new candidate name. Never relabel legacy pixels as linear evidence.
Use new names when shared source/contract hashes change. Current runner `--output`
is an exact output directory, so pass the full owner/candidate namespace:

```sh
python3 Renderer/terrain_lab/v2/app/runner.py compose --fixture Renderer/terrain_lab/v2/fixtures/sampling/hardware_lod.fixture.json --candidate checker-hardware-lod02 --output Renderer/terrain_lab/v2/audits/sampling/out/Q1/Q1-sampling/checker-hardware-lod02
python3 Renderer/terrain_lab/v2/app/runner.py compose --fixture Renderer/terrain_lab/v2/fixtures/sampling/ground_lod.fixture.json --candidate ground-lod01 --output Renderer/terrain_lab/v2/audits/sampling/out/Q1/Q1-sampling/ground-lod01
python3 Renderer/terrain_lab/v2/app/runner.py check --fixture Renderer/terrain_lab/v2/fixtures/sampling/linear.fixture.json --candidate linear01 --output Renderer/terrain_lab/v2/audits/sampling/out/Q1/Q1-sampling/linear01
"$C3X_Q1_PYTHON" Renderer/terrain_lab/v2/tests/sampling/verify_linear_capture.py Renderer/terrain_lab/v2/audits/sampling/out/Q1/Q1-sampling/linear01/report.json
```

Inspect a study, generate final-pixel crops/metrics, or align a pan:

```sh
"$C3X_Q1_PYTHON" Renderer/terrain_lab/v2/systems/sampling/inspect_study.py Renderer/terrain_lab/v2/audits/sampling/out/Q1/Q1-sampling/c006mixedab/report.json
python3 Renderer/terrain_lab/v2/systems/sampling/audit_packet.py Renderer/terrain_lab/v2/audits/sampling/out/Q1/Q1-sampling/c006mixedab/report.json
"$C3X_Q1_PYTHON" Renderer/terrain_lab/v2/systems/sampling/pan_metrics.py Renderer/terrain_lab/v2/audits/sampling/out/Q1/Q1-sampling/c019real-pan/report.json
python3 Renderer/terrain_lab/v2/tests/sampling/verify_evidence.py
python3 Renderer/terrain_lab/v2/tests/sampling/verify_evidence.py --accept
```

The final command must fail while any handoff acceptance gate is pending.
Passing numeric checks never substitutes for direct image inspection. The
deduplication manifest preserves original hashes for byte-identical repeat BMPs
reclaimed during shared disk pressure (278,806,128 bytes); original images are
retained. A rerun can regenerate repeats. No shared asset/cache cleanup is part
of these recipes. No injected-code compile or Integration promotion is performed.

## Adopted source-linear input

The context generator also emits `linear.fixture.json` and `linear.module.json`
using Q6's declared shader owner and Q0's linear adapter. Core later commands:

```sh
python3 Renderer/terrain_lab/v2/systems/sampling/render_study.py --fixture Renderer/terrain_lab/v2/fixtures/sampling/real-mixed-q1/linear.fixture.json --candidate c021linear-matrix --mode matrix --post scene_linear_box
python3 Renderer/terrain_lab/v2/systems/sampling/render_study.py --fixture Renderer/terrain_lab/v2/fixtures/sampling/real-holdout-q1/linear.fixture.json --candidate c022linear-holdout --mode matrix --post scene_linear_box
python3 Renderer/terrain_lab/v2/systems/sampling/render_study.py --fixture Renderer/terrain_lab/v2/fixtures/sampling/real-mixed-q1/linear.fixture.json --candidate c023linear-pan --mode pan --post scene_linear_box
"$C3X_Q1_PYTHON" Renderer/terrain_lab/v2/tests/sampling/verify_linear_capture.py Renderer/terrain_lab/v2/audits/sampling/out/Q1/Q1-sampling/c021linear-matrix/report.json --source-scene
"$C3X_Q1_PYTHON" Renderer/terrain_lab/v2/systems/sampling/review_media.py phases Renderer/terrain_lab/v2/audits/sampling/out/Q1/Q1-sampling/c021linear-matrix/report.json
```

Use the corresponding `real-animation-N/linear.fixture.json` with `--mode pair
--post scene_linear_box` for poses 0..3. Published names are c026linear-animation0,
c024linear-animation1, c027linear-animation2 and c028linear-animation3.
`review_media.py animation REPORT0 REPORT1 REPORT2 REPORT3 --output OWNED_PATH`
creates lossless films; `pan_metrics.py REPORT` creates aligned pan witnesses.

Q2 published read-only composed on/off module consumers are pinned in
`fixtures/sampling/q2-linear-on.fixture.json` and `q2-linear-off.fixture.json`.
Run `render_study.py` with those paths, `--mode pair --post scene_linear_box`,
and candidates c030q2-linear-on / c031q2-linear-off. This is an integration input
comparison, not a replacement evaluator or geometry repair by Q1.

The native parity report is prepared with the ordinary Mac `runner.py compose`
and the Q1 source-linear fixture, output `c029linear-parity`. Do not launch a
competing native run while Q0 is diagnosing transport. Use the existing parity
and renderer_dev dispatcher after Q0 confirms its working per-process path
configuration. No global VM mapping or renderer configuration is edited.

Independent Q7 source/world/serialized geometry audit (read-only owner inputs):

```sh
"$C3X_Q1_PYTHON" Renderer/terrain_lab/v2/systems/sampling/audit_q7_world.py Renderer/terrain_lab/v2/fixtures/objects/generated/all-pools-01
"$C3X_Q1_PYTHON" Renderer/terrain_lab/v2/systems/sampling/audit_q7_world.py Renderer/terrain_lab/v2/fixtures/objects/generated/ancient-earth-02
```

The Q7 metadata audit rejects source UV/normal/affine mismatches and marks
planar full-3D ambiguity explicitly. It reports declared but unbound material
channels rather than silently assuming that base color equals full material
fidelity. The generated audit JSONs contain hashes and portable source identities.

Final shared metadata and approved parity witnesses:

```sh
python3 Renderer/terrain_lab/v2/app/runner.py compose --fixture Renderer/terrain_lab/v2/fixtures/sampling/real-mixed-q1/linear.fixture.json --candidate c034source-audit --output Renderer/terrain_lab/v2/audits/sampling/out/Q1/Q1-sampling/c034source-audit
python3 Renderer/terrain_lab/v2/tests/sampling/verify_source_metadata.py Renderer/terrain_lab/v2/audits/sampling/out/Q1/Q1-sampling/c034source-audit/report.json
C3X_RENDERER_WINDOWS_ROOT="$C3X_Q1_NATIVE_ROOT" python3 Renderer/terrain_lab/v2/app/parity.py Renderer/terrain_lab/v2/audits/sampling/out/Q1/Q1-sampling/c029linear-parity/report.json
```

`C3X_Q1_NATIVE_ROOT` is the Q0-confirmed installed Conquests/C3X_Districts
shared-directory link for this VM session. The earlier drive-letter share was
not visible in the exec session; the per-process installed-link override worked.
Do not change a global mapping or commit a machine-specific root. D3D outputs
and repeats are owned Q1 evidence; maximum channel delta1 and silhouetteIoU1.

The final Q7 matrix/channel sidecar validation is:

```sh
"$C3X_Q1_PYTHON" Renderer/terrain_lab/v2/tests/sampling/verify_q7_sidecars.py
```
