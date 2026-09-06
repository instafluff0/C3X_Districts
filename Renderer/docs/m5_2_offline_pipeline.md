# M5.2 Offline Scene-Export Pipeline

## Boundary

`tools/process_scene_export_batch.py` implements the parallel-safe offline part of M5.2. It starts with complete `c3x.visible_scene.v0` files. It does not know the layout of injected capture state, read a Civ III process, parse a save/BIQ, insert a rendered surface, or decide which Civ III draw hook owns an overlay.

The live M5.2 exporter produces each scene automatically. M5.1 supplies the bounded capture primitives, authoritative anchors, and proven composition boundary, but it does not itself claim to export the full scene contract. The offline tool supports optional durable fixture naming and matched review when deeper regression or release evidence is useful.

## Live Capture

With `enable_custom_rendering = true`, the first successfully composited map frame is exported automatically. `Ctrl+Shift+F12` remains available as an optional recapture shortcut. The renderer atomically writes:

```text
Renderer/validation/live/civ3-live.scene.json
```

Export does no work on the configuration-off path. The export contains the authoritative tile anchors and captures terrain plus any visible feature, road, river, improvement, resource, city, unit, and active-effect categories found in the viewport. Non-terrain categories remain owned and drawn by Civ III.

Screenshots are lightweight game-health evidence. Routine development does not require the user to locate, rename, or pair files; strict schema and replay behavior are exercised automatically with synthetic scenes, and the user reports visible breakage or odd behavior when encountered.

## Batch Plan And Names

A `c3x.scene_export_batch_plan.v0` plan identifies one save, BIQ, paired save-from-BIQ, or synthetic source and one or more named camera fixtures. The fixture ID is a lowercase stable slug. Its files use these names:

```text
<fixture-id>.scene.json
<fixture-id>__ingame.png
```

The plan records required captured categories independently for every camera. An early terrain-only diagnostic plan can therefore require only `terrain`; an accepted populated M5.2 fixture must require every category present in its chosen save/BIQ viewport without changing the scene schema or the batch tool.

`source.artifact` is optional so tracked synthetic plans never name a copyrighted or machine-local save. A local integration plan may point to a relative `.sav` or `.biq`; the report hashes it but never copies it.

## Outputs And Gates

For every fixture the command:

1. Validates the exported scene and writes its byte-stable canonical serialization under `canonical_scenes/`.
2. Counts captured tiles and instances by renderer category and fails the offline gate when a required category is absent.
3. Runs the existing deterministic viewport/hour/season matrix, PNG metrics, mapping diagnostics, and per-fixture PNG contact sheet.
4. Records the exact scene, definitions, pack inputs, source artifact, and optional matched in-game screenshot by stable path and SHA-256.
5. Writes deterministic `report.json` and `contact_sheet.html` files across all named cameras.

The command exits successfully when `summary.offline_passed` is true. The separate `summary.full_m5_2_evidence_passed` field remains conservative: it becomes true only when an optional formal fixture has a real matched screenshot and a human has marked all six retained-layer checks (`fog`, `borders`, `labels`, `highlights`, `hud`, and `ui`) as `pass`. Routine development uses the automated M5.2 gate plus an ordinary game-health screenshot instead.

Generated exports, screenshots, matrices, and reports belong under ignored `Renderer/validation/`. Only the tool, tests, schema example, and documentation are intended to be tracked.

The portable M5.2a verification gate is:

```powershell
py -m unittest Renderer.tools.test_process_scene_export_batch
```

It proves plan validation, canonical export serialization, category inventory, deterministic batch PNG/report output, honest offline failure, and matched-evidence review state. Together with the native export gate, injected compile, and supplied game-health capture, it is the automated acceptance path for completed M5.2.

## Command

For an optional formal matched-fixture review, prepare a local batch plan and run from the C3X root:

```powershell
py Renderer\tools\process_scene_export_batch.py `
  --plan Renderer\validation\reference-map\batch-plan.json `
  --default Renderer\samples\config\m5_2_capture.custom_rendering.txt `
  --mod-root . `
  --references Renderer\samples\validation\reference_metadata.json `
  --output Renderer\validation\reference-map\report
```

For a synthetic contract check, replace `--plan` with `Renderer\samples\validation\scene_export_batch_plan.json`. Rendering still requires the normalized pack selected by the definition file.

## Live M5.2 Export Contract

The live exporter hands the offline pipeline only:

- A valid canonical or noncanonical JSON serialization of `c3x.visible_scene.v0`.
- The stable camera/fixture ID used as the scene filename when a durable fixture is promoted.
- Optional local save/BIQ identity and matched `__ingame.png` capture for formal review.

No injected C structure, pointer, hook-private buffer, or native renderer object becomes part of the offline interface.

The fixed `civ3-live` name is intentionally overwritten atomically. Promotion to a durable named fixture with screenshot pairing and explicit review is optional and outside the routine development path.
