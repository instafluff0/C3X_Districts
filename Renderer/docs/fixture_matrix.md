# C3X Fixture Matrix And Reference Validation

## Implemented M4.2 Boundary

`tools/render_fixture_matrix.py` turns one validated visible-scene fixture into a deterministic matrix of standalone renders. The default matrix is:

- Viewports: `640x480`, `1024x768`.
- Hours: midnight `0`, sunrise `6`, noon `12`, sunset `18`.
- Seasons: summer, fall, winter, spring.

The resulting 32 PNGs, `contact_sheet.png`, and `manifest.json` are written beneath an ignored output directory. No output contains a generation timestamp or absolute output path, so identical inputs produce identical bytes in a different output directory or on a repeated run.

## Deterministic Manifest

The `c3x.fixture_matrix.v0` manifest records:

- Renderer ID and explicit implementation version.
- Relative source paths and SHA-256 hashes for the scene, renderer definitions, reference catalog, merged typed catalog, pack manifest, mesh, material, and base-color texture.
- Exact viewport, hour, and season axes plus stable cell ordering.
- Every image path and SHA-256 hash.
- Per-cell mapping coverage, map-bound violations, anchor misses, hidden-fragment depth rejections, invalid depth values, unique colors, mean RGB, luminance range/mean, and a 16-bin luminance histogram.
- Named structural checks with their actual values, thresholds, and pass/fail results.
- Noon-versus-midnight and summer-versus-winter comparisons for every applicable row.
- Contact-sheet dimensions, labels, and hash.

Mapping coverage excludes categories explicitly retained by Civ III. An owned renderer category that lacks a rule or usable payload counts as unresolved and fails the default 100% required-coverage gate.

## Viewport Replay

A viewport override preserves the scene's recorded map/HUD margins, recenters the captured projection origin and every authoritative tile/object anchor by the same integer offset, derives a new deterministic scene ID, and revalidates the result before rendering. It does not infer a perspective camera or replace Civ III's pixel basis.

## Structural Versus Art-Direction References

`samples/validation/reference_metadata.json` contains two deliberately separate reference kinds:

- `structural_regression` uses exact hashes of generated C3X output and exact recorded render state.
- `art_direction` is qualitative, local/untracked, and identified only by lighting phase unless a source capture is independently proven to have an exact hour.

The validator rejects cross-engine art-direction entries configured as pixel-equality gates and rejects an inferred exact hour on a phase-only reference. This keeps optional human or AI observations outside the deterministic acceptance decision.

## Command

After building the local normalized grassland pack, run from the C3X root:

```powershell
py Renderer\tools\render_fixture_matrix.py `
  --scene Renderer\samples\scenes\grassland_viewport.scene.json `
  --default Renderer\samples\config\default.custom_rendering.txt `
  --mod-root . `
  --references Renderer\samples\validation\reference_metadata.json `
  --output Renderer\validation\grassland_viewport
```

The command exits with status 1 if any structural cell or required environment comparison fails. Generated images may contain local derived art and remain ignored; the tool, schemas, synthetic tests, thresholds, and reference metadata are tracked.

