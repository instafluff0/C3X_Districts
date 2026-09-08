# Visual workbench

Renderer Lab operates on the current C3X checkout. Category definitions live
under `categories/`; `catalog.json` is their index. There are no Lab or
Integration release numbers and no pending/integrated category ledger.

Current commands from the repository root:

```sh
python3 Renderer/renderer.py list
python3 Renderer/renderer.py show resources
python3 Renderer/renderer.py lab resources
python3 Renderer/renderer.py compare resources
python3 Renderer/renderer.py test resources
python3 Renderer/renderer.py lab resources --affected
python3 Renderer/renderer.py test resources --affected
python3 Renderer/renderer.py integration resources
python3 Renderer/renderer.py check
```

The ordinary loop is edit → `lab` → `test`. `lab CATEGORY` prepares only the
asset builders needed by that category, builds or reuses an isolated candidate
DLL and renders the current code. `--case` gives a focused diagnostic preview.
`--affected` explicitly adds the bounded visual consumers declared by category
dependencies.

`compare` puts the current render beside the category's fixed reference image.
The reference is a review aid only: different pixels are reported, not treated
as an Integration failure. Comparison is read-only. If the user explicitly
wants the new appearance to become the comparison point, `approve CATEGORY
--user-approval "actual user statement"` replaces that one category's fixed
reference. Git, rather than numbered reference folders, preserves history.

`integration CATEGORY` is the broader automated check for the current code. It:

- prepares inputs and builds or reuses the isolated candidate;
- runs the category and native capture/compositing regression suites;
- selects only the assets and behavior witnesses needed by the category and its
  declared dependents; and
- exercises scrolling, wrapping, edits and the relevant object animation paths.

Integration does not read or compare reference images. Visual comparison is an
explicit Lab action, so a stale or intentionally different snapshot cannot block
the current code.

The disposable receipt under `lab/out/integration/` is tied to the exact current
input signatures and candidate DLL hash. Integration does not stage the DLL,
run `INSTALL.bat`, launch Civ III, claim a live-game pass or update a separate
release-status file. A strategic live-game checkpoint remains a real observed
test, not a bookkeeping prerequisite for current-code verification.

## Preparation and builds

`lab CATEGORY` and `test CATEGORY` prepare shared shader bindings plus only the
asset jobs used by that category. `integration CATEGORY` prepares the affected
category closure. Plain `prepare` and `build` intentionally cover every current
asset job. Validated output replaces generated files atomically; warm runs skip
unchanged work. Shader-only edits do not rebuild textures, and changed source-art
bytes rebuild only their pack. Preparation and candidate builds never stage the
production DLL.

Edit shared sources or builders/adapters, not generated runtime files.
Preparation rejects edited generated files so an intentional change can be moved
to its source without silently losing it. Runtime textures are independent
copies, not hard links to editable source art.

Native fixtures keep one disposable log and completion record per invocation.
The dispatcher requires the matching invocation ID and executed witness markers;
an old output file or successful process exit cannot certify a new run.

## References and generated output

Generated previews, receipts and caches live under `out/` and `.cache/`; fixed
local comparison images live under `references/CATEGORY/approved/`. These are
ignored and disposable except for any reference image the user wants to retain
as a comparison aid. Source-derived art remains local. Complete category renders
use the production D3D11 renderer through the Windows VM dispatcher. `check`
validates the catalog and dependency graph without rendering or hashing images.

Current known limitations include connected-route gaps, incomplete resource
mapping and a production edit-reuse witness that rebuilds all visible tiles.
The current resource code fixes the animated black-backdrop defect; an older
fixed reference may still show it until the user elects to replace that visual
comparison point. That difference does not block Integration.
