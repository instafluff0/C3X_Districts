# Visual workbench

Renderer Lab operates on the current C3X checkout. Category definitions live
under `categories/`; `catalog.json` is their index. There are no Lab or
Integration release numbers and no pending/integrated category ledger.

For renderer performance engineering, follow the
[benchmark workflow](../docs/benchmark_workflow.md). It extends the existing
harness and preserves the category verification rules below. Read
[the architecture](../docs/renderer_architecture.md) for the destination and
[the retained plan](../docs/retained_renderer_plan.md) for the single next task.
The bounded tooling phase supersedes old performance handoffs, not ordinary
category art work; its completion returns work to scene/rendering implementation.

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
python3 Renderer/renderer.py integration resources --full
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

`integration CATEGORY` is the focused automated check for the current code. It:

- prepares only that category's inputs and builds or reuses the isolated candidate;
- runs its category tests plus the shared source/capture contracts; and
- runs only its owned behavior witness, such as resource playback/compositing.

For resources, the ordinary post-approval check is the resource animation,
scroll-parity, removal and background-compositing witness. It does not run unit
actions or the three generic terrain scrolling variants. `integration CATEGORY
--full` explicitly adds the affected-category closure, exhaustive unit proofs,
and the scrolling, reduced-zoom and world-wrap sweep for strategic checkpoints.

Integration does not read or compare reference images. Visual comparison is an
explicit Lab action, so a stale or intentionally different snapshot cannot block
the current code.

For standalone renderer work in a checkout containing another task's injected
C edits, use `integration CATEGORY --renderer-only`. It retains the category
tests and production behavior witnesses and records that injected compilation
was not requested. Omit this option when verifying changes to `C3X.h` or
`injected_code.c`. New diagnostic times/zooms without a fixed image are shown by
`compare` as unpaired current renders; they do not replace or acquire references.

## Visual acceptance and promotion

Automated `test` and `integration` checks may run before visual acceptance so
technical problems are found early. They do not grant visual approval. When a
change materially alters rendered output, the agent must present the relevant
focused and gameplay-context comparison to the user and receive explicit visual
acceptance before it:

- calls the visual change accepted, ready, promoted or integrated;
- ordinarily stages the candidate DLL into `Renderer/bin/`; or
- replaces a fixed reference image.

User acceptance applies to the shown result; replacing the optional fixed
reference remains a separate explicit choice. Once the user explicitly accepts
the shown result, stage the exact tested candidate into `Renderer/bin/` for their
game check and verify that the candidate and staged DLL hashes match. Do not ask
for a second staging approval unless the user said not to stage it. If the user
explicitly asks to stage before visual acceptance, stage that exact tested
candidate as an evaluation build and leave reference images unchanged. Staging
never implies permission to run `INSTALL.bat` or launch Civ III unless the user
also requests that action.

The disposable receipt under `lab/out/integration/` is tied to the exact current
input signatures and candidate DLL hash. Integration does not stage the DLL,
run `INSTALL.bat`, launch Civ III, claim a live-game pass or update a separate
release-status file. A strategic live-game checkpoint remains a real observed
test, not a bookkeeping prerequisite for current-code verification.

## Preparation and builds

`lab CATEGORY`, `test CATEGORY` and focused `integration CATEGORY` prepare shared
shader bindings plus only the asset jobs used by that category. `integration
CATEGORY --full` prepares the affected category closure. Plain `prepare` and
`build` intentionally cover every current asset job. Validated output replaces
generated files atomically; warm runs skip unchanged work. Shader-only edits do
not rebuild textures, and changed source-art bytes rebuild only their pack.
Preparation and candidate builds never stage the production DLL.

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

Current known limitations include connected-route gaps and incomplete resource
mapping. The production edit witness checks partial terrain reuse and exact
warm/cold pixel parity on an existing coast.
The current resource code fixes the animated black-backdrop defect; an older
fixed reference may still show it until the user elects to replace that visual
comparison point. That difference does not block Integration.
