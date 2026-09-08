# Visual workbench

The current C3X build is approved baseline revision 1. Category definitions live
under `categories/`; `catalog.json` is their index. Production delivery revisions
live in `Renderer/integration/status.json`. The catalog does not imply that every
existing technical limitation is fixed.

Current commands from the repository root:

```sh
python3 Renderer/renderer.py list
python3 Renderer/renderer.py show grassland
python3 Renderer/renderer.py gallery grassland
python3 Renderer/renderer.py lab grassland
python3 Renderer/renderer.py compare grassland
python3 Renderer/renderer.py test grassland
python3 Renderer/renderer.py test grassland --backends both
python3 Renderer/renderer.py affected shadows
python3 Renderer/renderer.py lab shadows --affected
python3 Renderer/renderer.py integration pending
python3 Renderer/renderer.py integration verify grassland
python3 Renderer/renderer.py check
```

`compare` and `gallery` use Pillow; city source lighting tests also need NumPy.
Use a Python environment with those installed. `gallery` produces one compact overview of the
captured baseline. Each category normally renders a detail view and a gameplay
context; shared environment and city recipes additionally select day phases.
`gallery CATEGORY` shows all that category's reference phases and contexts;
plain `gallery` is the small catalog overview. `show` hides internal hash metadata.
`--case detail` is a focused preview, insufficient for approval by itself.
With `--affected`, that case applies only to the requested category; dependent
categories use their complete recipes, including differently named animation cases.
Complete `lab` and `compare` runs automatically include dependent categories.
They also include categories affected by changed source bytes, even if the
command names another category. `affected CATEGORY` shows the combined set;
the same selection supplies regression tests and required approval previews.
City and resource edits include fixtures containing those objects; unit edits
include animation. Global lighting, shadows, transitions and unclassified shared
inputs conservatively select every consumer. Generated shader copies do not
broaden a change beyond their prepared source inputs.

Each category keeps only one reviewed-input signature for its approved revision.
Missing records require fresh complete comparisons. An exact pixel match to all
existing approved views establishes equivalent inputs without changing approval,
reference images, revision numbers or integration status. Different pixels keep
the category in the review set until explicit approval; partial comparisons do
not clear it. Internal signatures are not historical handoff manifests.

Texture edits invalidate existing previews. Category renders and integration
verification automatically build a candidate when compiled inputs change or the
candidate is missing; unchanged builds are reused. Internal hash caches are
disposable and are not part of the review process.
Unchanged input inventories reuse their existing hash cache without rewriting
it. Cache and category records use independent atomic temporary files so
concurrent writers cannot collide on a shared temporary filename.

`lab`, `test`, `build` and integration verification automatically prepare shared
shader bindings and the current natural, hill/cliff, city, unit and resource assets. `prepare`
runs those offline steps alone. Adapters/builders generate disposable output
first; only validated results replace current files. Warm runs skip generation.
Shader-only edits do not rebuild textures. Changed source-art bytes rebuild their
pack, and newly referenced textures become tracked inputs. Runtime textures are
independent copies, not hard links to editable sources. Preparation does not
stage a DLL or change approved images. Embedded unit-shader changes select a
candidate rebuild; dynamically loaded shader changes do not. Comparison,
approval and delivery reject unprepared inputs.
City preparation observes its actual catalog, mesh, material and texture reads;
edited shared layout or facade-light code also invalidates the pack. It clears
parsed mesh caches between builds. Unit preparation preserves aliases, source
normals, sampler addressing and complete action palettes. Resource preparation
preserves the current static bodies, clip-unit calibration and marine facing;
its inputs no longer depend on a historical preview-output report.

Edit shared sources or their builders/adapters, not generated runtime files.
Preparation preserves edited generated files and reports the conflict. Move an
intended edit into its source/adapter and restore the corresponding generated
file from the matching version before retrying. After cache loss, preparation
must reproduce the existing generated files before accepting new source edits.

Only after the user explicitly accepts the displayed affected set, record the
decision with `approve CATEGORY --user-approval "actual user statement"`.
Approval rejects missing affected previews, stale inputs, incomplete case sets
and modified images. It creates new references and leaves integration revisions
unchanged. It must never be invoked merely because tests pass.

`test CATEGORY --backends metal` additionally checks texture-array mip sampling
and vertex/pixel constant-buffer bindings on the Mac GPU, then translates and
compiles the current production terrain, mountain, object, water, feature, city
and unit shader programs (including reflection and city emission passes).
`--backends both` also compares the binding witness against D3D11. These are
transport and shader-compilation checks, **not full production-scene parity**.
They do not create or approve category reference images. Cached shader tools and
backend builds live under `lab/.local/shader-tools` and `lab/.cache`.

The first production-scene Mac preview is available explicitly:

```sh
python3 Renderer/renderer.py lab grassland --backend metal --case detail
python3 Renderer/renderer.py compare grassland --backend metal
```

It uses the current natural pack, shared terrain grid and production shader at
noon/zoom 128. Only the flat detail fixture is supported; gameplay, other
categories and affected-set runs are rejected. Its side-by-side comparison is
under `out/grassland/metal/`. These partial results neither replace a complete
D3D11 candidate nor certify approval or integration. A warm measured run took
12.3 seconds including input preparation, with 2.1 seconds in cached scene/render
work. Reducing the remaining preparation overhead is part of migration.

Generated candidates belong in `out/`, approved local images in `references/`,
and build caches in `.cache/`. These are ignored: source-derived art remains local.
Git preserves source history. The category pages describe current choices.

`build` compiles the candidate DLL and standalone preview tools without staging.
`integration verify CATEGORY` reuses a current candidate or builds one when needed;
`--build` forces a rebuild. It runs current behavior tests and compares the
affected views against their approved D3D11
images. It neither starts Civ III nor records a game pass. After the normal
delivery process and an actual game test, `integration record CATEGORY
--game-check "actual checks and result"` records the verified revision. The
receipt must be fresh and the staged DLL must match. Baseline revision 1 already
records the user's acceptance of the existing build; do not fabricate a new check.

Each native fixture keeps one disposable log and a completion record tied to
that invocation. A confirmed child-process result can survive a Parallels
return-code reporting error. Missing or mismatched completion is not a pass;
the dispatcher checks Windows and rechecks late completion before allowing one
retry after confirmed process absence. Live, unknown or unrecognized process
state is not retried automatically. Old files cannot certify
a new invocation merely because the image already exists.

Migration remains in progress. Complete category renders use the production
D3D11 DLL through the Windows VM dispatcher. Completing the Mac fast path,
expanded current native behavior replays and removal of the old campaign tree
remain required.
`check --complete` reports missing native/Metal reference coverage; it must pass
before the catalog is considered fully migrated. See `MIGRATION.md` for the
remaining implementation work and `Renderer/docs/visual_fidelity_playbook.md`
for the preserved graphics guidance.

Current exposed limitations: resource studies show black background patches
around animated bodies, connected route fixtures show segment gaps, and the
production edit-reuse witness fails because all visible tiles are rebuilt.
These are not silently repaired or treated as passed integration checks.
