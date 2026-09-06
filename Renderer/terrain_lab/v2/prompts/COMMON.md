You are the persistent owner of the Renderer Lab v2 `<TRACK_ID>` work package.

Before acting, read `AGENTS.md`, `Renderer/MASTER_PLAN.md`,
`Renderer/ROADMAP.md`, `Renderer/project_status.json`,
`Renderer/docs/renderer_lab_v2.md`,
`Renderer/docs/renderer_workstreams.md`,
`Renderer/docs/visual_validation_plan.md`,
`Renderer/docs/environment_lighting_and_ambient_effects.md`,
`Renderer/terrain_lab/PLAN.md`, the Q1 campaign manifest, your work-package
manifest, your status file,
`Renderer/terrain_lab/v2/REAL_MAP_VALIDATION.md`, and the relevant
canonical-reference audits.

Read and obey `Renderer/terrain_lab/v2/SOURCE_ART_POLICY.md` before creating or
changing visible art. Reuse Civ VI meshes/materials and the explicitly selected
source skin first; investigate missing source channels and import support before
inventing replacements. Procedural topology/joins are allowed, but generated
rock faces, mountain bodies, textures, or other dominant art are not an approved
substitute. Label diagnostic proxies and keep them out of beauty acceptance.
Record component-level source versus generated provenance in your owned audit.
This rule limits the earlier proxy permission: proxies enable independent tests,
not silent promotion of made-up art. Continue other source-backed work when an
asset or importer is missing; do not turn that issue into a whole-track wait.

Follow `Renderer/terrain_lab/v2/PLACEMENT_CLEARANCE.md`: Q5 road/rail and Q3
river envelopes constrain Q4 forest/jungle and Q7 city placement. Trees must
leave readable transport corridors; buildings must avoid rivers, roads, and
rails. Publish and consume owned interface witnesses without new start gates.

Every cast shadow must derive from the actual transformed caster mesh and its
authored cutout alpha, including current unit pose and building/vegetation parts.
Generic oval/blob/ribbon or footprint-only shadow substitutes cannot pass visual
acceptance. Project onto actual receivers using Q6's shared light direction;
soft penumbrae are allowed, but must preserve the caster-derived silhouette.
Do not confuse valid cast shadows across roads with physical placement overlap.
Legacy/proxy shadows may remain in immutable historical evidence only; missing
proper shadows in a new candidate remain pending, not a pass by omission.

Start now and work in parallel with the other owners. This user-authorized
schedule supersedes earlier instructions to wait for Q0 or another track to be
accepted. Empty `dependencies` permit immediate work; `integration_inputs` are
results to adopt before convergence, not launch gates. LQ0 remains the next
global closure checkpoint while LQ1 implementation proceeds concurrently.

Use the current Mac runner at `Renderer/terrain_lab/v2/app/runner.py` and read
its README. Pin the interface/source versions used by your candidate. Consume
available published candidates or frozen inputs; build a small explicit proxy
inside your owned paths if an upstream system is unfinished. Mark that evidence
provisional and replace the proxy when the real input arrives. Do not implement
a competing shared backend or edit another owner's system to bypass ownership.

Record interface requests and reproducible witnesses in your owned audit
directory, then continue all independent rendering and implementation. Q0 owns
platform/registry support; Q6 supplies color/alpha/lighting semantics to Q0.
Routine interface choices do not need user approval. Missing shared inputs,
real-map registration, or Windows parity must not stop the whole track. Q8
starts fixture design and candidate composition immediately using available
inputs. Stop only when no useful in-scope work remains, and identify exactly
which implementation or acceptance items are pending.

Work autonomously until your deliverable is ready for convergence. Do not stop
after planning, a successful
build, a deterministic hash, or the first plausible render. Render, inspect the
actual images, identify the highest-impact defect, revise, and repeat. Preserve
source evidence and clearly distinguish confirmed source behavior from inferred
rendering behavior.

Immutable v1 rules:

- Never edit `Renderer/handoffs/L*.json`, accepted `L*_AUDIT.md` records,
  accepted v1 reference hashes, or existing v1 output names.
- Do not modify the v1 `terrain_lab.cpp` or `terrain_lab.hlsl` unless the
  coordinator explicitly grants migration authority for a named section.
- Write candidates only under versioned Lab v2 paths and namespaced output
  directories.
- Do not promote into Game Integration or change an I# gate. Deliver a v2
  candidate handoff for coordinator review.

Ownership rules:

- Edit only `owns_paths` plus your own declared `status_file` from your
  work-package manifest. Record pending evidence separately from start blockers.
- Treat every other Lab v2 path as read-only.
- If a shared contract or another owner's implementation must change, report
  the exact requested interface change and a reproducing fixture for its owner
  and the coordinator. Continue with available inputs while it is resolved.
  Do not patch across the boundary.
- Preserve generic runtime packs. Civ VI-specific discovery and conversion
  remain offline adapter concerns.

Primary visual benchmark:

- Read `Renderer/terrain_lab/v2/GAMEPLAY_BENCHMARKS.md`. The frozen crowded
  all-assets scene is a coverage/regression fixture, not the primary beauty
  target. Preserve it; do not tune object size or layout to make that gallery
  attractive at the expense of gameplay readability.
- Lead visual reports with an actual-gameplay-scale contextual before/after,
  then focused diagnostic crops; put the inventory/stress overview last.
  Use Q8's published recipes read-only when available. Until then, use a small
  plausible contextual fixture in your own paths and label its provenance.
  Neither Q8 nor the verified-map registry is a new start gate.
- Ordinary edits still use one small crop, phase, and zoom. Broader contextual,
  day/night, two-zoom, and scrolling checks belong at candidate checkpoints,
  not after every edit. Keep assets, placement, camera, and settings pinned
  across A/B comparisons; report intentional changes explicitly.

Real-map acceptance:

- Pass relevant named regions of the actual user `test.biq`, with a verified
  source hash and coordinates, plus a neighboring or held-out region before
  acceptance. The historical `test_biq_l13_rivers_192.csv` name is insufficient:
  its preparation script currently reads Ancient Treasures instead.
- Consume Q0's cached dataset/region registry on the Mac. Preserve source terrain,
  river topology, wrapping, and enough neighboring tiles for correct crop edges.
  Keep small synthetic fixtures for exhaustive cases and quick diagnosis; log
  real-map coverage gaps instead of fabricating missing terrain.
- Add absent cities, roads/rails, units, resources, and improvements only as
  separate deterministic Lab layers. Record source and augmentation hashes,
  preserve legal terrain/domain placements, and render augmentation-off controls.
  Never label added objects as captured Civ III state or modify the source BIQ.
- Keep shared region metadata read-only, outputs in your own namespace, and
  include source/region/layer provenance in acceptance evidence. If the source
  or adapter is unavailable, continue independent work and record this gate as
  pending; synthetic-only evidence cannot close the visual track.

Fast-loop cadence:

1. Run `python3 Renderer/tools/renderer_dev.py state` once at the start.
2. Run only your package's portable/static tests during ordinary edits.
3. Use one cached small real region or 4x4-8x8 microfixture, one relevant phase,
   one zoom, and the smallest useful output for normal visual iterations.
4. Reuse compiled binaries, content-addressed assets, prepared geometry, and
   the Mac Metal or resident/batched renderer when available.
5. Inspect every candidate image directly and continue until the package rubric
   passes.
6. Then run the package matrix and relevant named real-map regions, including
   the neighboring/held-out witness, at both zooms and relevant day phases,
   followed by a deterministic repeat.
7. Run the 192-tile/four-phase promotion suite only when closing the package or
   when the coordinator requests a shared-contract check.

Do not ask the user for ordinary in-game screenshots or subjective feedback.
Exhaust canonical screenshots, accepted evidence, standalone renders,
isolation views, contact sheets, metrics, and feasible local capture first.
Manual evidence belongs only to a coordinator-declared strategic checkpoint.
If unavailable, record it as pending and continue every independent task.

Every delivery must include changed files, fixtures and exact commands,
before/after and isolation outputs, deterministic and perceptual metrics,
direct visual observations including remaining weaknesses, the consumed
contract version, a proposed v2 candidate handoff that references but does not
alter v1, and confirmation that touched files contain no personal paths or
sensitive information.
