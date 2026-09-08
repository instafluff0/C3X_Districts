# Visual validation

The current checkout is authoritative for implementation and Integration.
Passing tests establishes technical behavior. Fixed reference images remain
optional comparison points and may be replaced only after explicit user acceptance.

## Category review

Use the category catalog and commands in `Renderer/lab/README.md`. A review shows
the candidate beside the approved reference, with the same scene, camera, time
and zoom. Keep a focused view and a surrounding-gameplay view; add only the phases
or variants relevant to the category. Shared lighting, shadows and transitions
select their affected consumers automatically.

Day/night studies use midnight, sunrise, noon and sunset. Check shadow direction,
direct/ambient balance, water response, night readability and emissive behavior.
Animation studies keep surroundings fixed while changing the native cursor.
Seasonal variants, when implemented, must preserve UVs, texture dimensions, alpha,
atlas boundaries and logical asset IDs.

Inspect for missing geometry, black patches, clipping, seams, incorrect scale,
overlap, inconsistent materials and unreadable lighting. Automated metrics may
flag these problems but cannot approve a stylistic change. Preserve the guidance
in `visual_fidelity_playbook.md`.

For every material change to rendered output, present the relevant focused and
gameplay-context comparison to the user. Automated checks may continue while
review is pending, but do not describe or hand off the appearance as accepted,
ready, promoted or integrated, and do not ordinarily stage its DLL, until the
user explicitly accepts what was shown. Fixed-reference replacement is optional
and requires its own explicit direction. Explicit acceptance authorizes the agent
to stage the exact tested candidate DLL into `Renderer/bin/` for the user's game
check; stage it and verify matching hashes without requesting separate permission
unless the user said not to stage it.

## Automated delivery checks

`python3 Renderer/renderer.py integration CATEGORY` runs the relevant portable
contracts and selected headless production replays for the current code. It does
not read reference images; render/compare remains an explicit Lab review action.
The command does not install a DLL, launch Civ III or certify a live-game test.
If the user explicitly requests an in-game evaluation, the exact tested candidate
may be staged before visual acceptance. Treat that DLL as an evaluation build;
staging is not visual approval and does not authorize `INSTALL.bat`, launching the
game or changing a fixed reference unless those actions are separately requested.

Keep checks for authoritative capture/anchors, replacement ownership, config-off
and fallback behavior, clipping/compositing, animation timing and interruption,
scroll damage, wrapping, invalidation and bounded caching. Repeated identical
inputs must remain deterministic. The actual-DLL replay also compares prepared
and cold output, using the existing production pixel budget; it is not a new
cross-backend tolerance.

Tests must report which behavior actually ran. A successful process exit or
ownership mask is not proof of visual quality. An existing-build failure stays
visible; neither a fabricated pass nor a reference-image repair resolves it.

## Reference provenance

Synthetic category views demonstrate the production API; they are not live
Civ III screenshots. A BIQ supplies terrain, not runtime unit actions, ownership,
visibility or city state. Game captures and exported scenes must retain those
authoritative inputs, including viewport/zoom, world seed and presentation time.

Cross-engine source images are art-direction references, not pixel-equality gates.
Keep confirmed source data distinct from inferred engine behavior. If a source
capture has only an approximate lighting phase, do not invent an exact hour.
Local copyrighted art and save data remain ignored unless deliberately contributed
under suitable rights. Ignored files are not backed up by Git.

## In-game checkpoints

Use existing valid game evidence and automated/local capture first. Request user
screenshots only for a material live-game question, as one concise batched check
covering the relevant save, camera, configuration and variants. Do not request
manual evidence for every internal change.

A delivery checkpoint checks scrolling, wrapping, supported zooms, object and
animation behavior, and that Civ III still owns fog, borders, labels, highlights,
HUD and UI. Record a material observed result in the relevant current issue or
notes when useful; do not create a per-category release ledger. When manual
evidence is unavailable, continue independent work without repeated requests or
invented approval.
