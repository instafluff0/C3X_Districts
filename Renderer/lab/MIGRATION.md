# Current migration work

Production at commit `65b02cfc` is the accepted baseline by explicit user direction.
The integration task completed before migration began. The staged and candidate
DLLs both match `baseline.json`. Preserve `docs/visual_fidelity_playbook.md`.

Completed foundation: 20 category definitions point to current production code;
all begin at approved/integrated revision 1. Grassland detail and gameplay replays
are pixel-identical between staged and candidate DLLs and were visually inspected.
Plains, desert and tundra also have fresh native references. Five approval workflow
tests and five existing production terrain tests pass. Approximately 1.1 GiB of
regenerable shader/object/executable compiler caches were deleted; local packs,
geometry packets and retained candidate references were preserved.

Current tool: `python3 Renderer/renderer.py` supports list/show/affected, native
lab/baseline renders, compare (Pillow), explicit approval and integration pending.
`check --complete` correctly reports missing reference coverage. Native replay uses
the existing VM dispatcher through a short generated batch file. The current VM
maps Y: to iCloud and Z: to Home; set `C3X_RENDERER_WINDOWS_ROOT` to the existing
UNC checkout share when using the older dispatcher default. No native source,
DLL, injected code or source art has changed.

Remaining work, in execution order:

1. Complete focused fixture coverage and baseline renders for the other categories.
   Units/animation currently reject terrain-only replay; migrate the dedicated
   production roster/action harness rather than fabricating category evidence.
2. Establish the Mac fast path from current production code, with explicit
   comparisons against the Windows references. Old isolated Lab appearance is
   insufficient evidence of parity.
3. Migrate the requested categories, dependencies, current reference visuals,
   useful candidate recipes and relevant tests.
4. Replace milestone/campaign dispatch with the category interface and update
   AGENTS.md and the concise operational documentation.
5. Remove obsolete tracked experiments and dependency-checked generated output.
   Preserve local asset inputs and current reference data before pruning caches.
6. Verify the production build and behavior, inspect current previews, and audit
   every requirement in the user goal before declaring completion.

This file is temporary migration state, not a replacement experiment ledger.
