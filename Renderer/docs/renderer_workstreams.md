# Lab and Integration

Use [the category workbench](../lab/README.md), not a milestone ladder or campaign.
The accepted production build starts every category at approved/integrated r1.

Lab owns current visual recipes, source intake, shared geometry/materials/lighting,
focused and gameplay-context previews, comparison and explicit user approval.
Integration owns game capture, cache/invalidation, redraw, anchors, clipping,
zoom/wrap/scroll, compositing, native ownership, fallback, timing and config-off.
An integration defect may reveal a visual problem; redesign returns to Lab.

A category uses shared implementation through its `standard.json` entry.
Day/night and shadows are lighting entries; transitions belong under terrain.
Their dependency lists select affected consumers. Keep a few useful views per
consumer, not every combination ever tested. Preserve reusable source findings.

The normal sequence is edit, render/compare/test the requested category, receive
explicit user approval, verify the wider delivery regression scope, test the game,
and record the integrated revision. Dependent Lab previews are available
explicitly when they help review; unrelated stale work does not silently expand a
category task into a full-catalog render.
Approval does not install code or transfer native ownership. Integration never
silently adopts an unapproved candidate. A replay pass is not a live-game pass.
Neither agents nor tests may invent a user approval or game-check statement.

Use Git for source history. Keep candidate output disposable and approved local
reference images intact until superseded by an explicit decision. Ignored licensed
assets need separate preservation before cleanup; they are not backed up by Git.

The architectural boundaries and deferred wonder/District contracts are in
[Renderer's entry guide](../README.md). The current production DLL and selected
assets are the baseline, not older isolated Lab fixtures. The Mac fast path is
still being reconciled with production; see [migration state](../lab/MIGRATION.md).
