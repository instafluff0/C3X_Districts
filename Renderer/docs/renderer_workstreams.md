# Lab and Integration

Use [the category workbench](../lab/README.md), not a milestone ladder or campaign.
The current C3X checkout is the implementation authority.

Lab owns current visual recipes, source intake, shared geometry/materials/lighting,
focused and gameplay-context previews, comparison and explicit user approval.
Integration owns game capture, cache/invalidation, redraw, anchors, clipping,
zoom/wrap/scroll, compositing, native ownership, fallback, timing and config-off.
An integration defect may reveal a visual problem; redesign returns to Lab.

A category uses shared implementation through its `standard.json` entry.
Day/night and shadows are lighting entries; transitions belong under terrain.
Their dependency lists select affected consumers. Keep a few useful views per
consumer, not every combination ever tested. Preserve reusable source findings.

The normal sequence is edit, render/compare/test the requested category, then run
`integration CATEGORY` on the current code. Integration checks the requested
category and its declared dependents; unrelated work does not silently expand a
category task into a full-catalog render. A fixed reference may be replaced after
explicit user acceptance, but reference differences do not block Integration.
Neither comparison nor Integration installs code. A replay pass is not a
live-game pass, and agents must not invent one.

Use Git for source history. Keep candidate output disposable and fixed local
reference images intact until superseded by an explicit decision. Ignored licensed
assets need separate preservation before cleanup; they are not backed up by Git.

The architectural boundaries and deferred wonder/District contracts are in
[Renderer's entry guide](../README.md). Current source and selected assets, not
older isolated Lab fixtures, handoff records or cross-backend parity campaigns,
determine behavior.
