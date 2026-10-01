# Bounded native HUD composition

At `Session::world/world_end`, map-attached native HUD primitives now become
one immutable generation recipe. The recipe captures the pre-HUD projected
packed/full-color world and each external operand version. It executes the
original fill, text, blend, sprite and copy sequence into one owned canvas pair.
Working-pair aliases read the preceding command's result through the existing
native compositor self-alias rules. External operands are attached and retired
one command at a time. Command anchors and clips remain authoritative; their
logical placement coverage no longer requires a retained backdrop per glyph.

A generation owns both completed outputs. Saved versions can retain a generation
absent from the current front; retiring its camera preserves completed pixels
and drops placement/input recipes. It never captures the mutable live
`world_selection` as its pre-HUD input. Partial commits preserve the previously
committed outside pixels. Both output growths are admitted before either result
is initialized. The existing 256 MiB retained and live ceilings and 128 MiB
replay ceiling are unchanged. Direct units and other projected operations retain
their existing ownership; this change does not replace the whole compositor.

`CompositionStorage` tracks unique application-reachable composition texture
bytes, including live canvases, retained/projected outputs, replay scratch and
recycled scratch. Aliases share a lease. New leases are acquired before old ones
are released, so its high-water mark includes application-owned replacement
overlap. It excludes driver padding, resources outside these owners, diagnostic
readback staging and driver-retained in-flight allocations. It does not measure
total physical GPU memory. Retained admission also charges direct CPU payloads;
that conservative payload charge is separate from the unique texture peak.
Node descriptions now traverse projected children and batch operands. Admission
failures report the exact site, request, current charge and cap.

## Native verification

`test_fullscreen_hud_recipe.py` executes the production Session at 2240x1260
with 140 labels / 700 ordered native fill/text/blend/self-tint commands. Both 555/565
arms match an independent production GPU compositor exactly across 30 frames,
five actual map generations and repeated/reversed zoom. A separate fullscreen
oracle preserves a saved pair absent from the front through source replacement,
paired self-copy, reversed zoom and partial restoration. Reset releases outputs
and replay working storage. Existing 126 pixel oracles and retained view lifetime
regressions pass as well.

A private frozen accepted-baseline replay of the same dense Session workload,
with only a print observer added after its exception, rejects at 267,974,112
retained bytes / 708 nodes: `retained composition texture budget`. Candidate retained
outputs are 79,052,288 bytes; unique application-reachable composition texture peak
is 225,842,176 bytes, including old/new map overlap. The saved-pair oracle peaks
at 124,185,600 bytes. The approximately 30–32 ms submit-plus-forced-readback mean is
a diagnostic wall time, not displayed FPS or causal GPU timing.

Raw sources, binaries, logs, manifests and comparison receipts live in the
private `Renderer/.cache/composition-storage-step` evidence directory. Gameplay
qualification and installed identity are recorded separately at completion.
