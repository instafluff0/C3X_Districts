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

## Integrated game qualification

Composition and cancellation were merged into the authoritative checkout at
`43c6523b595546cf4183cc5f3ea40aa5b4a04c98`; the material-plan candidate remains
outside this integration because its native gate demonstrated little actual
coverage and no useful frame benefit. The final renderer build, fullscreen HUD
and cancellation regressions, config-off delegation check, asynchronous bridge
fixture and staged startup probe pass. The asynchronous fixture adopts all 32
camera transitions without composition errors and reports 59.89 successful
fixture presentations/sec during its warm interval. This is a fixture result,
not game FPS or physical scanout.

The matching staged binaries are:

| Binary | SHA-256 |
| --- | --- |
| `C3XRenderer.dll` | `d27666ec6ba0415de36e8f16319b9bea3b10f00141a0164f05fddd3e597c6058` |
| `C3XRenderer_x64.dll` | `3cf70de3e5aa262e697b0dde054b32440d09809abb9a9e15a71a1bb5d8651432` |
| `C3XRendererHelper64.exe` | `fd32c36deee58878fa74d340892bf247e4b5dd41f758144ea2447d8e303ffbc1` |

One bounded 120-second automated game run uses the user-accepted 1498 AD save,
normal effects and the 2240x1260 game window. It completes all 32 scroll commands
with no recorded renderer failure or early game exit. The sampled coastline and
city positions change, while city labels, population/production text, units,
selection, minimap and HUD remain visible. Eight representative samples and two
full-resolution frames were reviewed. This establishes the composition repair's
functional game qualification; it is sampled inspection, not new art acceptance.

The read-only successful-presentation counter measures 5.57 presentations/sec
during the scroll command window and 13.34 during the subsequent stationary
interval. Detailed trace and a 2 Hz window observer are enabled. These are
instrumented game measurements, not physical scanout, a paired speedup ratio or
an uninstrumented ceiling. Zoom stays at its normal scale, so this run makes no
live zoom-cadence claim. The 60 FPS idle/scroll target remains unmet. Logged
retained composition charge peaks at 210,089,468 bytes, below its unchanged cap;
this logged charge is distinct from the fullscreen oracle's unique texture peak.
Successful sampled composition calls have a 36.321 ms median, while final API
presentation calls have a 0.126 ms median. Remaining composition work and scene
preparation/drawing require further performance work; these CPU submission
intervals do not establish causal GPU time.

Private evidence is preserved under
`Renderer/.cache/composition-integration-step/`: source/shader/binary identities,
original and corrected fixture receipts, original capture, cadence analysis,
loaded helper/DLL identities, and cleanup receipts. The complete original game
capture contains 221 verified files / 293,027,325 bytes. All 22,072 frozen runtime
dependencies and original/disposable save bytes remain unchanged. The installed
game executable, JGL DLL and INI bytes match their pre-test hashes; the cursor and
environment are restored; game, helper, collector, observer and temporary test
tasks are absent. The previous matching binary trio is retained privately for
rollback. No injected source or patch-table changes were needed.
