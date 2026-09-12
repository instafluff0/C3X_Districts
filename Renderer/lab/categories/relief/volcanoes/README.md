# Volcanoes

Ordinary Civ III terrain volcanoes (`real_terrain_type == 10`). The user accepted
the rock skin, existing shape, inherited relief detail, static crater lava and
shared cast shadow on 2026-09-12, and authorized production staging. Smoke,
particles, emissive glow and animation are excluded. Natural-wonder volcanoes
remain deferred. No fixed reference has been replaced.

```sh
python3 Renderer/renderer.py lab volcanoes
python3 Renderer/renderer.py test volcanoes
python3 Renderer/renderer.py integration volcanoes --renderer-only
```

Every review case retains mountains for comparison. Detail/activity, gameplay
and coastal cases use the accepted adjacent pair; `isolated` moves the pair away
to exercise the ordinary ground surface. Noon previews use tile widths 128 and
224. Static lava remains present in both activity states, as in the approved
image; captured activity does not enable emission or attached effects.

## Production implementation

The legacy raised-land pass omits ordinary volcanoes in the natural fidelity
profile. Their existing height survives in the natural ground and unified
mountain meshes, but those replacements previously lost the volcano material.
Both surface families now carry local volcano offsets from captured terrain
identity through indexed GPU geometry and the CPU mesh cache. Queries observe
the authoritative neighbor dependencies and preserve wrapped occurrence space.
The shared material uses the already loaded rock and lava DDS textures; source
art, height, normals, crater shape and inherited mountain detail are unchanged.

The mountain caster formerly clipped the volcano body against mountain-only
coverage. It now also accepts the raised volcano footprint using that same
local ownership. Existing shared shadow pages and receiver lighting remain
responsible for casting and receiving shadows.

The native lifecycle witness removes a volcano, checks exact warm/cold image
parity, creates one at a different captured placement, repeats parity, restores
the original and checks exact cached repeat with no geometry upload. It also
checks simultaneous volcanoes, scrolling and wrapped occurrences against cold
renders. The ordinary
terrain-edit witness and portable ownership/wrap tests also remain selected.
Generated renders and verification receipts live under `lab/out/volcanoes/` and
`lab/out/integration/`; the staging receipt records the exact tested DLL identity.

## Preserved source evidence

All four material DDS files and three macro channels reproduce exactly from
installed Expansion2 source. This was a material-ownership defect, not an import
resolution failure. The BC5 channels' intended source shader roles remain
unconfirmed; no speculative normal decode has been added. The lava UV registration
(+.015, -.002) is the accepted measured C3X calibration, not a recovered engine
transform. Detailed source findings and the fixed-coordinate diagnostic controls
remain in `lab/studies/volcanoes/`; those controls are not production code.

The ordinary inland ground retains its existing 16 subdivisions per tile; the
mountain neighborhood retains 64 and its inherited detail. This work preserves
both geometry paths rather than redesigning their silhouettes. Actual injected
Civ III observation remains the user's game check after staging.
