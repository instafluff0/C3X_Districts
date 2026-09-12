# Mountains

Five authored macro height/blend variants form a continuous relief surface.
Connected mountains broaden along captured adjacency; shared-edge geometry and
shadow coverage match. Grass, plains, tundra or desert climbs the lower slope
before stone takes over. Terrain decals remain later.

## Accepted material

The user accepted the cleaner mountain body and slope-aware base projection on
2026-09-12: "Put it in production, please." The implementation preserves the
snow caps, rocky feet, geometry, material channels and prior zebra-band fix.
No fixed reference was replaced.

The Civ V Environment Skin's nine 2048×2048 base/top/snow color, height and
specular files match the installed package byte for byte. Base/top height and
specular happen to be identical in this skin; other packs keep independent
channels. Runtime materials remain source-independent.

Ground-to-rock color/detail/specular coverage still uses final rise 0.02–0.48
world units. Source footprint and face steepness cannot override coverage.
The patchy-snow top layer blends at normalized source height 0.52–0.68; full snow
blends at 0.62–0.78 with the existing 0.02–0.25 slope gate. Per-projection height
derivatives retain broad amplitude 0.04 and fine 3.7× detail amplitude 0.12.

Added grain, height-color and crevice contrast stays intact at the rocky foot,
then fades over rise 0.38–0.75, retaining 8% above that. It does not return near
the snow line. Full snow's color multiplier remains intact. These masks and
amplitudes are accepted C3X artistic choices, not recovered source equations.

The ground portion retains its original world-XY texture mapping on flat land.
On rising, steep slopes the 20 fine terrain/hill color, height and specular
samples blend toward three-axis projection, preserving each family's original
scale, rotation and offsets. This corrects the vertically stretched grass base.
The broad terrain color field and ground-to-rock coverage remain unchanged.
Both new calibrations use captured local volcano coverage to retain the existing
volcanic material response; there are no fixed Lab coordinates in production.

## Verification

`python3 Renderer/renderer.py integration mountains --renderer-only` checks the
current shader adapters, source/capture contracts and native terrain-edit reuse.
The executable material test checks neutral flat ground, monotonic projection,
volcano protection and relocation/wrap independence with and without volcano
bindings. Volcano Integration checks the shared surface's material lifecycle.

Accepted experiments remain under `lab/out/mountains/body-study/` and their
reproduction tools under `lab/studies/mountains/`. Current-code visual checks,
integration and staging identities are recorded under
`lab/out/mountains/body-promotion/`. Staging never launches Civ III.

Production staging is complete. The two mountain close-ups reproduce the accepted
Lab pixels exactly; volcano-context views differ only by one color step in 53
and 17 pixels. Mountain and volcano Integration passed 249 and 253 tests,
respectively (one skip each), including edit reuse and volcano lifecycle,
scrolling and wrap parity. The staged DLL matches the tested candidate hash.

Earlier source evidence and controlled snow/banding probes remain in the study
notes and `Renderer/docs/visual_fidelity_playbook.md`; Git preserves history.
