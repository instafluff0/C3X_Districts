# Volcanoes

Ordinary Civ III volcano terrain (`real_terrain_type == 10`) now has a
current-code Renderer64 candidate with sixteen stable visual forms. Slot 00
keeps the previous cone; the others combine the ordinary authored crater with
the five existing mountain height fields, including the raised broad and eroded
forms and their corrected B facings. The form is selected from canonical tile
coordinates, independently of Civ III's four-neighbor PCX sprite mask. Wrapped
occurrences keep the same form and rock orientation.

The staged evaluation build uses dormant rock without inner lava or smoke.
When the four diagonal neighbors select forest or jungle, the fidelity
renderer places four small natural trees toward the lower edge of the same
rock mesh. Raised tiles receive no vegetation-floor decals. The fixed
references have not been replaced. Snow-specific art and natural-wonder
volcanoes remain deferred.

The repeatable current-code [Lab capture script](../../../studies/volcanoes/current_port.py)
produces [sixteen isolated grassland examples](../../../out/volcanoes/current-port/grassland-16-current.png),
[forest context](../../../out/volcanoes/current-port/05-forest/preview.png) and
[jungle context](../../../out/volcanoes/current-port/05-jungle/preview.png).
Each capture records the exact DLL, preview, scene and image hashes and requires
zero fallback tiles. The active-state check uses the same bare rock appearance.

The current native bridge and 64-bit Renderer64 companion compile with
`BUILD_RENDERER64.bat no-stage`. The private candidate shader bundle is built
with `Renderer/tools/prepare_renderer64_materials.py terrain mountain` and
keeps all other pinned shaders byte-identical. The connected asynchronous
bridge witness ran against that bundle before staging. The matching bridge,
64-bit renderer and helper plus the two changed shader sources are now staged
for an in-game evaluation. Windows-side hashes match the checked candidate,
and the staged startup probe is healthy. A small task-local backup under
`Renderer/native/build/volcano-port/pre-stage-backup/` retains the five
previous files for rollback. No injected code or fixed reference was changed
for this stage. Source art is reused; the generic runtime pack format is
unchanged.

Portable checks:

```sh
python3 -m unittest Renderer.native.test_render_core Renderer.native.source_fidelity.test_contract Renderer.lab.test_volcano_fixture Renderer.lab.test_natural
```

The ordinary category dispatcher remains available for routine category
checks. It currently cannot complete its all-source preparation because an
unrelated ignored city-study input is missing; the focused native captures
above use the built candidate directly and do not reconstruct that input.
Installed source evidence and older isolated studies remain in the
[volcano study notes](../../../studies/volcanoes/README.md). Their images are
exploration history, not this Renderer64 candidate or a live-game check.
