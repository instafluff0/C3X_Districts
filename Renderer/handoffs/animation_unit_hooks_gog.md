# Unit animation bridge: GOG CSV pickup

The DLL now has a body-only unit renderer and the injected source has three thin
capture/forwarding wrappers. The requested CSV rows have now been supplied as inleads and approved injected
compilation passes. The rows below are retained as the exact GOG reference. The project reserves CSV edits for the
human; no CSV file was modified by this work.

For the **GOG Windows VM prototype**, the three existing rows now match the
following. The reduced routine needs **nine stack arguments**, not the four in
its current declaration. Do not merely change its capability to `inlead` while
retaining the old signature.

```csv
inlead, 0x5CBF50, 0x0, 0x0, "Unit_tick_anim", "void (__fastcall *) (Unit * this, int edx, PCX_Image * canvas, int pixel_offset_x, int pixel_offset_y, bool include_status_layer)"
inlead, 0x5F88B0, 0x0, 0x0, "Sprite_draw_unit_body_normal", "int (__fastcall *) (Sprite * this, int edx, PCX_Image * background, PCX_Image * canvas, int pixel_x, int pixel_y, char * palette_path, PCX_Color_Table * color_table)"
inlead, 0x5F8940, 0x0, 0x0, "Sprite_draw_unit_body_reduced", "int (__fastcall *) (Sprite * this, int edx, PCX_Image * background, PCX_Image * canvas, int pixel_x, int pixel_y, int scale_x, int scale_y, int scale_divisor, char * palette_path, PCX_Color_Table * color_table)"
```

`palette_path` retains the existing legacy argument name and is forwarded
unchanged. Reduced zoom passes scale factors `1, 1, 2` followed by that same
argument and the palette selected by Civ III. All other scale combinations pass
through unchanged.

## Evidence and build scope

- `verification/animation/unit-hook-audit.json` records exact entry/call bytes
  read from the VM's unmodified executable. Normal returns with `RET 0x18`,
  reduced with `RET 0x24`; the ordinary reduced caller pushes all nine arguments.
- `verification/animation/unit-injected-compile.json` covers the approved
  `TEST_INJECTED_CODE_COMPILE.bat` smoke. Approved compilation now includes the supplied inleads and corrected signature.
- `native/test_unit_bridge.cpp` compiles the actual extracted wrapper source
  against native mocks. It checks both zooms, unchanged original arguments,
  disabled/invisible/unmapped handling, palette capture, nested context
  restoration, body fallback and native underlay/HUD ordering. The Windows x86
  version exercises the nine-argument calling convention.
- GOG addresses are confirmed. **Steam and PCGames.de addresses remain
  unverified. Zero is not a patch address:** do not install these inleads for
  those builds. Leave all three as definitions/native body rendering there
  until matching addresses and bytes are audited.

## Runtime boundary and fallback

`Unit_tick_anim` establishes a scoped unit/canvas context and still calls the
complete native routine. Only its exact current-frame Sprite body is eligible
for replacement. Original visibility, movement, timing, selection and HUD code
continues. The body renderer uses the effective palette passed by the native
body call rather than assuming the owner is the displayed civilization.

Configuration-off, unsupported unit/action/material, unrelated Sprite/canvas,
and pre-transfer failures retain the native body. Armies remain entirely native
in this initial prototype; their paired bodies are never partially replaced.
If these rows remain definitions, all unit bodies remain native.

The animation checkpoint is now staged for a user-run GOG test through normal
`INSTALL.bat`, with `enable_custom_rendered_units` controlling optional custom
unit bodies. See `docs/animation_integration_checkpoint.md` for coverage, passing
checks, open unrelated regressions and the batched gameplay checklist.
