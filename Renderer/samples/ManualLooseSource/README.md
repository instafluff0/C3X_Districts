# Manual Loose Source Sample

This sample demonstrates the source-agnostic input shape for C3X renderer packs. It intentionally references a placeholder mountain `.glb` path so the compiler can prove it reports missing conversion outputs without failing the whole pack.

Run from the C3X root:

```powershell
py Renderer\tools\asset_compiler\c3x_asset_compiler.py import-loose
py Renderer\preview\render_iso.py --pack Renderer\packs\ManualLoosePrototype\manifest.json --output Renderer\preview\out\manual_loose_1024.bmp --width 1024 --height 768 --grid 16
```

Future Civ VI conversion work should produce this kind of manifest after `.fgx -> .cn6 -> .glb` normalization succeeds.
