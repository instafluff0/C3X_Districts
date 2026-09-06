# L4 Authored Hill Audit

The normalized pack contains four hill height families and no hill-specific blend or region-ID channels. Raw LOD0 PNGs are generated locally under `Renderer/preview/out/terrain_lab/hill_sources/` for `standard`, `continental`, `continental_plains`, and `continental_snow`.

The three 2048x2048 continental fields contain substantially more high-frequency structure. They remain valid candidates for later regional variation, but admitting one of them first would make it difficult to distinguish asset semantics from scale and filtering problems.

`standard` is the first admitted family because its 512x512 field has the clearest broad structures. The first direct trial used 0.26 authored UV units per world tile and 64 pixels of relief. It failed: the patch became a dense field of narrow ridges. That rejected configuration is not retained as a baseline.

The accepted lab configuration uses the same unmodified source at 0.085 authored UV units per world tile and 42 pixels of relief. At that scale the 2x2 patch samples one continuous crop with broad readable rises, no per-tile restart, and no topology seam. The raw heightfield remains visible on the left of `grass_authored_hill_standard.bmp`; the grass-shaded geometry is on the right.

This establishes the asset as usable macro input only at a much larger world footprint than the earlier renderer assumed. It does not establish that every map tile should receive authored hill displacement. The production design will still need an explicit topology mask that blends selected hill regions into surrounding flat terrain.
