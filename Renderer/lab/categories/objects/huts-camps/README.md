# Goody huts and barbarian camps

Neutral, static site bodies consume the shared world light and paged shadows.
Civ III supplies viewer-conditioned presence and camp tribe identity. Removal
invalidates both geometry and shadow pages; guards and resources remain separate.
The native m19 map boundary already owns this draw layer when custom rendering
is enabled. Config-off retains Civ III rendering.

Three normalized hut variants use a stable eight-bucket selection; primitive
camps remain the default in every era. Source body textures and UVs are retained.
The established vertical calibration is applied with inverse-transpose normals.
Ground decals and optional animated attachments are excluded from this static
site pass. Terrain surface height supplies the object anchor.

The fixture contains both sites, including a camp sharing a resource tile,
at four hours and two zooms. Game capture uses the current viewer's visibility
accessors; no omniscient site bits enter the visible scene.

`integration huts-camps` runs the production site lifecycle witness: removal
must clear ownership and shadows, match a cold render exactly, and restoration
must reproduce the original pixels. The fixture also exercises hillside
grounding and resource coexistence.

The tested API 17 DLL is staged in `Renderer/bin` with the existing 768 MiB
cache configuration. Run `INSTALL.bat` to update the matching injected game
executable. Installation and live-game verification are left to the user.
Individual visual categories are `goody-huts` and `barbarian-camps`.
