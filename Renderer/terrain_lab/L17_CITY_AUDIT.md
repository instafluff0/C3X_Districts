# L17 City Audit

Status: complete; deterministic 192-tile alternate-skin candidate critically inspected and approved under the user's 2026-09-05 autonomous-review authorization.

## Scene and provenance

- Terrain and all prior object layers are the unchanged authoritative 16x12 BIQ viewport inherited from L13-L16.
- `fixtures/l17_cities_192.csv` is a separate deterministic, source-independent Lab augmentation. It adds twelve route-connected city witnesses without changing BIQ terrain or gameplay fields.
- The matrix covers four eras, all three Civ III city sizes, five culture labels, four owner colors, six walled settlements, and two capitals.
- Components, base-color materials, wall bodies, and emissive channels come from `CityComponentsNormalized` and `CityAdjunctsNormalized`. No procedural buildings, smoke, fire, bloom, or invented light is present.

## Critical visual review

- Town, city, and metropolis compositions remain distinct at native and reduced Civ III scales while staying subordinate to the map's mountain silhouettes.
- Compact deterministic golden-angle placement reads as a settlement rather than repeated isolated props and leaves room around each city for retained labels and population displays.
- Era silhouettes are immediately distinct. Restrained owner tint does not overwrite source materials, and source wall segments remain a secondary perimeter rather than a competing landmark.
- Noon deactivates emissive contribution. At midnight, source-authored windows and lamps remain localized and legible without bloom or invented analytic lights.
- Coast, river, relief, vegetation, resources, roads, and railroads remain visible around the city envelopes. The no-city control is byte-identical to the approved L16 scene.

## Deterministic evidence

Two successful `python3 Renderer/tools/renderer_dev.py lab` renders produced byte-identical outputs; one intervening Parallels dispatch attempt failed before rendering and did not alter evidence:

- noon complete: `53d9717775789cee0e3a67fd936447817818c90672c7c8945c6b8a72ae5a2ed6`
- midnight complete: `5910b2088685cb54ee3b1cc779f57ca6e566ad36ad56fc7c04968be8b34d02dc`
- reduced: `8a26cd75c6494ec8822457e957c89cf647427c5f60481458e3cfc597e07129fe`
- no cities: `3c112b32e677df37950498d681637e8e5c1a668710bccac9f2a10b0089533a9a`
- cities only: `73bff06a14142036428aa4e0da6a87c7444cc1d7f42791979c3fa7540401e5b4`
- Lab scenario: `ef7b24cb0cec97c7b6de7971552ebdea9725c09ec666f3eb57a5423aeab8ed47`
- city runtime: `2e2f480466a23304df7ec655c959757172895b747ea4e163e52c432c98538646`
- wall runtime: `7a6516ff61a9ba72765dcf99612f54047934b6c9db1e4da0fce9cdff10de1bf7`

The no-city control is byte-identical to the approved L16 native render.
