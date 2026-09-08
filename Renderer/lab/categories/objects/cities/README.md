# Cities

Current city-fidelity templates, single-era growth, central modern palace, source materials, paving and facade lighting.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.

Reusable layout, footprint coverage and emissive-facade sampling live in
`Renderer/lab/shared/cities`. The current compiler is
`Renderer/native/city_fidelity/prepare_pack.py`; category commands automatically
rebuild when consumed assets or shared layout/lighting code change. The builder
records actual file reads and validates disposable output before replacing the
current pack. Its `--output` option remains available for isolated diagnosis.
The local normalized inputs and selected layouts are under
`Renderer/packs/CityFidelitySources/current`, not the old experiment folders.
The six selected layouts plus ordinary culture/era growth retain the production
122-model, 37-material, 72-template pack without visual changes.

The reference views currently show the modern American capital across four day
phases. They do not certify every culture/era/site. Other palace styles and
constrained-site placement retain the existing limitations in the native city
checkpoint; accepting the current build does not erase those limits.

The unselected alternate central-city recipe/data is retained in
`Renderer/packs/RendererSourceStudies/city-alternate-central-inland`, separate
from current templates and approvals. Selected inland, wooded and coastal-fallback
source comparisons remain in `Renderer/lab/references/source-studies`.
The coastal problem is footprint/foundation placement, not permission to change
terrain clearance or lighting. See [source findings](../../../../docs/source_art_findings.md).
