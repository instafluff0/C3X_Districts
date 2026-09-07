# City opacity and wilderness check r34–r39

Status: partial Lab material improvement, no overall city acceptance or milestone
promotion. One era per city remains the user's policy. Roads remain deferred.
The broader palace roster is tracked in `Renderer/docs/city_palace_asset_import.md`;
its four unresolved Gran Colombian tree attachments remain explicit.

## Changed pixels and rejected experiments

Modern coastal r34 preserves r32 bodies, scale, camera, terrain, lighting and
ground placement. Restored opacity masks open roof vents/details that were filled
before. At normal gameplay size, 259 pixels change by more than 2/255 in each of
the noon and midnight images, within [822,364,935,434]. This is a small material
completion gain, not a transformation of city composition. See
`out/city-material-r2/modern-opacity-native.png` (previous left, candidate right).
Four r36 disabled-channel images reproduce r32 exactly, including the new wire
layout and texture bindings. Earlier candidates remain available.

r35 adds direct-light metalness to the same scene. It darkens broad facades
(6,123 noon and 6,670 midnight pixels change by more than 2/255) without the
filtered environment specular that should contribute to those surfaces. It is
an unselected diagnostic. Further diffuse/specular coefficient tuning alone is
not the next fix; recover the missing environment response first.

The previously untuned wilderness city site is fixed at tile [6,6] in the existing
100-tile test.biq region. r37 exposes a real placement defect: the old dry-ground
query accepts forest cells. Preserve that rejected composition despite its
passing Windows comparisons. Optional vegetation clearance now excludes sampled
forest/jungle cells, with a 0.12-tile margin beyond normal placement padding.

r38 cannot place seven source bodies within the 0.8-tile half-extent after 33
bounded greedy alternatives preserving the four-body growth prefix. This is a
search failure, not proof that no layout exists. No r38 render was made. r39
places the four-body stage legally at the same anchor/scale and 0.8 extent;
its initial unrendered 0.65-extent failure is retained separately. Neither terrain
nor trees were moved. The small stage still reads as a compact tower cluster;
it does not resolve medium/large city growth. No new sea night-reflection gain
is claimed from this wilderness view. Wilderness is now a regression witness,
so another untouched site is required for a future untuned check.

## Source evidence and generic adapter

Installed rigid-model opacity shader at DX11 library offset 1870480 samples UV0
opacity and emits rounded sample coverage, while output alpha remains instance
fade. The source-family evidence is repeatable with
`python3 Renderer/terrain_lab/v2/qa/city_shader_material_probe.py` (add
`--disassemble` only when regenerating local disassembly). Exact active city
permutations/constants remain unproven. Lab single-sample cutoff 0.5 is an
explicit adaptation of this coverage behavior, not source MSAA parity.

The compact modern overlay contains 96 normalized material records, 69 metalness
and 31 opacity bindings, and eight unique textures (about 2.6 MiB). Each target
material is fingerprinted. Source slots 0x2c/0x30 are interpreted offline in
`qa/prepare_city_extra_materials.py`. Runtime-facing records use generic roles.
BC4 opacity blocks are losslessly placed in BC3 alpha with white RGB and the
same mip chain; independent DDS decoding verifies alpha values. Shared city
packs and the core importer remain unchanged. This overlay is not general
importer coverage for all cultures, eras or palaces.

The extended Lab wire uses magic 0x3B514353, seven texture paths and 92-byte
vertices, with separate emission UV at byte 84. Opacity uses UV0 in body,
emission, reflection and shadow paths. Masked bodies cast coverage-aware shadows;
the additive emission pass does not cast a duplicate shadow. Existing wire
layouts remain supported. The four masked bodies and one masked emission draw
are inspected in actual replay packets, including decoded shadow alpha.
The historical `material_channels_enabled` summary in r34–r39 predates the new
labels; `extra_materials`, `source_surface` and actual packet records are the
authoritative detailed evidence. Future producer summaries include these roles.

## Verification and next work

Run `qa/city_extra_material_evidence.py` with Python providing Pillow/NumPy and
clang++ on PATH. It verifies frozen placements/terrain, exact disabled images,
bounded-failure records, independent clearance against stored terrain samples,
texture hashes and actual packet coverage flags. It produces
[CITY_EXTRA_MATERIAL_r34_r39_EVIDENCE.json](CITY_EXTRA_MATERIAL_r34_r39_EVIDENCE.json).
Eight Windows comparisons pass: four r34, two rejected-composition r37, and two
r39. Twelve focused opacity/frame/ground/palace tests pass. These support the
pixel assessment; they do not constitute visual acceptance. No injected/native
code was changed by this pass and no full-Lab clean result is claimed.

The three largest remaining visible gaps are coherent urban ground and building
arrangement, incomplete facade environment lighting, and medium/large growth
without vegetation or relief conflicts. Solve placement with a broader layout
strategy or appropriate source component alternatives; do not keep retrying the
same late-slot perturbations or shrink the city to conceal failure. Preserve
single-era style, capital focal visibility, fixed cameras and prior candidates.
Local night light pools, the full culture/era/size matrix, and human/milestone
gates remain open. Cleanup of this pass's disposable linear render sidecars is
recorded in `CITY_EXTRA_MATERIAL_CLEANUP.json`.
