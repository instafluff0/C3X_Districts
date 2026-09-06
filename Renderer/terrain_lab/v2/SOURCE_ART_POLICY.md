# Source-art-first Lab policy

The user requires reuse of Civ VI source art where it supplies the needed look.
Agents must not invent substitute art merely because generating it is easier
than finding, decoding, or correctly rendering the source. This applies to all
visual tracks, including terrain detail, cliffs, dunes, vegetation, roads,
buildings, resources, and units. Preserve explicitly selected alternate source
skins and label their provenance; do not silently switch the campaign's skin.

## Required decision order

1. Inspect existing source inventories, normalized packs, source metadata and
   relevant installed packages for the needed mesh, part, texture, normal,
   height/displacement, mask, decal, and material roles. A missing mesh in the
   current normalized pack does not establish absence from the source game.
2. Render the best matching source asset/material faithfully first: preserve
   rigid-body proportions and UVs, use uniform scale/rotation, and recover the
   intended material channels and source-height construction where supported.
   Diagnose missing channels, bad sampling, deformation, or lighting before
   replacing the art. Screenshots define appearance targets, not proof of the
   source engine's implementation or an available standalone mesh.
3. Adapt placement and topology to Civ III with the smallest necessary joins,
   masks, clipping, terrain-following seams, and source-piece composition. Keep
   the dominant visible form and material detail source-authored. Compare the
   adapted result with an unmodified source witness at matched display scale.
4. If a suitable source is inaccessible, undecoded, or genuinely absent, record
   the exact search/evidence and missing capability. Request extraction/import
   support through the existing ownership boundary. Continue other work; a
   missing art family does not stop the whole track. Do not silently substitute
   invented art or claim that an unsuccessful search proves none exists.

## Procedural adaptation is not replacement art

Allowed implementation includes Civ III terrain topology, route paths,
shoreline classification, continuous blending, deterministic source-asset
placement, lighting, and source-height/material-driven surfaces. Procedural
joins must remain subordinate and must not become the main visible cliff,
mountain, dune, or other feature. A generated rock mass covered with a Civ VI
texture is still generated geometry; disclose it as such, not as source art.

Do not use noise-generated rock faces, invented mountain/dune bodies,
hand-authored substitute textures, AI-generated textures/models, or generic
primitive buildings/resources as final beauty candidates when source art should
be used. Do not add invented high-frequency detail to hide missing source
normals, height, material channels, or inadequate sampling.

Simple synthetic geometry and diagnostic colors remain allowed for interface
and topology tests, explicitly labeled `diagnostic_proxy`. They cannot satisfy
source-fidelity or beauty acceptance. Keep existing experiments recoverable and
separate; do not delete another owner's work. A proposed original-art fallback
needs explicit user authorization before becoming the selected beauty solution;
log it as an unresolved exception and continue source-backed work meanwhile.

## Evidence and review

Each candidate's owned audit must identify the important visible components:
source pack/asset IDs and hashes, mesh and material-channel provenance,
transform/UV changes, and any generated geometry or detail. Classify each as
`source_reuse`, `source_adaptation`, `diagnostic_proxy`, or
`proposed_original_fallback`. Separate confirmed package data from inference.
Reuse cached inventories and witnesses; do not repeat a full extraction search
on every visual edit. Keep local installed art and source-specific conversion
offline; runtime packs stay generic and no Firaxis assets are redistributed.

Q8 checks provenance as well as appearance. An attractive result cannot pass
if its dominant feature is an unapproved invented substitute. Q0 owns shared
platform support, not artistic permission. Each visual owner remains responsible
for the source fidelity of its output; missing importer capability is a precise
interface request, not permission to cross-edit or replace art silently.
