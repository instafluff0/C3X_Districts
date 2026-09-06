# Compound Landmark and TileBase conversion

`compound_landmark_importer.py` is the reusable lab-side source adapter for
compound landmark assets. Its mapping accepts multiple source packages and
assigns stable C3X asset IDs; package structure is decoded through reflected
types rather than asset-specific offsets.

## Component model

One `c3x.compound_landmark.v0` asset may compose:

- one or more indexed mesh components;
- state-selective draw bindings and material variants;
- one or more skeletons with stable bone names, parents, rest transforms, and
  inverse-bind matrices;
- compact normalized vertex weights;
- a terrain-surface decal component; and
- a future animation clip selected by its explicit skeleton/track-group
  binding.

The importer decodes model-to-mesh and mesh-to-primitive ranges, deduplicates
primitive geometry reused by multiple states, recomputes deterministic normals,
preserves UVs outside 0..1 with per-material repeat addressing, omits isolated
zero-area source triangles when valid geometry remains, and supports optional LEAN,
gloss, and ambient-occlusion material channels. Positions, bone translations,
inverse-bind translations, and LOD error use the mapping's uniform tile-unit
conversion. Quaternions and scale/shear are not scaled.

The admission profile fails closed on malformed state masks, incomplete
container ranges, missing material roles, incomplete LEAN pairs, invalid index
ranges, geometry made entirely of degenerate triangles, non-topological skeletons, non-unit rest
quaternions, invalid bone indices, weights that do not total 255, or non-empty
terrain-edit data without a separate conversion profile.

Decal requirements are component-specific. Dune and marsh mappings continue to
require height, while a TileBase may legally contain a base/fog-only decal.

## Initial proof set

The initial mapping compiles both source container classes through the same
normalization path. The complete resource-facing TileBase set already identified
by the resource inventory is:

- oasis: state table plus base/fog decal;
- fish: skinned mesh, 28-bone skeleton, two state-selective materials; and
- whales: wrapping-UV skinned mesh, 17-bone skeleton, two state-selective
  materials with base color, LEAN, gloss, and ambient occlusion.

The environment/clutter proof set now includes:

- both elephant variants: two models, two meshes, two materials, and two
  skeleton records; the elephant component uses an 18-bone vertex skin while
  its grass component uses a one-bone rigid-model binding, and its terrain
  contact decal supplies base color, fog color, and height channels;
- five ordinary and five snow silk variants containing two or three rigid
  models, per-material clamp/repeat addressing, and terrain-contact decals; and
- two shared snow-boulder clump entries whose source wrappers intentionally
  carry a one-bone model but no mesh containers; their actual visual payload is
  the normalized ground decal, represented as `skeleton_only_decal` rather than
  rejected as incomplete geometry.

The generated proof pack contains 17 compound landmarks in total: 14 with
geometry, 4 with skinned geometry, and 15 with decals. These mixed Landmarks prove that component ranges and bindings are preserved
without flattening the source into one assumed mesh or one assumed skin mode.

This creates normalized source assets only. Runtime skeletal rendering and
category ownership are not enabled by the importer.

Run from the project root:

```sh
python3 Renderer/tools/asset_compiler/compound_landmark_importer.py
```

The ignored local pack is written to
`Renderer/packs/CompoundLandmarksNormalized`; source names, installed paths,
pointer evidence, and hashes remain in
`Renderer/preview/out/compound_landmarks/build.json`.
