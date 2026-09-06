# Required shadow silhouette

User requirement: **For all things that cast shadows, the shadow must be in the
shape of the thing itself.** This is a hard acceptance gate for every category.

Use the same transformed source mesh that supplies the visible object, including
compound components, current pose, uniform placement transform and authored
cutout alpha. One shared light projects that geometry onto actual receivers.
Soft penumbra may soften its edge; it may not replace the silhouette with an
oval, blob, rectangle, tile mask or generic ribbon. Contact comes from the same
blocker geometry, not an independently painted footprint. Missing source/alpha
or receiver data is an explicit incomplete gate, never permission to fabricate
a substitute shadow.

Q6 implementation: `systems/lighting/scene_shadow.cpp` collects actual source
terrain/feature triangles from Q0 authoritative world positions and evaluates
source alpha before light-depth writes. It removes **all** inherited projected
shadow layers (kinds 7, 10, 12, 14), including generic feature ribbons. Real-map
candidate `source-world-16` is the first directly inspected application; prior
complete-map candidates retain legacy shadow layers and are superseded for this
gate. Existing source-category diagnostic fields already rasterize actual mesh
triangles and cutout coverage. Analytic proxy scenes remain diagnostic only.

Do not claim source completeness from this rule alone: Q7 owns complete compound
source bindings/poses and Q4 owns actual raised terrain and vegetation geometry.
The lighting field must consume their final source-complete geometry at
convergence, not silently promote inherited incomplete source import paths.
