# Rivers

Current river corridor and bank response, with authoritative topology and bounded world-page queries.

The two synthetic review views share one deterministic watershed rather than an
isolated channel. It includes an inland source pool, a winding connected course,
a mountain chain and pass, foothills, mixed terrain, river-aware forest, nearby
jungle, an irregular coast, and a mouth that continues into coast, sea, and ocean.
This makes the fixture useful for judging curvature, relief avoidance, vegetation
clearance, bank carving, terminal treatment, and water blending together.
`detail` centers the inland source and mountain corridor; `gameplay` centers the
forested/jungle lowlands and river mouth so the paired captures cover the whole
course at readable scale.

The mountain corridor includes multiple mountain-owned tiles directly incident
to river edges, while never pinching one edge between two peaks. Production
mountain geometry consumes the same continuous river field and carves its height,
material coverage, and finite-difference normals back from the wet bank. This
keeps genuinely adjacent peaks visible as valley walls without hiding the river.

River banks use seamless world-anchored variation at three scales: broad
sediment patches alternate sand and darker soil, a finer field breaks up the
damp waterline, and sparse authored clutter supplies gravel-scale detail. The
bank width follows the same fields so the material variation and irregular
silhouette agree instead of reading as a uniform painted ribbon. A wider
world-noise feather affects only coverage at the dry outer edge, preserving the
fine bank texture while smoothly revealing the underlying grassland or plains.

The current Lab candidate keeps the established river course and applies a
very narrow, shallow bed depression to terrain, underlay, and river surfaces.
The bank is mostly exposed terrain with low-opacity, irregular deposits
instead of a continuous dark soil strip. Sparse grit reuses authored shoreline
crack cells and the river gravel atlas, with a little shallow gravel through the
water. This visual candidate has not replaced a fixed reference or been staged
for game integration.

At the mouth, the river surface alpha now fades across the optical shore so
the ocean remains visible under the last reach instead of a solid blue cap.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.

The accepted surface transition uses the river base-color alpha's fine
pebble pattern for bank contrast, restrained normals and outer coverage breakup.
The authored river material supplies most bank color; sand and wet-soil blending
preserve its grain. The clearer teal water response and bank shading remain
separate, and channel width/topology are unchanged. See
`Renderer/docs/shore_river_material_findings.md` for the source audit. Matched
native previews and production-path verification are under
`lab/out/shorelines/surface-transition/`. The user approved this appearance for
production; optional fixed comparison references remain unchanged.
