# Seas and oceans

Current source water optics and planar object reflections, using the shared environment.

Renderer64 derives its optical view direction from the continuous projected
water-plane basis around the viewport center. Wave textures still repeat, but
choosing a wrapped world copy cannot switch the reflection direction. The
focused D3D test covers world-copy changes and zoom while preserving specular
glint. The separate scene-wide warm ray overlay is removed; ordinary shared
lighting, shadows, sun/moon glint and coast/sea/ocean color families remain.
See [runtime import notes](../../../../docs/lab_material_integration.md#coast-and-water).

The separate [coastal-wave feasibility study](../../../studies/waves/README.md)
recovers the source crest atlas and evaluates surf in an isolated snapshot.
It is pending approval and does not enable waves in production.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.
