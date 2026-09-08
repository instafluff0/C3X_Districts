# Shorelines

Current world-continuous shore contour, beaches, rocky shores, depth and relief
join. Desert coasts carry the desert sand family to a narrow wet edge and into
the submerged shelf instead of exposing a second dry-beach ribbon.

Rock geometry remains exclusive to hill tiles adjoining water. The shoreline
witness includes both that rocky join and an ordinary lowland beach so changes
to the cliff composition cannot hide a regression in the unrocked coast.

The installed Civ V Environment Skin supplies four large and four small cliff
rock bodies, one shared base texture, two LEAN normal channels and one gloss
channel. The renderer consumes those authored meshes, UVs, normals and material
channels directly. Its continuous terrain-to-water cliff join remains generated:
the inspected terrain package describes cliffs as terrain material plus clutter,
not as a standalone authored wall mesh. ArtDef counts are retained as selection
weights; the source engine's final scattering algorithm is not claimed recovered.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.
