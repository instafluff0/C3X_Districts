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

The current cliff candidate preserves raised coastal height through the natural
terrain adapter and binds the existing Civ V cliff materials on steep faces.
It embeds source rocks into that rim using the imported scale/variation recipe.
The user selected the Civ V skin and rejected the initial boulder-only pass.
See `Renderer/docs/coastal_cliff_findings.md` for evidence and current previews.
The corrected cliff pack and placement must be promoted together after visual
acceptance; the candidate is not staged for game use.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.

The accepted surface transition carries authored gravel patches through
the dry margin and shallow bed, with larger submerged rock/crack patches farther
out. Fine sand grain comes from the beach base-color alpha; its separate height
channel is nearly flat. The grass edge uses authored detail to break up material
coverage. Broad patch variation preserves sandy stretches and desert sand remains
dominant. These are C3X material choices, not recovered Civ VI shader equations.
See `Renderer/docs/shore_river_material_findings.md` for the source audit and
matched native previews under `lab/out/shorelines/surface-transition/`. The user
approved this appearance for production; optional fixed comparison references
remain unchanged.
