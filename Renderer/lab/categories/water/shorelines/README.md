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

The cliff candidate now joins the actual visible hill surface: the coastal floor
is a minimum rather than an added ledge, hills use the steep cliff envelope,
and rocky coverage is opaque before the face rises. Rock attachments sample
that same surface; upper details sample their own coordinates. The earlier
projection, normal-basis and depth corrections remain. Less regular upper/foot
scatter and a narrow wet-rock material band refine the selected Civ V skin.
Ordinary beaches and inland terrain keep their existing formulas. Source UVs
and texture bytes are unchanged. See `Renderer/docs/coastal_cliff_findings.md`
for evidence, reconstructed choices and native comparisons. Promote the
corrected pack and compatible DLL together. The regular diagnostic now covers
noon, evening, midnight and dawn with the shared light basis, including a lowland
control. Explicit deployment requests do not replace fixed reference images.

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
