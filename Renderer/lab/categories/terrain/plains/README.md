# Plains

Warm source plains materials now combine world-scale source detail, exact
source-triangle surface patches, hill-top blending and the shared environment. Patch placement
is deterministic in world coordinates and fades against the authoritative biome
field, so neighboring tiles do not expose their boundaries.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.
