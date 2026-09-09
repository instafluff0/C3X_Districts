# Shadows

One shared light basis drives terrain, buildings, resources and native unit
sprites. Noon casts screen-left, 18:00 down, midnight right, and 06:00 up;
the fixed slope preserves height-to-length scale. Source object normals use
the inverse transpose of the same world transform as their caster geometry.

The mixed fixture includes mountains, forest, a city, Iron, animated Horses
and two native unit bodies at all four phases and both ordinary zooms. The
shared receiver filter owns paging, depth bias and contact; units and animated
resources retain bounded local ground shadows. See
`Renderer/docs/shared_shadow_contract.md` for explicit caster/receiver limits.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.
