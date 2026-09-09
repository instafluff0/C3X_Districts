# Units

Current full Conquests unit roster, source normals and material addressing, native actions and current shadow behavior.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.

Unit pack preparation is automatic. It consumes the current animation runtime
and authored-normal catalog, builds disposable output and preserves native keys,
source geometry, material addressing and animation palettes. Edit source assets
or `Renderer/native/environment_refresh/prepare_units.py`, not generated payloads.
See `Renderer/native/environment_refresh/UNIT_FIDELITY.md` for current contracts.

The opt-in `sizing`, `sizing-gameplay` and `sizing-move` cases compare a separate
six-subject anatomy-sizing pack and 1x/2x material sampling. They use a larger
diagnostic canvas and do not promote that pack or change production sizing.
See [the study](../../../studies/units/README.md) for results and the long-weapon
dirty-bounds requirement. The 192-pixel tile view is true 1.5x projection.
