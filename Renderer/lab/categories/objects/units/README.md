# Units

Current full Conquests unit roster, source normals and material addressing, native actions and current shadow behavior.

The current build is the approved revision 1. `standard.json` identifies the
shared implementation, dependencies, fixture recipe and focused regression tests.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. New visual changes need explicit
approval before replacing these references. See `Renderer/docs/visual_fidelity_playbook.md`.

Unit pack preparation is automatic. It consumes the current animation runtime
and authored-normal catalog, builds disposable output and preserves native keys,
source geometry, material addressing and animation palettes. Edit source assets
or `Renderer/native/environment_refresh/prepare_units.py`, not generated payloads.
See `Renderer/native/environment_refresh/UNIT_FIDELITY.md` for current contracts.
