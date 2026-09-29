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

The runtime compiler also emits optional generic joint metadata for live pose
transitions. It preserves the original baked palettes; the fidelity preparation
step preserves the metadata when adding source normals and tangent frames.
See [joint and direction transitions](../../../../docs/renderer64_scene_and_motion.md#joint-and-direction-transitions)
for binding identity, lifecycle, timing and current GPU evidence. The whole-pack
`Renderer/native/test_unit_rig_pack.cpp` oracle reconstructs first/middle/last
palettes for every used joint; `test_unit_pose_transition.py` covers intermediate
blends, interruption and retirement. Live-game evidence is recorded in the
[scripted testing guide](../../../../tools/scripted_game_test.md#combat-diagnostic).

The [settler carrier study](../../../studies/units/README.md#single-settler-carrier-lab-study)
selects one of Civ VI's two identical backpack members. Its private preview key
remains separate; the current game-facing recipe also selects this one member
for `PRTO_Settler`. The seven-action pack covers idle, run, fidget, run-stop,
capture, founding and death. The backpack-to-pelvis socket is an offline visual
calibration rather than confirmed source-engine attachment behavior.

The opt-in `sizing`, `sizing-gameplay` and `sizing-move` cases compare a separate
six-subject anatomy-sizing pack and 1x/2x material sampling. They use a larger
diagnostic canvas and do not promote that pack or change production sizing.
See [the study](../../../studies/units/README.md) for results and the long-weapon
dirty-bounds requirement. The 192-pixel tile view is true 1.5x projection.
