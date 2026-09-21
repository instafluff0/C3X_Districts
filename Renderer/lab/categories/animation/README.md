# Animation

Movement/combat playback follows native action cursors and anchors. Selected
idle/work loops, resources, water and selection pulses use renderer-owned time
and immutable captured scene/visibility inputs. Unselected idle bodies freeze;
fog hides units and freezes explored-but-not-visible ambient content.

An owned cadence thread schedules retained frames independently of Civ III's
message pump, through the existing D3D worker and composition presenter on the
same window. It targets 33 ms without accumulating work; native/gameplay updates
do not need to run for ambient animation. Native actions, UI setup and focus changes do not pause ambient motion on a
visible map. Hidden/minimized windows, native ownership and lifecycle boundaries
still apply. See
[`visual_frame_ownership.md`](../../../docs/visual_frame_ownership.md).

The native 66 ms loop continues gameplay advancement and compatibility recovery
when there is no eligible retained front. Its scheduler traces diagnose that
fallback; they are not the normal resident animation cadence. Renderer visual
status and actual blocked-UI desktop tests measure independent delivery.

The detail and gameplay studies each have matching `start` and `mid` views:
six production unit families, the same surroundings, native action cursors 0
and 7. They show pose changes rather than comparing different terrain scenes.
They are a small visual sample, not proof of every action or interruption;
the unit lifecycle and resource temporal/reuse tests cover those contracts.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.
