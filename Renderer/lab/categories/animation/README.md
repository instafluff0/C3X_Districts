# Animation

Current unit and resource playback follows native cursors, timestamps, paused state and bounded caches.

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
