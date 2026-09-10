# Animation

Gameplay unit playback follows native cursors. Idle/work loops and resources use
their authored source duration on a shared pause-filtered presentation clock, so
Civ III interturn stalls resume without a catch-up jump. Idle redraw eligibility
is 50 ms on Civ III's existing 66 ms timer so callback jitter cannot halve the
effective cadence. Mouse input keeps a bounded 250 ms click-decision guard;
the guard is bypassed immediately when Civ III reports a selected-unit map/
pathfinder hold. Other clicks and the release interval remain guarded, without
a separate timer or input hook.

Ambient unit loops have a stable phase derived from the native unit identity;
nearby units do not share a zero-offset loop. Camera, zoom and callback order
leave that phase unchanged. Movement, combat and explicitly native-directed
fidget clips retain their authoritative native cursors. Exact posed-image reuse
keys the resulting frame, so reuse cannot synchronize different unit phases.

The `OutputDebugStringA` test stream emits one `scheduler-callback` record per
eligible Civ III timer callback. Its callback/presentation gaps, mouse-button
mask and hold time, authoritative pathfinder state, pre/post-guard decisions and reason correlate
with `map-complete` records carrying requested/presented/pending counts and the
logical animation time. A matching `timer-return` record splits time between
the scheduler and Civ III's native timer/Animator path and reports request and
presentation deltas. Together these distinguish callback starvation, scheduler
suppression, a blocked native handler and requested redraws Civ III did not
present.

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
