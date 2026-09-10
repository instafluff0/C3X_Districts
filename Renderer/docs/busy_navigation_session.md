# Busy-map navigation session

User-requested continuation after staging the verified September 9 navigation
improvements. This extends the existing native preview and camera queue. It is
not a new presenter, simulation engine or permission to draw uncaptured objects.

## Workload

Run one continuous renderer instance with waves and reflections enabled at
2240×1192. Populate fixed world positions with cities, connected roads and
railroads, farms, mines, camps, supported resources and units from several
families. Units keep stable identities as they enter and leave the view. Report
actual visible counts after every camera change; a requested count alone is not
coverage. Exercise 24 units first, then 64 where sufficient visible sites exist.

The sequence starts with a cold view and no hidden pose warm-up:

1. Idle for ten seconds while waves, resources and independently phased units
   animate. Include first-use costs in the startup report.
2. Scroll continuously for ten seconds, reverse, and return through visited
   content. Continue unit movement, direction changes and ambient animation.
3. Cycle 128 → 160 → 192 → 160 → 128 while the map remains busy. Include first-use
   zoom costs separately from later returns.
4. Jump to several distant minimap destinations, pause long enough to observe
   completion, and scroll locally at each destination. Keep unseen, overlapping
   and previously visited destinations distinguishable.
5. Return to the initial area and idle again. Include attack, fortify, fidget and
   interruption/held-endpoint transitions throughout the session.

Do not reset caches between these phases. Do not keep every unit permanently
screen anchored or make all units change action together. Preserve native action
cursors and identity-based ambient offsets. A view with fewer eligible units
must report that fact rather than silently claiming the requested density.

## Timing and correctness

Maintain two explicitly separate modes: a deterministic script for repeatable
pixel/ownership comparisons, and wall-clock playback for responsiveness. The
latter advances clocks by actual elapsed time, records intended and dispatched
input times, and reports skipped/coalesced input and delayed completions. A slow
render must not slow the authored world clock or silently remove missed updates.

Report cold startup, initial idle, scrolling, first/repeated zooms, distant jumps,
return travel and final idle separately. Include total completion, capture,
geometry/assembly, map/GPU wait, unit-body drawing, pose misses, allocation/cache
pressure and address-space samples. Preserve detailed event records without
writing a full BMP on every measured frame; use bounded representative images
and output hashes, keeping evidence writes outside measured intervals.

Check replacement ownership against each current capture, unit bounds/clipping,
stable identities, action transitions and exact current-camera results. Sample
independent redraws after the timed session so correctness verification does not
warm or evict the measured session's caches. Include returning to an edited
region in a separate invalidation witness.

These are standalone completed frames. Actual input-to-display and the plan's
1,000 native-presented-frame/30 FPS target remain a separate integration check.
The native bridge is synchronous and its existing approximately 66 ms animation
timer does not establish 30 Hz. This test must expose those limits rather than
label an unpaced render loop as live-game performance.

## Status

Planned; the existing isolated idle, mixed-action, pan, zoom and distant witnesses
do not yet implement this continuous session. Implement and run it after the
authorized production staging, then use its slowest phases to select further
optimization. Preserve [the original targets](navigation_implementation_plan.md)
and [native presentation constraints](native_async_presentation_audit.md).
