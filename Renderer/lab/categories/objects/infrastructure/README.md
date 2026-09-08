# Infrastructure

Current roads, railroads, mines and farms; preserve category ownership and fallback.

The fixtures supply two connected runs and a cross-branch, plus a separate mine
and irrigated farm, with and without surrounding relief/vegetation. All four
custom ownership flags are checked through the production API. The present
headless output shows gaps between route segments despite connected input
nodes; this is an exposed current limitation, not a new design.
This category does not imply new support for pollution, craters, colonies or
other tile improvements that the current renderer does not replace.

`standard.json` identifies the shared implementation, dependencies, fixture recipe
and focused regression tests. The current checkout is authoritative.
Reference captures reproduce production using small synthetic diagnostic scenes;
they are not claimed to be live Civ III captures. References are optional review
aids. See `Renderer/docs/visual_fidelity_playbook.md`.
