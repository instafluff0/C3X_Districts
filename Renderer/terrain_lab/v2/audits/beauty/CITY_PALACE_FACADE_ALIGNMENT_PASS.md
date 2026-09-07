# Palace light reaches the paving border

The paving material already calls the same shared local-irradiance function as
terrain. The user's observation nevertheless exposed a real placement defect:
the palace's four light proxies were pushed to axis-aligned bounding-box planes,
away from the rotated emitting walls. Their outward directions sent much of the
light beyond the actual paving.

The offline `facade_plane_proxy` helper now uses the sampled wall normals and
emissive surface positions for the capital. It places each proxy 0.012 tiles
beyond that facade plane. In this matched pass only four palace positions and
directions change. Their intensity, color, range, all house lights, blockers,
window emission, paving wire, city bodies and shadow textures remain unchanged.
This remains an authored approximation derived from emissive geometry, not
decoded source point/area-light bindings.

A separate control disables local light on the added paving only. In the fixed
palace-border rectangle [770,450,830,485], the previous positions illuminate 139
pixels above 2/255; corrected positions illuminate 467, with maximum difference
32/255 against the paving-off control. Warm light now covers the border near the
walls instead of mainly appearing beyond it. Dark asphalt still reflects less
light than grass; no paving-only gain or artificial emission was added. Daylight
is pixel-identical on all three scenes.

[Paving-only control and corrected positions](out/city-palace-facade-alignment-r1/paving-light-diagnostic.png)
is a three-times diagnostic; use the full 1360x800 day/night outputs for visual
acceptance. Eight Windows comparisons and eighteen independent frame/ground/
material packet checks pass. [Evidence](CITY_PALACE_FACADE_ALIGNMENT_EVIDENCE.json)
also verifies unchanged light parameters and identical paving geometry. The
`city_paving_light_control.py` tool reproduces the receiver isolation.

The provisional same-layout results are the three `*-environment/render`
directories under `out/city-palace-facade-alignment-r1`. The user's subsequent
central, orthogonal city-composition preference supersedes these placements;
carry forward the facade-light and paving fixes. Source geometry occlusion,
broader culture coverage and all native/manual/milestone gates remain open.
[Cleanup](CITY_PALACE_FACADE_ALIGNMENT_CLEANUP.json) removes 249.0 MiB of completed
linear readbacks while retaining images and replay resources.
