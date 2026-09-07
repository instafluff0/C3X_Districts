# City environment material trial

Modern facades now have a bounded sky/ground reflection contribution alongside
their existing metalness maps. The local inland-large and wilderness-medium
candidates show a modest improvement in cool facade separation at gameplay
size. They retain their exact source buildings, placement, paving, window maps,
local light recipe, terrain, camera and shared shadow resources. This adds a
missing material response; it does not add geometry or texture detail.

[Modern gameplay comparison](out/city-environment-r2/selected-modern-native.png)
shows the prior material at left and the provisional candidate at right, day
above night. There are 7,398 daytime and 4,730 nighttime pixels above 2/255 of
change, within `[741,354,827,490]` and `[739,347,827,489]` respectively. Outside
the city region pixels are exact. The cooler reflective facades move toward the
material separation seen in `Renderer/canonical/nightlights.jpg`, while the
reference's finer facade detail, varied architecture and open ground remain
clearly beyond this result. This is not Civ VI-level quality acceptance.

[Wilderness comparison](out/city-environment-r2/wilderness-medium-native.png)
uses the same recipe without region-specific changes. Its 3,170 daytime and
2,079 nighttime changed pixels remain within the building silhouette; outside
the recorded city region pixels are exact. The wilderness was previously a
city-placement benchmark but was not used to choose this environment recipe.
These are unchanged 100-tile `test.biq` scenes with 1360x800 gameplay outputs.

## Why the first trial was rejected

The earlier direct-metalness trial removed diffuse energy without providing
environment reflection. New r1 restores an analytic reflected hemisphere and
recovers some modern facade light, but washes out the Asian palace roofs.
Simply changing its brightness would hide a missing material term.

Inspection of the preserved installed rigid-model shader found a directional
environment attenuation controlled by the blue cooked-roughness channel and
view angle. r2 adds that attenuation and the separate broad reflectance term.
It reduces the washout, but the [Asian roof comparison](out/city-environment-r2/asian-medium-native.png)
still shifts too far toward gray. That result remains **unselected**. Retain
the prior `city-palace-composition-r2/asian-medium/render` appearance. Do not
apply the fallback globally or call dielectric material coverage complete.

`CITY_ENVIRONMENT_SOURCE.json` records the source-family evidence and preserved
disassembly hash. The inspected source uses two filtered cubearray lobes,
weighted one third and two thirds, plus SH irradiance. The directional lobe
attenuation is approximately:

```text
saturate((.315 - roughness_blue) / .315)
/ (1 + 10 * sqrt(pi * roughness_blue) * (1 - sqrt(saturate(N dot V))))
```

Exact active city permutations, source environment radiance/normalization,
cubearray and SH payload bindings, and LEAN1 variance scaling remain unresolved.
The source-family attenuation does not prove those engine bindings.

## Generic Lab implementation and composition

`shaders/lighting/city_environment.hlsl` is an explicit authored fallback. It
derives analytic sky/ground radiance from the existing shared ambient, sun and
moon state. Two cooked lobe variances control analytic broadening; this is not
source cubearray filtering. Its ground and sky colors are authored values,
and the broad irradiance term approximates the missing source SH evaluation.
AO also bounds indirect reflection. No new time policy, texture pack, native
backend code or source-specific runtime branch is introduced.

`qa/city_environment_probe.py` composes the helper into both the body shader
and the existing water reflection prepass. It preserves the complete
postprocess and source replay inputs. `--enable-bound-metalness` activates only
nonempty generic metalness textures already present in the preserved packet.
This opt-in probe is not a global city-material default or a native promotion.

The packet adapter originally assumed a uniform material value per draw. The
current producer batches by texture tuple, so one draw can contain different
per-triangle addressing flags. That initial preparation was rejected before
producing a packet. The corrected adapter preserves every vertex's original
flags and adds only the metalness bit. Independent checks verify that all
positions, normals, UVs, other material bits, texture bytes, draw state,
lighting constants and shadow-frame bytes remain exact. No original packet is
rewritten. The modern large and wilderness cases change 27,954 and 14,178
material vertices per frame, respectively; those are
transport counts, not visible acceptance.

The old American capital/lake fixture uses an earlier city material layout
without bound metalness. Its preparation failed explicitly; no missing channel
was fabricated and no replacement capital render was produced. Complete
palace/house material intake for that fixture is the next concrete composition
gap. Its existing night-water reflection evidence remains preserved. This pass
makes no new strong water-reflection quality claim.

## Verification and continuation

The r2 disabled helper control is pixel-exact to the prior inland city in both
daylight and midnight. A separate direct-only control shares the candidate's
exact metalness-enabled packets and isolates the new environment contribution.
Four independent packet inspections check the two modern cases, and the Asian
trial reuses its original packets without any data edit. Eight corrected
Windows/Metal comparisons pass, including the unselected Asian result and the
disabled control; two earlier r1 inland comparisons also passed. One interrupted
VM dispatch was resumed only after confirming no live replay and no night output.
Passing parity did not select the Asian appearance.

Recheck with Python providing NumPy/Pillow and clang++ available:

```sh
python3 Renderer/terrain_lab/v2/qa/city_environment_evidence.py
python3 Renderer/tools/lab_v2.py validate
```

`CITY_ENVIRONMENT_EVIDENCE.json` freezes pixel bounds, source/packet/shader
hashes, disabled and direct-only controls, and backend comparisons. Replays use
the saved per-case reports, shader closures and shared packets. Keep the r1
diagnostics and all earlier bests. `CITY_ENVIRONMENT_CLEANUP.json` records
removal of completed new linear readbacks only.

The three largest remaining visible gaps are facade/roof material fidelity,
more varied architectural and open-ground composition, and complete coherent
growth across the culture/era/site matrix. Next restore the older American
capital's complete material inputs together, then evaluate them with its
preserved lake reflection and local lights. Broaden single-era culture coverage
and investigate actual environment probe data rather than repeatedly tuning
the analytic hemisphere. The independent coastal capital fit remains unresolved.
Connecting roads, human visual approval and all native/milestone gates remain
open; the full city-quality goal is still active.
