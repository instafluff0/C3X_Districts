# Upstream Asset Family Intake Status

This is the resume point for local licensed-source art that has been discovered
but is not yet runtime-owned by C3X. Conversion here prepares generic packs for
the owning Renderer Lab gate; it does not advance a gate or authorize native
suppression.

## Prepared in the current intake pass

| Family | Offline result | Lab ownership still required |
| --- | --- | --- |
| City walls | 19 pieces across complete ancient, medieval, and industrial role kits | L17 perimeter topology, scale, grounding, and both-zoom approval |
| Capital accent | The generic `BUILDING_PALACE` marker remains composition-only, but two culture-specific installed palace compounds now normalize with emissives and exact fire/smoke sockets | L17 compares the optional candidates at both zooms and uses them only through explicit city-style mapping |
| Naval unit | Galley body normalized from three skinned meshes/two materials; eight basic actions converted, validated, and included in the family model-aware pose-cache bake | L20 multi-part rendering, scale/facing, formation, and visual approval |
| Army commanders | Dedicated Classical and Modern Great General ArtDefs resolved; Modern foot officer is direct, while Classical horse+rider now passes the generic socket/paired-clip compiler; the two-child Army contract preserves Civ III's exact displayed member | L20 calibrates both commander profiles and the full Army matrix; I20 captures both native bodies atomically |
| Compound units | Horseman, Classical Great General, Catapult, and Tank compile as eight independent animated nodes, four resolved joints, 30 components, 50 textures, 52 model-aware pose caches, and 62 logical node/action bindings across 31 actions through one arbitrary-tree schema | L20 renders/measures the eight-facing/two-zoom matrix, resolves Catapult death if truthful compatible art is found, and runs the promotion render |
| Mines/farms | Full discovered library accepts 18 mine and 204 farm roots with zero top-level conversion rejects | L18/L19 selection, adjacency, terrain conformance, readability, and promotion renders |
| Rice terrain edits | 18 typed records retained as hashed source evidence while visual art converts | Semantic decoding or authored terrain behavior; application remains disabled |
| Analytic lights | 12 production-like Base lights normalized with six typed parameters each; four test/negative fixtures excluded | Owning Lab category must bind sockets and calibrate values |
| Ambient effect textures | Eight fire/glow/smoke/steam textures normalized | M7.5 sprite layout, blend/timing, particle behavior, and attachment ownership |
| Combat effect textures | 22 muzzle/projectile/explosion/smoke/debris/water/nuclear textures normalized; audio-preserving pixel suppression and nuclear outcome boundaries audited | M7.5 sprite layout, particle behavior, runtime event binding, and visual calibration |
| Goody huts | Exact `IMPROVEMENT_GOODY_HUT -> LM_GOODY_HUT` chain; three roots plus recursive bodies normalized, with eight deterministic Civ III buckets and day/night attachment policy | L19A grounding, facing, visibility/fog, night-light, both-zoom, and 192-tile approval |
| Civ III colonies | Three preindustrial and three industrial resource-camp roots plus recursive logistics props normalize strictly as a reduced owned outpost; resource coexistence, extraterritorial owner color, four-era selection, and night cues are frozen | L19A scale, density, owner-color, resource-coexistence, day/night, both-zoom, and 192-tile visual approval |
| Forts, barricades, airfields, outposts | Five roots recursively normalize with road-connected fort attachments, fort walls/cannon/flags, complete airstrip props and runway lights, and two standalone watchtower candidates | L19B visual calibration; barricades reuse denser fort perimeter pieces rather than unrelated art |
| Pollution and craters | Four exact crater decals normalize; `NUCLEAR_FALLOUT -> FX_Radiation` is the preferred pollution family and five radiation textures now normalize to generic IDs | L19B binds both lifetimes to Civ III state and calibrates a restrained tile-local radiation layout |
| Future-gate candidates | Seven source-independent compounds: two palaces, one observatory/radar candidate, and four crater decals; 39 textures, five decals, three geometry roots | L17/L19B visual-semantic comparison and promotion only |

## Remaining source-decoding backlog

- Exact `VFX_FireFX.blp` script/resource dependencies, sprite layout, blend
  constants, and timing. The Base package has 64 big-data entries and its
  qualified resource strings live there, so source-equivalent behavior requires
  a separate chunk profile. Bounded generic fire/smoke/steam behavior now exists
  independently and makes no source-equivalence claim.
- `VFX.blp` and `VFX_A/B/C.blp` combat particle scripts, emitter graphs,
  sprite layouts, blend constants, and timing. Their named families and 22
  conservative standalone texture payloads are proven, while the new generic
  land/water impact graphs remain authored behavior rather than a decoded source
  effect system.
- Exact Light/VFX resource scripts and activation/state rules. The consolidated
  attachment catalog now proves 2,620 exact name/socket/transform joins (88 VFX
  candidates and six analytic-light candidates), but identity alone does not
  decode the resource graph.
- A culture-neutral capital centerpiece. Two culture-specific palace compounds
  are now normalized and may be selected only by explicit style mapping.
- A proven standalone Radar Tower body and Victory Location marker. An
  observatory body is structurally normalized, but its isolated calibrated
  sheet reads as a flat ornamental plaza rather than a tower or antenna. The
  missile silo remains rejected. Victory Location is explicitly set aside.

Road/rail, static resources, resource animation clips, city component pools,
city emissives, unit owner color, and the basic unit action contract are already
covered by their dedicated intake documents. Natural wonders, constructed
wonders, and Districts retain their explicit M9/M10/M11 ownership and are not
pulled forward by this backlog.
