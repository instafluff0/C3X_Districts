# Lab v2 real-map validation

Every visual track must pass relevant regions of the user's actual `test.biq`
before acceptance. Small constructed fixtures remain useful for diagnosis and
exhaustive cases. They supplement real-map evidence rather than replacing it.
This is a Lab evidence requirement; dependency order and Integration gates are
unchanged. Ordinary replay runs on the Mac.

## Source identity and acquisition

The current historical `PREPARE_L13_BIQ_VIEWPORT.bat` reads
`Intro1 Ancient Treasures.biq` while naming its output
`test_biq_l13_rivers_192.csv`. That output is not proof of `test.biq` provenance.
Keep the historical runner, files, and frozen handoffs intact.

Q0 must resolve the intended `test.biq` from the configured Windows installation
or a verified shared copy. Record the original file's SHA-256, byte length,
parsed map dimensions, wrap flags, parser/exporter version, and import settings.
Use a configurable local source path (for example `C3X_LAB_TEST_BIQ`), with
machine-specific paths kept out of tracked records. If multiple distinct
`test.biq` candidates exist, establish identity from available project evidence
and request clarification only if that evidence cannot resolve them. Never
silently substitute another BIQ based on output naming or convenient content.

Copy/read the source once and cache its export by content hash on the Mac.
Recheck identity when deliberately refreshing the source, not through a VM
round trip on every render. Keep the source and cached payloads local/ignored;
track portable metadata and recipes. A source refresh creates a new dataset
revision and invalidates dependent fixture caches and acceptance evidence.
Do not modify the original BIQ or populate it in the game for Lab testing.

## Three fixture classes

1. **Real terrain:** exact source tile coordinates, base/real terrain, rivers,
   and any existing map features. Preserve neighboring source tiles outside
   each viewport to the extent required by smoothing, shore, shadow, and other
   dependency footprints. A crop boundary is not a shoreline or map boundary.
2. **Augmented real terrain:** the same immutable terrain with separate,
   deterministic Lab layers for missing cities, roads/rails, resources,
   improvements, and units. Each layer records its own hash, generator/profile,
   seed, placements, and the source dataset/region it consumes. Respect terrain
   and domain constraints and distinguish additions from existing BIQ features.
   Label these as Lab augmentations, not captured Civ III runtime state. Include
   an augmentation-off control proving the underlying terrain is unchanged.
3. **Constructed stress cases:** small explicitly synthetic fixtures for rare
   pairs, junctions, orientations, negative controls, and absent features.
   Document which coverage gap each addresses. They never acquire real-map
   provenance merely by using the same material pack.

## Named regions and coverage

Q8 must also deliver a developed-gameplay composition over verified real terrain,
guided by canonical `civ3_real_example.jpg` (`civ3.real_gameplay_layout`). Use
observed city spacing, route/improvement density, unit presence, open space, and
label/HUD crowding as layout references. Keep exact placements in deterministic
Lab layers governed by source terrain and declared scenario constraints; the
screenshot is not an authoritative export or proof of hidden state. Include a
sparser gameplay variant and a separately labeled crowded stress case. Final
acceptance requires plausible play layouts as well as diagnostic asset galleries.
Q8 owns these recipes and Q5/Q7 consume the reference early for their own tests;
there is no new dependency on Q8 before those tracks can proceed.

Q0 publishes an initial versioned region registry and replay adapter inside
its owned `shared/real_map/` and `app/` paths before visual tracks depend on it.
All tracks consume that registry read-only and keep their fixtures, overlays,
reports, and output names in their own directories. Q8 later audits campaign
coverage; it is not a prerequisite for Q1-Q7 to begin real-map validation.
Region additions go through the coordinator/Q0 instead of parallel edits to a
shared registry. Reuse the existing BIQ parser/exporter's explicit origins and
neighbor export where suitable; account for each consumer's required halo.

Select named regions from an actual map inventory, not assumed coordinates:

| Track | Relevant real-map coverage |
| --- | --- |
| Q1 sampling | Mixed ground/foliage/water with separate city, unit, and route layers |
| Q2 terrain | Terrain-family boundaries and multiway junctions |
| Q3 hydrology | Sandy shores, rocky hill coasts, coves, islands, channels, river mouths |
| Q4 relief | Hills, mountain chains, hill-to-water edges, dunes, biome shoulders |
| Q5 networks | Route layers over flat land, relief, water crossings, and wrap where present |
| Q6 lighting | Representative relief and dense object/city layers, all four day phases |
| Q7 presentation | City/object layers on representative inland/coastal settings with neighbor clearance |
| Q8 beauty | Campaign-wide combined views and the coverage gaps reported by all tracks |

Each region records a stable ID, raw BIQ origin, tile extent, coordinate basis,
wrap behavior, neighbor extent, camera/zoom recipes, feature tags, and dataset
hash. Select a compact shared set that covers observed features. Maintain an
explicit present/absent/not-yet-checked ledger: e.g. if `test.biq` has no volcano,
record that absence and cover volcanoes synthetically. Do not edit real terrain
to manufacture a missing feature. Every track must still exercise the real
regions applicable to its system.

Reserve an additional neighboring or held-out region for each applicable
track's candidate check. Select it before tuning, and do not change it simply
because it reveals a defect. If a defect is found, fix it and retain that case
as a regression witness; record that it is no longer an untouched holdout.

## Cadence and acceptance

During edits, use one cached small real region or a diagnostic microfixture,
one phase/zoom, and only necessary layers. At candidate checkpoints render the
track's relevant named regions plus its neighboring/held-out witness at both
zooms and relevant lighting phases. Inspect actual-size images, isolation
controls, and camera scrolling. Hydrology stays static; scrolling is not an
animation requirement. Reserve broad contact sheets, the combined 192-tile
matrix, and Windows parity for acceptance/promotion.

The acceptance report must identify dataset hash, region IDs/coordinates,
neighbor coverage, fixture class, augmentation hashes, effective render
settings, consumed contracts, image paths, deterministic checks, direct visual
observations, and outstanding coverage gaps. Screenshots supplement the BIQ
data; they do not replace structured terrain or prove live-game object state.
Later Integration verifies runtime capture, anchors, clipping, and compositing.

If source acquisition or the replay adapter is unavailable, record real-map
validation as pending and continue independent synthetic/asset work. Do not
mark a track accepted on substitutes alone. This policy is an instruction for
the implementation and evidence gates, not a claim that the source has already
been acquired or any region has passed.
