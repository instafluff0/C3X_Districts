# Hills beside rivers

## Report (2026-10-07)

In game a hill on the far bank seemed to cover part of the river and to shade
the water. Civ III rivers run on tile edges, but the authored hill bodies
(`composed_hill`, radius 0.62–0.86 tiles) reach across those edges.
`SurfaceQueries::terrain_height` flattened hill relief only at coasts, while
low relief (`low_height`) and mountains (`mountain_river_scale`) already step
down to the continuous river corridor. The river layer draws at the flat
datum with a 1.1-unit channel cut, so the raised hill ground stood over it,
hid it under the depth test and cast its shadow on the water.

Measured with the real hill field (river surface = within 7.4 source pixels
of the centreline): a hill stood up to 40 units over the water; between two
facing hills 57% of the river surface lay more than the channel cut below
the ground. The river spline already bends away from a lone hill, which is
why a single far-bank hill only pinches the water.

## Candidate (accepted 2026-10-07)

The user accepted slope14 with the river distance fix. Production code has no
switch: `queries.h` applies the 1.4 units/px bank and `river_corridor.h`
the .95-tile buckets; `Renderer/native/test_hill_river_banks.py` covers both.

Hill relief above the datum is lowered toward the river corridor in
`SurfaceQueries::terrain_height` (`lab/shared/natural/queries.h`). The
corridor is the one low relief uses; the river's route still reads the raw
`NaturalData::height`, so flattening cannot move the river. Every consumer of
the natural height (ground mesh, hill decals, forests on hills, routes,
mines, resources, cities) follows automatically. Hill material needs rise, so
the flattened banks lose the rock band on their own.

The study compared four profiles through a private tree (`private_tree.py`,
deleted after promotion) with a `C3X_LAB_HILL_BANKS` switch:

- `0`: checkout behavior;
- `1`: rise × smoothstep((d − 7.5) / 24);
- `2`: rise limited to a 1.4 units/px bank, smooth-min joined (k = 6);
- `3`: the same with a 0.9 units/px bank.

`2` (slope14) was the clear pick: `1` cut steep shaded faces into hill flanks
and `3` flattened hills into mounds. The kept renders are `before` (checkout
before both fixes) and `candidate` (the promoted code) with the review sheets.
To re-render the current code, pass `--tree` a fresh tree built from the
checkout:

```sh
python3 Renderer/lab/studies/mountains/private_tree.py sync current
python3 Renderer/lab/studies/mountains/private_tree.py build current
python3 Renderer/lab/studies/hills/river_banks.py render after --tree current
$C3X_RENDERER_PYTHON Renderer/lab/studies/hills/river_banks.py sheet before after
```

## Seam: river distance beyond the bucket reach

A thin dark line across flat grass near the far-bank hill was a 9-unit cliff
in low relief on the tile line wx = -3. `river::Corridor` filed each segment
only into cells within .65 tiles of it and a sample reads only its own cell,
so distances were exact to 37 screen pixels; beyond, a neighboring cell could
miss the segment and read 1000. Low relief ramps to 52 pixels, so it jumped
(16.5 to 25.4 units there). On test.biq 485 cell boundaries jumped by up to
13.8 pixels. The same gap would cut the hill banks, which reach 53 pixels.

The candidate files segments within .95 tiles (exact to 54 pixels; +54% bucket
entries, about +40% per sample on test.biq) and keeps `affects()` on the
original .65 cells, so river terrain detail selection is unchanged (522 of
5000 test.biq tiles either way). Distances below 37 pixels are unchanged.

The case (`river_banks.py`) runs a river left to right into a coast past a
far-bank hill the river wraps, a near-bank hill, a four-hill valley crossed
by a bridged road, a mined far-bank chain and a forested hill facing it.
Outputs are disposable, under `Renderer/lab/out/hills/river-banks/`.
