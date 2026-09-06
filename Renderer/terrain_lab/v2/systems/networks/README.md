# Q5 networks candidate

`network.py` owns graph validation, exact nodes, smooth degree-two tangents,
explicit wrap translations, curve tessellation and bridge grade fitting.
`source_routes.py` adapts normalized authored route atlases to those ribbons;
`bridges.py` uses rigid normalized source bodies with unchanged UVs and uniform
scale. No game state, river topology or ground geometry is modified.

The chosen local geometry is `fixtures/networks/source-10/scene.bin` (relative
to Lab v2). Rebuild it from the repository root:

```sh
python3 Renderer/terrain_lab/v2/systems/networks/fixtures.py --name source-10 --context --source --rail-width 12
python3 -B -m unittest discover -s Renderer/terrain_lab/v2/tests/networks -v
python3 Renderer/terrain_lab/v2/app/runner.py check --fixture Renderer/terrain_lab/v2/fixtures/networks/source-10/fixture.json --settings Renderer/terrain_lab/v2/fixtures/networks/scroll.settings.json --candidate final-10
```

Use `linear.module.json` for new composition. `module.json` preserves the
explicit diagnostic display branch for comparisons. The current
`fixtures/networks/source-linear/fixture.json` selects the linear module and
pins the corridor sidecar; its `linear-14` checkpoint renders four phases/two
zooms with repeats. The renderer owns GPU work and final transfer; Q5 emits
linear premultiplied color and validity, and never clears shared category depth.

```sh
python3 Renderer/terrain_lab/v2/systems/networks/exchange.py Renderer/terrain_lab/v2/fixtures/networks/source-linear/fixture.json
python3 Renderer/terrain_lab/v2/app/runner.py check --fixture Renderer/terrain_lab/v2/fixtures/networks/source-linear/fixture.json --candidate linear-14
```

`clearance.py` publishes capsule chains and rigid bridge polygons from the same
rendered curves/source bounds. `exchange.py` converts them to Q0's shared
polygon schema. Full transformed footprints/crowns are tested; the declared
4-pixel additional margin does not clear whole tiles. Capsule polygonization
is conservative by at most 0.212 pixels at the 6-pixel occupied radius. Consumers
must supply the same raw-coordinate origin and canonical wrap period, preserve
halo geometry, and union road, rail, junction, bridge and Q3 river/bank envelopes.
Do not use the synthetic witness as a real-map clearance result.

Real-map preparation uses Q0's cached, verified `test.biq` registry. Exported
terrain is immutable. `hydrology_adapter.cpp` consumes Q3 read-only and converts
its water tangent/width into route-aligned crossing spans. The last argument
explicitly supplies horizontal wrap (1 for the current dataset); never read
wrap from the CSV's halo-count field. It exports ordered data with null deck
heights until the final terrain sampler is supplied.

```sh
python3 Renderer/terrain_lab/v2/app/real_map.py export mixed-holdout --owner Q5-networks --output Renderer/terrain_lab/v2/fixtures/networks/real-holdout --augment
clang++ -std=c++17 -O2 Renderer/terrain_lab/v2/systems/networks/hydrology_adapter.cpp -o /tmp/q5-hydrology-adapter
/tmp/q5-hydrology-adapter Renderer/terrain_lab/v2/fixtures/networks/real-holdout/terrain.csv Renderer/terrain_lab/v2/fixtures/networks/real-holdout/q5-crossings.json 1
```

Multiple non-overlapping crossing anchors are tessellated as ordered bridge
spans while keeping one stable gameplay edge. Overlapping spans fail explicitly
until hydrology supplies a single merged crossing envelope. Source road height BC5 remains unbound until its channel meaning
is established; existing geometric normals follow the supplied terrain.
Background grass/relief, worked plots, and city markers are diagnostic proxies,
not approved original-art fallbacks. The final surface/scene composition and
placement-clearance gates remain open. See the owned audit and handoff proposal.
