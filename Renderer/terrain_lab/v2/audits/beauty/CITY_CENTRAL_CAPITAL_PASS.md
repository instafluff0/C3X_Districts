# Central palace and orthogonal city grid

The user now requires the palace at the city core, surrounded by ordinary
buildings, with all buildings aligned to right-angle city axes. This supersedes
the earlier preference that put the palace toward the visible foreground.
Keep single-era neighborhoods, uniform source scale, illuminated paving and
source-informed facade lights.

Selected r111 inland and r112 freshcanopy use `--central-capital` and
`--orthogonal-buildings`. The palace stays at the shared city core. The first
four houses occupy four sides with courtyard frontage; later houses extend the
surrounding neighborhood. Four- and seven-house prefixes contain the palace
inside their convex hull and connect through its courtyard. This deliberately
replaces the earlier requirement for a house-only connected cluster: a central
courtyard is now part of the neighborhood's connection.

The normalized American palace has a baked approximately -30-degree footprint.
The offline minimum-area footprint rectangle identifies a +30-degree correction;
the ordinary American buildings already have zero correction. Constrained layout
then adds only quarter turns. Source bodies, UVs, tangents and proportions remain
intact; the same transform reaches ground, light derivation and shadow geometry.
Independent transformed-footprint checks put all eight buildings within 0.02
degrees of the shared grid. This is geometric inference from the normalized
pack, not a claim of recovered source placement metadata.

The combined scene now reads as a palace between surrounding blocks, and its
foundation lines agree with the neighboring buildings. The palace remains visible
between the lower foreground buildings. Warm windows, the corrected paving border
and facade spill survive the composition. See the matched gameplay-size
[inland comparison](out/city-central-capital-r2/inland-native.png) and
[previously city-untuned freshcanopy comparison](out/city-central-capital-r2/holdout-native.png).
These retain the same real 100-tile terrain, cameras and 1360x800 outputs.

Four Windows comparisons, twenty independent terrain/frame/light/ground/material
checks, axis/clearance checks and 33 focused tests pass. Evidence and images can
be regenerated with `city_central_capital_evidence.py`. `city_central_capital_probe.py`
composes a prepared central fixture with the corrected source-hull paving,
facade-plane lights and provisional modern environment material.

The coastal site remains unresolved for this exact seven-house surrounded recipe.
After 25 legal-core attempts, the required third source body still has no legal
placement; an alternate side-assignment diagnostic also fails. r106-r110 and
r113-r115 preserve failed placement evidence. r108/r109 initially wasted search
budget on disconnected courtyard combinations; pruning impossible first-stage
frontages fixes the inland/holdout search in seven nodes with zero backtracks.
The alternate r116 inland arrangement is unselected. A storage guard stopped
r117 before capture; completed readbacks were cleaned and no threshold was lowered.
Preserve the preceding coastal material/light scene rather than erase terrain,
move vegetation or claim a successful central arrangement. The next coastal fix
needs suitable footprint/foundation composition, not lighting or texture tuning.

The selected render directories are `out/city-central-capital-r2/inland/environment/render`
and `holdout/environment/render`. The coastal fallback remains
`out/city-palace-facade-alignment-r1/environment/render`. This validates one style
at two sites, not every palace/culture/era/size. General material richness,
source environment/LEAN1 reconstruction, city coverage and all native/manual/
milestone gates remain open. [Cleanup](CITY_CENTRAL_CAPITAL_CLEANUP.json) retains
all images, packets, shared resources and failed-layout evidence.
