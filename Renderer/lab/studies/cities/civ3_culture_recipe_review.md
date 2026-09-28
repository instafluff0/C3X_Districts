# Five-culture city recipe audition

`civ3_culture_recipe_candidates.json` proposes one offline art profile for each
Civ III culture group and era. Each era names a base source-art pool and optional
historical layers. Its Civ VI source selectors are preparation inputs only; the
result is a generic list of normalized city instances with a stable profile ID.
The candidate gallery is separate from `city_render_strategy.json` and is not
promoted to the game pack.

Run from the project root with the local source-family study packs prepared:

```sh
python3 -m Renderer.lab.studies.cities.civ3_culture_recipe_review
python3 -m Renderer.lab.studies.cities.civ3_culture_recipe_review --seed 1 --overview-only
```

The ignored `Renderer/lab/out/cities/all-era-source-auditions/review/civ3-mixed-culture-candidates/seed-0/`
directory holds an overview, a larger Industrial/Modern comparison, a manifest,
and a full four-era variant sheet for each culture. Town walls use the previously
selected era kit; City and Metropolis each show only Base and Capital, with no
empty wall columns. There are no
ground decals or artificial building foundations in these recipes. The sheets
are software material previews; map-context rendering still needs review.

The Middle Ages examples use denser civic mixing for the three cultures whose
Ancient and Middle Ages base pools are the same: American Cree/Maya, Roman
Mediterranean/Portugal, and Asian EastAsian/Korea. Industrial and Modern City
recipes retain four older buildings near the inhabited edge; Metropolises retain
six. The taller fitting source pieces are preferred so their roofs stay legible
next to later-era buildings. Towns have none of these additions. Positions are
selected from existing plots and remain stable for a given seed. The glass and
ordinary modern accents remain clustered downtown.

For a future user-facing config, resolve a **generic compiled art-profile ID**
from the city's culture group, then let an explicit civilization override select
a different profile ID. An optional profile definition can inherit a culture
default and override individual era recipes. That selection should happen
before constructing the visible scene; rendering consumes only the selected
normalized pack metadata and the city's stable variation seed. Scenario civ IDs
must be data keys, not renderer branches. No runtime config syntax is fixed by
this Lab audition.
