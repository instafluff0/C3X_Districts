"""Recommendation text for the district survey page (page.py). HTML snippets."""

FINDINGS = [
    ("Civ VI already works the way C3X does",
     "Each Civ VI district is one <b>base variant</b>, chosen by the set of buildings present and the art era, plus a "
     "separate <b>hero building</b> model for each building, modelled in place in the hex. When a building completes, "
     "Civ VI swaps the base and adds that building's model. That is what C3X does with sprite columns: a C3X column is a "
     "Civ VI building set, and a C3X row is a Civ VI art era."),
    ("Key the 3D kit on which buildings exist",
     "C3X's default <code>by-count</code> picks a column from how many dependent building types any city in range has, so "
     "a city with a SAM battery and no Barracks shows the Barracks picture. Civ VI keys its art on the actual set. The 3D "
     "kit should follow the set, which C3X already computes for <code>by-building</code>; with Civ III's building "
     "prerequisites the two agree almost everywhere."),
    ("Drafted onto the Civ III tile, densely",
     "Civ VI lays each district out on a hex: a ring of buildings at 60° angles that spills past a Civ III tile. Every "
     "land district here is re-drafted onto the square tile as one dense, irregular cluster grown from its own seed: "
     "hero buildings toward the back, a statue or fountain near the middle, then every kit building, then copies of "
     "the kit's own buildings until the tile is about half built over, a few props, and Civ VI's trees scattered "
     "into the gaps (never in rings). Real wall kits become a closed square ring along the tile edges with towers at "
     "the corners. Every piece is square to the tile, its walls aligned from their own normals, with its front toward "
     "south-west or south-east at random. Civ VI's largest heroes (airport, factory) are shrunk to share the tile. "
     "Drafted districts are drawn at 2.0× Civ VI's authored size; coastal kits keep Civ VI's layout."),
    ("Civ VI art where its district kits are thin",
     "Civ VI's own Commercial Hub is market stalls in every era; its skyline comes from the city generator. Industrial "
     "Commercial Hubs here gain Civ VI's industrial row houses and Modern ones its glass towers (the same models "
     "Renderer64's cities use), more as the hub grows. The Colosseum is Civ VI's own Colosseum wonder at district size, "
     "taken from its wonder-movie model (Civ VI has no Colosseum district building), and the Stadium in the Modern era. "
     "The Data Center adds a glass tower to the Spaceport pad and Research Lab."),
    ("Night lights come from Civ VI's own emissive maps",
     "Districts share the city's sun, shadows and night darkening. Civ VI ships an emissive map for about half of the "
     "district materials (lit windows, lamps, signs, stadium floodlights). The Lab bakes them into emissive atlases "
     "laid out like the colour atlases, and a Lab-only Renderer64 build lights them through the same night response "
     "the cities use, so after dark districts glow at the city's scale and by day nothing changes. One difference "
     "remains: Civ VI's pale stone is brighter at noon (mean luminance about 125 against about 94 for city houses). "
     "Production kits should use the city material path, with these emissive maps and the city readability tone."),
    ("City ground cover, cut close",
     "Civ VI paints the paved district hex with its terrain system; the landmark itself holds only buildings, props and "
     "trees. Each drafted district here uses the city pack's era paving, but only just round each building's own "
     "standing footprint (its placed outline above the ground, so the pad turns with the building), with short "
     "paths between near neighbours; open ground between buildings stays grass. The Lab composition path "
     "alpha-tests, so the edge is crisp; production should draw it with the city plate's material. Context scenes "
     "include a close-up without it for comparison."),
    ("Culture: Civ VI barely varies districts",
     "Outside City Centers, Civ VI districts have no regional styles. C3X varies the Holy Site, Entertainment Complex, "
     "Commercial Hub, Central Rail Hub and Neighborhood by culture. The plan uses Civ VI's civilization-unique districts "
     "and buildings as regional swaps (Mosque, Pagoda, Prasat, Meeting House, Cothon, Royal Navy Dockyard, Tlachtli, "
     "Grand Bazaar, Suguba) and the production city pack for Neighborhoods."),
    ("Coverage",
     "Of the 21 default districts, 14 have a direct Civ VI match (the Great Wall from Civ VI's wall kit on C3X's tile "
     "topology) and 3 a close one: Neighborhood from the city pack's houses, Energy Grid from Industrial Zone pieces and "
     "Distribution Hub from Civ VI's warehouses. Data Center and Bridge use stand-ins; Central Rail Hub needs a mod or "
     "authored art; the Wonder "
     "District belongs to M10. The SAM Missile Battery has no "
     "Civ VI building. Gathering Storm's Power Plant is a nuclear plant, so the Nuclear Plant is covered; the seed "
     "ledger still lists it as authored."),
    ("Import limits found",
     "Poland's Sukiennice package fails a size check in this install. The Great Wall pieces import only through the "
     "city-wall path, not as a landmark. Lab-only importer tolerances (empty base models, extra attachment policies, "
     "skin weights) were needed and are not in the shared importer. Constructed wonders exist only in "
     "<code>WonderMovie.blp</code>; with two more Lab tolerances (its entries read as landmarks, its animated pieces "
     "have no clean bind pose) the Colosseum imports in about 15 minutes, which also opens M10's wonder art."),
]

DISTRICTS = [
    {"id": "encampment", "name": "Encampment", "status": "exact", "sprites": ["Encampment"],
     "source": "Civ VI Encampment base by era", "culture": "Era: 4 Civ VI bases. Culture: none needed (C3X does not vary it); "
     "civ-unique Ordu, Ikanda, Basilikoi Paides and Thanh are optional extras.",
     "buildings": "Barracks → Civ VI Barracks (classical model, industrial model from Modern). SAM Missile Battery → no "
     "Civ VI building; stand-in in the Armory slot.",
     "text": "<p>Columns: no buildings → Civ VI <code>EMPTY</code>; Barracks → <code>BARRACKS</code>; Barracks + SAM "
             "(Modern only, since SAM needs Rocketry) → the <code>BARRACKS, ARMORY</code> base with a SAM stand-in "
             "seated in the Armory's slot. The scaled Missile Silo shown is a placeholder; a static Civ VI Mobile SAM or "
             "Anti-Air Gun unit model, converted through the existing unit pipeline, would read as an air-defence "
             "battery. No mod provides one.</p><p>Ancient and Medieval Encampments are walled with Civ VI's city walls "
             "of the era (the walls Renderer64's cities use): a closed square ring with corner towers and the gate on "
             "the side facing the viewer, in place of the kit's stake palisade, whose loose clumps of stakes read "
             "poorly on the tile.</p>",
     "galleries": [{"catalog": "encampment", "title": "Encampment kits",
                    "note": "Column 3 shows the Civ VI Fort improvement as an alternative base, and Barracks + Armory "
                            "for reference."}]},
    {"id": "holy-site", "name": "Holy Site", "status": "exact",
     "sprites": ["HolySite_AMER", "HolySite_EURO", "HolySite_ROMAN", "HolySite_MIDEAST", "HolySite_ASIAN"],
     "source": "Civ VI Holy Site base by era", "culture": "Era: add it (C3X's 2D art does not vary by era; Civ VI has 4 "
     "bases). Culture: worship building per culture, see the culture plan.",
     "buildings": "Temple → Civ VI Temple (the marble tholos for Roman, a Japanese temple compound for Asian). "
     "Cathedral → the culture's worship building: Meeting House, Cathedral, domed basilica, Mosque or Pagoda.",
     "text": "<p>Two buildings, as in C3X. No buildings → a pond in the middle ringed by a loose circle of trees; "
             "Temple → the culture's temple; Temple + Cathedral → temple and worship building. Civ VI's marble "
             "pavilion (its Shrine) is now the Roman temple only. The pond stays in every column: with temples "
             "built it moves toward the front of the tile so the temple keeps the back. Each column keeps one layout through the eras: the pond, trees and buildings stay where they are, and only the buildings' era models change. Civ VI's worship buildings are the culture switch: "
             "Meeting House (American), Gothic Cathedral (European), the Synagogue model's domed basilica, which reads "
             "as Byzantine (Roman), Mosque (Middle Eastern) and Pagoda (Asian). The Asian temple is a Japanese temple "
             "compound from Civ VI's East Asian city buildings, among Japanese-style halls.</p><p>The temple grounds "
             "reuse a few pieces of Civ VI's City Park, scaled to the district: hedge beds and a tree planter, "
             "braziers (lamps from the Industrial era) and, for the classical cultures, a stone arch as the front "
             "gate and votive pillars; benches by the pond. Civ VI's water is a shader with no texture, so the pond "
             "is a terrain decal with a still-water texture made for the Lab.</p>",
     "galleries": [{"catalog": "holy-euro", "title": "Holy Site (European)"},
                   {"catalog": "holy-asian", "title": "Holy Site (Asian, Japanese style)"},
                   {"catalog": "holy-culture", "title": "Worship buildings and regional temples"}]},
    {"id": "campus", "name": "Campus", "status": "exact", "sprites": ["Campus"],
     "source": "Civ VI Campus base by era", "culture": "Era: 4 Civ VI bases. Culture: none needed; Seowon (Asian), "
     "Madrasa (Middle Eastern University) and Observatory (American) are optional extras.",
     "buildings": "Library → Civ VI Library. University → Civ VI University.",
     "text": "<p>The closest one-to-one match: C3X's three columns are exactly Civ VI's <code>EMPTY</code>, "
             "<code>LIBRARY</code> and <code>LIBRARY, UNIVERSITY</code> sets. Each column keeps one layout through the "
             "eras, so a Campus does not reshuffle when only its buildings' era models change, with only a few filler "
             "buildings (none repeated more than twice) so the Library and University stand out.</p>",
     "galleries": [{"catalog": "campus", "title": "Campus kits"}]},
    {"id": "entertainment", "name": "Entertainment Complex", "status": "exact",
     "sprites": ["EntertainmentComplex_EURO", "EntertainmentComplex_ROMAN"],
     "source": "Civ VI Entertainment Complex base by era", "culture": "Era: 3 Civ VI bases (Ancient and Medieval share "
     "one). Culture: the Colosseum in every culture; the Roman version stands on the Hippodrome kit.",
     "buildings": "Colosseum → Civ VI Colosseum wonder at district size; Civ VI Stadium in the Modern era.",
     "text": "<p>Columns: no buildings → <code>EMPTY</code>; Colosseum → Civ VI's Colosseum on the <code>ARENA</code> "
             "fairground base. Civ VI has no Colosseum district building (its Entertainment Complex building is the "
             "Arena, a wooden jousting ground), so the Colosseum here is Civ VI's Colosseum <i>wonder</i>: its "
             "wonder-movie model's shell, seating and arena floor, without the construction scaffolds and stone piles, "
             "scaled to the Stadium's size. The movie model is about 200,000 vertices in thousands of animated pieces, "
             "so production needs a decimated, merged copy. The Colosseum stands in every culture, drawn squat (as is the "
             "Modern Stadium) and the same size in every cell, among fewer, larger fair tents; Industrial and Modern "
             "complexes add the Water Park's Ferris wheel, scaled to stand only a little taller than the Stadium. Columns 2–3: the Colosseum on the Byzantine Hippodrome kit (Roman) and Civ VI's "
             "jousting Arena for comparison.</p>",
     "galleries": [{"catalog": "entertainment", "title": "Entertainment Complex kits"}]},
    {"id": "commercial", "name": "Commercial Hub", "status": "exact",
     "sprites": ["CommercialHub_EURO", "CommercialHub_MIDEAST"],
     "source": "Civ VI Commercial Hub base by era", "culture": "Era: 3 Civ VI bases. Culture: Grand Bazaar or Suguba for "
     "Middle Eastern; Sukiennice for European once its package reads.",
     "buildings": "Marketplace → Market. Bank → Bank (Grand Bazaar for Middle Eastern). Stock Exchange → Stock Exchange. "
     "Industrial hubs add Civ VI row houses, Modern hubs glass towers.",
     "text": "<p>C3X's four columns are Civ VI's <code>EMPTY</code>, <code>MARKET</code>, <code>MARKET, BANK</code> and "
             "<code>MARKET, BANK, STOCK_EXCHANGE</code>. To sit well beside a city, the market buildings stand among "
             "the city pack's own houses for that culture and era (recovered from its frozen recipes), on the same "
             "paving. The hub grows with each building: a wider footprint, more houses and denser blocks per column, "
             "so a hub with a Stock Exchange is visibly more than one with a Marketplace.</p>",
     "galleries": [{"catalog": "com-euro", "title": "Commercial Hub (European)"},
                   {"catalog": "com-asian", "title": "Commercial Hub (Asian)"},
                   {"catalog": "commercial-culture", "title": "Regional alternatives",
                    "note": "Poland's Sukiennice (a European market option) is missing: its package fails a size check "
                            "in this install."}]},
    {"id": "industrial", "name": "Industrial Zone", "status": "exact", "sprites": ["IndustrialZone"],
     "source": "Civ VI Industrial Zone base (Industrial, Modern)", "culture": "Era: Industrial and Modern only "
     "(Industrialization). Culture: none needed; Hansa and Oppidum are optional European extras.",
     "buildings": "Factory → Factory. Manufacturing Plant → Japan's Electronics Factory, seated in the power-plant slot.",
     "text": "<p>Civ VI's own Industrial Zone buildings, warehouses and tanks, packed onto the tile. Civ VI's Factory "
             "model reads small on the tile, so the Factory borrows the Energy Grid's coal-plant works (brick halls "
             "and stacks) with a factory smokestack beside them, and the Manufacturing Plant adds the oil plant's "
             "industrial complex. Civ III has no Workshop, so the Workshop set is the 'no buildings' look.</p>",
     "galleries": [{"catalog": "industrial", "title": "Industrial Zone kits"}]},
    {"id": "data-center", "name": "Data Center", "status": "standin", "sprites": ["DataCenter"],
     "source": "Spaceport pad + Research Lab (proposed)", "culture": "Modern only.",
     "buildings": "Research Lab → Civ VI Research Lab.",
     "text": "<p>Civ VI has no data-centre district and no mod provides one. Proposed: a dense modern tech park of "
             "Civ VI's glass towers only round a modern Campus plaza, with the Research Lab as the building. The "
             "Spaceport pad version is the alternative.</p>",
     "galleries": [{"catalog": "data-center", "title": "Data Center candidates"}]},
    {"id": "offshore", "name": "Offshore Extraction Zone", "status": "exact", "sprites": ["OffshoreExtractionZone"],
     "source": "Civ VI Offshore Oil Rig", "culture": "Era: Civ VI has industrial and modern rig variants.",
     "buildings": "Offshore Platform → Offshore Oil Rig. Before it: the same rig's deck without the derrick.",
     "text": "<p>C3X shows a bare platform, then a rig with a derrick, out on the sea. The Civ VI Offshore Oil Rig is "
             "the second picture; the 'no platform' picture is the same rig cut below its derrick, so the two match "
             "(Civ VI's unbuilt and construction states draw the full rig). The Offshore Wind Farm is an alternative.</p>",
     "galleries": [{"catalog": "coastal", "title": "Offshore candidates", "rows": [0]}]},
    {"id": "park", "name": "Park", "status": "exact", "sprites": ["Park"],
     "source": "Rise and Fall City Park by era", "culture": "Era: Civ VI has classical, industrial and modern parks.",
     "buildings": "None.",
     "text": "<p>The City Park improvement is a direct match, with era variants (Ancient and Middle Ages share the "
             "classical park). C3X builds Parks on forest, but the Lab's forest tile drew no trees under the park, so "
             "how Civ III's forest and the park share the tile still needs a check. National Park, Golf Course and the "
             "Preserve are alternatives.</p>",
     "galleries": [{"catalog": "leisure", "title": "Park candidates", "rows": [0, 1]}]},
    {"id": "ski", "name": "Ski Resort", "status": "exact", "sprites": ["SkiResort"],
     "source": "Gathering Storm Ski Resort", "culture": "One model.", "buildings": "None.",
     "text": "<p>Civ VI's Ski Resort pieces laid out on the Civ III mountain itself, as in Civ VI: the lodge on the "
             "front foot, the gondola climbing from beside it to a station at the summit, a cabin on each flank and a "
             "slalom of flags, on a snow-capped tile; it adds no trees of its own. Renderer64 seats each piece on the drawn "
             "mountain; the gondola has Civ VI's own slope taken out and every vertex seated on this mountain, so its "
             "cable and pylons follow the real slope. A snow cover decal drapes the mountain the way Civ VI's ski "
             "slopes are white; Renderer64's snow cap alone covers only the summit. Production would raise the snow "
             "cap on Ski Resort tiles instead of drawing a decal.</p>",
     "galleries": [{"catalog": "leisure", "title": "Ski Resort", "rows": [2]}]},
    {"id": "water-park", "name": "Water Park", "status": "exact", "sprites": ["WaterPark"],
     "source": "Rise and Fall Water Park (full set)", "culture": "Era: Civ VI has 3 bases; C3X's Water Park is Modern.",
     "buildings": "None in C3X; show Ferris Wheel + Aquarium + Aquatics Center as one fixed picture.",
     "text": "<p>Direct match, out in the water as C3X builds it. C3X has a single picture, so use one complete Civ "
             "VI stage. Some Aquarium and Aquatics Center materials have no base colour (glass) and are left out of "
             "these renders, so the later stages look close to the first. Its Ferris wheel also appears in Industrial "
             "and Modern Entertainment Complexes.</p>",
     "galleries": [{"catalog": "coastal", "title": "Water Park stages", "rows": [1]}]},
    {"id": "neighborhood", "name": "Neighborhood", "status": "close",
     "sprites": ["Neighborhood_AMER", "Neighborhood_EURO", "Neighborhood_ROMAN", "Neighborhood_MIDEAST", "Neighborhood_ASIAN"],
     "source": "The city pack's houses for the culture and era, in four layouts", "culture": "Era and culture: the city "
     "pack's houses for each of the five culture styles and four eras; C3X's four tile-hash variants are four layouts.",
     "buildings": "None.",
     "text": "<p>C3X Neighborhoods exist from the Ancient era and vary by culture; Civ VI's Neighborhood appears only in "
             "the Modern era and has no regional style. Each Neighborhood here is drafted from the city pack's houses "
             "for its culture and era, so it matches the cities, in four clearly different layouts for C3X's four "
             "tile-hash variants: a dense block, houses round a square, houses along a street, and garden houses among "
             "trees. Each variant also draws its own selection of houses. Only low houses are used (no towers or "
             "blocks), joined by Civ VI's residential fillers in the Industrial era and its suburban Neighborhood "
             "houses in the Modern era.</p><p>Civ VI recolours those houses per building through a tint mask in the "
             "colour texture's alpha (it covers their roof planks); left untinted, with white weatherboard walls, "
             "they read as bare white houses with bright orange chimneys. Each house type here takes its own roof "
             "colour through the mask (slate, moss, terracotta, charcoal, brown, teal, red) and a pastel wall colour "
             "on its weatherboard (cream, butter, sage, pale blue, peach, stone, rose), and the chimney brick is "
             "toned down. The wall palette is a Lab choice; Civ VI's own tint palette is not in the art files.</p>",
     "galleries": [{"catalog": "nbh-euro", "title": "Neighborhood (European)"},
                   {"catalog": "nbh-asian", "title": "Neighborhood (Asian)"},
                   {"catalog": "neighborhood", "title": "Civ VI Neighborhood bases (alternative)"}]},
    {"id": "wonder", "name": "Wonder District", "status": "delegated", "sprites": ["WonderDistrict"],
     "source": "M10 constructed wonders; construction props for the empty plot", "culture": "Era: per M10.",
     "buildings": "The wonder itself (M10).",
     "text": "<p>The constructed wonder belongs to renderer M10. For a planned but empty plot, Civ VI's district bases "
             "carry construction props (scaffolds, lumber, stone) in their under-construction state, shown under "
             "district states below.</p>"},
    {"id": "distribution", "name": "Distribution Hub", "status": "close", "sprites": ["DistributionHub"],
     "source": "Civ VI warehouses by era (no Civ VI district)", "culture": "Era: one picture per era in C3X.",
     "buildings": "None.",
     "text": "<p>C3X's sprite is a storage yard: warehouses with goods stacked between them. Civ VI has no such "
             "district, but it has warehouses, so each era's yard is drafted from them: the Harbor's classical "
             "sheds and the Encampment's shed round the Granary as storehouse (Ancient, Medieval), the Outback "
             "Station's sheds and barns with the Industrial Zone's tanks (Industrial), and the Harbor's modern warehouses (Modern), with "
             "barrels, crates, shipping containers and trucks between them.</p>",
     "galleries": [{"catalog": "distribution", "title": "Distribution Hub warehouse yards"}]},
    {"id": "aerodrome", "name": "Aerodrome", "status": "exact", "sprites": ["Aerodrome"],
     "source": "Civ VI Aerodrome", "culture": "One Civ VI era.", "buildings": "Airport → Civ VI Hangar + Airport.",
     "text": "<p>Columns: no buildings → <code>EMPTY</code>; Airport → <code>HANGAR, AIRPORT</code>. Civ VI's own "
             "Aerodrome layout, not re-drafted: the whole kit turned so the airport lines up with the tile. Civ VI "
             "paints its runway as a ground decal the import cannot place, so a runway decal (asphalt, edge lines, "
             "centre line and thresholds) is laid along a tile axis beside the buildings. The Airstrip improvement is "
             "a smaller alternative.</p>",
     "galleries": [{"catalog": "aerodrome", "title": "Aerodrome kits"}]},
    {"id": "port", "name": "Port", "status": "exact", "sprites": ["Port_NE", "Port_SW"],
     "source": "Civ VI Harbor, aligned to the coast", "culture": "Era: 2 Civ VI bases. Culture: Cothon for Roman, Royal "
     "Navy Dockyard for European.",
     "buildings": "Harbor → Lighthouse. Commercial Dock → Shipyard (Seaport is the alternative).",
     "text": "<p>C3X draws one Port picture per coast direction. Civ VI's Harbor is aligned to the coast, so the 3D kit "
             "rotates to the same four directions. The Lab tiles here are small bays, so the docks do not yet face open "
             "water.</p>",
     "galleries": [{"catalog": "port", "title": "Port kits"},
                   {"catalog": "port-culture", "title": "Regional harbors"}]},
    {"id": "rail", "name": "Central Rail Hub", "status": "authored",
     "sprites": ["CentralRailHub_EURO", "CentralRailHub_ASIAN"],
     "source": "No Civ VI art; mod or authored kit", "culture": "C3X varies it by era and culture.",
     "buildings": "Mass Transit System → no Civ VI building.",
     "text": "<p>Civ VI ships railroads as route decals only: no station, depot or metro model. JNR's Suburbs Transit "
             "Station and a newer Transport Hub mod with a Railway Station are the nearest; nothing covers Mass Transit. "
             "An authored kit from Civ VI industrial buildings beside the existing Renderer64 rail art is the practical "
             "route.</p>"},
    {"id": "energy", "name": "Energy Grid", "status": "close", "sprites": ["EnergyGrid"],
     "source": "Industrial Zone power-plant base + one plant", "culture": "Era: Industrial and Modern.",
     "buildings": "Coal Plant → Coal Power Plant. Hydro Plant → Dam + Hydroelectric Dam. Nuclear Plant → Power Plant "
     "(Gathering Storm's nuclear plant). Solar Plant → Solar Farm.",
     "text": "<p>C3X shows the first plant found. Every plant has Civ VI art: the Coal Power Plant's brick halls and "
             "stacks, the nuclear Power Plant's cooling towers and the Solar Farm's panels read at gameplay zoom. The "
             "Oil Power Plant looks almost the same as the coal plant. The Hydroelectric Dam is built across a river in "
             "Civ VI and looks stranded on dry land; prefer river tiles or show only the plant building. RUIVO's "
             "Generator District mod has custom power-plant models.</p>",
     "galleries": [{"catalog": "energy-grid", "title": "Energy Grid plants"}]},
    {"id": "bridge", "name": "Bridge", "status": "standin", "sprites": ["Bridge"],
     "source": "Civ VI road bridges by era, scaled across the coast tile", "culture": "Era: medieval, industrial, "
     "modern bridge models.", "buildings": "None.",
     "text": "<p>Civ VI's route bridges come in medieval, industrial and modern styles but are sized for a river "
             "crossing; spanning a C3X coast tile needs about three times their length, built from repeated spans. Mods "
             "offer a full era ladder: Old London Bridge (medieval, CC-BY source), Tower Bridge, Port Lime's bridge "
             "district and the Hong Kong–Zhuhai–Macao bridge.</p>",
     "galleries": [{"catalog": "topology", "title": "Civ VI bridge models", "rows": [1]}]},
    {"id": "canal", "name": "Canal", "status": "exact", "sprites": ["Canal"],
     "source": "Gathering Storm Canal (straight, bent and port pieces)", "culture": "Era: ancient, classical, industrial "
     "and modern pieces.", "buildings": "None.",
     "text": "<p>Direct match. The pieces need C3X's canal neighbour topology to choose straight, bent and port "
             "segments; the Lab shows straight pieces only.</p>",
     "galleries": [{"catalog": "topology", "title": "Canal pieces", "rows": [0]}]},
    {"id": "great-wall", "name": "Great Wall", "status": "exact", "sprites": ["GreatWall"],
     "source": "Civ VI Great Wall kit (Walls.artdef GreatWallSet) on C3X's tile topology",
     "culture": "One style; C3X stops new building at Metallurgy.",
     "buildings": "None.", "scenes": ["great-wall", "great-wall-city"],
     "text": "<p>C3X draws a pillar on every Great Wall tile and a half wall toward each edge-sharing neighbour (NE, SE, "
             "SW, NW) that is also Great Wall of the same owner. The 3D version keeps exactly that topology with Civ VI's "
             "own Great Wall kit, which imports through the city-wall path:</p><ul>"
             "<li>wall runs from the tile centre to each shared edge, built from Civ VI's half-wall pieces;</li>"
             "<li>a tower at every tile centre, the tower with a gatehouse building at corners, and Civ VI's end tower "
             "where a run stops;</li>"
             "<li>the gate where a road crosses the wall.</li></ul>"
             "<p>Civ VI bends these pieces over the terrain. Here each short piece sits on the lowest ground under it, so "
             "a run climbs hills and mountains in steps. Production should bend the pieces to the ground the way Civ VI "
             "does. Pillaged versions of every piece exist for C3X's Abandoned state.</p>"},
    {"id": "states", "name": "Under construction and Abandoned", "status": "exact", "sprites": ["Abandoned"],
     "source": "The same kit's construction and pillaged states", "culture": "Same as the kit.", "buildings": "—",
     "text": "<p>C3X draws nothing while a district is being built and a ruin when it is outside any territory. Every "
             "Civ VI base also has construction and pillaged states, so the 3D kit can show construction with no extra "
             "art. For the Abandoned ruin, a pillaged kit still reads as a working district, so the ruin here is "
             "Civ VI's ruined-city blocks (the pieces it uses for razed cities), taken back by trees and scrub. "
             "C3X districts on water (Offshore Extraction Zone, Water Park) leave a wreck among floating debris.</p>",
     "galleries": [{"catalog": "abandoned", "title": "Abandoned: ruined blocks and a wreck at sea"},
                   {"catalog": "topology", "title": "Campus: completed, under construction, pillaged", "rows": [3]}]},
]

CULTURES = {
    "intro": "<p class=\"intro\">Civ III culture groups: American (America, Aztecs, Iroquois, Inca, Maya), European "
             "(Germany, Russia, France, England, Spain, Scandinavia, Celts, Netherlands, Portugal), Roman (Rome, Egypt, "
             "Greece, Carthage, Byzantines), Middle Eastern (Babylon, Persia, Zululand, Ottomans, Arabia, Sumeria, "
             "Hittites) and Asian (China, Japan, India, Mongols, Korea). Bold entries are the proposed defaults for the "
             "districts C3X already varies by culture; the rest are optional extras the renderer could add.</p>",
    "head": ["District", "American", "European", "Roman", "Middle Eastern", "Asian"],
    "rows": [
        ["Holy Site (temple / worship)", "<b>Temple / Meeting House</b>", "<b>Temple / Cathedral</b>",
         "<b>Temple / domed basilica</b> (Civ VI Synagogue model); Lavra optional", "<b>Temple / Mosque</b>; Dar-e Mehr optional",
         "<b>Prasat / Pagoda</b>; Wat, Stupa optional"],
        ["Entertainment Complex", "<b>Tlachtli</b>; Street Carnival optional", "<b>Colosseum</b>",
         "<b>Colosseum on the Hippodrome kit</b>", "<b>Colosseum</b>", "<b>Colosseum</b>"],
        ["Commercial Hub", "<b>Default</b>", "<b>Default</b>; Sukiennice once readable", "<b>Default</b>",
         "<b>Grand Bazaar</b>; Suguba optional", "<b>Default</b>"],
        ["Neighborhood", "<b>City pack: American</b>", "<b>City pack: European</b>", "<b>City pack: Roman</b>",
         "<b>City pack: Middle Eastern</b>", "<b>City pack: Asian</b>"],
        ["Central Rail Hub", "Authored", "Authored", "Authored", "Authored", "Authored"],
        ["Port", "Harbor", "Royal Navy Dockyard", "Cothon", "Harbor", "Harbor"],
        ["Campus", "Observatory", "Campus", "Campus", "Madrasa as University", "Seowon"],
        ["Encampment", "Encampment", "Encampment", "Basilikoi Paides", "Ikanda", "Ordu or Thanh"],
        ["Industrial Zone", "Industrial Zone", "Hansa or Oppidum", "Industrial Zone", "Industrial Zone", "Industrial Zone"],
    ],
    "outro": "<p class=\"intro\">Selection should follow the city-pack rule already written for palaces: an explicit "
             "scenario or civilization override, then the Civ III culture group, then the pack default. Renderer code "
             "never tests Civ VI civilization names.</p>",
}

MODS = """<p class="intro">Web survey of Steam Workshop and CivFanatics, pages read only. No mod states a licence that
allows reuse, and kitbashed mods are Firaxis-derived, so anything below is for local prototypes, or needs the author's
permission.</p>
<div class="table-wrap"><table><thead><tr><th>Need</th><th>Best candidates</th><th>Notes</th></tr></thead><tbody>
<tr><th scope="row">Central Rail Hub</th><td><a href="https://steamcommunity.com/sharedfiles/filedetails/?id=2669325299">JNR – District Expansion: Suburbs</a> (Transit Station);
<a href="https://steamcommunity.com/sharedfiles/filedetails/?id=3795434442">District Transport Hub</a> (Railway Station);
<a href="https://steamcommunity.com/sharedfiles/filedetails/?id=2479197624">Leugi – Monopoly++</a> (Station improvement)</td>
<td>JNR claims unique 3D art; Transport Hub is probably vanilla kitbash. No metro, tram or depot model exists.</td></tr>
<tr><th scope="row">Distribution Hub</th><td><a href="https://steamcommunity.com/sharedfiles/filedetails/?id=2402394695">JNR – District Expansion: Industry</a> (Logistics Center);
<a href="https://steamcommunity.com/sharedfiles/filedetails/?id=3516350860">Manor District</a> (food storage, custom models)</td>
<td>Monopoly++ adds Warehouses and Container Ports.</td></tr>
<tr><th scope="row">Energy Grid</th><td><a href="https://steamcommunity.com/sharedfiles/filedetails/?id=3504749688">Generator District</a> (RUIVO, custom models);
<a href="https://steamcommunity.com/sharedfiles/filedetails/?id=1660936201">JNR – Renewable Energy Complexity</a></td>
<td>JNR's Battery Power Station uses a cut Firaxis model still in Gathering Storm's files.</td></tr>
<tr><th scope="row">Bridge</th><td><a href="https://steamcommunity.com/sharedfiles/filedetails/?id=3117103358">Old London Bridge</a>;
<a href="https://steamcommunity.com/sharedfiles/filedetails/?id=2068247220">Tower Bridge</a>;
<a href="https://steamcommunity.com/sharedfiles/filedetails/?id=2331527458">Port Lime – Bridge Districts</a>;
<a href="https://steamcommunity.com/sharedfiles/filedetails/?id=3634834658">Hong Kong–Zhuhai–Macao Bridge</a></td>
<td>An era ladder. Old London Bridge is based on a CC-BY Sketchfab model.</td></tr>
<tr><th scope="row">Park</th><td><a href="https://steamcommunity.com/sharedfiles/filedetails/?id=2937953680">Leugi's Garden District</a>;
<a href="https://steamcommunity.com/sharedfiles/filedetails/?id=3574374656">District: Town</a> (era park models)</td>
<td>Vanilla City Park already covers the need.</td></tr>
<tr><th scope="row">Culture styles</th><td><a href="https://steamcommunity.com/sharedfiles/filedetails/?id=2169097417">Leugi's City Styles Pack</a>;
<a href="https://steamcommunity.com/sharedfiles/filedetails/?id=2427195031">Project Metropolis</a>;
<a href="https://steamcommunity.com/sharedfiles/filedetails/?id=2190389813">City Lights</a></td>
<td>About 20 regional city styles, mostly recombined Gathering Storm art.</td></tr>
<tr><th scope="row">Harbor, docks</th><td><a href="https://steamcommunity.com/sharedfiles/filedetails/?id=2056401784">JNR – District Expansion: Commerce</a>;
<a href="https://steamcommunity.com/sharedfiles/filedetails/?id=3668683945">C's Maritime District Expansion</a></td>
<td>Trade Dock, Offshore Terminal, Dry Dock.</td></tr>
<tr><th scope="row">Nothing found</th><td colspan="2">Data Center, SAM Missile Battery, subway or metro, and a better offshore
oil platform. For SAM, a static Civ VI Mobile SAM or Anti-Air Gun unit model is the best vanilla source.</td></tr>
</tbody></table></div>"""

ERA_MAP = """<h3>Era mapping</h3><div class="table-wrap"><table><thead><tr><th>Civ III era</th><th>Civ VI art era used</th>
<th>Falls back to</th></tr></thead><tbody>
<tr><th scope="row">Ancient</th><td>ARTERA_ANCIENT</td><td>DEFAULT</td></tr>
<tr><th scope="row">Middle Ages</th><td>ARTERA_CLASSICAL</td><td>DEFAULT</td></tr>
<tr><th scope="row">Industrial</th><td>ARTERA_INDUSTRIAL</td><td>CLASSICAL, DEFAULT</td></tr>
<tr><th scope="row">Modern</th><td>ARTERA_MODERN</td><td>INDUSTRIAL, CLASSICAL, DEFAULT</td></tr>
</tbody></table></div><p>This is the production city pack's mapping. Civ VI's DEFAULT tag is its ancient art; a building set
with no art for an era uses the nearest earlier one.</p>"""

QUIRKS = """<h3>C3X district behaviour the 3D kits must reproduce</h3><ul>
<li><b>Era</b> comes from the territory owner's current era, not the era the district was built in, so every district
changes when its civilization advances. Outside any territory, the builder's era and culture are used.</li>
<li><b>Culture</b> is the owner's Civ III culture group. Only districts with <code>vary_img_by_culture</code> change; the
3D renderer may vary others.</li>
<li><b>Buildings</b> come from every city in the district's radius, so a shared district shows a building completed in a
neighbouring city. <code>by-count</code> picks the column from the number of distinct dependent building types present;
<code>by-building</code> layers one picture per building present.</li>
<li><b>Abandoned</b>: a completed district outside any territory draws the ruin (land or sea), except Wonder, Natural
Wonder, Bridge and Canal districts.</li>
<li><b>Neighborhood</b> picks one of four variants from a hash of the tile coordinates.</li>
<li><b>Energy Grid</b> shows the first plant found among its dependent buildings; <b>Port</b> picks one of four coast
directions; <b>Bridge</b>, <b>Canal</b> and <b>Great Wall</b> pick pieces from neighbouring districts.</li>
<li>Nothing is drawn while a district is under construction.</li></ul>
<h3>How these pictures were made</h3><ul>
<li><code>Renderer/lab/studies/districts/kits.py</code> reads every installed <code>Landmarks*.artdef</code>, picks the base
variant Civ VI would choose for each building set and era, compiles it and its hero buildings into a Lab-only pack
(<code>DistrictKitSourcesLab</code>), bakes Lab-only compositions (<code>DistrictKitCatalogLab</code>) and renders them
through a pinned copy of the current candidate renderer on the Windows VM.</li>
<li>Every building is turned with the city pack's rule (<code>footprint_alignment</code>, as in
<code>city_fidelity/prepare_pack.py</code>): its minimum-area footprint rectangle square to the tile axes and its front
toward the nearer of south-west and south-east.</li>
<li>Land district kits are then re-drafted onto the tile by role, read from the Civ VI entry names: walls
(<code>*_Wall_*</code>) become a square ring with towers at the corners; hero buildings and large landmarks take the
back slots of a 3 × 3 lattice; a statue, fountain or square takes the middle; filler buildings, tallest first, take the
remaining slots; up to five props go toward the front. Generic trees, shrubs and hex ground platforms are left out. Drafted districts are drawn at 2.0× Civ VI's authored size, other kits at
2.5×. Coastal kits, improvements (parks, farms, rigs) and the lone-building galleries keep their own layout.</li>
<li>The grids face south-west; the context scenes alternate south-west and south-east tile by tile. Each part stands
on the ground under its own pivot; Civ VI would also flatten the ground.</li>
<li>Not shown: Civ VI's terrain-painted district ground, night window emissive, and Civ VI's player-colour tint on
banners.</li>
<li>Nothing here changes production packs, the renderer, the seed ledgers or the game install.</li></ul>"""


# --- PCX-layout sheets: which Renderer64 cell fills each C3X sprite cell -------
# map(row, col) -> (catalog, cell row, cell col) or None (C3X draws nothing there).
ERA_ROWS = ("Ancient", "Middle Ages", "Industrial", "Modern")
LATE_ONLY = "Not reachable before the district's tech"


def _culture_sheets(stem, district, cols, catalog_for, note=""):
    return [{"file": f"{stem}_{code}", "district": district, "culture": code, "rows": ERA_ROWS, "cols": cols,
             "map": (lambda r, c, k=catalog_for(code): (k, r, c)), "note": note}
            for code in ("AMER", "EURO", "ROMAN", "MIDEAST", "ASIAN")]


SHEETS = [
    {"file": "Encampment", "district": "Encampment", "rows": ERA_ROWS,
     "cols": ("no buildings", "Barracks", "Barracks + SAM", "unused"),
     "map": lambda r, c: ("encampment", r, c) if c < 2 else ("encampment", 3, 2) if c == 2 and r == 3 else None,
     "note": "SAM needs Rocketry, so a second building exists only in the Modern row; its model is a placeholder."},
    *_culture_sheets("HolySite", "Holy Site", ("no buildings", "Temple", "Temple + Cathedral", "unused"),
                     lambda code: f"holy-{code.lower()}",
                     "C3X draws one Holy Site row for every era; Renderer64 adds the four era rows."),
    {"file": "Campus", "district": "Campus", "rows": ERA_ROWS,
     "cols": ("no buildings", "Library", "Library + University", "unused"),
     "map": lambda r, c: ("campus", r, c) if c < 3 else None},
    *_culture_sheets("EntertainmentComplex", "Entertainment Complex", ("no buildings", "Colosseum", "unused", "unused"),
                     lambda code: {"AMER": "ent-amer", "ROMAN": "ent-roman"}.get(code, "entertainment")),
    *_culture_sheets("CommercialHub", "Commercial Hub", ("no buildings", "Marketplace", "+ Bank", "+ Stock Exchange"),
                     lambda code: f"com-{code.lower()}"),
    {"file": "IndustrialZone", "district": "Industrial Zone", "rows": ERA_ROWS,
     "cols": ("no buildings", "Factory", "+ Manufacturing Plant", "unused"),
     "map": lambda r, c: ("industrial", r - 2, c + 1) if r >= 2 and c < 3 else None, "note": LATE_ONLY + "."},
    {"file": "DataCenter", "district": "Data Center", "rows": ERA_ROWS,
     "cols": ("no buildings", "Research Lab", "unused", "unused"),
     "map": lambda r, c: ("data-center", 0, c) if r >= 2 and c < 2 else None,
     "note": "Dense modern tech park + Research Lab proposal, the same in both late rows."},
    {"file": "Aerodrome", "district": "Aerodrome", "rows": ERA_ROWS, "cols": ("no buildings", "Airport", "unused"),
     "map": lambda r, c: ("aerodrome", r - 2, c) if r >= 2 and c < 2 else None, "note": LATE_ONLY + "."},
    {"file": "EnergyGrid", "district": "Energy Grid", "rows": ERA_ROWS, "cell_h": 88,
     "cols": ("no plant", "Coal Plant", "Hydro Plant", "Nuclear Plant", "Solar Plant"),
     "map": lambda r, c: None if r < 2 else (("energy-grid", r - 2, 0), ("energy-grid", r - 2, 1), ("energy-grid", 2, 0),
                                               ("energy-grid", r - 2, 2), ("energy-grid", 2, 1))[c],
     "note": LATE_ONLY + ". Hydro and Solar have one Civ VI model for both rows."},
    {"file": "Port_SW", "district": "Port", "rows": ERA_ROWS,
     "cols": ("no buildings", "Harbor", "Harbor + Commercial Dock", "unused"),
     "map": lambda r, c: ("port", r, c) if c < 3 else None,
     "note": "One of C3X's four coast directions; the kit turns to the others. Port keeps Civ VI's layout so its docks meet the water."},
    {"file": "OffshoreExtractionZone", "district": "Offshore Extraction Zone", "rows": ("All eras",),
     "cols": ("no platform", "Offshore Platform"),
     "map": lambda r, c: ("coastal", 0, c)},
    {"file": "Park", "district": "Park", "rows": ERA_ROWS, "cols": ("Park", "unused", "unused", "unused"),
     "map": lambda r, c: ("leisure", 0, r) if c == 0 else None},
    {"file": "SkiResort", "district": "Ski Resort", "rows": ("All eras",), "cell_h": 88, "cols": ("Ski Resort", "unused", "unused", "unused"),
     "map": lambda r, c: ("leisure", 2, 0) if c == 0 else None},
    {"file": "WaterPark", "district": "Water Park", "rows": ("All eras",), "cell_h": 88, "cols": ("Water Park", "unused", "unused", "unused"),
     "map": lambda r, c: ("coastal", 1, 3) if c == 0 else None},
    {"file": "DistributionHub", "district": "Distribution Hub", "rows": ERA_ROWS, "cols": ("stand-in", "unused", "unused", "unused"),
     "map": lambda r, c: ("distribution", 0, r) if c == 0 else None,
     "note": "A storage yard of each era's warehouses, from Civ VI's harbour, encampment and industrial kits."},
    *_culture_sheets("Neighborhood", "Neighborhood", ("variant 1", "variant 2", "variant 3", "variant 4"),
                     lambda code: f"nbh-{code.lower()}",
                     "The city pack's houses in four layouts; C3X picks one of the four variants from the tile."),
    {"file": "GreatWall", "district": "Great Wall", "rows": ("arms", "joined"), "cell_h": 88,
     "cols": ("1", "2", "3", "4", "5"),
     "map": lambda r, c: ("great-wall-pieces", r, c),
     "note": "C3X's sheet holds a pillar and eight half walls, of which it draws the four edge directions. "
             "Renderer64 tiles: the pillar alone, each arm, two runs, two corners and a crossing."},
    {"file": "Abandoned", "district": "Abandoned", "rows": ("All eras",), "cols": ("land", "sea", "unused", "unused"),
     "map": lambda r, c: ("abandoned", 0, 0) if c == 0 else ("abandoned", 0, 3) if c == 1 else None,
     "note": "Land: a ruined block taken back by trees and scrub. Sea: a wreck among floating debris."},
]
