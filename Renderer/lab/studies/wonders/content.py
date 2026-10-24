"""Prose for the wonder survey page: per-wonder recommendations, Workshop candidates, findings."""
from __future__ import annotations

STEAM = "https://steamcommunity.com/sharedfiles/filedetails/?id="


def workshop(item: str, title: str, author: str, note: str) -> dict:
    return {"id": item, "title": title, "author": author, "note": note}


# Per C3X wonder (ledger c3x_key): match status, the recommendation in a sentence,
# notes, and the Workshop items considered (installed ones are rendered).
WONDERS = {
    "pyramids": {"status": "exact", "text": "Civ VI's Pyramids: the Giza complex with its temples, palms and "
                 "obelisks. Its build movie raises the three pyramids course by course with ramps and stone piles."},
    "hanging_gardens": {"status": "exact", "text": "Civ VI's Hanging Gardens: terraced ziggurat planted with "
                        "trees, as C3X draws it."},
    "oracle": {"status": "exact", "text": "Civ VI's Oracle on its hillside sanctuary; C3X places it on hills."},
    "temple_of_artemis": {"status": "exact", "text": "Civ VI's Temple of Artemis (Rise and Fall)."},
    "mausoleum_of_mausollos": {"status": "exact", "text": "Civ VI's Mausoleum at Halicarnassus (Persia and "
                               "Macedon pack): the same monument under its usual name."},
    "statue_of_zeus": {"status": "exact", "text": "Civ VI's Statue of Zeus: the seated statue in its temple "
                       "precinct."},
    "great_library": {"status": "exact", "text": "Civ VI's Great Library. Its build movie works differently from "
                      "the Pyramids': scaffolded walls and a crane, then the finished library replaces them."},
    "copernicus_observatory": {
        "status": "standin",
        "text": "No Civ VI or Workshop wonder depicts Copernicus' observatory. The Civ VI Campus University has a "
                "domed observatory tower on a stone college, close to C3X's castle with a dome. The Workshop's Ulugh "
                "Beg Observatory (installed) is a Timurid observatory, further from C3X's sprite.",
        "workshop": [workshop("3283744069", "Ulugh Beg Observatory (World Wonder)", "c",
                              "Timurid observatory with its gardens; no build movie the Lab can stage")]},
    "js_bachs_cathedral": {
        "status": "close",
        "text": "Sukritact's Notre-Dame (installed Workshop mod) matches C3X's white twin-tower Gothic cathedral and "
                "has a full build movie. Civ VI's own Gothic Cathedral (Holy Site) is the base-game fallback.",
        "workshop": [workshop("3391287227", "Notre-Dame de Paris (2024) (World Wonder)", "Phantagonist",
                              "installed, but its model is assembled from attached pieces the Lab places as a "
                              "scatter; not used")]},
    "newtons_university": {
        "status": "close",
        "text": "Civ VI's Oxford University: a Gothic English college with a build movie, as C3X draws Newton's. "
                "Albro's National College (installed) is a white Beaux-Arts hall, further from C3X's sprite.",
        "workshop": [workshop("2503076363", "CIVILIZATION VI: NATIONAL WONDERS (National College)", "Albro",
                              "white hall with mansard roofs; no build movie")]},
    "seti_program": {
        "status": "close",
        "text": "Albro's Arecibo Observatory (installed Workshop mod): a radio dish slung between three towers, the "
                "closest thing to C3X's dishes in a crater. Its mod reuses another wonder's animation, so it rises "
                "from the ground beside Civ VI's crane and material piles while under construction.",
        "workshop": [workshop("2317145428", "CIVILIZATION VI: ARECIBO OBSERVATORY", "Albro, Deliverator23",
                              "installed; the recommendation"),
                     workshop("3243323274", "S.E.T.I. (World Wonder)", "JNR",
                              "an array of about eight dishes; not installed")]},
    "shakespeares_theater": {
        "status": "exact",
        "text": "Deliverator23's Globe Theatre (installed Workshop mod): the round timber Globe that C3X draws. It "
                "has no build movie of its own, so it rises from the ground beside Civ VI's construction props.",
        "workshop": [workshop("1770688835", "Globe Theatre (World Wonder)", "Deliverator23",
                              "installed; the recommendation")]},
    "sistine_chapel": {
        "status": "close",
        "text": "Albro's St. Peter's Basilica (installed Workshop mod): the domed basilica C3X draws for the Sistine "
                "Chapel, which stands beside it. No build movie of its own; it rises from the ground while built.",
        "workshop": [workshop("2511437298", "CIVILIZATION VI: ST PETERS BASILICA", "Albro",
                              "installed; the recommendation")]},
    "great_lighthouse": {"status": "exact", "text": "Civ VI's Great Lighthouse, on the coast tile as in C3X."},
    "colossus": {"status": "exact", "text": "Civ VI's Colossus on its harbour moles. C3X's second art direction "
                 "becomes a turn of the same model."},
    "hoover_dam": {
        "status": "standin",
        "text": "WindFly's Three Gorges Dam (installed Workshop mod) is a concrete river dam that reads on its own "
                "tile, unlike Civ VI's Dam district, which is built across a river between two hexes. A true Hoover "
                "Dam would still need an authored layout: the dam across a river gorge cut into the mountain tile, "
                "as C3X's sprite draws it.",
        "workshop": [workshop("1710103474", "【Wonder】Three Gorges Dam", "WindFly",
                              "installed; the recommendation for now")]},
    "apollo_program": {"status": "close", "text": "Civ VI's Space Race rocket on its launch tower, with its "
                       "assembly buildings, as C3X draws a Saturn V on its gantry. Its model has no build movie, so "
                       "its stages rise from the ground."},
    "forbidden_palace": {"status": "exact", "text": "Civ VI's Forbidden City: the same palace complex under its "
                         "usual name."},
    "intelligence_agency": {"status": "close", "text": "C3X draws a low, wide modern office complex like CIA "
                            "headquarters. Civ VI's own Intelligence Agency (Government Plaza) is a classical "
                            "colonnaded hall and looks out of place, so two of Civ VI's Campus Research Labs, low "
                            "grey-white modern offices, form a small campus instead. Civ VI's modern city blocks and "
                            "Broadcast Center were tried and read as towers. No build movie: it rises from the ground."},
    "pentagon": {
        "status": "exact",
        "text": "IthildinX's The Pentagon (installed Workshop mod): the five-sided building itself. Its movie only "
                "reveals the finished building at the end, so its stages rise from the ground beside Civ VI's "
                "construction props.",
        "workshop": [workshop("3337911322", "The Pentagon (World Wonder)", "IthildinX",
                              "installed; the recommendation")]},
    "military_academy": {"status": "exact", "text": "Civ VI's Encampment building of the same name."},
    "iron_works": {
        "status": "close",
        "text": "Civ VI's Ruhr Valley: blast furnaces, chimneys and a rail spur, with a full build movie. Albro's "
                "Ironworks (installed) is a compact brick works; compare both below.",
        "workshop": [workshop("2612473657", "CIVLIZATION VI: NATIONAL WONDERS PACK 1 (Ironworks)", "Albro",
                              "installed; an alternative")]},
    "wall_street": {
        "status": "close",
        "text": "Albro's Financial Center (installed Workshop mod): a cluster of 1920s Manhattan towers with gold "
                "tops, as C3X draws Wall Street. The Stock Exchange among glass towers is the base-game fallback.",
        "workshop": [workshop("2612473657", "CIVLIZATION VI: NATIONAL WONDERS PACK 1 (Financial Center)", "Albro",
                              "installed; the recommendation"),
                     workshop("2384183359", "CIVILIZATION VI: EMPIRE STATE BUILDING", "Albro",
                              "a single tower; not installed")]},
}

STATUS = {"exact": "Same wonder", "close": "Close match", "standin": "Stand-in", "gap": "Gap"}

FINDINGS = (
    ("Civ VI's wonders are their build movies",
     "Every Civ VI wonder model (base game, DLC and Workshop mods alike) is the construction movie that plays when "
     "it completes. Each block, scaffold pole and crane is a piece on its own animated bone with an on/off "
     "visibility timeline. Posing that movie gives real construction stages at any progress, and its last frame "
     "is the finished wonder with every scaffold gone."),
    ("Construction art at any progress",
     "C3X has one construction sprite per wonder. Renderer64 could show the site grow with the city's real build "
     "progress: 25%, half and 75% below are exactly that. Each movie's own building span is measured from its "
     "visibility timeline, so the mapping is generic."),
    ("Every C3X wonder now has a match",
     "With the Workshop mods you installed, 13 of the 23 default C3X wonders have the same wonder (Civ VI or "
     "Workshop) and 8 a close match. Only Copernicus' Observatory and Hoover Dam still use stand-ins: Civ VI's "
     "domed University and the Workshop's Three Gorges Dam."),
    ("Workshop mods use the same format",
     "Workshop wonders ship their own WonderMovie package and import, stage and render through the same path. "
     "Sukritact's Notre-Dame has a full build movie. Arecibo, the Globe, St. Peter's, the Pentagon and Albro's "
     "Financial Center reuse or lack a build movie, so they rise from the ground beside Civ VI's own crane and "
     "material piles while under construction."),
    ("Laid on the square Civ III tile, not the hex",
     "Civ VI wonders are authored in a hex, with buildings at hex angles and props in the hex ring. Each one is now "
     "turned square to the Civ III tile, fitted so its buildings span it, and trimmed to it, with ground-level "
     "paving pressed to the tile edge. Raised geometry stays within 0.1 tile of the square everywhere."),
    ("Night lights as Civ VI draws them",
     "Civ VI lights a wonder at night with emissive maps laid out by a separate texture coordinate set, the "
     "same way it lights city buildings. The wonders here are drawn with Renderer64's city body path, which "
     "keeps those coordinates, Civ VI's normal, gloss and occlusion maps and the night response of the "
     "production cities: lit doorways, windows and lamps rather than glowing stonework. Many build movies swap "
     "the finished, lit building in on their last frame (the Temple of Artemis); the finished stage uses it."),
    ("Civ VI's fires and lights, as separate effects",
     "Besides their lit windows, Civ VI wonders start effects from their model's timeline: braziers, a "
     "lighthouse beacon, tower lights, marquee blinkers. Each trigger names its effect and the bone it sits on. "
     "Renderer64 draws them in its city effect layer: fires as its flames, lights as new steady or blinking "
     "lamps. Shown here: the Great Lighthouse's beacon and braziers and the Mausoleum's corner braziers. The "
     "Great Library's, Hanging Gardens' and Statue of Zeus' braziers sit on bones of a model the Lab does not "
     "import yet, so they are left out rather than misplaced. Most other wonders, and every Workshop mod, start "
     "no idle effects: their windows alone light them at night."),
    ("Sources without a build movie",
     "Workshop models without a usable build movie and district buildings used as stand-ins (Intelligence Agency, Military Academy, War Department, Stock Exchange, "
     "Spaceport, Dam, University) have no build movie. Here they rise from the ground at the built share of their "
     "height."),
)

PRODUCTION = (
    "Each staged wonder needs C3X's build progress for its wonder tile (shields toward the wonder over its cost) "
    "in the scene capture. Today the renderer only knows construct or complete, which C3X's sprite selection "
    "already follows.",
    "Stages would be baked offline into a few static meshes per wonder (for example 25/50/75/100%) or kept as "
    "per-piece reveal times. Either form is source-independent; neither needs Granny or Civ VI at runtime.",
    "The Lab conversion uses the existing CivNexus6 animation converter plus a small Lab exporter for the "
    "visibility tracks (the managed wrapper hides them; they are read from the native track group and evaluated "
    "by Granny's own sampler).",
    "Civ VI stages stand on Civ VI's painted ground decals (the movie has a decal track too); here they stand on "
    "plain Civ III terrain.",
)
