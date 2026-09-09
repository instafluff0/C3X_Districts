# Goody huts

Three stable hut variants; presence comes from Civ III’s viewer-conditioned goody-hut accessor.

Uses the shared world lighting and shadow pages. Detail and gameplay fixtures
cover noon, evening, midnight and dawn at normal and reduced zoom. The noon
normal-zoom witness removes and restores the site, checking ownership, shadow
cleanup and exact warm/cold pixel parity. Gameplay includes hillside grounding.

Run `python3 Renderer/renderer.py lab goody-huts` or
`python3 Renderer/renderer.py integration goody-huts`.
The combined `huts-camps` category retains the interaction fixture.
Source ground decals and optional animated attachments remain excluded.
