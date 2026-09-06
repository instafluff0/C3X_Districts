# L15 Railroad Audit

Status: complete; deterministic 192-tile alternate-skin candidate critically inspected and explicitly approved under the user's 2026-09-05 autonomous-review authorization.

## Scene and provenance

- Terrain remains the unchanged authoritative 16x12 BIQ viewport inherited from L13-L14.
- `fixtures/l15_railroads_192.csv` is a separate deterministic source-independent Lab augmentation derived from the accepted L14 road graph, never represented as captured Civ III railroad state.
- The candidate contains one connected 72-node / 75-edge railroad network with 15 junctions, ten intentional ends, 14 exact river crossings, six wrap continuations, relief traversal, normal/pillaged states, and road coexistence.
- Ballast, sleepers, paired steel color, and railroad bridge bodies come from the normalized route packs. No trains, signals, traffic, smoke, animation, or invented lights are present.

## Rendering contract and visual review

- Railroad segments use the road graph's exact-node, terrain-conforming curve contract at a narrower calibrated width.
- The authored sleeper/ballast atlas is combined with its separate authored paired-steel strip; the narrower second candidate corrected the first candidate's ladder-like read.
- Railroads are drawn over their selected road backbone, matching Civ-style railroad upgrades while leaving non-upgraded roads legible.
- Railroad bridge bodies use the normalized worked/pillaged models and exact shared-edge river-crossing selection. Apparent route gaps behind relief in isolation are true depth occlusion.
- Native and reduced views retain paired-rail rhythm, continuous junctions, route hierarchy, terrain grounding, and readable crossings. The intentionally bridge-heavy fixture is stress coverage, not a target average density.

## Deterministic evidence

Two complete `python3 Renderer/tools/renderer_dev.py lab` runs produced byte-identical outputs:

- complete: `69c6f30ebe70ec47eafd208f2ab0ff70cd30423589924e3a32cef39f6f72242f`
- reduced: `d76ff4ed45d7d6bab42268885f468b247192dd1215878ac65b1703589dde25f4`
- no railroads: `038328159315349f56e86090bc491ba964d8ba84a8508b34bcd78c12d3bbf93f`
- railroads only: `b9c023d6a0df30c3977444541d3e975c17278f8de4985e80b7d97eff8010d748`
- crossings: `9c843d5d54c2fcea4716d74b7bcb951ba553f3c31c5a6b90727971cd85791d8f`
- Lab scenario: `8cbeccbb1806425654cf19c8a36952b6b710161e6e3f9346ce8476118d9d013b`
- bridge runtime: `30f2784e92ec063701410c45282004a025cb96dc56ad11ce01e3ce14d8b01099`

The no-railroads control is byte-identical to the approved L14 native render.
