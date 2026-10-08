# Unit readability study

Lab check of units against Civ III's own sprites (2026-10-07): shadow artifacts,
sizes, brightness and ownership. The shadow fix, Civ III-matched sizes and
look A were accepted and promoted into the pack pipeline and renderer; a soft
owner disc (preferred over a thin vanilla-style ring) answers Civ III's
team-colour disc preference.

## Findings

Naval and air (accepted 2026-10-07): ships float at their authored waterline
(hull below it clipped; sized on the visible area); aircraft hover at half of
Civ III's body-to-shadow gap (23-36 px at full). Whole-vessel tint looked
silly, so hulls carry a side stripe and pennants and aircraft their wing and
tail tips. The owner ramp no longer brightens a civ colour past itself. Sea
reflections stay at the shared water's noon Fresnel (about 4%), so unit
reflections are faint; any boost belongs to the sea's object-reflection weight.

Measured at 128-pixel tiles, idle, median over eight directions.

- Shadow: every part (and face) of a unit drew its own 0.28 ground shadow, so
  overlaps compounded into dark patches (GPU-measured 0.259 vs 0.360 for one
  layer). Now one shadow per unit, at the shared dynamic-shadow strength.
- Size: the old fit put each unit's largest idle extent at 56 px, weapons and
  mounts included. Foot units were 1.2-1.4x Civ III's linear size; mounted,
  siege and ships 0.4-0.85x, with same-class pairs disagreeing.
- Brightness: median body luma 52 against Civ III's 89 (owner palette ntp01).
- Ownership: civ colour covered about 5% of a unit against Civ III's 24%; several
  modern units showed none. Look A brings coverage to 24% and luma to 63.

## Tools

From the repository root (NumPy; PNGs are written without Pillow):

```sh
python3 Renderer/lab/studies/unit_readability/audit.py          # sizes vs Civ III
python3 Renderer/lab/studies/unit_readability/civ3_sheet.py --civ 1
python3 Renderer/lab/studies/unit_readability/sheet.py build     # x64 live-path fixture (VM)
python3 Renderer/lab/studies/unit_readability/sheet.py render LABEL [--pack P] [--owner HEX] [--env K=V ...]
python3 Renderer/lab/studies/unit_readability/sheet.py compose OUT.png LABEL ... --civ3 1
```

`sheet.py render --reflect` also writes the production reflected pass and
`compose --sea` places sheets on an in-game ocean patch with the water shader's
reflection weight. `prepare_units.py --quality FILE --output PACK` builds a
candidate pack; `run_scripted_game_test.ps1 -Scenario units -UnitPack PACK`
captures it in the 1498 AD save without installing it (staged renderer).

`unit_sheet.cpp` loads a unit pack and calls the production `prepare_real` and
`draw_real` into the transparent unit layer; `sheet.py` composites it and applies
the production display transfer. The Lab `units` category instead previews the
older sprite path. Lab overrides: `C3X_RENDERER_UNIT_LOOK=gain,saturation,owner`
and `C3X_RENDERER_UNIT_OWNER_RINGS=1`. Outputs stay in `lab/out/unit-readability/`.

The Civ III FLC reader (`Renderer/tools/asset_compiler/civ3_flc.py`) is ported
from the C3X Editor decoder. `UnitSizingLab` is the disposable pack that was
reviewed; production sizing now comes from `prepare_units.py`.
