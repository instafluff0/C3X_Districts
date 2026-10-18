# Unit civ colours and look (Lab, Mac only)

Match each 3D unit's civ-colour sections to its paired Civ III sprite, and
try lighting looks that read like Civ III units. Started 2026-10-11 with
Infantry, Tank, Rifleman, Knight, Battleship and Bomber; to expand to all
units once the method is accepted. No pack, staged or game files change.

```sh
python3 Renderer/lab/studies/unit_civ_colors/transfer.py [--units PRTO_Infantry,...] [--civ 1]
python3 Renderer/lab/studies/unit_civ_colors/look.py [--units ...] [--looks now,proposed,...] [--zoom 2] [--out look]
```

`survey.py` runs both on every unit in the pack (`--jobs`, `--polite` pauses
while a VM game capture runs, `--units` redoes single units) and writes
`survey-NN.png` contact sheets: Civ III | now | proposed, SW and NE, at map
scale. Each unit is paired with its art through its binding's Civ III keys,
the BIQ art folder or the first era variant (Leader, Army). The proposal
keeps the pack's colours under the new look when placement cannot carry over:
the sprite/model silhouette overlap is under 0.45 (a Huey for Civ III's
Chinook), or no consistent area reaches a quarter of Civ III's share
(one-material hulls and missiles: Nuclear Submarine, ICBM, Nuke). First full
run (2026-10-11): 72 of 78 transferred, 6 kept.

Outputs go to `Renderer/lab/out/unit-civ-colors/`: `review.png` (Civ III |
current | proposed, SW and NE), `gallery*.png` (four directions plus the
evidence field), `look*.png` and their JSON statistics.

## Civ colour transfer (`transfer.py`)

The user asked for close and consistent colour placement, not spotty or
pixel-exact. The method is generic: no unit names select behaviour.

- **Civness:** how much each palette index varies across the 32 `ntpNN`
  palettes. Indices 0-15 are the civ ramp, even 16-63 are civ blends, odd
  16-63 are fixed neutrals.
- **Fit:** in each of the eight directions, the DEFAULT sprite is fitted to
  the idle pose at the live projection: per-axis scale and shift by
  silhouette overlap, then coarse and fine control-grid warps on silhouettes
  and edges.
- **Sampling:** every texel is a surface point. Each view that sees it
  samples the sprite, weighted by facing to the eighth power, so the most
  frontal view decides.
- **Consistency:**
  - mirror pooling (left/right);
  - light surface smoothing within a material;
  - per mesh piece, either a ring by height (upright pieces with evidence
    all round; walls only) or bands along the piece's longest axis;
  - 30% of the evidence comes from the UV-island and colour segment;
  - a small prior toward the source asset's authored owner mask
    (`UnitAnimationRuntime`).
- **Threshold:** a scanned threshold makes the visible share equal Civ III's.
  Hysteresis growth completes connected areas.
- **Look:** under the mask the albedo is flattened, so the owner ramp reads as
  one colour.

Current state:

- **Close:** infantry tunic only, rifleman coat, knight tabard and horse
  cloths, battleship bow, stern and bridge, and a band around the tank turret.
- **Still weak:**
  - the tank's front marks;
  - the bomber's wings. The 3D B-24's long wings don't meet the B-17
    sprite's wingtip marks; a stronger warp smears marks.

## Look (`look.py`)

NumPy preview of the live unit material and the unit BeautyFrame:

- **Key light:** follows the sun/moon azimuth at slope 1.35, 2.05 x warm
  (1, .73, .57).
- **Sky:** .62 x cool (.34, .45, .60).
- **Rest of the response:** fixed view, owner ramp, AO, GGX, rim, look
  gain/saturation and roll-off, then the filmic display.
- **Missing:** no self-shadow or normal maps.
- **Calibration:** 0.45, fitted to the D3D render
  `lab/out/unit-readability/tone-new` (noon, SW); about +/-12 luma per unit.

Measured on body pixels at map scale (display 0-255):

- **Civ III:** luma about 90, std about 45, saturation .3-.5, no dark
  outline, key from the upper right.
- **Production at noon:** similar luma and spread. Softer, more saturated, and
  warm-tinted (greys turn pink).
- **Production at dusk and night:** luma drops to 35-55 and 16-38.

The candidate (`PROPOSED`) combines:

- unsharp masking of the unit layer (0.6);
- a key and sky moved 60% toward neutral;
- a light floor of 75% of noon at any hour, keeping the hour's tint;
- AO in the key;
- stronger specular with a higher knee;
- no extra saturation;
- optionally a 35% one-pixel outline.

It keeps noon luma, raises contrast and rim separation, and keeps units
legible at dusk and night. A port would change `prepare_unit_shader.py`, the
unit frame values and the unit-layer compositor, so it needs user acceptance
and in-game verification.
