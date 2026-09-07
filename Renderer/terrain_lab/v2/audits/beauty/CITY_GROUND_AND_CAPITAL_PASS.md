# Single-era ground and palace composition

Status: partial visual improvement and composition diagnostics. The city goal,
human review and all Lab/integration gates remain open. r8 night, r11 growth and
r13 capital evidence remain preserved; r17 mixed-era appearance remains rejected
by the user. This pass does not replace the selected overall city best.

## r18: source paving under every selected building

`qa/prepare_city_ground_probe.py --pool` recovers exact descriptor-0 triangles
and atlas UVs for all 12 selected components in each of the American modern and
European medieval pools. Package readers are reused and existing normalized DDS
files are referenced; no new texture copies are needed. Descriptor state
selection and source height/specular behavior remain unproven.

Both rendered cities retain all seven building bodies, sizes, rotations and
positions. The same 100-tile coastal terrain, cameras, noon/midnight, output sizes
and both gameplay zooms are retained. Modern r18 adds ground beneath the six
standalone buildings that r16 omitted. Medieval r18 compares with r8.

The medieval source pads extend beyond the buildings' legal dry footprints.
The initial pre-render failure exposed 275 wet input vertices. These are ground
overhangs, not permission to move buildings into water. The producer now clips
each finely tessellated ground triangle to the sampled signed shore boundary
at -0.02, interpolating position, depth and UV attributes. This is a local linear
approximation on cells of about 0.06 tiles, not a recovered Civ VI algorithm.
Dry building and foundation checks remain unchanged. Ground receives shadows,
does not cast them and does not write depth.

At normal gameplay size, modern paving changes 672 noon / 595 night pixels;
medieval paving changes 767 / 613. The changed pixels sit around building bases.
They introduce short paving edges and reduce some immediate grass-to-wall joins.
Most pad area is hidden by buildings, so this does not recover the broad urban
ground coverage visible in the reference. Other scene pixels remain unchanged
within 1/255. The fixed interior lake sample is unchanged.

- [Modern, actual gameplay size](out/city-scene-r18/review/all-ground-native.png)
- [Medieval, actual gameplay size](out/city-scene-r18/review/medieval-ground-native.png)

## r19: first combined use of the broader palace pack

The completed [47-root intake](../../../../docs/city_palace_asset_import.md)
is reused directly. A per-style pack override maps the modern American Lab
capital to `city/palace/root/0d0c35f4a4c9651a` in `CityPalacesNormalized`.
The surrounding house pool stays entirely modern. Source culture selectors are
provenance; runtime city style and capital status still require explicit generic
mapping and authoritative Civ III state. No universal palace mapping is implied.
The Gran Colombian tree-child limitation remains open.

The existing 0.8-tile half-extent failed to fit the palace and seven city pieces.
An explicit 0.95 half-extent, within the user's permitted modest cross-tile
overlap, also failed at the initial palace site. The actual dependency was greedy
placement: a legal palace position stranded the last tower. Bounded alternative
palace-site search now tries up to 25 deterministic sites without changing
building scale, order, count, dry-land limits or collision checks. The third site
fits this fixture. The no-palace control reserves exactly that same site and
retains identical surrounding buildings.

The palace changes 908 noon / 1,062 night gameplay-size pixels, including its
distinct dome and window material. Foreground towers obscure much of its facade.
The wider arrangement is less coherent than the earlier ordinary city and is
**not a new visual best**. The palace changes no pixels beyond 1/255 in the fixed
interior lake sample, so this is not evidence of improved palace reflections.
Existing r8 city-window reflection evidence remains separate and valid.

[Matched palace/control comparison](out/city-scene-r19/review/capital-control-native.png)

## Next three visible gaps

1. Palace visibility and compact city organization: score its visibility within
   the combined composition, rather than accepting the first collision-free fit.
   Preserve the source bodies and single-era house selection.
2. Coherent ground coverage between buildings: individual source pads are largely
   occluded. Investigate recovered grounding-material and block/filler metadata
   together before scaling arbitrary pads. Connecting roads remain deferred.
3. Material and night richness: facades still read flat; local light pools and
   complete material-channel interpretation remain unresolved. Brighter emission
   alone does not address these defects.

The next combined acceptance pass must also include size changes, other cultures,
and a fresh terrain region. These two previously used coastal anchors are not
an untuned-region acceptance witness. Do not claim full benchmark coverage.

## Verification and storage

Six focused shoreline-geometry/palace-importer tests pass. The analytic clipping
check verifies retained area and atlas parameterization at an oblique triangle
boundary. Four medieval ground and four palace frames pass standalone Windows
D3D11 comparisons. These are supporting implementation checks, not visual
acceptance. No injected compile or milestone closure occurred; this pass makes
no full Lab pass claim.

`qa/city_ground_capital_evidence.py` rechecks matching inputs/buildings, localized
pixel changes, lake samples and all eight parity results. The retained report is
[CITY_GROUND_CAPITAL_r18_r19_EVIDENCE.json](CITY_GROUND_CAPITAL_r18_r19_EVIDENCE.json).
Each render uses shared-resource packets and removes the disposable pre-shadow
packet. Source ground maps are ignored regenerable local derivatives. The 8 GiB
free-space guard remains enabled; no broad capture matrix or duplicate palace
import was run.

After parity and pixel checks, the regenerable r18/r19 linear GPU intermediates
were removed (174.3 MiB reclaimed). PNG/BMP references, validity masks, source
inputs, replay packets and shared blobs remain. Evidence revalidation passes
after cleanup; retained previous-best revisions were not altered.

## Subsequent composition work

[r21/r22](CITY_COMPOSITION_PASS.md) improves the American palace facade and
combines the new layout with source paving. It also resolves categorical water
tiles with a negative smoothed shore through exact dry-cell clipping. The r19
first-legal layout remains a preserved diagnostic rather than the current
capital-composition candidate.
