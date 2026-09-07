# Mountain material coherence: provisional visual improvement

The r2 diagnostic uses the same triplanar coordinates for rock color, height
and specular response. Filtered height gradients are transformed into the
actual surface tangent plane, so steep faces retain material relief. Height
amplitude 0.025 world units remains an explicit Lab calibration; it is not a
recovered Civ VI engine setting.

Eight previously absent channels are added: snow height/specular and height/
specular for each of three desert stripe materials. Normalized mountain base
and top height/specular are byte-identical aliases, verified before reuse.
All channels use the same existing height/slope layer weights as color. The
weights themselves remain inherited Lab approximations. Confirmed source
selectors and thresholds are in [GROUND_LAYER_SELECTORS.json](GROUND_LAYER_SELECTORS.json);
they do not establish the engine's exact blend evaluation or unit conversion.

## Changed pixels and composition

[Inland, native zoom-1 crops](out/rock-channels-r2/review/inland-native-z1.png)
has the baseline on the left and r2 on the right, noon above midnight. The
exposed faces of the central range and the isolated mountain beside the pool
have more visible small light/dark relief. These marks follow the projected
rock detail while the silhouettes and cast shadows remain fixed. At night,
the faces retain their form and develop finer relief without darkening the
whole scene. This is a modest improvement toward the granular faces in
canonical `mountain.png`, not a claim of equivalent detail or approval.

The fixed coastal and wilderness views contain only one mountain each, so
their changes are intentionally small. The freshground region has seven and
shows the same effect without local tuning. Its first use was the preceding
ground-patch pass; it is a regression witness here, not a newly unseen region.

- [Freshground full native zoom 2, day/night](out/rock-channels-r2/review/freshground-full-z2.png)
- [Coastal full native zoom 2, day/night](out/rock-channels-r2/review/coastal-full-z2.png)
- [Wilderness full native zoom 2, day/night](out/rock-channels-r2/review/wilderness-full-z2.png)

| Noon / zoom 1 | Changed pixels | Bounds |
| --- | ---: | --- |
| Coastal | 2744 | 630,541–727,626 |
| Inland | 36461 | 163,72–798,560 |
| Wilderness | 2511 | 374,484–476,565 |
| Freshground | 25822 | 413,95–1244,544 |

This diagnostic is retained for further composition, but has not replaced the
preserved overall best. Sixteen matched combined frames cover four regions,
two zooms and noon/midnight. No source terrain, object placement, projection,
output size, material color texture or lighting setting is changed.

## Material coverage discovery

The complete test.biq dataset contains 72 mountain tiles; all 72 have grassland
base terrain, and none have desert base terrain. The shader selects desert
mountains with `input.base_terrain < 0.5`, so the three fixed benchmarks cannot
prove desert stripe appearance. A nearby desert tile does not change that
source field. Do not claim that the wilderness render covers desert mountains.

`rock-desert-witness-baseline` and `rock-desert-witness-r2` explicitly force
desert material selection on unchanged inland geometry. These are synthetic
material tests, not gameplay candidates and not modified BIQ fixtures. Four
matched views exercise the missing channels. The noon zoom-1 comparison changes
23615 pixels within [166,68–795,562]. Source desert heights have less variation
than the gray rock height map; the finer response remains subtle. Texture
presence and branch execution alone do not prove source-equivalent appearance.

## Binding and engineering evidence

`qa/extend_packet_materials.py` appends generic DDS textures to copied packets
and binds them only to unused slots 108..115 in non-feature draws. It refuses
occupied slots, preserves feature bindings, and verifies that every byte of
the original buffer/draw tail is unchanged after undoing the recorded binding
edits. This uses the existing packet format, not a new runtime format. It is a
diagnostic adapter; production needs a named terrain material binding stage
instead of shader aliases named for legacy bridge slots.

`qa/rock_channel_pass.py` resolves the source descriptors, preserves original
packets and renders the candidate. `qa/verify_rock_channels.py` verifies actual
uploaded payloads/bindings in one packet for each region, source/candidate
packet hashes for all sixteen frames, output sizes, changed-pixel bounds and
four byte-identical disabled-branch controls. The evidence is in
[ROCK_CHANNEL_r2_EVIDENCE.json](ROCK_CHANNEL_r2_EVIDENCE.json).

Local GPU samples are inconsistent: inland is 108 ms versus a historical
46 ms baseline, while freshground is 60 ms versus 63 ms. These are not
controlled performance measurements. The gradient shader still performs
multiple height samples per plane and layer; cost needs a controlled check
and likely compiled material gradients or another efficient representation.
No performance, production-binding, crop/wrap or milestone gate is closed.

## Next work

1. Trace `StandardFlat` continental terrain elements together with grass/plains
   high materials. The inherited renderer has analytic rolling ground; the
   old geometry resolver explicitly inventories source packages without
   decoding their geometry. Do not assume that adding high textures alone
   reproduces their placement or the source land form.
2. Complete the effective flat/high/hill material graph, including source
   overrides, blend masks and coherent channels; preserve authoritative Civ
   III tiles and ground anchors.
3. Compose the correction through the normal fixture pipeline, verify world
   crop/wrap stability and practical cost, and inspect an additional unseen
   region before promoting a combined candidate.

The all-applicable-ground-layer requirement remains incomplete. The active
surface-richness goal, LQ0 and all human-review gates remain open. The frozen
Integration preparation is unchanged.
