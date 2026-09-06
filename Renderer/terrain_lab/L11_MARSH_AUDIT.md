# L11 Marsh Promotion Audit

Status: complete; explicitly approved by the user on 2026-09-05.

## Authoritative map input

- Windows source: `C:\Program Files (x86)\GOG Galaxy\Games\Civilization III Complete\Conquests\Scenarios\test.biq`
- Source size: 17,173 bytes (compressed BIQ; 254,089 bytes after parser decompression)
- Source SHA-256: `a6a88d7fffcc567c3500bbd5aa947398dd48170d4f412aa1e518bb45ffe8453e`
- Read-only parser result: 100x100 Civ III map dimensions, 5,000 TILE records
- The similarly named Mac-side scenario file is not used.

## Deterministic 96-tile selection

`PREPARE_BIQ_VIEWPORT.bat` invokes the C3X Editor parser and asks the selector for a marsh-bearing 12x8 `diamond` window. The two local axes are raw BIQ `(x,y)` steps `(1,1)` and `(1,-1)`, which are Civ III edge adjacencies. This produces a 96-cell parallelogram/diamond instead of a rectangular stack of display rows.

- User-selected raw BIQ origin: `(53,55)`
- Local-to-source mapping: `sourceX = 53 + column + row`; `sourceY = 55 + column - row`
- Covered raw coordinate bounds: x 53..71, y 48..66
- Export fixture SHA-256: `b59b3310c52768a0e2b065f839be309c22ff6b6bd8abd82f82c287352f7a9b3f`
- Real-terrain counts: desert 6, plains 3, grassland 14, floodplain 2, hills 14, mountains 14, forest 21, jungle 0, marsh 7, coast 10, sea 5
- Base-terrain counts: desert 8, plains 8, grassland 65, coast 10, sea 5

The exact requested window contains no BIQ jungle cells. A post-approval L12
vegetation calibration re-used this authoritative viewport as the forest
witness: 36 smaller authored forest bodies per cell form a closed canopy while
remaining clearly shorter than the adjacent mountains. The L12 Mesoamerica
viewport supplies the complementary jungle witness.

The fixture format carries exactly 96 visible cells plus a read-only two-cell BIQ adjacency halo. Halo cells do not enlarge the rendered or promoted footprint; they preserve the authoritative off-crop terrain needed by filtered material and shoreline samples. This matters at visible cell `(11,4)`: inside the crop it appears to be a lone coast cell, but its native BIQ neighbors continue the water beyond the right edge. Treating the crop boundary as land incorrectly produced a round pond; the halo makes the contour open toward the real off-crop water.

Feature ownership remains exactly the BIQ value for each cell. Ground materials blend near compatible shared edges; relief tapers only where its source feature stops; shoreline width varies continuously while the land/naval transition remains near the shared grid edge.

The first render exposed two false discontinuities. Material UVs were derived independently from raw BIQ coordinates, so even identical grass or water families restarted their texture phase at local cell edges. A later audit found that the first attempted continuous coordinates still mapped the second BIQ adjacency axis as `row+v`; its actual shared vertices are `v=0` on one tile and `v=1` on the next, so that expression jumped by two cells. The corrected surface uses `row+(1-v)` for material, water, and world-space relief coordinates, giving identical values on both types of Civ III shared edge. Coast/sea optical depth is reconstructed by smooth interpolation between BIQ tile centers rather than retaining a flat per-cell interior. The diagnostic tile grid has zero raster width in the beauty candidate because depth-hidden fragments looked like arbitrary cracks. Finally, land, submerged bed, and transparent water meet at the same 2.5-pixel geometric datum; water depth remains in the material instead of exposing the black clear color through an unrendered vertical slit.

That continuity correction still left the visible shoreline assembled from four independent edge curves. They matched along an edge but met as visible points at tile corners. A later revision built one viewport-wide land/water scalar, but still emitted ground only for land-owned diamonds and bed/water only for water-owned diamonds. The scalar was continuous while the draw coverage was not, so later same-depth triangles forced the visible result back onto Civ III tile edges. The current candidate emits pass-major ground, bed, and water coverage across all 96 cells, then clips the bed and water per pixel against the single signed contour. Ground remains underneath the translucent shallows, and optical depth is derived from the signed field rather than from the owning tile family.

The common contour is built from the filtered BIQ ownership field and perturbed continuously in authoritative source coordinates. Cell-center anchors keep every land/naval assignment visually legible, including a bounded round basin for an isolated coast cell, while shared edges are free to form broad coves and points. Shore distance and optical water depth are separate vertex fields: the former drives the grassland/plains/desert/marsh-to-Beach blend and narrow source-authored foam, while the latter drives Shallows/ocean-bed visibility, coast density/scatter, opacity, and the multi-scale surface response. Foam is composited into the continuous water surface from the extracted crash, ripple, and turbulence channels rather than emitted as per-tile geometry. Land-material weights are likewise reconstructed from a shared, source-coordinate-warped center field, so both incident cells evaluate the identical mixture at their edge without exposing a straight diamond boundary.

The first `(53,55)` hill candidate treated the four normalized hill families as interchangeable shape variants and mapped a large crop independently into every cell. It was rejected because this promoted source micro-relief into repeated clusters of sharp miniature peaks. All fourteen hills in the locked BIQ window have grassland as their authoritative base terrain, so the revised candidate samples only the normalized `standard` hill heightfield. A nine-tap source-space low-pass separates its real macro landform from fine breakup, and world-space coordinates keep adjacent hill cells continuous. A calibrated `0.22..0.60` remap suppresses the positive floor of the normalized source field, allowing its authored irregular macro contours—not the BIQ ownership diamond—to define the visible hill body. A rounded continuous support field lets the authored skirt cross onto adjacent compatible land while tapering to the common datum at water, mountain, and viewport boundaries; exact hill ownership remains separately encoded by the BIQ. Flat source-material underlays cover the projected skirts without black voids. The other normalized hill families remain available but unused here because their exact Civ VI selection relationship is unresolved; they are not mislabeled as random shape variants.

The first `(53,55)` mountain candidate reused only standard variant 02 at 150 source-height pixels; it was rejected because its low shoulders were removed by a multiplied height/blend threshold and all fourteen cells read as identical narrow snow cones. The revised candidate deterministically distributes all five normalized standard mountain variants (3/4/1/4/2 instances), preserves their authored height and blend channels independently, expands the central source footprint to fit the Civ III 2:1 diamond, and reduces vertical amplitude to 104 source-height pixels. The resulting bodies retain the actual asymmetric multi-peak silhouettes, full rocky skirts, and height/slope-controlled snow instead of interpolating replacement shapes. The terrain pass writes height-aware depth and the vegetation pass reuses it, preventing later neighboring cells or feature bodies from clipping raised mountain geometry. Flat source-material underlays beneath raised relief prevent exposed background at the outer viewport edge. The later BIQ vegetation calibration uses more, smaller authored bodies on a deterministic jittered lattice: forest scale `0.42` with 36 instances per cell and jungle scale `0.40` with 49. This closes random canopy holes without enlarging individual trees or palms relative to the mountains.

## Marsh source truth

The marsh ground uses the normalized Civ VI `GrassMarsh` base-color, height, and specular channels. Surface breakup uses the reachable `CLUTTER_MARSH` projected-decal records and their shared source base-color, height, and specular payload. The audited source defines no conventional marsh reed/grass mesh set, so the preview invents none.

## Candidate outputs

- Complete SHA-256: `1b954f2190ceda40c13ac8b0f5a8c01af6a03d0633e2938d37bb67e1d7002544`
- No-marsh SHA-256: `74929466816b5a7fa15c202d2f56119408096bf316cd09f39bb5aa037eb17ea5`
- Marsh-only SHA-256: `e159aa30e8d319f0ae909793c3cc999bbd81806a179f989aaca31685f34b415c`
- Thumbnail SHA-256: `59904531efd9fe5ea2667f3defa8f99083f10855721221d4629f853ee399405c`

Two consecutive L11 batch runs produced these exact hashes. `test_continuous_surface_contract.py` also enforces pass-major BIQ surface ordering, contour clipping for both bed and water, shared-coordinate material reconstruction, and tile-independent optical depth; `test_export_biq_terrain_scene.js` proves the visible count remains exactly 96 and the adjacency halo is deterministic. These establish repeatability and the structural continuity contract. The user explicitly approved the complete L11 promotion render on 2026-09-05, including the corrected off-crop water continuation at the lower-right edge.
