# L9.5 Shoreline Revision Audit

## Source recipe

The standard Civ VI coast is not a single beach bitmap and is not the sum of
every shore-adjacent texture in the installation. `TerrainStyle.artdef` assigns
the Beach terrain material to the land edge and the Shallows material to a
25-unit coast profile. It records shallow heights of -3.5 and -10, material
cutoffs of 0.1 and 0.8, and position roughness of 0.1 for a neutral beach and
0.3 for a convex beach. `Water.artdef` selects `Water/Coast` and supplies the
0.001 Fresnel F0, exponent 4, refraction scale 0.4, dynamic specular exponent
850, and sun specular exponent 5000. `Wave.artdef` supplies a 20-pixel wave
width, 128-pixel length, and 8-pixel crash distance.

L9.5 uses the normalized Beach and Shallows base-color, height, and specular
channels; coast dark/scatter density profiles; large, small, and secondary
LEAN pairs; tiling and non-tiling normal/variance pairs; gloss and surface
masks; and crash-foam, ripple, and turbulence effects. The beach blanket/FOW
pair, Expansion 2 submerged-coast textures,
ocean decal, cliffs, and rocks are excluded because they are not part of the
standard smooth-beach recipe requested for this gate.

## Transition correction

The rejected draft used one smooth contour and clean constant-width strips.
It also began the seafloor eight height units below the beach at the shoreline,
which exposed the clear background through the transparent water as a dark
ribbon.

The revision gives the land/beach edge, beach width, and surf envelope separate
restrained contour roughness. Beach material height perturbs the grass-to-sand
mix, the beach material continues into the first part of the submerged bed,
and the shallow bed now slopes continuously from waterline height to the
authored -10 low contour. The water density ramps control shallow transmission
and offshore opacity instead of multiplying the shallow coast by an arbitrary
dark tint.

The follow-up material pass centers its erosion inputs on the measured Beach
and Grassland height-channel means (approximately 0.426 and 0.421), blends a
small amount of the real Shallows material into dry and wet beach, compresses
the coast density-ramp domain to optical rather than full mesh depth, and
reaches opaque muted water sooner offshore. A dedicated no-surf render permits
the grass, beach, submerged sand, and coast-water transition to be judged
without wave foam.

Following visual review, the grass-to-sand transition returns to the preferred
wider smoothstep while retaining height-driven boundary erosion. The water
surface now combines the real tiling/non-tiling normal pairs with the three
LEAN scales, and it uses those same slopes to refract a deterministic phase of
the Shallows/Ocean bed. That phase deliberately exposes one representative
authored shallow-rock cluster in the fixture. The water remains transparent
enough for the bed base color and strengthened material-height response to read
through the non-foam surface.

## Verification

The Windows Direct3D 11 lab builds cleanly under `/W4 /WX`. Nine focused
clutter and shore extractor tests pass. Current deterministic BMP hashes are:

- complete: `4dbfaf0a0c249142deefbc3d2f6169616c5303ab27576328cf1f531c4b73a600`
- no vegetation: `0202c9d14b8beaa1e7eca381a8790ee0d58d77d04850332ddb937850964fb9ba`
- no water: `646042e61bf482c5ac931c9b9e273bc6b34ce3e6bb07dea540f7a48ee97cb50e`
- no surf: `d397aa78a72baedb4ce53be4d6c3f9865ef0323a1dff019f16e62b9d46a772a1`
- thumbnail: `f53faa4c7346d88714ef7865f7266abb32554aa13e859ba7df7eadc3ca8ea2b3`

These hashes establish reproducibility, not acceptance. The shoreline direction
has now advanced into the 48-tile L9 promotion candidate documented in
`L9_PROMOTION_AUDIT.md`; that combined scene remains at the explicit visual
approval checkpoint.
