# L5 Authored Mountain Audit

The first admitted mountain is `standard/variant_01`. Its three LOD0 channels are shown together in `grass_authored_mountain_standard_01.bmp`: confirmed height at upper left, inferred HBLEND footprint at upper right, and false-colored region IDs at lower left.

The height channel contains one asymmetric multi-ridge massif rather than a radial cone. HBLEND contains a smooth oval footprint. The lab multiplies height by HBLEND as an explicit experiment; this produces a bounded mountain spanning the center of the whole 2x2 patch with no per-tile restart. Geometry normals, directional shadow, and broad occlusion all derive from that same combined field.

Region IDs contain background value 0 and three nonzero values: 115, 128, and 153. Their visual nesting is proven, but their material meanings are not. The lab displays them and deliberately does not guess a mapping.

Mountain material zoning uses confirmed source height thresholds instead: base material at low altitude, mountain-top material through the upper rock, and snow blended from 0.75 to 0.8125 normalized source height. The base mountain material also supplies micro-height and specular response. This is sufficient to prove the visual stack without conflating unconfirmed region-ID semantics with observed package data.
