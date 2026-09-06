# L6 Coast Topology Audit

The lab now has two coast fixtures driven by explicit continuous contour functions rather than a bilinear terrain scalar plus periodic edge warp.

`coast_straight_beach.bmp` uses one gently curved contour. Inland grass, a sloped beach strip, and diagnostic water are tessellated independently but share the same boundary coordinates. The beach uses the normalized Civ VI beach base, height, and specular material channels.

`coast_corner_cliff.bmp` uses a monotonic smoothstep bend to form a rounded corner without repeated scallops. Elevated land ends at the contour, the water begins at the same contour at water level, and a vertical cliff wall connects those exact vertices. Cliff UVs advance along the contour and descend from rim to waterline, avoiding the vertical texture streaks produced by ordinary ground-plane UVs.

The water is intentionally a flat diagnostic color. No claim about water quality belongs to L6; seafloor, transparency, depth response, Fresnel, waves, reflection, and foam are isolated in L7. Likewise, the cliff is a clean structural wall rather than final authored rock clutter. L8 may break that silhouette with normalized rock assets after the water boundary is stable.
