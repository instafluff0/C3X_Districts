# L7 Water Stack Audit

The L7 coast fixtures replace the diagnostic blue slab with three separate meshes that share the L6 coastline exactly.

The seafloor sits below water level and blends the normalized shallows and ocean-bed materials by offshore distance. This confirms that the brown ocean source is a submerged material ingredient, not a finished blue water surface.

The water surface is independently tessellated at water level and alpha-blended over that bed. The current local pack supplies authored large- and small-scale lean-map pairs (`R16G16B16A16_UNORM` slope data plus `R16G16_UNORM` variance data). Terrain Lab combines those two scales for the surface normal and attenuates its restrained sun glint with the variance channel. This replaces the earlier use of the shallow-bed height map as a synthetic wave source. A shallow-to-deep absorption color and increasing offshore opacity still expose the near-shore bed.

A narrow foam mesh begins on the same coast contour. Its width is geometric rather than inferred from a blurred tile blend, and a normalized crash-foam texture breaks its opacity along the shore. Beach, cliff, seafloor, water, and foam therefore cannot drift apart as their shading evolves.

The outer brown seafloor rim visible at the isolated patch boundary is expected lab evidence: the bed is physically lower than the water plane. Adjacent production water geometry will continue beyond the viewport and cover that artificial outer edge.

## Local normalized water bundle

`tools/asset_compiler/water_pack_builder.py` imports 70 renderer-relevant Base textures into generic runtime roles and, when installed, 27 directly useful expansion water/terrain textures. It additionally decodes the three 256x256 expansion height-edit blobs for flooded coast, submerged coast, and flood plains to generic R8 DDS. The set includes four lean-map pairs, surface masks/normals/gloss, all observed profile density/scatter ramps, mist/waterfall/ripple/splash/turbulence/foam effects, beach and cliff support textures, ocean and river-source decals, river clutter channels, snow decals, flood/coast-submergence/flood-plain channels, volcano channels, and the three terrain-water references. It also extracts 20 R8 terrain-element channels: two LODs each for oasis height/blend/region IDs, river-bank noise, and flat/hill/mountain river origins.

The generated DDS and reports remain ignored local artifacts. Tracked code contains only the conversion rules and generic runtime schema; C3X does not ship or require the source assets.
