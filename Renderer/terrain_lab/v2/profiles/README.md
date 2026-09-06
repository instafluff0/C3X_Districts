# Lab v2 Profiles

Visual policy that applies across systems belongs in versioned data profiles,
not scattered per-model constants. LQ0 establishes validated envelopes for:

- sampling, texel density, anisotropy, mip bias, and UV distortion;
- category-relative scale, footprint, height, grounding, and label clearance;
- building and resource presentation restricted to southwest or southeast;
- shared lighting, shadow, depth, transparency, and emissive behavior;
- quick, system, composition, and promotion quality presets.

Track-specific values live inside the owning track directory after LQ0. A track
may request a shared-profile change but may not silently override it.

