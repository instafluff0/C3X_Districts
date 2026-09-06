ROLE: Continuous terrain-material and universal-blending owner.

Own the continuous base surface, terrain material weights, shared-world noise,
detail/clutter interface, wrap-safe sampling, and an explicit transition rule
for every reachable terrain-family pair. Do not own shore water, rivers, raised
relief bodies, or shared lighting.

Generate the exhaustive unordered pair matrix across every effective terrain
family, both Civ III adjacency axes, reversed ownership, wrap aliases, and
selected three-way junctions. Seamless means an intentional smooth, shoulder,
shore-mediated, or hard-material transition with no crack, exposed diamond,
independently randomized edge, or raw-coordinate jump. Require matching shared
edge height, normal, and weights, no tile-frequency grid signal, and stable
world-space detail at both zooms.

