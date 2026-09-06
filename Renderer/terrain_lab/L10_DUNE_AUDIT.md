# L10 Dune Promotion Audit

## Candidate

The L10 candidate is a deterministic 2560x1440 render of a 12x8,
96-tile Civ III promotion fixture. It retains all accepted L9 terrain,
vegetation, relief, shore, water, and surf layers while adding dune, desert-hill,
and desert-mountain presentation from normalized local assets.

Sixteen contiguous desert cells form one dune field. Broad geometry is evaluated
in world coordinates and therefore does not restart at tile boundaries. The
confirmed Civ VI `DuneAngle`, `DuneWidth`, `DuneHeight`, and `DuneNoise`
relationships guide its scale and direction; broad and fine phase bends turn
the crests into irregular S-curves. The exact unpublished Civ VI engine formula
is not claimed or reconstructed.

The dune surface uses the normalized desert-hills base-color, height, and
specular material. Four overlapping placements sample the real shared
base-color and height textures referenced by the five normalized desert sand
decal variants. The decals add local breakup and micro-normal response only;
they do not masquerade as macro relief. Separate flat-desert cells, eight
authored hill cells, and one authored mountain cell using the normalized desert
base/stripe materials make those categories independently readable.

## Shoreline ownership correction

The accepted smooth grass/sand/water stack is unchanged in construction. For
the L10 promotion layout, its medium-scale bays and points are centered around
the x=10 Civ III grid boundary rather than the middle of the coastal cells.
Medium- and fine-scale contour noise, variable beach width, the separately
perturbed surf envelope, and submerged-bed continuation remain active. This
makes the ten columns west of the boundary read as land and the two columns
east of it read as naval water without reducing the coast to a straight line.

## Evidence

The Windows Direct3D 11 lab builds under `/W4 /WX`. Fifteen focused dune,
generic-decal, vegetation, and shore adapter tests pass. The current BMP hashes
are:

- complete: `5b2a145196afd62e7cacf4b9b0af15d780a0f64e7820a62e817e1e27e9b3bc7d`
- no dunes: `975b9df3a9b7c370bdb56e655d2b43b4f92ed1bf35df78fdcfcd1fe8033f7124`
- dunes only: `200672a204b471ffff08bf1d1506536eeac6133fe023a14eec749e8cd092aeff`
- 640x360 thumbnail: `06477a205ec2f48076e354e7fcf680cee3e558a2b63da2ec6d3042bfcc24de6a`

These hashes establish reproducibility. The user explicitly approved this
complete promotion render on 2026-09-05. L10 is complete and L11 is active.
