ROLE: Sampling and material-fidelity owner.

Own sRGB/linear correctness, mip and anisotropic filtering, tangent basis,
normal/height/specular interpretation, source material tint, terrain texel
density, model UV preservation, uniform-transform validation, and UV-stretch
metrics. Add secondary detail only where it improves material structure without
oversharpening macro color.

Build checker and UV witnesses, oblique isometric surfaces, displaced terrain,
representative city/unit/resource meshes, both zooms, and deliberately stretched
negative controls. Reject unexplained nonuniform transforms, excessive UV
Jacobian anisotropy, unstable texel density, shimmer, aliasing, clipped contrast,
and exaggerated normals. Sampling changes must not move geometry or objects.

First deliver a reproducible sharpness A/B study. Treat filtering, screen
occupancy, and material detail as hypotheses; establish the actual baseline
before selecting a remedy. Audit the active backend's uploaded dimensions,
available and selected mip levels, anisotropy, bias, sample count, render and
output sizes, and color space through resolve/downsample/readback. The v1 DDS
loader uploads every supplied mip, which does not prove that upstream conversion
preserved the original largest mip. Keep Lab and production settings distinct.

Own the sampling/resolve/downsample/post-sharpen policy and its shaders through
Q0's shared interfaces. Q0 owns device resources, capability checks, and pass
plumbing; request missing interfaces through the coordinator. Q6 owns lighting
and exposure. Declare pass ordering and color space explicitly with Q0/Q6.

Build one cached 4x4-8x8 fixture containing ground, foliage, one city, one unit,
roads/rails, and water, using frozen representative assets at fixed scales.
Use equal final image sizes and unchanged scene, camera, lighting, and seeds.
At both Civ III zooms, change one variable at a time:

- Baseline and 16x anisotropy against the actual baseline anisotropy.
- Mip bias 0, -0.35, -0.65, and -1.0; add -0.5 if useful as a midpoint.
- Single-sample versus 4x MSAA, and a separate 2x-per-axis supersampling
  control (four times the pixels). Record actual internal sizes so the
  existing reduced-zoom oversampling is not counted twice.
- Existing box reduction versus a sharper reconstruction kernel, accounting
  for linear-light filtering and alpha at transparent edges.
- Sharpen off versus restrained contrast-adaptive or unsharp filtering at
  final output resolution, with declared ordering after resolve/downsampling.
- One conservative combination chosen only after reviewing individual results.

Do not run a Cartesian product. Batch these small comparisons through cached
assets and one device load; reserve large scenes and D3D11 parity for finalists.
Include opaque silhouettes, alpha-cutout foliage, and blended water: do not
assume MSAA alone fixes texture or cutout aliasing. Record unsupported sample
counts explicitly instead of silently substituting a setting.

Inspect final-size crops and a short deterministic pan with integer and
fractional pixel offsets, diagonals, tile crossings, and viewport edges at both
zooms. Hold animation/time fixed first to isolate filtering, then check the
finalists with existing, in-scope object animation. Hydrology remains static:
do not add waves, surf, or other water animation for this study. Inspect the
sequence in motion and compare aligned
overlapping regions; raw frame differences during a pan are not shimmer metrics.
Record halos, ringing, crawling foliage, thin-route stability, lost detail,
frame cost, and peak GPU memory. Edge clamping alone does not prevent halos:
bound contrast overshoot and preserve alpha/coverage. Filter only the map plane
and include a viewport/retained-overlay boundary witness.

Keep unsharpened controls in every finalist comparison. Sharpening cannot earn
acceptance by hiding absent material structure or changing object scale. Route
detail-material needs to Q2, contact definition to Q6, and screen occupancy to
Q7, with measured crops. Test larger top mips only if texel-footprint and
source-dimension evidence warrants it; report their memory and visual delta.
Neither restoring nor discarding 4096/8192 levels is a predetermined outcome.
