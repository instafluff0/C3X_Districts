# Q1 interface response — resolved

Q6 supplied `systems/lighting/color_alpha_v1.json`; Q0 implemented linear
RGBA16F, independent R8 validity, explicit blending/depth, matching MSAA,
linear resolve, shared display output, and Q1 contract2 reconstruction on Metal
and D3D11. The frozen encoded-byte branch remains the default. See
[source_adapters_v1.md](source_adapters_v1.md) for exact bindings and commands.

Q1 independently verified the invalid-pixel reconstruction correction against
raw captures (c033validity-fixed: previous two built-in failures now zero).
Q0 linear-post-01 passed all eight native pairs with MSAA4/scale2/post2 and
stable repeats. Q1 actual-source c029linear-parity passed both zooms with
max channel delta1 and p99 zero. This resolves the earlier attachment blocker;
it does not accept the visual tracks or their final quality policies.

Source mesh/UV/normal/texture and instance calibration metadata now accompanies
frozen packets. Per-material sampler overrides are not implemented; current
Q1 policy uses the declared global controls and owns its shader diagnostics.
