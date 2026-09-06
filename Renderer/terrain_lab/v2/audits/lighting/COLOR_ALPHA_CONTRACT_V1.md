# Q6 color, alpha, and lighting boundary v1

Status: Q6 semantic specification adopted by Q0 wire3/4 attachments and per-draw
state. The source-complete city/category matrices exercise the shared Metal HDR
branch. Full real-map composition and native parity remain separately gated.
The immutable `legacy_l21_display_straight_v1` branch remains untouched.

## Attachments and transforms

New branch ID: `q6_scene_linear_premultiplied_v1`.

- Working primaries: Rec.709/sRGB, D65 white; scene-linear float radiance.
- Base-color textures: hardware sRGB decode exactly once. Normal, height,
  roughness, coverage and emissive intensity/mask channels are linear data.
  Colored emissive textures obey their normalized pack's declared color space.
- Scene attachment: RGBA16_FLOAT, linear RGB premultiplied by coverage/opacity;
  clear `(0,0,0,0)`. Do not clamp HDR emission before composition.
- Explicit map validity: separate R8_UNORM coverage attachment, OR an equivalent
  independently preserved coverage field. Half-open valid map rectangle in
  final output pixels `[x0,y0,x1,y1)` is mandatory. Black content is valid;
  never infer validity from luminance. Clear outside the valid map extent.
- Geometry stage: shared normal-driven lighting, direct visibility and bounded
  ambient contact; emission added independently before any exposure or tone map.
- Q1 reconstruction input: scene-linear premultiplied RGBA16_FLOAT plus map
  validity; no tone mapping, transfer encoding, clipping or per-category grading.
- Reconstruct/filter premultiplied RGB AND alpha consistently, retaining
  transparent black at alpha zero. Q1 owns final sampling and sharpening policy.
- After scene composition/reconstruction, unpremultiply where alpha > 1e-6,
  apply ONE shared `exposure = frame.exposure * profile.exposure`, then the
  versioned Q6 shoulder `c / (1 + max(c.r,c.g,c.b))` to nonnegative RGB.
  Reference implementation: `../../shaders/lighting/response_v1.hlsl`.
  This is a bounded, hue-preserving C3X response, not decoded Firaxis behavior.
- Tone-mapped display-linear output may be supplied as a second named attachment
  if Q1 wants a display-stage operation. It must not be called scene-linear.
- Encode once with IEC sRGB piecewise transfer (0.0031308 linear breakpoint),
  then quantize BGRA8_UNORM. Never also use an sRGB target to double-encode.
  Opaque final map pixels have alpha 1. Where fractional final coverage is
  needed, preserve explicit alpha/validity and declare final RGB straight sRGB;
  do not gamma-blend this over retained Civ III art inside the lighting graph.
- Retained Civ III labels, overlays and UI are outside this graph and never
  receive Q6 exposure, tone mapping, bloom, shadows, or sharpening.

## Depth and composition

One shared geometric depth field in authoritative anchor/projection coordinates.
Opaque: depth test/write, blending disabled. Cutout: source alpha threshold
before color AND shadow depth writes; surviving samples opaque, rejected samples
write neither. Resolve multisample coverage only once. No category depth clear.
Equal-depth opaque ties require deterministic stable-ID ordering; otherwise
opaque submission permutation must be byte-identical per backend.

Caster depth/visibility is computed from actual opaque and cutout geometry using
one frame light direction and the documented fixed stylized projection slope.
A receiver's world position and normal, category/material receiver eligibility,
and source baked-occlusion metadata are required. Shadows attenuate direct light
at that receiver, not a dark alpha silhouette blended over the map. Roofs,
facades, ground and raised receivers therefore see the same blocker geometry.
A light-space depth pass/visibility query must not use camera depth as shadow depth.

Decals/routes: depth read, no depth write, explicit surface owner and bounded
receiver bias. Water: opaque bed first; transparent water reads opaque depth,
does not write opaque depth, uses linear premultiplied over with declared
surface opacity. Ambient contact is excluded from water; directional occlusion
is allowed only with material receiver eligibility. Underwater objects remain
subject to water depth/attenuation. Other translucency: deterministic back-to-front
order with stable-ID ties and depth read only. Alpha is coverage/opacity, not emission.

Emissive opaque/cutout surfaces keep their body's depth and source mask. If
emission is a separate pass, depth-equal body visibility and coverage are required;
no light through an occluding building and no whole-wall glow. Lit pixels do not
imply analytic local lights. Unresolved source attachments stay disabled.
Effects follow opaque/water ordering by declared depth semantics; premultiplied
additive emission must not change map coverage. No private category clocks.
Static environment/emissive changes invalidate only on captured-state change and
request zero continuous redraws. Optional bloom is off in initial v1.

## Q0 adoption request

Keep `platform_v1`/frozen packets intact. Add an explicitly selected packet/render
branch carrying attachment format, alpha convention, transfer function, map
validity/rectangle, depth mode (off/read/read-write), blend mode
(off/premultiplied-over/additive), and per-draw shader entry/module. Publish
source-independent mesh world positions/normals/transforms and material caster,
receiver, alpha and emissive semantics rather than preprojected dark silhouettes.
Expose the linear input to Q1 before Q6 display conversion. Do not silently
reinterpret existing `Draw.depth` (currently only boolean) as this new enum.

Executable witnesses in this package are provisional single-pass analytic
reference scenes on the existing Metal runner: opaque boxes, a clipped lattice,
window masks and transparent water. They perform linear scene composition before
one output transform, without requiring a competing backend. Q0 must exercise
these same semantics with actual separate attachments/draws before this shared
boundary passes. The analytic witness alone does not prove hardware blend or
depth-write conformance of the new branch.
