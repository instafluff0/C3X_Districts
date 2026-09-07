# One shadow frame, with explicit posed-object gaps

The audit confirms a real direction mismatch, not just missing darkness.
`package.py shadows` compiles the actual native environment evaluator and unit
shadow header alongside the actual Lab Q6 shadow-frame builder. At noon, for a
half-unit-high marker at normal zoom, current unit ground displacement is
**(-50, +25) px**, while the same marker converted to the pickup world projects
**(+38.7147, -19.3573) px**. The directions are opposite. Midnight reverses both.
Sunrise/sunset also differ; see `manifest.shadow_probe` for all eight phase/zoom
rows. The probe performs 64 phase/zoom/yaw checks. This is coordinate evidence,
not a newly rendered combined visual acceptance result.

## Actual coverage at the pinned native baseline

| Category | Face illumination | Self occlusion | Cast / receive |
| --- | --- | --- | --- |
| Static land, relief, trees, cliffs and eligible static features | Shared sun/moon normals | Shared source-depth field | Alpha-aware static source casters and world receivers; bounded cached pages |
| Static resources | Shared normals | Eligible in static source field | Same static feature path |
| Animated resources | Shared normals, posed geometry | No independently updated posed self-shadow | Receive the existing static field. `submit_geometry` excludes `chunk.animation_texture` from casters, so no posed silhouette reaches neighbors/terrain. |
| Supported native animated units | Shared sun/moon normals in unit-local basis | Posed 128×128 max-height field, 3×3 filter, .006 bias | Flat local ground only; up to .48 × frame shadow strength, 3-pixel edge fade. No terrain/object receiving, cross-unit occlusion or casting onto relief. |
| New city Lab pipeline | Extended material normals and local facade light | Q6 source caster/receiver composition | Bodies/paving/terrain share source frame; central transform reaches all geometry. Broader style/wall coverage remains open. |
| Legacy raised infrastructure | Authored-normal face response | Do not infer geometric self occlusion from the historical word “self shading” | Source-mesh cast projection in Lab; must join the common field when ported. |
| Territorial ribbons, pollution and crater ground art | Ground material treatment | Not raised casters | Receive appropriately on their surface; do not invent elevated shadows. |

Unit body alpha-cutout is applied in the color shader, but its caster rasterizer
currently fills mesh triangles without sampling material alpha. Preserve this
as a known limitation until coverage is supplied. Arbitrary translucent shadow
transport is not implemented; cutout, opaque and non-casting translucent roles
must be explicit. Emission and ambient illumination must not be erased by a
direct-light shadow term.

## Canonical math and adapter

The candidate helper [directional_shadow_contract.h](directional_shadow_contract.h)
contains executable, source-independent math. It is **not wired into native
production by this preparation** and changes no selected Lab pixels.

Keep pickup/Lab Q6's intensity-weighted sun/moon horizontal direction:

```text
h = sun.xy * sun.intensity + moon.xy * moon.intensity
d = normalize(h) if length(h) > 1e-6, otherwise (-1, 0)
L = normalize(d.x, d.y, 1.35)       // toward the light, fixed stylized slope
U = normalize(cross((0,0,1), L))
V = cross(L, U)
receiver.xy = caster.xy - L.xy/L.z * (caster.z - receiver.z)
```

This matches `systems/lighting/shadow_field_v1.h::build_shadow_frame` and
`native/c3x_renderer.cpp`'s pickup basis. The unit renderer currently chooses
the higher-intensity light and independently uses slope `150/96`; neither its
light selection nor its coordinate convention is interchangeable with Q6.
Do not merely replace the slope while keeping the reflected Y axis wrong.

Current unit and resource posed-local geometry projects as:

```text
sx = anchor_x + (x-y) * W/2
sy = anchor_y + (x+y) * H/2 - z * 150 * W/224
```

The pickup world uses:

```text
sx = origin_x + (u+v) * W/2
sy = origin_y + (u-v) * H/2 - w * 112 * .82 * W/224
k = 150 / (112 * .82)
(u,v,w) = canonical_anchor + (x, -y, k*z)
normal_world = normalize(normal_local.x, -normal_local.y, normal_local.z/k)
```

Apply skeletal pose, asset scale and heading before this adapter; transform
normals by the inverse transpose. The probe verifies pixel projection equality,
tangent/normal orthogonality and heading-independent cast direction. Translation
and ground height come from the captured authoritative anchor, never a private
camera reconstruction. Resources already construct receiver coordinates as
`(owner_u+u+lx, owner_v+1-v-ly, (ground+2.5+lz*150/.82)/112)`; preserve the owner,
tile offset and ground terms when adding posed casters. Units require an explicit
world anchor/receiver association beyond their current flat sprite-local plane.

The shared environment's existing comment about noon casting down-screen is
written for another coordinate basis; code/projection evidence takes precedence
over that comment. This preparation retains the already-selected terrain field
as the common target. If global artistic direction is later changed, change it
once for every category and rerender all fixed phases; do not compensate units
by rotating only their visible shadows.

## Bounded integration work orders

1. **S1, frame adapter:** port the common weighted light and pose-to-world
   transform. Feed lighting normals and self/cast receivers in the same basis.
   Keep the native unit crop/background/dirty-bounds fixes. Compare actual unit,
   resource, tree and mountain shadow vectors in one scene at 00/06/12/18 hours,
   both zooms, all eight unit headings. Numerical probe parity alone is insufficient.
2. **S2, dynamic resource shadow field:** use each current posed triangle and
   its alpha/material coverage. Add a bounded dynamic field beside static pages,
   sample it for posed self-shadow and local terrain/object receivers. Do not
   rebake the entire terrain every 15 Hz animation quantum. Preserve resource
   subject transforms (including all calibrated fish/whale members).
3. **S3, units and shared receivers:** use actual posed kit/tool triangles,
   authoritative world association and same coverage policy. Combine static and
   dynamic visibility without double-darkening. Receive on relief and nearby
   objects and handle cross-unit casts inside the bounded influence region.
   Terrain and caster-only tiles must not acquire false replacement claims.
4. **S4, retained compositing and caching:** rebuild affected dynamic receiver
   overlays from unchanged linear/MSAA backdrop data; invalidate old ∪ new
   pose/body/cast bounds on move/removal. Keys include source/material revision,
   action/pose quantum, heading, canonical transform, environment and relevant
   receiver/caster revisions. Screen translation alone remains reusable.

Keep memory caps explicit and measured. The unit sprite path currently resolves
coverage against a native background and copies a keyed result later in the
animator plane; it cannot cast onto arbitrary terrain by simply drawing one
larger flat shadow bitmap. Integration must choose a bounded world receiver
overlay that respects map/animator ordering and underlying retained layers.
Use a declared no-shadow/fallback behavior when inputs cannot be resolved;
never infer absent world data or create a competing presenter.

Required visual controls: caster-off, receiver-off, cutout leaf/cloth holes,
raised/slope receiver, adjacent moving bodies, different headings, day/night,
both zooms, scroll/translation, wrap, occluded/removed objects and skipped pose
time. Images must show changed shadow pixels without changing unrelated terrain.
Compare cached and cold redraws and measure frame/memory cost against the pinned
native baseline. S1–S4 are open native implementation items, not completed by
the contract header or existing unit shadow smoke tests.
