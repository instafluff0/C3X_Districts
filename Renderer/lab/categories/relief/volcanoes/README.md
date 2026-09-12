# Volcanoes

Ordinary Civ III terrain volcanoes (`real_terrain_type == 10`). Smoke and particle
attachments are excluded. The user accepted the `skin-probe` gameplay appearance
on 2026-09-12: "looks fantastic in terms of texture, shape, and everything" and
"Let's move forward with this." Preserve its shape, texture, inherited fine detail,
crater and mountain join. Production generalization remains pending; the accepted
Lab appearance is not yet a general runtime implementation. No fixed reference
was replaced. Natural-wonder volcanoes remain deferred.

```sh
python3 Renderer/renderer.py lab volcanoes
python3 Renderer/renderer.py test volcanoes
python3 Renderer/renderer.py compare volcanoes
```

Every review case retains the adjacent mountain pair for direct visual comparison.
The cases are a dormant body, the identical body with captured activity enabled,
a joined mountain/forest/biome context, and coastal shoulders. Each runs
at tile widths 128 and 224 at noon. `detail` and `active` have identical geometry;
the existing preview activity switch supplies the only state difference. Shared
lighting, shadow and transition changes select this category. The ordinary
terrain-edit witness and scrolling tests remain selected for Integration.

## Findings

The current category renders confirm that the volcano is grass-covered, with
neighboring mountain rock appearing across it in the context case. The dedicated
source textures are present: re-extracting all four material DDS files and the
three macro channels from the installed Expansion2 packages reproduces the
selected runtime files exactly. This is not an asset-resolution/import failure.

The primary problem is missing material ownership in the natural surface. In
`c3x_renderer.cpp`, the legacy raised-land pass is emitted only when
`!fidelity_profile || draw_marsh`, excluding ordinary volcanoes in the selected
production profile. Its volcano UV/coverage/activity shader therefore has no
raised volcano surface to shade. `source_fidelity/geometry.h` retains volcano
height through `natural_height_at` but emits ordinary ground with grass/plains/
desert/tundra weights and no volcano material ownership. Near a mountain,
`relief_mesh_body.h` emits its unified mountain surface over that same height.
Neither path carries the volcano skin. Editing the legacy material shader alone
cannot repair the visible result.

The isolated `current` control exactly reproduces the category close-up.
Discarding the natural surface in the synthetic volcano footprint removes the
cone entirely, exposing flat ground and its remaining cast shadow. This proves
that a skinned raised body is missing, rather than merely hidden underneath.
Dormant and active category renders are byte-identical at both zooms. A second
Lab-only probe routes the existing volcano color onto the visible natural
surface; tan/gray rock and a dark crater return immediately. The user subsequently
accepted that gameplay result, including its cone and crater shape. The earlier
proposal to broaden the cone, open the crater or retune its texture is superseded.
Material routing is the production task; preserve the shown visual result.

Preserved source findings (not authorization to change the approved appearance):

- The ordinary natural ground grid has 16 subdivisions per tile inland, while
  the unified mountain surface has 64. The source volcano has a 256×256 macro
  field. The accepted gameplay probe inherits the existing mountain-neighborhood
  surface and its detail. Preserve this mesh treatment; an isolated-volcano
  discrepancy should be evaluated against the chosen target, not trigger a general
  tessellation or silhouette redesign.
- The legacy shader treats the two BC5 channels as X/Y normal offsets. Source R
  contains fine radial ridges; G is a smooth footprint-like field, with 93.6% of
  texels below normal-neutral 0.5. This is strong evidence against that XY-normal
  interpretation. Exact source shader semantics remain unproven; R-as-height
  with surface derivatives is a Lab hypothesis, not a recovered engine equation.

## Remaining work for the accepted target

1. Generalize the approved `skin-probe` material routing. Replace its fixed world
   center with captured ordinary-volcano identity, stable source UVs, coverage and
   activity. Keep the approved color, geometry, normals and mountain shading. Make
   it work on both natural ground and unified mountain surfaces, across multiple
   volcanoes, terrain boundaries and wrapped copies. Source textures remain generic
   normalized pack data; no source-game branches enter the runtime.
2. Add the existing active lava color inside the crater, controlled by captured
   activity. Keep dormant rock intact; the dark background of the active map must
   not replace the whole cone. The user selected static lava art only: ordinary
   scene-lit color, with no emissive glow, bloom, animation, smoke or particles.
3. Correct and verify the volcano's missing cast-shadow coverage in mountain
   context. The user's accepted scene visibly has a mountain shadow but no clear
   volcano shadow. `collect_shadow_casters` includes the shared surface, but
   `source_caster.hlsl` clips class 42..43 vertices using `i.coverage`, supplied by
   the mountain-only blend field in `relief_mesh_body.h`. Volcano height can exist
   outside that mask. The earlier claim that system participation proved a
   complete volcano shadow was too broad. Carry the volcano's actual raised
   footprint into the caster mask and preserve existing mountain/coastal coverage.
   Check crater self-shadow, neighboring receivers, lighting directions, wrap and
   cache invalidation with mountains present. Do not add a separate shadow system.
   The matched `skin-shadow-probe` confirms this diagnosis: a bounded fixture
   override of that mask restores a visible westward volcano shadow. The static
   lava/shadow preview retains the accepted body and uses no emissive output.
4. Compare the generalized dormant appearance with the accepted gameplay probe,
   then validate active/dormant transitions, isolated/mountain-context cases,
   supported zooms, deterministic repeats, scrolling/wrapping and config-off.
   Run focused renderer Integration before staging the tested general candidate.
   The user's acceptance authorizes proceeding with this appearance and staging
   its faithful production implementation once verified; the fixed-coordinate
   diagnostic itself must not be staged as a runtime solution. Show the added lava
   as a separate visible extension. No new Civ III patch symbol is indicated:
   capture already supplies ordinary volcano identity and activity.

Source audit and controlled production-renderer probes live in
[`lab/studies/volcanoes`](../../../studies/volcanoes/README.md). The probe's fixed
fixture exclusion is diagnostic only and must never become runtime logic.
