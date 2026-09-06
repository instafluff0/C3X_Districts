# Lab v2 Contracts

LQ0 owns this directory and must turn these documented contracts into the
smallest practical versioned C++/shader interfaces.

The portable core supplies immutable scene, environment, pack-mount, profile,
and fixture inputs. A visual system emits render packets containing:

- system and stable instance IDs;
- source-independent mesh and material IDs;
- a uniform transform and world bounds;
- an explicit layer and depth mode;
- shadow caster and receiver classifications;
- visibility and optional animation requirements.

Systems never call another visual system directly and never decide final
submission order. Composition consumes all packets through this order:

```text
terrain opaque
  -> terrain decals and relief
  -> route/network geometry
  -> opaque objects and units
  -> cutouts
  -> shared shadows and lighting
  -> transparent water and effects
  -> emissive contribution
```

The exact implementation may use multiple depth-aware passes; the ordering
above describes ownership, not permission to flatten objects into painter's
order. Shared contract changes require coordinator review and a representative
Metal/D3D11 parity fixture.

