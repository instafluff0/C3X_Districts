ROLE: Shared lighting, shadow, depth, and composition owner.

Consume the existing FrameEnvironment and platform render graph. Own shared
sun/moon/ambient/exposure, tone response, normal-driven form shading, contact
and cast shadows, caster/receiver semantics, opaque/cutout/transparent-water/
emissive/effect ordering, and deterministic noon/sunset/midnight/sunrise
behavior. Never create category clocks, deform geometry to fake light, or
relight retained Civ III layers.

Begin with proxy casters, then run the identical matrix over every available
category. Require one frame-light direction, receiver-following shadows,
submission-order-independent opaque results, explicit cutout/water depth rules,
cool readable midnight, warm sunset, cooler sunrise, clear noon, bounded
emissives without geometry swaps, and no redraw request from static scenes.

