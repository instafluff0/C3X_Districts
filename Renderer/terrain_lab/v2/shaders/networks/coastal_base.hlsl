// Read-only Q8 composition; replace only inherited route surfaces (kind 11).
// The source bodies, receiver depth and all object placements remain intact.
#define PSMain q5_inherited_main
#include "../../fixtures/beauty/coastal-r01/combined.hlsl"
#undef PSMain
Q6SceneOutput PSMain(PixelInput input) {
 if(input.panel>.5 && input.surface_kind>10.5 && input.surface_kind<11.5)clip(-1);
 return q5_inherited_main(input);
}
