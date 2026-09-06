#define PSMain inherited_main
#include "../coastal-r02/combined.hlsl"
#undef PSMain
Q6SceneOutput PSMain(PixelInput input) {
 if(input.panel>.5 && input.surface_kind>10.5 && input.surface_kind<11.5)clip(-1);
 return inherited_main(input);
}
