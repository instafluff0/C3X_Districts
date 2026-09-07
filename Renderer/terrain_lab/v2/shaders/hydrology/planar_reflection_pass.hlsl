// Include the unchanged scene with entry points renamed before this adapter.
// Mirror only projection: original normals, materials and shadow coordinates
// retain the object's original illumination. Water never reflects itself.
PixelInput VSMain(VertexInput input) {
 PixelInput o=Q3OriginalVSMain(input);
 o.position.y-=input.q6_world.z*Q3_REFLECTION_HEIGHT_NDC;
 o.position.z=.5+(o.position.z+input.q6_world.z*.2688-.5)*.25;
 return o;
}
FeaturePixelInput VSFeature(FeatureVertexInput input) {
 FeaturePixelInput o=Q3OriginalVSFeature(input);
 o.position.y-=input.q6_world.z*Q3_REFLECTION_HEIGHT_NDC;
 o.position.z=.5+(o.position.z+input.q6_world.z*.2688-.5)*.25;
 return o;
}
Q6SceneOutput PSMain(PixelInput input) {
 clip(input.q6_world.w-.5);clip(input.q6_world.z-.0001);
 // The composed terrain has a supporting underlay beneath water. It is not
 // an above-water reflector even where its stored support height is positive.
 clip(input.hydrology_data.x);
 if(input.surface_kind>3.5)clip(-1);
 return Q3OriginalPSMain(input);
}
Q6SceneOutput PSFeature(FeaturePixelInput input) {
 clip(input.q6_world.w-.5);clip(input.q6_world.z-.0001);
 return Q3OriginalPSFeature(input);
}
