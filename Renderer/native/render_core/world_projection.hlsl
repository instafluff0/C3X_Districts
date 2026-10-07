// b1 supplies the native occurrence, independently of immutable source content.
// Kind 1 (and existing natural bindings) projects world positions; kind 2
// preserves feature depth; kind 3 scales normalized ground geometry.
float3 project_world_content(float3 position, float3 world, float4 projection, float kind) {
    if(projection.z==0) return position;
    float width=projection.z,h=world.z*112-2.5;
    if(kind>2.5 && kind<3.5) return position*width;
    if(kind>1.5 && kind<2.5) {
        float relief=width/224*.82;
        float2 xy=position.xy*width;
        float base=xy.y+h*relief;
        return float3(xy,base+(h-position.z)*relief*.75+position.z*.0012*projection.w);
    }
    float dx=world.x-projection.x,dy=world.y-projection.y;
    float base=(dx-dy+1)*width*.25;
    return float3((dx+dy)*width*.5,base-h*(width/224*.82),base+h*.0016*projection.w);
}
// Resource bodies (integer slots 21-28), baked resource ground decals
// (fraction .325-.385) and farm kit pieces (fraction .0135/.0235/.0335) take
// the natural terrain's height-depth basis, so the part of a body standing on
// raised natural ground is not hidden beneath it.
bool farm_kit_material(float material) {
    float f=frac(material);
    return material>20.5 && material<28.5 && f<.05 && abs(frac(f*100)-.35)<.05;
}
float resource_natural_depth(float3 projected, float world_z, float4 projection, float kind, float material) {
    float f=frac(material);
    if(projection.z==0 || kind<1.5 || kind>2.5 || material<20.5 || material>28.5 ||
       !(f<.005 || abs(f-.355)<.03 || farm_kit_material(material)))
        return projected.z;
    float h=world_z*112-2.5;
    return projected.y+h*(projection.z/224*.82)+h*.0016*projection.w;
}
