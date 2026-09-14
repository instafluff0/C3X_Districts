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
