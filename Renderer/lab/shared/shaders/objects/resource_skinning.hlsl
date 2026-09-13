// Flattened after the selected production material shader. Both entries use
// the same resident source mesh and instance pose; pixel shading is unchanged.
cbuffer ResourceInstance : register(b8) {
    float4 resource_shape; // cosine, sine, scale, unused
    float4 resource_offset; // source XYZ offset, ground height
    float4 resource_anchor; // center XY, world UV
    float4 resource_projection; // half width/height, tile projection, target height
    float4 resource_bones[1792]; // position matrix (4), inverse transpose (3)
};
struct ResourceVertex {
    float3 position:POSITION;float3 normal:NORMAL;float2 uv:TEXCOORD0;
    uint4 joints:BLENDINDICES;float4 weights:BLENDWEIGHT;
};
void resource_pose(ResourceVertex input,out precise float3 position,out precise float3 normal) {
    precise float3 p=0,n=0;
    [unroll]for(uint i=0;i<4;++i){
        float weight=input.weights[i];if(weight==0)continue;
        uint at=input.joints[i]*7;
        p+=weight*(input.position.x*resource_bones[at].xyz+
            input.position.y*resource_bones[at+1].xyz+
            input.position.z*resource_bones[at+2].xyz+resource_bones[at+3].xyz);
        n+=weight*(input.normal.x*resource_bones[at+4].xyz+
            input.normal.y*resource_bones[at+5].xyz+input.normal.z*resource_bones[at+6].xyz);
    }
    precise float len=sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
    position=p;normal=len>1e-12?n/len:input.normal;
}
void resource_local(ResourceVertex input,out precise float3 local,out precise float3 normal) {
    precise float3 source,n;resource_pose(input,source,n);
    precise float x=source.x+resource_offset.x,y=source.y+resource_offset.y;
    local=float3((x*resource_shape.x-y*resource_shape.y)*resource_shape.z,
        (x*resource_shape.y+y*resource_shape.x)*resource_shape.z,
        (source.z+resource_offset.z)*resource_shape.z);
    precise float3 transformed=float3(n.x*resource_shape.x-n.y*resource_shape.y,
        -(n.x*resource_shape.y+n.y*resource_shape.x),n.z/resource_shape.w);
    precise float len=sqrt(transformed.x*transformed.x+transformed.y*transformed.y+transformed.z*transformed.z);
    normal=len>1e-6?transformed/len:float3(0,0,1);
}
FeaturePixelInput VSResourceBody(ResourceVertex vertex) {
    precise float3 local,normal;resource_local(vertex,local,normal);
    precise float feature_height=local.z*150.0/.82;
    precise float relief=resource_projection.z*.82;
    precise float sx=resource_anchor.x+(local.x-local.y)*resource_projection.x;
    precise float sy=resource_anchor.y+(local.x+local.y)*resource_projection.y-local.z*150.0*resource_projection.z;
    precise float depth=resource_anchor.y+resource_offset.w*relief+(local.x+local.y)*resource_projection.y+
        resource_offset.w*relief*.75+feature_height*.0012*resource_projection.w;
    PackedFeatureInput packed;
    packed.position=float3(sx,sy,depth);packed.uv=vertex.uv;packed.normal=normal;packed.material=21;
    packed.world=float3(resource_anchor.z+local.x,resource_anchor.w-local.y,
        (resource_offset.w+2.5+feature_height)/112.0);
    return VSIntegratedFeature(packed);
}
PixelInput VSResourceShadow(ResourceVertex vertex) {
    precise float3 local,normal;resource_local(vertex,local,normal);
    precise float height=(local.z*150.0/.82)/112.0;
    // The frame's authoritative key light supplies the same CPU ground offset.
    precise float2 cast=Q6ShadowL.z>.0001?-Q6ShadowL.xy/Q6ShadowL.z*height:0;
    precise float sx=resource_anchor.x+(local.x-local.y)*resource_projection.x+(cast.x+cast.y)*resource_projection.x;
    precise float sy=resource_anchor.y+(local.x+local.y)*resource_projection.y+(cast.x-cast.y)*resource_projection.y;
    IntegratedVertexInput input=(IntegratedVertexInput)0;
    input.position=float3(sx,sy,sy+resource_offset.w*(resource_projection.z*.82)*1.75);
    input.uv=vertex.uv;input.panel=1;input.geometry_normal.z=1;input.shape_visibility=1;input.surface_kind=15;
    return VSIntegrated(input);
}
