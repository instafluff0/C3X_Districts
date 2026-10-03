#pragma once
#include <string>

// Runtime shader adaptation: omit texture families with exactly zero weight.
// Explicit gradients preserve mip selection through divergent biome branches.
// Frozen art/pack inputs remain untouched. Unknown shader layouts are refused.
inline bool sandbox_terrain_material_fast(std::string& source) {
    auto begin=source.find("        float3 grass_fine = GrassColor.Sample(Wrap, uv0).rgb;");
    auto end=source.find("        float hill_ratio =",begin);
    if(begin==std::string::npos || end==std::string::npos ||
            source.find("float desert_weight = saturate(input.biome.y);",begin)>end)return false;
    source.replace(begin,end-begin,R"(
        float desert_weight=saturate(input.biome.y);
        float tundra_weight=saturate(input.material.w/max(1-desert_weight,.00001));
        float plains_weight=saturate(input.biome.x/max(1-desert_weight-input.material.w,.00001));
        float slope=1-saturate(geometric.z);
        float rocky_band=input.material.z*smoothstep(.09,.43,input.material.x)*
            (1-tundra_weight)*saturate(.42+slope*2.2);
        float2 tundra_uv=input.world.xy*(Detail.x*.84)+float2(.19,.71);
        float2 gx=ddx(uv0),gy=ddy(uv0),px=ddx(uv1),py=ddy(uv1);
        float2 tx=ddx(tundra_uv),ty=ddy(tundra_uv);
        float3 grass=0,plains=0,tundra=0,desert=0;
        float grass_h=0,plains_h=0,tundra_h=0,desert_h=0;
        float grass_s=0,plains_s=0,tundra_s=0,desert_s=0;
        [branch]if(desert_weight<1 && tundra_weight<1 && plains_weight<1){
            float3 fine=GrassColor.SampleGrad(Wrap,uv0,gx,gy).rgb;
            float3 low=GrassColor.SampleGrad(Wrap,uv0,gx*128,gy*128).rgb;
            float3 broad=(low+
                GrassColor.SampleGrad(Wrap,uv0+float2(.37,.11),gx*128,gy*128).rgb+
                GrassColor.SampleGrad(Wrap,uv0+float2(.13,.53),gx*128,gy*128).rgb+
                GrassColor.SampleGrad(Wrap,uv0+float2(.61,.71),gx*128,gy*128).rgb)*.25;
            grass=fine+broad-low;
            grass_h=GrassHeight.SampleGrad(Wrap,uv0,gx,gy).r;
            grass_s=GrassSpecular.SampleGrad(Wrap,uv0,gx,gy).r;
        }
        [branch]if(desert_weight<1 && tundra_weight<1 && plains_weight>0){
            plains=PlainsColor.SampleGrad(Wrap,uv1,px,py).rgb;
            plains_h=PlainsHeight.SampleGrad(Wrap,uv1,px,py).r;
            plains_s=PlainsSpecular.SampleGrad(Wrap,uv1,px,py).r;
        }
        [branch]if(tundra_weight>0 && (desert_weight<1 || rocky_band>0)){
            tundra=TundraColor.SampleGrad(Wrap,tundra_uv,tx,ty).rgb;
            tundra_h=TundraHeight.SampleGrad(Wrap,tundra_uv,tx,ty).r;
            tundra_s=TundraSpecular.SampleGrad(Wrap,tundra_uv,tx,ty).r;
        }
        [branch]if(desert_weight>0){
            desert=DesertColor.SampleGrad(Wrap,uv0,gx,gy).rgb;
            desert_h=DesertHeight.SampleGrad(Wrap,uv0,gx,gy).r;
            desert_s=DesertSpecular.SampleGrad(Wrap,uv0,gx,gy).r;
        }
        float3 base=lerp(lerp(lerp(grass,plains,plains_weight),tundra,tundra_weight),desert,desert_weight);
        float base_h=lerp(lerp(lerp(grass_h,plains_h,plains_weight),tundra_h,tundra_weight),desert_h,desert_weight);
        float base_s=lerp(lerp(lerp(grass_s,plains_s,plains_weight),tundra_s,tundra_weight),desert_s,desert_weight);
        float3 hill=base;float hill_h=base_h,hill_s=base_s;
        [branch]if(rocky_band>0){
            float3 gh=0,ph=0;float ghh=0,phh=0,ghs=0,phs=0;
            [branch]if(plains_weight<1){
                gh=GrassHillColor.SampleGrad(Wrap,uv0*1.08,gx*1.08,gy*1.08).rgb;
                ghh=GrassHillHeight.SampleGrad(Wrap,uv0*1.08,gx*1.08,gy*1.08).r;
                ghs=GrassHillSpecular.SampleGrad(Wrap,uv0*1.08,gx*1.08,gy*1.08).r;
            }
            [branch]if(plains_weight>0){
                ph=PlainsHillColor.SampleGrad(Wrap,uv1*1.08,px*1.08,py*1.08).rgb;
                phh=PlainsHillHeight.SampleGrad(Wrap,uv1*1.08,px*1.08,py*1.08).r;
                phs=PlainsHillSpecular.SampleGrad(Wrap,uv1*1.08,px*1.08,py*1.08).r;
            }
            hill=lerp(lerp(gh,ph,plains_weight),tundra,tundra_weight);
            hill_h=lerp(lerp(ghh,phh,plains_weight),tundra_h,tundra_weight);
            hill_s=lerp(lerp(ghs,phs,plains_weight),tundra_s,tundra_weight);
        }
)");
    // The flat-ground cliff contribution is exactly zero. Avoid six texture
    // fetches there, retaining the complete triplanar material on steep faces.
    begin=source.find("        float3 weights=pow(abs(face),4);");
    end=source.find("        geometric=normalize(lerp(geometric,face,exposure));",begin);
    if(begin==std::string::npos || end==std::string::npos)return false;
    auto finish=source.find(';',end)+1;
    auto body=source.substr(begin,finish-begin);
    for(auto axes:{"yz","xz","xy"}){
        std::string from=".Sample(Wrap,uv."+std::string(axes)+")";
        std::string to=".SampleGrad(Wrap,uv."+std::string(axes)+",cliff_dx."+axes+",cliff_dy."+axes+")";
        std::size_t at=0;while((at=body.find(from,at))!=std::string::npos){body.replace(at,from.size(),to);at+=to.size();}
    }
    source.replace(begin,finish-begin,"        float3 cliff_dx=ddx(input.world)*1.5,cliff_dy=ddy(input.world)*1.5;\n"
        "        [branch]if(exposure>0){\n"+body+"\n        }");
    return true;
}
