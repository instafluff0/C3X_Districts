#pragma once
#include <fstream>
#include "pass_workload.h"
#include "scroll_region.h"
#include "static_raster_state.h"
#include "terrain_material_fast.h"
#include "../native/render_core/raster_contributors.h"
#include "../native/render_core/process_environment.h"
#include "../native/render_core/shadow_sampling_grid.h"
#include "../native/render_core/shadow_page_contents.h"
#include "../native/render_core/receiver_cells.h"
#include "../native/render_core/body_placement_requirements.h"
#include "../native/render_core/prepared_draw_parameters.h"
#include "../native/render_core/submission_census.h"
#include "../native/render_core/pulled_mesh_pages.h"
#include "../native/render_core/overlay_composite.h"
#include "../native/render_core/gpu_event_timeline.h"
#include <limits>
#include <sstream>
#include "../native/gpu_territory_borders.h"
#include "../native/gpu_city_site_overlay.h"

inline auto& sandbox_active_reflection() {
#ifdef C3X_RENDERER64_FRESH
    // The fresh path initializes the camera reflection, not legacy region targets.
    return renderer.reflection;
#else
    return renderer.scene_region_size==128?renderer.reflection:renderer.region_reflection;
#endif
}

// Keep production's material equations and preparation. Change only receiver
// addressing from regional pages to the camera-framed field.
struct SandboxVisualShaders {
    bool installed = false;
    ID3D11SamplerState* shadow_comparison=nullptr;
    ID3D11PixelShader* water_surface = nullptr;
    ID3D11PixelShader* water_lighting = nullptr;
    ID3D11PixelShader* river_surface = nullptr;
    ID3D11PixelShader* underlay = nullptr;
    ID3D11PixelShader* material_albedo = nullptr;
    ID3D11PixelShader* material_normal_world = nullptr;
    ID3D11PixelShader* material_relight = nullptr;
    ID3D11PixelShader* terrain_material = nullptr;
    ID3D11PixelShader* mountain_material = nullptr;
    ID3D11PixelShader* reflected_terrain_material = nullptr;
    ID3D11PixelShader* reflected_mountain_material = nullptr;
    ID3D11PixelShader* terrain_relight = nullptr;
    ID3D11PixelShader* aquatic_feature = nullptr;
    ID3D11PixelShader* vegetation_depth[2] = {};
    ID3D11VertexShader* vegetation_instances[2] = {};
    // Vertex-pulled VSIntegrated/VSIntegratedFeature for batched non-rigid
    // records; optional.
    ID3D11VertexShader* pulled_vertex = nullptr;
    ID3D11VertexShader* pulled_feature = nullptr;
    ~SandboxVisualShaders() {
        if(pulled_vertex) pulled_vertex->Release();
        if(pulled_feature) pulled_feature->Release();
        if(shadow_comparison)shadow_comparison->Release();
        if(water_surface) water_surface->Release();
        if(water_lighting) water_lighting->Release();
        if(river_surface) river_surface->Release();
        if(underlay) underlay->Release();
        if(material_albedo) material_albedo->Release();
        if(material_normal_world) material_normal_world->Release();
        if(material_relight) material_relight->Release();
        if(terrain_material) terrain_material->Release();
        if(mountain_material) mountain_material->Release();
        if(reflected_terrain_material) reflected_terrain_material->Release();
        if(reflected_mountain_material) reflected_mountain_material->Release();
        if(terrain_relight) terrain_relight->Release();
        if(aquatic_feature) aquatic_feature->Release();
        for(auto* shader:vegetation_depth)if(shader)shader->Release();
        for(auto* shader:vegetation_instances)if(shader)shader->Release();
    }
    static std::string source(std::string const& relative) {
        char const* profile=renderer.city_profile?"city_fidelity":
            renderer.environment_profile?"environment_refresh":"source_fidelity";
        std::ifstream file(renderer.shader_root + "/Renderer/native/" + profile + "/" + relative,
            std::ios::binary);
        if (!file) return {};
        return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    }
    static bool patch(std::string& shader) {
        auto start = shader.find("float c3x_paged_visibility(");
        if (start == std::string::npos) return false;
        auto body = shader.find('{', start);
        if (body == std::string::npos) return false;
        int braces = 1;
        std::size_t end = body + 1;
        for (; end < shader.size() && braces; ++end)
            braces += (shader[end] == '{') - (shader[end] == '}');
        if (braces) return false;
        shader.replace(start, end-start, R"(
SamplerComparisonState SandboxShadowComparison : register(s15);
// Only the shadow query uses a continuous occurrence near the map seam.
// Material, water, local-light and canonical mesh coordinates stay unchanged.
float3 sandbox_shadow_world(float3 world) {
 float4 center=pickup_pages[1];
 if(center.z>0)world.xy-=floor((world.x+world.y-center.x+center.z*.5)/center.z)*center.z*.5;
 if(center.w>0){float turn=floor((world.x-world.y-center.y+center.w*.5)/center.w)*center.w*.5;
  world.xy-=float2(turn,-turn);}
 return world;
}
// Global cell coordinates precede physical page lookup. The same canonical
// page has the same caster projection regardless of its physical array slice.
float sandbox_shadow_load(Texture2DArray field,int2 cell) {
 float4 range=pickup_pages[0]; // integer low page x/y and resident page count x/y
 int2 page=int2(floor(float2(cell)/1024.));
 int2 local_page=page-int2(range.xy);
 if(any(local_page<0) || any(local_page>=int2(range.zw)))return -1e6;
 int logical=local_page.y*int(range.z)+local_page.x;
 int layer=int(pickup_pages[3+logical].x);
 if(layer<0)return -1e6; // A pending page must never expose recycled atlas pixels.
 int2 local=cell-page*1024;
 return field.Load(int4(local,layer,0)).r;
}
float c3x_paged_visibility(Texture2DArray field,float4 world,float3 normal,bool water,
 float4 ShadowU,float4 ShadowV,float4 ShadowL,float4 ShadowFlags) {
 if(world.w<=.5 || ShadowFlags.x<=.5)return 1;
 world.xyz=sandbox_shadow_world(world.xyz);
 float2 inverse_pitch=pickup_pages[2].xy;
 float texel=1/min(inverse_pitch.x,inverse_pitch.y);
 float3 offset=world.xyz+normal*(water?6./1024.:texel*1.5);
 float2 uv=float2(dot(offset,ShadowU.xyz),dot(offset,ShadowV.xyz))*inverse_pitch;
 float z=dot(offset,ShadowL.xyz);
 float2 plane=float2(dot(world.xyz,ShadowU.xyz),dot(world.xyz,ShadowV.xyz))*inverse_pitch;
 float plane_z=dot(world.xyz,ShadowL.xyz);
 float2 ux=ddx(plane),uy=ddy(plane);float zx=ddx(plane_z),zy=ddy(plane_z);
 float determinant=ux.x*uy.y-ux.y*uy.x;
 float2 gradient=0;
 if(abs(determinant)>1e-12)gradient=float2(zx*uy.y-zy*ux.y,zy*ux.x-zx*uy.x)/determinant;
 int2 center=int2(floor(uv));float sum=0;
#ifdef SANDBOX_FILTERED_SHADOW
 // Four bilinear comparisons cover a 3x3 tent. Keep the exact per-texel
 // receiver-plane correction on steep surfaces and across page boundaries.
 int2 page=int2(floor(float2(center)/1024.));int2 local=center-page*1024;
 float bias=water?.00060:texel*.35;
 if(all(local>=1) && all(local<1023) && max(abs(gradient.x),abs(gradient.y))<bias*.25) {
  float4 range=pickup_pages[0];int2 selected=page-int2(range.xy);
  if(any(selected<0)||any(selected>=int2(range.zw)))return 1;
  int layer=int(pickup_pages[3+selected.y*int(range.z)+selected.x].x);
  if(layer<0)return 1;
  [unroll]for(int j=0;j<2;++j)[unroll]for(int i=0;i<2;++i){
   float2 delta=float2(i,j)-.5;
   float2 location=float2(local)+.5+delta;
   float receiver=z+dot(gradient,float2(center)+.5+delta-uv);
   sum+=field.SampleCmpLevelZero(SandboxShadowComparison,float3(location/1024.,layer),receiver+bias);
  }
  return sum*.25;
 }
#endif
 [unroll]for(int y=-1;y<=1;y++)[unroll]for(int x=-1;x<=1;x++) {
  int2 sample=center+int2(x,y);
  float blocker=sandbox_shadow_load(field,sample);
  float receiver=z+dot(gradient,float2(sample)+.5-uv);
  sum+=step(blocker,receiver+(water?.00060:texel*.35));
 }
 return sum/9;
}
)");
        UINT support=0;char reference[8]={};
        bool legacy=c3x_renderer::render_core::cached_environment("C3X_RENDERER_SHADOW_PCF_REFERENCE",reference,sizeof(reference)) && reference[0]=='1';
        if(!legacy && SUCCEEDED(renderer.device->CheckFormatSupport(DXGI_FORMAT_R32_FLOAT,&support)) &&
                (support&D3D11_FORMAT_SUPPORT_SHADER_SAMPLE_COMPARISON))shader.insert(0,"#define SANDBOX_FILTERED_SHADOW 1\n");
        return true;
    }
    static bool compile(char const* file, char const* entry, ID3D11PixelShader** output) {
        auto fail_source=[&](char const* reason){
            char detail[192];sprintf_s(detail,"file=%s entry=%s reason=%s",file,entry,reason);
            renderer.trace.write("fresh-shader-source",detail,true);
            return false;
        };
        bool water_variant = std::strcmp(file,"water_surface.hlsl")==0;
        std::string shader = source(water_variant?"hydrology.hlsl":file);
        bool albedo_variant=std::strcmp(entry,"PSSandboxMaterialAlbedo")==0;
        bool normal_variant=std::strcmp(entry,"PSSandboxMaterialNormalWorld")==0;
        bool terrain_variant=std::strcmp(entry,"PSSandboxTerrainMaterial")==0 ||
            std::strcmp(entry,"PSSandboxReflectionTerrainMaterial")==0;
        if(albedo_variant)shader.insert(0,"#define Q8_DEBUG_ALBEDO 1\n");
        if(normal_variant)shader.insert(0,"#define Q8_DEBUG_NORMAL 1\n");
        if(terrain_variant)shader.insert(0,"#define SANDBOX_TERRAIN_MATERIAL 1\n");
        // At this map scale the source hill decal and fine material normals
        // turn small patches into isolated dark marks. Preserve the sculpted
        // terrain normal while softening only the material-scale response.
        if(renderer.city_profile && std::strcmp(file,"terrain.hlsl")==0){
            char reference[8]={};
            if(!(c3x_renderer::render_core::cached_environment("C3X_RENDERER_TERRAIN_MATERIAL_REFERENCE",reference,sizeof(reference))&&reference[0]=='1') &&
                    !sandbox_terrain_material_fast(shader))return fail_source("terrain-material-fast-markers");
            auto replace=[&](char const* old_text,char const* new_text){
                auto at=shader.find(old_text);
                if(at==std::string::npos)return false;
                shader.replace(at,std::strlen(old_text),new_text);
                return true;
            };
            if(!replace("albedo = lerp(decal.rgb, stone, rock * 0.94);",
                    "albedo = lerp(decal.rgb, stone, rock * 0.75);") ||
               !replace("stone_height * 0.58 + rock * 0.42, 0.11);",
                    "stone_height * 0.58 + rock * 0.42, 0.08);") ||
               !replace("        alpha *= decal.a;",
                    "        alpha *= decal.a * 0.75;"))return fail_source("terrain-markers");
            if(!replace("albedo = lerp(base, hill, rocky_band * 0.90);",
                    "albedo = lerp(base, hill, rocky_band * 0.85);"))return fail_source("hill-marker");
            auto material=shader.find("#ifdef SANDBOX_TERRAIN_MATERIAL\n    // The sandbox retains");
            if(material==std::string::npos)return fail_source("terrain-material-marker");
            shader.insert(material,
                "    float sandbox_hill = smoothstep(0.02, 0.22, input.material.x);\n"
                "    geometric = normalize(lerp(geometric, normalize(input.normal), "
                "sandbox_hill * 0.35));\n");
            auto cavity=shader.find("    float cavity = lerp(0.79, 1.0,",material);
            for(int branch=0;branch<2;++branch){
                if(cavity==std::string::npos)return fail_source("terrain-cavity-marker");
                auto end=shader.find(';',cavity);
                shader.insert(end+1,
                    "\n    cavity = lerp(cavity, max(cavity, 0.97), sandbox_hill);");
                cavity=shader.find("    float cavity = lerp(0.79, 1.0,",end+1);
            }
        }
        if(renderer.city_profile && std::strcmp(file,"objects.hlsl")==0){
            auto at=shader.find("    radiance += Ambient.rgb * rim * 0.08;");
            if(at==std::string::npos)return fail_source("object-rim-marker");
            shader.insert(at,
                "    float low_sun=smoothstep(.10,.25,Sun.w)*"
                "(1-smoothstep(.52,.69,Sun.w));\n"
                "    if(kind>.5 && kind<1.5)radiance+=SunColorExposure.rgb*"
                "Sun.w*low_sun*pow(1-saturate(dot(normal,normalize(View.xyz))),2)"
                "*shadow*.24;\n");
        }
        if(water_variant){
            std::ifstream variant(renderer.shader_root+
                "/Renderer/sandbox/"+file,std::ios::binary);
            if(!variant)return fail_source("water-variant-missing");
            shader.append(std::istreambuf_iterator<char>(variant),
                std::istreambuf_iterator<char>());
            // Runtime-owned overlay works with inherited packs as well as
            // newly prepared ones; never edit an ignored frozen shader pack.
            auto light_begin=shader.find("    float shadow = 1;",shader.find("float4 ShadeWaterSurface("));
            auto light_end=shader.find("    // 0 A.D. separates",light_begin);
            auto surface=shader.find("float4 ShadeWaterSurface(");
            if(light_begin==std::string::npos || light_end==std::string::npos || surface==std::string::npos)
                return fail_source("water-lighting-markers");
            shader.replace(light_begin,light_end-light_begin,
                "    float4 illumination=water_view.z>.5 ? resource_base_texture_6.Load(int3(input.position.xy,0)) : SandboxWaterLighting(input);\n"
                "    float shadow=illumination.a;float3 light=illumination.rgb;\n\n");
            shader.insert(surface,R"(
float4 SandboxWaterLighting(PixelInput input) {
    float depth=max(0,input.hydrology_data.w);
    float coastal_detail=1-smoothstep(.30,.39,depth);
    float shadow=1;
    if(coastal_detail>0)shadow=q6_receiver_visibility(input,float3(0,0,1),1);
    shadow=lerp(1,shadow,coastal_detail);
    return float4(frame_illumination(float3(0,0,1),shadow,1),shadow);
}
float4 PSWaterLighting(PixelInput input):SV_Target {
    clip(-input.hydrology_data.x-.0001);
    return SandboxWaterLighting(input);
}
)");
        }
        if(std::strcmp(entry,"PSSandboxVegetationDepth")==0 ||
           std::strcmp(entry,"PSSandboxReflectedVegetationDepth")==0)
            shader.append(R"(
void PSSandboxVegetationDepth(P input) {
    if(input.secondary.y>.5)clip(Opacity.Sample(Wrap,input.uv).r-.5);
}
void PSSandboxReflectedVegetationDepth(P input) {
    clip(input.world.z-NativeReflection.z-.0001);
    PSSandboxVegetationDepth(input);
}
)");
        if(std::strcmp(entry,"PSSandboxUnderlay")==0)
            shader.append(R"(
float4 PSSandboxUnderlay(PixelInput input) : SV_Target {
    input.surface_kind = 0.5;
    return PSMain(input).color;
}
)");
        if(albedo_variant)shader.append(R"(
float4 PSSandboxMaterialAlbedo(PixelInput input) : SV_Target {
    input.surface_kind = 0.5;
    return PSMain(input).color;
}
)");
        if(normal_variant)shader.append(R"(
struct SandboxMaterialNormalWorld {
    float4 normal : SV_Target0;
    float4 world : SV_Target1;
};
SandboxMaterialNormalWorld PSSandboxMaterialNormalWorld(PixelInput input) {
    input.surface_kind = 0.5;
    SandboxMaterialNormalWorld result;
    result.normal = q6_raw_main(input);
    result.world = input.q6_world;
    return result;
}
)");
        if(std::strcmp(entry,"PSSandboxMaterialRelight")==0)shader.append(R"(
float4 PSSandboxMaterialRelight(float4 position : SV_Position) : SV_Target {
    int3 pixel = int3(position.xy, 0);
    float4 albedo = resource_base_texture_0.Load(pixel);
    if(albedo.a < 0.000001) discard;
    float3 normal = normalize(resource_base_texture_1.Load(pixel).xyz * 2 - 1);
    float4 world = resource_base_texture_2.Load(pixel);
    PixelInput receiver = (PixelInput)0;
    receiver.position = position;
    receiver.q6_world = world;
    receiver.surface_kind = 0.5;
    float3 light = q6_receiver_illumination(receiver, normal, 1, 1);
    return float4(albedo.rgb * light, albedo.a);
}
)");
        if(terrain_variant)shader.append(R"(
Output PSSandboxTerrainMaterial(P input) { return shade(input); }
)");
        if(std::strcmp(entry,"PSSandboxReflectionTerrainMaterial")==0)shader.append(R"(
Output PSSandboxReflectionTerrainMaterial(P input) {
    clip(input.world.z-NativeReflection.z-.0001);
    return shade(input);
}
)");
        if(std::strcmp(entry,"PSSandboxTerrainRelight")==0)shader.append(R"(
Texture2D<float4> SandboxTerrainAlbedo : register(t116);
Texture2D<float4> SandboxTerrainNormal : register(t117);
Texture2D<float4> SandboxTerrainWorld : register(t118);
Texture2D<float4> SandboxTerrainProperties : register(t119);
Texture2D<float> SandboxTerrainDepth : register(t120);
struct SandboxTerrainLit { float4 color : SV_Target0; float depth : SV_Depth; };
float sandbox_shadow_blocker(float3 world) {
    world=sandbox_shadow_world(world);
    float2 uv=float2(dot(world,ShadowU.xyz),dot(world,ShadowV.xyz))*pickup_pages[2].xy;
    return sandbox_shadow_load(ShadowField,int2(floor(uv)));
}
SandboxTerrainLit PSSandboxTerrainRelight(float4 position : SV_Position) {
    int3 pixel = int3(position.xy, 0);
    float4 sample_color = SandboxTerrainAlbedo.Load(pixel);
    clip(sample_color.a - 0.000001);
    float inverse_alpha = rcp(sample_color.a);
    float3 albedo = sample_color.rgb * inverse_alpha;
    float3 normal = normalize(SandboxTerrainNormal.Load(pixel).rgb *
                              (2 * inverse_alpha) - 1);
    float3 world = SandboxTerrainWorld.Load(pixel).rgb * inverse_alpha;
    float3 properties = SandboxTerrainProperties.Load(pixel).rgb * inverse_alpha;
    float rock = saturate((properties.x - 0.95) * 0.5);
    float cavity = properties.x - 2 * rock;
    float3 light_direction = ShadowL.xyz;
    float sky = saturate(normal.z * 0.5 + 0.5);
    float3 ambient = Ambient.rgb * Ambient.a *
                     lerp(lerp(0.56, 0.52, rock), 1.0, sky) *
                     cavity * properties.y;
    float shadow = q6_shadow_visibility(ShadowField, world, normal,
        ShadowU, ShadowV, ShadowL, ShadowFlags.x > 0.5, true);
    shadow = lerp(lerp(1.0, shadow, 0.48), shadow, properties.z);
    float wrap_bias = lerp(0.20, 0.18, rock);
    float wrap = saturate((dot(normal, light_direction) + wrap_bias) /
                          (1 + wrap_bias));
    float diffuse_floor = lerp(0.07, 0.055, rock);
    float3 sun = SunColorExposure.rgb * Sun.w;
    float3 radiance = albedo * (ambient + sun *
        (diffuse_floor + (1 - diffuse_floor) * wrap) * shadow *
        lerp(1, properties.y, 0.7 * rock));
    radiance += sun * ggx(normal, light_direction, normalize(View.xyz),
        lerp(0.86, 0.70, rock), lerp(0.035, 0.045, rock)) * shadow;
    // Low-angle light between nearby tree casters creates restrained beams
    // across the forest floor. Use the actual shared light-space occluders so
    // the beams follow the same sun direction as the trees' cast shadows.
    float low_sun=smoothstep(.10,.25,Sun.w)*
                  (1-smoothstep(.52,.69,Sun.w));
    if(low_sun>.001 && world.z<.32 && rock<.4 && shadow>.35) {
        float2 across=normalize(float2(-ShadowL.y,ShadowL.x));
        float3 left=world+float3(across*.27,0);
        float3 right=world-float3(across*.27,0);
        float left_canopy=smoothstep(.018,.09,
            sandbox_shadow_blocker(left)-dot(left,ShadowL.xyz));
        float right_canopy=smoothstep(.018,.09,
            sandbox_shadow_blocker(right)-dot(right,ShadowL.xyz));
        float gap=shadow*max(left_canopy,right_canopy);
        radiance+=sun*gap*low_sun*.34;
    }
    radiance += albedo * q8_local_irradiance(float4(world, 1),
        normalize(normal * float3(1, -1, 1 / Q8_LOCAL_Z_METRIC)), 1);
    SandboxTerrainLit output;
    output.color = float4(max(radiance, 0) * sample_color.a, sample_color.a);
    output.depth = SandboxTerrainDepth.Load(pixel);
    return output;
}
)");
        if(std::strcmp(entry,"PSSandboxAquaticFeature")==0)shader.append(R"(
float4 PSSandboxAquaticFeature(FeaturePixelInput input) : SV_Target {
    float4 body = PSIntegratedFeature(input);
    // Premultiplied marine color is accumulated for the water's refraction.
    return float4(body.rgb * float3(0.55, 0.72, 0.84) * 0.40,
                  body.a * 0.40);
}
)");
        if (shader.empty() || !patch(shader)) {
            char detail[192];sprintf_s(detail,"file=%s entry=%s source_or_patch=0",file,entry);
            renderer.trace.write("fresh-shader-failed",detail,true);
            return false;
        }
        std::uint64_t key=14695981039346656037ull;
        for (unsigned char c:shader) key=(key^c)*1099511628211ull;
        for (unsigned char const* p=reinterpret_cast<unsigned char const*>(entry);*p;++p)
            key=(key^*p)*1099511628211ull;
        char name[160]={};
        sprintf_s(name,"%s.%s.%016llx.cso",file,entry,
            static_cast<unsigned long long>(key));
        std::string directory=renderer.fidelity_root+"/Renderer/sandbox/out/compiled";
        CreateDirectoryA(directory.c_str(),nullptr);
        std::string cache=directory+"/"+name;
        std::ifstream cached(cache,std::ios::binary);
        if (cached) {
            std::vector<char> code((std::istreambuf_iterator<char>(cached)),
                std::istreambuf_iterator<char>());
            if (!code.empty() && SUCCEEDED(renderer.device->CreatePixelShader(
                    code.data(),code.size(),nullptr,output))) return true;
        }
        ID3DBlob* code = nullptr;
        ID3DBlob* errors = nullptr;
        HRESULT hr = D3DCompile(shader.data(), shader.size(), file, nullptr, nullptr,
            entry, "ps_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
        if (errors) {
            if (FAILED(hr)) {
                std::printf("SANDBOX_SHADER_ERROR file=%s entry=%s %s\n", file, entry,
                    static_cast<char const*>(errors->GetBufferPointer()));
                std::fflush(stdout);
            }
            errors->Release();
        }
        if (SUCCEEDED(hr)) hr = renderer.device->CreatePixelShader(
            code->GetBufferPointer(), code->GetBufferSize(), nullptr, output);
        if (SUCCEEDED(hr)) {
            std::ofstream saved(cache,std::ios::binary);
            saved.write(static_cast<char const*>(code->GetBufferPointer()),
                std::streamsize(code->GetBufferSize()));
        }
        if (FAILED(hr)) {
            char detail[192];sprintf_s(detail,"file=%s entry=%s hr=0x%08x",file,entry,unsigned(hr));
            renderer.trace.write("fresh-shader-failed",detail,true);
        }
        if (code) code->Release();
        return SUCCEEDED(hr);
    }
    // Build VSIntegratedPulled from the selected hydrology source: the same
    // VSIntegrated body, fed from a page record and raw vertex/index bytes.
    // Viewport occurrence terms are redirected to per-vertex statics after the
    // C3XViewportSettings declaration; shared terms still come from b1.
    // Any unexpected layout leaves batching disabled (ordinary path).
    static bool compile_pulled_vertex(ID3D11VertexShader** output,ID3D11VertexShader** feature) {
        auto shader=source("hydrology.hlsl");
        auto cbuffer=shader.find("cbuffer C3XViewportSettings");
        auto close=cbuffer==std::string::npos?cbuffer:shader.find("};",cbuffer);
        auto layout=shader.find("struct IntegratedVertexInput");
        auto body_begin=layout==std::string::npos?layout:shader.find('{',layout);
        auto body_end=body_begin==std::string::npos?body_begin:shader.find("};",body_begin);
        if(close==std::string::npos||body_end==std::string::npos||shader.find("PixelInput VSIntegrated(IntegratedVertexInput input)")==std::string::npos)return false;
        struct Element {char const* semantic;unsigned index,offset;};
        static Element const elements[]={{"POSITION",0,0},{"TEXCOORD",0,12},{"TEXCOORD",1,20},{"NORMAL",0,24},
            {"TEXCOORD",2,36},{"TEXCOORD",3,44},{"TEXCOORD",4,52},{"TEXCOORD",5,56},{"TEXCOORD",6,60},{"TEXCOORD",7,64},
            {"TEXCOORD",8,68},{"TEXCOORD",9,84},{"TEXCOORD",10,92},{"TEXCOORD",11,96},{"TEXCOORD",12,112},{"TEXCOORD",13,116},
            {"TEXCOORD",14,120},{"TEXCOORD",15,136},{"TEXCOORD",16,152}};
        std::string decode;
        std::istringstream fields(shader.substr(body_begin+1,body_end-body_begin-1));
        for(std::string line;std::getline(fields,line);){
            auto colon=line.find(':');if(colon==std::string::npos)continue;
            std::istringstream left(line.substr(0,colon));std::string type,name;left>>type>>name;
            std::string semantic;for(char c:line.substr(colon+1))if(std::isalnum(static_cast<unsigned char>(c))||c=='_')semantic+=c;
            if(type.empty()||name.empty()||semantic.empty())return false;
            unsigned width=type=="float"?1:type=="float2"?2:type=="float3"?3:type=="float4"?4:0;
            if(!width)return false;
            auto digits=semantic.find_first_of("0123456789");
            unsigned index=digits==std::string::npos?0:unsigned(std::stoul(semantic.substr(digits)));
            auto base=semantic.substr(0,digits);
            Element const* match=nullptr;
            for(auto const& e:elements)if(base==e.semantic&&index==e.index)match=&e;
            if(!match||match->offset+width*4>168)return false;
            decode+="    input."+name+"=asfloat(c3x_pulled_vertices.Load"+(width>1?std::to_string(width):std::string())+
                "(base+"+std::to_string(match->offset)+"u));\n";
        }
        if(decode.empty())return false;
        // Shared by both pulled entries: redirect the occurrence terms after
        // the viewport cbuffer, then load one page record per vertex.
        std::string const redirect=R"(
static float2 c3x_pulled_translation;
static float c3x_pulled_depth_translation;
static float c3x_pulled_kind;
static float4 c3x_pulled_projection;
#define c3x_viewport_translation c3x_pulled_translation
#define c3x_viewport_depth_translation c3x_pulled_depth_translation
#define c3x_viewport_translation_padding c3x_pulled_kind
#define c3x_content_projection c3x_pulled_projection
)";
        std::string const helpers=R"(
struct C3XPulledRecord {uint first,count,vertex_byte,index_byte;uint index16,stride,vertices,r1;float2 translation;float kind,r2;float4 projection;};
StructuredBuffer<C3XPulledRecord> c3x_pulled_records : register(t60);
ByteAddressBuffer c3x_pulled_vertices : register(t61);
ByteAddressBuffer c3x_pulled_indices : register(t62);
cbuffer C3XPulledBatch : register(b12) {
    float2 c3x_pulled_base_translation;float c3x_pulled_depth_base;uint c3x_pulled_vertex_base;
    uint c3x_pulled_record_first;uint c3x_pulled_record_last;uint2 c3x_pulled_batch_pad2;
};
// Draw(count,0) plus an explicit base: D3D11 SV_VertexID excludes
// StartVertexLocation. Every raw load is clamped to its record, so no
// inconsistency can read outside a page. Selects the occurrence terms and
// returns the vertex's byte address.
uint c3x_pulled_vertex_address(uint vertex) {
    uint id=vertex+c3x_pulled_vertex_base;
    uint lo=c3x_pulled_record_first,hi=c3x_pulled_record_last;
    [loop] while(lo<hi){uint mid=(lo+hi+1)>>1;if(c3x_pulled_records[mid].first<=id)lo=mid;else hi=mid-1;}
    C3XPulledRecord r=c3x_pulled_records[lo];
    uint local=min(id-r.first,r.count-1),index;
    if(r.index16!=0){uint address=r.index_byte+local*2;uint word=c3x_pulled_indices.Load(address&~3u);index=(address&2u)!=0?(word>>16):(word&0xffffu);}
    else index=c3x_pulled_indices.Load(r.index_byte+local*4);
    c3x_pulled_translation=c3x_pulled_base_translation+r.translation;
    c3x_pulled_depth_translation=c3x_pulled_depth_base+r.translation.y;
    c3x_pulled_kind=r.kind;
    c3x_pulled_projection=r.projection;
    return r.vertex_byte+min(index,r.vertices-1)*r.stride;
}
)";
        shader.insert(close+2,redirect);
        shader+=helpers+R"(PixelInput VSIntegratedPulled(uint vertex : SV_VertexID) {
    uint base=c3x_pulled_vertex_address(vertex);
    IntegratedVertexInput input=(IntegratedVertexInput)0;
)"+decode+R"(
    return VSIntegrated(input);
}
)";
        // Feature records must keep the interpolant layout of the feature
        // pixel shader. Its FeaturePixelInput carries city AO/tangent/emissive
        // registers ahead of the five that hydrology's declares, so a feature
        // vertex shader compiled from hydrology.hlsl linked by position and
        // fed the pixel shader garbage UV, normal and material id (flat gray
        // coastal rocks, unshaded wherever eight or more records batched).
        // Compile it from the same feature source as the ordinary path; the
        // packed 48-byte layout must still match exactly.
        auto features=source("feature.hlsl");
        auto feature_cbuffer=features.find("cbuffer C3XViewportSettings");
        auto feature_close=feature_cbuffer==std::string::npos?feature_cbuffer:features.find("};",feature_cbuffer);
        bool feature_layout=feature_close!=std::string::npos &&
            features.find("struct PackedFeatureInput {\n float3 position:POSITION;float2 uv:TEXCOORD0;float3 normal:NORMAL;\n float material:TEXCOORD6;float3 world:TEXCOORD14;\n};")!=std::string::npos &&
            features.find("FeaturePixelInput VSIntegratedFeature(PackedFeatureInput packed)")!=std::string::npos;
        if(feature_layout){
            features.insert(feature_close+2,redirect);
            features+=helpers+R"(FeaturePixelInput VSIntegratedFeaturePulled(uint vertex : SV_VertexID) {
    uint base=c3x_pulled_vertex_address(vertex);
    PackedFeatureInput packed;
    packed.position=asfloat(c3x_pulled_vertices.Load3(base));
    packed.uv=asfloat(c3x_pulled_vertices.Load2(base+12u));
    packed.normal=asfloat(c3x_pulled_vertices.Load3(base+20u));
    packed.material=asfloat(c3x_pulled_vertices.Load(base+32u));
    packed.world=asfloat(c3x_pulled_vertices.Load3(base+36u));
    return VSIntegratedFeature(packed);
}
)";
        }
        auto compile=[&](std::string const& text,char const* entry,ID3D11VertexShader** target){
            Microsoft::WRL::ComPtr<ID3DBlob> code,errors;
            HRESULT hr=D3DCompile(text.data(),text.size(),"fresh_pulled",nullptr,nullptr,entry,"vs_5_0",
                D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,&errors);
            if(errors&&FAILED(hr))renderer.trace.write("fresh-pulled-shader",static_cast<char const*>(errors->GetBufferPointer()),true);
            return SUCCEEDED(hr)&&SUCCEEDED(renderer.device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,target));
        };
        if(!compile(shader,"VSIntegratedPulled",output))return false;
        if(feature_layout && !compile(features,"VSIntegratedFeaturePulled",feature))
            renderer.trace.write("fresh-pulled-shader","feature layout unavailable; feature records use per-record submission",true);
        return true;
    }
    bool install() {
        if (installed) return true;
        if(!shadow_comparison){
            D3D11_SAMPLER_DESC desc={};desc.Filter=D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
            desc.AddressU=desc.AddressV=desc.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP;
            desc.ComparisonFunc=D3D11_COMPARISON_GREATER_EQUAL;desc.MaxLOD=D3D11_FLOAT32_MAX;
            if(FAILED(renderer.device->CreateSamplerState(&desc,&shadow_comparison)))return false;
        }
        struct Replacement {char const* file; char const* entry; ID3D11PixelShader** slot;};
        Replacement targets[] = {
            {"hydrology.hlsl", "PSIntegrated", &renderer.pixel_shader},
            {"hydrology.hlsl", "PSCoastalWave", &renderer.wave_shader},
            {"feature.hlsl", "PSIntegratedFeature", &renderer.feature_pixel_shader},
            {"terrain.hlsl", "PSFeature", &renderer.natural.ps[0]},
            {"mountain.hlsl", "PSFeature", &renderer.natural.ps[1]},
            {"objects.hlsl", "PSFeature", &renderer.natural.ps[2]},
        };
        for (auto& target : targets) {
            ID3D11PixelShader* replacement = nullptr;
            if (!compile(target.file, target.entry, &replacement)) return false;
            (*target.slot)->Release();
            *target.slot = replacement;
        }
        if (renderer.city_profile) {
            char const* entries[]={"PSNativeCity", "PSNativeCityEmission",
                "PSNativeCityReflection", "PSNativeCityReflectionEmission"};
            for (unsigned i=0;i<4;++i) {
                ID3D11PixelShader* replacement=nullptr;
                if (!compile("city.hlsl",entries[i],&replacement)) return false;
                renderer.cities.ps[i]->Release();
                renderer.cities.ps[i]=replacement;
            }
        }
        if(!compile("water_surface.hlsl","PSWaterSurface",&water_surface) ||
           !compile("water_surface.hlsl","PSWaterLighting",&water_lighting) ||
           !compile("water_surface.hlsl","PSRiverSurface",&river_surface))return false;
        if(!compile("hydrology.hlsl","PSSandboxUnderlay",&underlay))return false;
        if(!compile("hydrology.hlsl","PSSandboxMaterialAlbedo",&material_albedo) ||
           !compile("hydrology.hlsl","PSSandboxMaterialNormalWorld",&material_normal_world) ||
           !compile("hydrology.hlsl","PSSandboxMaterialRelight",&material_relight))return false;
        if(!compile("terrain.hlsl","PSSandboxTerrainMaterial",&terrain_material) ||
           !compile("mountain.hlsl","PSSandboxTerrainMaterial",&mountain_material) ||
           !compile("terrain.hlsl","PSSandboxTerrainRelight",&terrain_relight))return false;
        if(!compile("terrain.hlsl","PSSandboxReflectionTerrainMaterial",
                &reflected_terrain_material) ||
           !compile("mountain.hlsl","PSSandboxReflectionTerrainMaterial",
                &reflected_mountain_material))return false;
        if(!compile("feature.hlsl","PSSandboxAquaticFeature",&aquatic_feature))return false;
        if(!compile("objects.hlsl","PSSandboxVegetationDepth",&vegetation_depth[0]) ||
           !compile("objects.hlsl","PSSandboxReflectedVegetationDepth",&vegetation_depth[1]))return false;
        char const* names[]={"hydrology.hlsl","feature.hlsl","terrain.hlsl",
            "mountain.hlsl","objects.hlsl"};
        auto& reflection=sandbox_active_reflection();
        for (unsigned i=0;i<5;++i) {
            ID3D11PixelShader* replacement=nullptr;
            if (!compile(names[i],"PSReflection",&replacement)) return false;
            if (!reflection.ps[i]) {
                char detail[160];
                sprintf_s(detail,"index=%u main=%p region=%p frame=%p region_size=%u",
                    i,renderer.reflection.ps[i],renderer.region_reflection.ps[i],
                    renderer.reflection.frame,renderer.scene_region_size);
                renderer.trace.write("fresh-reflection-slot",detail,true);
                return false;
            }
            reflection.ps[i]->Release();
            reflection.ps[i]=replacement;
        }
#ifdef C3X_RENDERER64_FRESH
        // Keep the authored projection and equations. Only the shared camera
        // is a draw constant; stable instance placement stays in immutable GPU data.
        auto instance_source=source("objects.hlsl");
        for(unsigned i=0;i<2;++i){
            Microsoft::WRL::ComPtr<ID3DBlob> code,errors;
            HRESULT hr=D3DCompile(instance_source.data(),instance_source.size(),"fresh_instances",nullptr,nullptr,
                i?"VSResidentReflectionInstance":"VSResidentInstance","vs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,&errors);
            if(errors)OutputDebugStringA(static_cast<char const*>(errors->GetBufferPointer()));
            if(FAILED(hr) || FAILED(renderer.device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&vegetation_instances[i])))return false;
        }
        char pulled_control[8]={};
        if(!(c3x_renderer::render_core::cached_environment("C3X_RENDERER_PULLED_SUBMISSION",pulled_control,sizeof(pulled_control))&&
             pulled_control[0]=='0') && !compile_pulled_vertex(&pulled_vertex,&pulled_feature))
            renderer.trace.write("fresh-pulled-shader","unavailable; ordinary per-record submission",true);
#endif
        installed = true;
        return true;
    }
};

// One light-space field for all receivers in the visible scene. The retained
// renderer supplies caster meshes, cutout textures and the light basis; the
// sandbox owns target selection, residency and the field's lifetime.
// Performance policy switches, read once. Defaults are the fast paths; each
// switch restores the previous behavior for A/B comparison and escape.
struct SandboxPerfOptions {
    bool legacy=false;          // C3X_RENDERER_STATIC_LEGACY=1: synchronous full-quality static redraws
    float bootstrap_scale=1.f;  // Native-resolution fallback; reduced scale is diagnostic only (0 disables).
    double refine_pixels=0;     // C3X_RENDERER_REFINE_PIXELS: fixed per-frame refinement budget
    bool shadow_tight=false;    // C3X_RENDERER_SHADOW_TIGHT_FIT=1: refit shadows to every view
    SandboxPerfOptions(){
        char value[32]{};
        legacy=GetEnvironmentVariableA("C3X_RENDERER_STATIC_LEGACY",value,sizeof(value)) && value[0]=='1';
        if(GetEnvironmentVariableA("C3X_RENDERER_BOOTSTRAP_SCALE",value,sizeof(value))){
            float scale=float(std::atof(value));bootstrap_scale=std::isfinite(scale)?std::clamp(scale,0.f,1.f):1.f;
            if(bootstrap_scale>0 && bootstrap_scale<.25f)bootstrap_scale=.25f;
        }
        if(GetEnvironmentVariableA("C3X_RENDERER_REFINE_PIXELS",value,sizeof(value))){
            double pixels=std::atof(value);refine_pixels=std::isfinite(pixels)?std::max(0.,pixels):0.;
        }
        shadow_tight=GetEnvironmentVariableA("C3X_RENDERER_SHADOW_TIGHT_FIT",value,sizeof(value)) && value[0]=='1';
    }
};
inline SandboxPerfOptions const& sandbox_perf_options(){static SandboxPerfOptions options;return options;}

struct SandboxSceneShadow {
    using Shadow = c3x_renderer::render_core::SourceShadow;
    using Grid = c3x_renderer::render_core::ShadowSamplingGrid;
    Grid sampling_grid;
    SandboxPassWorkload* work=nullptr;
    ID3D11Texture2D* texture = nullptr;
    std::array<ID3D11RenderTargetView*,Grid::max_pages> targets{};
    ID3D11ShaderResourceView* view = nullptr;
    ID3D11VertexShader* vertex = nullptr;
    ID3D11VertexShader* instance_vertex = nullptr;
    ID3D11VertexShader* rigid_vertex = nullptr;
    ID3D11PixelShader* opaque = nullptr;
    ID3D11PixelShader* cutout = nullptr;
    ID3D11InputLayout* layout = nullptr;
    ID3D11InputLayout* feature_layout = nullptr;
    ID3D11InputLayout* natural_layout = nullptr;
    ID3D11InputLayout* city_layout = nullptr;
    ID3D11InputLayout* instance_layout = nullptr;
    ID3D11Buffer* constants = nullptr;
    ID3D11BlendState* maximum = nullptr;
    ID3D11RasterizerState* raster = nullptr;
    struct InstanceGroup {
        using Instance=c3x_renderer::fidelity::MeshInstance;
        struct Part {Shadow::Bounds bounds;unsigned first=0,count=0;std::array<float,4> projected{};};
        Shadow::Caster source;
        std::vector<Part> parts;

    };
    using Submission=c3x_renderer::render_core::SharedInstanceSubmission;
    Submission::Lease shared_front,shadow_front;
    Submission::CpuLease shared_metadata,shadow_metadata,page_metadata;
    std::uint64_t submission_generation=0;
    std::vector<InstanceGroup> instance_groups;
    std::vector<Shadow::Caster> casters;
    std::vector<Shadow::Caster> caster_inputs;
    using Membership=c3x_renderer::render_core::SceneMembership<CachedVertexChunk,geometry_layer_count>;
    Membership::Lease caster_lease;
    using AtlasInputs=c3x_renderer::render_core::ShadowCasterProofs<CachedGeometryProof>;
    AtlasInputs atlas_inputs;
    // Exact keys of caster_inputs, computed once per membership. Every
    // camera step marks, updates and looks up each input several times.
    std::vector<AtlasInputs::Key> input_keys;
    // Footprints of input_keys. Both are values, so they outlive the borrowed
    // caster_inputs that retire_geometry_selection clears on every step.
    std::vector<std::array<int,4>> input_sources;
    std::uint64_t projection_serial=0;
    // Retained rasters bake the shadow field they sampled; their contributor
    // proofs cannot see a caster that enters or leaves the field (a reveal can
    // publish neighbor forests a moment after a repair drew the old field).
    // Each caster-set change journals the screen footprints of the casters
    // that entered or left. Rasters behind change_floor refresh completely.
    struct ShadowChange {std::uint64_t serial=0;std::array<int,4> source{};};
    std::vector<ShadowChange> shadow_changes;
    std::uint64_t change_serial=0,change_floor=0;
    std::uint64_t proof_membership_signature=~std::uint64_t(0);
    using Pages=c3x_renderer::render_core::ShadowPageContents<AtlasInputs::Key>;
    Pages page_contents;
    std::vector<Submission::Key> prepared_instances;
    std::uint64_t caster_collections=0,caster_reuses=0,group_builds=0,group_reuses=0,body_placement_builds=0,body_placement_reuses=0;
    std::uint64_t prepared_signature=~std::uint64_t(0);
    std::array<float,4> prepared_box{};
    std::array<float,12> prepared_light{};
    struct TerrainBatch {
        std::vector<AtlasInputs::Key> keys;
        std::vector<Shadow::Caster> draws;
        std::array<std::uint64_t,3> context{};
        ID3D11Buffer *vertices=nullptr,*indices=nullptr;
        std::size_t gpu_bytes=0;
        Submission::CpuLease gpu_charge;
        ~TerrainBatch(){if(vertices)vertices->Release();if(indices)indices->Release();}
    };
    std::map<std::tuple<int,int,unsigned>,std::unique_ptr<TerrainBatch>> terrain_batches;
    Submission::CpuLease terrain_metadata;
    std::uint64_t terrain_batch_builds=0,terrain_batch_reuses=0;
    bool atlas_complete=false;
    std::array<float,12> part_light_basis{};
    std::uint64_t part_group_builds=0,instance_projections=0,draw_projections=0;
    bool part_projection_valid=false;
    unsigned resolution=Grid::page_texels;
    bool ready=false;
    std::size_t production_field_bytes=0;
    std::size_t bytes()const{std::size_t result=(texture?Grid::texture_bytes:0)+(constants?80u:0)+production_field_bytes;
        for(auto const& batch:terrain_batches)result+=batch.second->gpu_bytes;return result;}
    std::size_t metadata_bytes()const{std::size_t result=sizeof(sampling_grid)+sizeof(targets)+page_contents.bytes();
        for(auto const& batch:terrain_batches)result+=sizeof(TerrainBatch)+96u+batch.second->keys.capacity()*sizeof(AtlasInputs::Key)+batch.second->draws.capacity()*sizeof(Shadow::Caster);return result;}
    ID3D11ShaderResourceView* production_view = nullptr;
    float box[4] = {};
    std::array<float,4> wrap_basis{};
    std::uint64_t signature = 0;
    std::uint64_t caster_signature = ~std::uint64_t(0), receiver_revision = 0;
    std::array<float,12> light_basis{};
    Grid receiver_grid;
    std::array<float,4> receiver_wrap{};
    std::array<std::uint64_t,9> receiver_key{};
    std::array<float,12> receiver_light{};
    bool receiver_grid_valid=false;
    std::uint64_t receiver_visits=0,receiver_builds=0,receiver_reuses=0;
    unsigned builds = 0, draws = 0;
    // Identity of the shadow *sampling* (span, light, wrap, scene). Retained
    // static pixels stay valid while it is unchanged, even when the page
    // window slides and new pages are drawn for newly exposed receivers.
    std::uint64_t sampling_identity=0;
    // World-anchored density: the span only changes when the receiver extent
    // leaves a hysteresis band, so scrolling never refits (and never redraws)
    // every page. A tight per-view fit made each scroll step a full rebuild.
    std::array<float,2> stable_span{};
    std::array<float,12> stable_light{};
    std::uint64_t span_refits=0;
    bool configure_stable(Grid& grid,float const* needed){
        if(sandbox_perf_options().shadow_tight)return grid.configure(needed,renderer.shadow_basis);
        float bounds[4]={needed[0],needed[1],needed[2],needed[3]};
        if(stable_light!=renderer.shadow_basis){stable_span={0,0};stable_light=renderer.shadow_basis;}
        for(unsigned axis=0;axis<2;++axis){
            float want=needed[axis+2]-needed[axis]+2*Grid::guard;
            if(!std::isfinite(want) || !(want>0))return grid.configure(needed,renderer.shadow_basis);
            if(!(stable_span[axis]>=want && stable_span[axis]<=want*1.45f)){
                // Even spans are exact for configure(); 1.15x+2 stays inside the band.
                stable_span[axis]=std::ceil(want*1.15f/2.f)*2.f;++span_refits;
            }
            // configure() derives exactly this (even) span from these bounds.
            bounds[axis+2]=bounds[axis]+stable_span[axis]-2*Grid::guard;
        }
        if(grid.configure(bounds,renderer.shadow_basis))return true;
        stable_span={0,0};
        return grid.configure(needed,renderer.shadow_basis);
    }
    void update_sampling_identity(std::uint64_t scene,std::array<float,4> const& wrap){
        std::uint64_t value=14695981039346656037ull;
        auto mix=[&](void const* data,std::size_t size){auto bytes=static_cast<unsigned char const*>(data);
            for(std::size_t i=0;i<size;++i){value^=bytes[i];value*=1099511628211ull;}};
        mix(sampling_grid.quality_span.data(),sizeof(float)*2);mix(renderer.shadow_basis.data(),sizeof(float)*12);
        mix(wrap.data(),sizeof(float)*4);mix(&scene,sizeof(scene));mix(&renderer.device_generation,sizeof(renderer.device_generation));
        sampling_identity=value;
    }
    template<class T> static void drop(T*& pointer) {if (pointer) pointer->Release(); pointer=nullptr;}
    ~SandboxSceneShadow() {
        if(renderer.source_shadow.borrowed_view==view)renderer.source_shadow.borrowed_view=nullptr;
        renderer.fresh_shadow_working_bytes=0;
        drop(production_view);production_field_bytes=0;
        shared_front.reset();shadow_front.reset();
        terrain_batches.clear();
        drop(texture); for(auto*& target:targets)drop(target); drop(view); drop(vertex); drop(instance_vertex); drop(rigid_vertex);
        drop(opaque); drop(cutout); drop(layout); drop(feature_layout); drop(natural_layout);
        drop(city_layout); drop(instance_layout); drop(constants); drop(maximum); drop(raster);
    }
    bool ensure() {
        if (ready) return true;
        // An interrupted setup keeps its actual owners charged. It cannot
        // silently create another field generation on a subsequent request.
        if(texture || view || vertex || instance_vertex || rigid_vertex || constants || maximum || raster || production_view)return false;
        // Pin the original SRV independently across renderer device reset.
        // The original field and this fixed field keep separate owning views;
        // only the sampling binding is temporarily borrowed.
        production_view=renderer.source_shadow.view;
        if(production_view)production_view->AddRef();
        production_field_bytes=0;
        if(renderer.source_shadow.view){
            ID3D11Resource* resource=nullptr;ID3D11Texture2D* held=nullptr;
            renderer.source_shadow.view->GetResource(&resource);
            if(resource && SUCCEEDED(resource->QueryInterface(__uuidof(ID3D11Texture2D),reinterpret_cast<void**>(&held)))){
                D3D11_TEXTURE2D_DESC desc{};held->GetDesc(&desc);
                if(desc.Format!=DXGI_FORMAT_R32_FLOAT){drop(held);drop(resource);return false;}
                for(unsigned mip=0;mip<desc.MipLevels;++mip)
                    production_field_bytes+=std::size_t(std::max(1u,desc.Width>>mip))*std::max(1u,desc.Height>>mip)*desc.ArraySize*desc.SampleDesc.Count*4;
            }
            drop(held);drop(resource);
        }
        renderer.fresh_shadow_working_bytes=Grid::texture_bytes+80u+production_field_bytes;
        renderer.preserve_process_headroom(true);
        auto root = renderer.shader_root + "/Renderer/native/";
        auto shader = [&](std::string const& path, char const* entry, char const* profile,
                ID3DBlob** result) {
            std::wstring wide(path.begin(),path.end()); ID3DBlob* errors=nullptr;
            HRESULT hr=c3x_renderer::render_core::compile_cached(wide.c_str(),entry,profile,result,&errors);
            if (FAILED(hr)) {
                char detail[256];sprintf_s(detail,"entry=%s hr=0x%08x",entry,unsigned(hr));
                renderer.trace.write("fresh-shadow-shader-failed",detail,true);
            }
            if (errors) {if (FAILED(hr)) std::printf("SANDBOX_CASTER_SHADER %s\n",
                static_cast<char const*>(errors->GetBufferPointer())); errors->Release();}
            return SUCCEEDED(hr);
        };
        ID3DBlob* code=nullptr;
        HRESULT hr=E_FAIL;
        auto caster=root+"environment_refresh/source_caster.hlsl";
        if (!shader(caster,"VS","vs_5_0",&code)) return false;
        hr=renderer.device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&vertex);
        D3D11_INPUT_ELEMENT_DESC e[]={
            {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"TEXCOORD",1,DXGI_FORMAT_R32_FLOAT,0,60,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"TEXCOORD",2,DXGI_FORMAT_R32G32B32A32_FLOAT,0,120,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"TEXCOORD",3,DXGI_FORMAT_R32_FLOAT,0,76,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"TEXCOORD",4,DXGI_FORMAT_R32G32B32A32_FLOAT,0,152,D3D11_INPUT_PER_VERTEX_DATA,0}};
        if (SUCCEEDED(hr)) hr=renderer.device->CreateInputLayout(e,5,code->GetBufferPointer(),code->GetBufferSize(),&layout);
        e[4].AlignedByteOffset=0;e[3].AlignedByteOffset=32;e[1].AlignedByteOffset=32;
        e[2].AlignedByteOffset=36;e[2].Format=DXGI_FORMAT_R32G32B32_FLOAT;
        if (SUCCEEDED(hr)) hr=renderer.device->CreateInputLayout(e,5,code->GetBufferPointer(),code->GetBufferSize(),&feature_layout);
        e[4].AlignedByteOffset=76;e[3].AlignedByteOffset=56;e[0].AlignedByteOffset=40;
        e[1].AlignedByteOffset=72;e[2].AlignedByteOffset=12;
        e[2].Format=DXGI_FORMAT_R32G32B32A32_FLOAT;
        if (SUCCEEDED(hr)) hr=renderer.device->CreateInputLayout(e,5,code->GetBufferPointer(),code->GetBufferSize(),&natural_layout);
        e[0].AlignedByteOffset=12;e[1].AlignedByteOffset=40;e[2].AlignedByteOffset=68;
        e[2].Format=DXGI_FORMAT_R32G32B32_FLOAT;e[3].AlignedByteOffset=52;
        e[4].AlignedByteOffset=80;e[4].Format=DXGI_FORMAT_R32G32_FLOAT;
        if (SUCCEEDED(hr)) hr=renderer.device->CreateInputLayout(e,5,code->GetBufferPointer(),code->GetBufferSize(),&city_layout);
        drop(code);
        if (SUCCEEDED(hr) && shader(caster,"PSOpaque","ps_5_0",&code))
            hr=renderer.device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&opaque);
        else hr=E_FAIL;
        drop(code);
        if (SUCCEEDED(hr) && shader(caster,"PSCutout","ps_5_0",&code))
            hr=renderer.device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&cutout);
        else hr=E_FAIL;
        drop(code);
        if (SUCCEEDED(hr) && shader(root+"source_fidelity/instance_caster.hlsl","VSResidentPlacedInstance","vs_5_0",&code)) {
            hr=renderer.device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&instance_vertex);
            if (SUCCEEDED(hr)) hr=c3x_renderer::render_core::create_resident_instance_layout(renderer.device,code,&instance_layout);
        } else hr=E_FAIL;
        drop(code);
        if (SUCCEEDED(hr) && shader(root+"city_fidelity/rigid_caster.hlsl","VSResidentPlacedCaster","vs_5_0",&code))
            hr=renderer.device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&rigid_vertex);
        else hr=E_FAIL;
        drop(code);
        D3D11_TEXTURE2D_DESC d={};d.Width=d.Height=resolution;d.ArraySize=Grid::max_pages;d.MipLevels=d.SampleDesc.Count=1;
        d.Format=DXGI_FORMAT_R32_FLOAT;
        d.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
        if (SUCCEEDED(hr)) hr=renderer.device->CreateTexture2D(&d,nullptr,&texture);
        D3D11_RENDER_TARGET_VIEW_DESC rt={};rt.Format=d.Format;
        rt.ViewDimension=D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
        rt.Texture2DArray.ArraySize=1;
        for(unsigned page=0;page<Grid::max_pages && SUCCEEDED(hr);++page){
            rt.Texture2DArray.FirstArraySlice=page;
            hr=renderer.device->CreateRenderTargetView(texture,&rt,&targets[page]);
        }
        D3D11_SHADER_RESOURCE_VIEW_DESC sr={};sr.Format=d.Format;
        sr.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
        sr.Texture2DArray.ArraySize=Grid::max_pages;sr.Texture2DArray.MipLevels=1;
        if (SUCCEEDED(hr)) hr=renderer.device->CreateShaderResourceView(texture,&sr,&view);
        D3D11_BUFFER_DESC b={};b.ByteWidth=80;b.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        if (SUCCEEDED(hr)) hr=renderer.device->CreateBuffer(&b,nullptr,&constants);
        D3D11_BLEND_DESC blend={};auto& color=blend.RenderTarget[0];color.BlendEnable=TRUE;
        color.SrcBlend=color.DestBlend=color.SrcBlendAlpha=color.DestBlendAlpha=D3D11_BLEND_ONE;
        color.BlendOp=color.BlendOpAlpha=D3D11_BLEND_OP_MAX;
        color.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;
        if (SUCCEEDED(hr)) hr=renderer.device->CreateBlendState(&blend,&maximum);
        D3D11_RASTERIZER_DESC raster_desc={};raster_desc.FillMode=D3D11_FILL_SOLID;
        raster_desc.CullMode=D3D11_CULL_NONE;raster_desc.DepthClipEnable=FALSE;
        if (SUCCEEDED(hr)) hr=renderer.device->CreateRasterizerState(&raster_desc,&raster);
        if (FAILED(hr)) {
            char detail[128];sprintf_s(detail,"resources hr=0x%08x",unsigned(hr));
            renderer.trace.write("fresh-shadow-setup-failed",detail,true);
            return false;
        }
        renderer.source_shadow.borrowed_view=view;
        ready=true;
        return true;
    }
    bool refresh_casters(std::uint64_t scene) {
        if(caster_signature==scene)return true;
        // Production may replace or evict vertex chunks without changing the
        // D3D device. Caster buffer and instance pointers must follow that
        // residency generation; the shadow target and shaders can remain.
        ID3D11Buffer* empty[]={nullptr,nullptr};
        UINT strides[]={0,0},offsets[]={0,0};
        renderer.context->IASetVertexBuffers(0,2,empty,strides,offsets);
        renderer.context->IASetIndexBuffer(nullptr,DXGI_FORMAT_UNKNOWN,0);
        std::vector<Shadow::Caster> next;
        renderer.collect_shadow_casters(renderer.geometry_vertex_buffers,next);++caster_collections;
        // collect_shadow_casters already includes cliffs. Their old extra
        // submission duplicated identical maximum-height samples.
        std::unordered_set<AtlasInputs::Key,AtlasInputs::Hash> unique;unique.reserve(next.size());
        std::vector<AtlasInputs::Key> keys;keys.reserve(next.size());std::size_t kept=0;
        for(std::size_t i=0;i<next.size();++i){auto key=caster_key(next[i]);
            if(!unique.insert(key).second)continue;next[kept++]=next[i];keys.push_back(key);}
        next.resize(kept);
        bool same=caster_inputs.size()==keys.size() && input_keys==keys;
        if(same)++caster_reuses;
        else if(input_keys==keys && input_sources.size()==keys.size())caster_inputs=std::move(next); // re-borrowed after retirement
        else{
            ++change_serial;
            std::unordered_set<AtlasInputs::Key,AtlasInputs::Hash> previous;
            bool keyed=input_keys.size()==input_sources.size();
            if(keyed){previous.reserve(input_keys.size());previous.insert(input_keys.begin(),input_keys.end());}
            std::vector<std::array<int,4>> changed;
            auto note=[&](std::array<int,4> const& source){
                if(source[0]<source[2] && source[1]<source[3])changed.push_back(source);};
            std::vector<std::array<int,4>> sources;sources.reserve(next.size());
            for(auto const& caster:next)sources.push_back(caster.source);
            if(keyed){
                for(std::size_t i=0;i<input_keys.size();++i)if(!unique.count(input_keys[i]))note(input_sources[i]);
                for(std::size_t i=0;i<next.size();++i)if(!previous.count(keys[i]))note(sources[i]);
            }
            // Wrapped copies and layers share footprints. A wholesale
            // replacement (first build, new scope) is not a local edit.
            std::sort(changed.begin(),changed.end());changed.erase(std::unique(changed.begin(),changed.end()),changed.end());
            constexpr std::size_t local_limit=16384,journal_limit=65536;
            if(!keyed || changed.size()>local_limit)change_floor=change_serial;
            else for(auto const& source:changed)shadow_changes.push_back({change_serial,source});
            if(shadow_changes.size()>journal_limit){
                auto drop=shadow_changes.size()-journal_limit/2;
                change_floor=std::max(change_floor,shadow_changes[drop-1].serial);
                shadow_changes.erase(shadow_changes.begin(),shadow_changes.begin()+std::ptrdiff_t(drop));
            }
            caster_inputs=std::move(next);input_keys=std::move(keys);input_sources=std::move(sources);
        }
        caster_lease=renderer.geometry_vertex_buffers.publish();
        caster_signature=scene;
        return true;
    }
    Submission::Key shared_caster_placement_key(Shadow::Caster const& caster)const{
        Submission::Key result{};auto facts=caster_key(caster);std::copy(facts.begin(),facts.end(),result.begin());
        result.back()=1;return result;
    }
    template<class BodyInputs> bool body_placements_covered(Submission::Generation const& candidate,BodyInputs const& inputs)const {
        return inputs.covers(candidate);
    }
    template<class BodyInputs> bool append_body_placements(Submission::Builder& builder,BodyInputs const& inputs) {
        bool complete=true;
        inputs.visit([&](unsigned,Submission::Key const& key,GeometryDrawRecord const& record,unsigned count){
            auto draw=GeometryDrawReference(record);auto const& mesh=draw.content();if(!complete)return;
            Submission::Range range;
            if(builder->delta && draw.occurrence){
                auto owner=draw.occurrence->owner;auto source=renderer.resident_content.lease(owner);
                auto cached=renderer.resident_content.resolve(owner);
                if(source && cached && cached->mesh && cached->mesh.get()==source.get() && cached->mesh->proof &&
                        renderer.raster_content_valid(*cached->mesh->proof)){
                    Submission::RetainedSource proof;proof.source=source;proof.owner={owner.slot,owner.generation};
                    proof.canonical_source=Submission::Generation::source_key(mesh.instances.get(),mesh.instance_material);
                    if(renderer.shared_instances.reuse_range(builder,key,count,proof,range))return;
                    if(builder->complete){complete=false;return;}
                }
            }
            complete=renderer.shared_instances.append(builder,key,mesh.instances.get(),
                mesh.instances->data(),count,draw.natural_projection(),
                float(draw.translation_x()),float(draw.translation_y()),float(draw.translation_y()),mesh.instance_material,range);
        });return complete;
    }
    template<class BodyInputs,class RetireCompletedPlans> bool prepare_instances(BodyInputs const& inputs,RetireCompletedPlans const& retire_completed_plans,
            Submission::Generation const* checked=nullptr,bool covered=false,bool shadow_only=false) {
        auto refused=[&](char const* stage){
            char detail[240];std::snprintf(detail,sizeof(detail),
                "stage=%s shadow=%u bytes=%zu rejected=%u body_records=%u shadow_records=%u",
                stage,unsigned(shadow_only),renderer.shared_instances.bytes(),renderer.shared_instances.rejected,
                shared_front?shared_front->records:0,shadow_front?shadow_front->records:0);
            renderer.trace.write("instance-placement-refused",detail,true);return false;
        };
        // Body/reflection and shadow keep independent active leases under
        // the same joint allowance; a page change cannot repack body values.
        // A serial identifies a new exact prepared caster selection; its input
        // equality is already checked by membership/box/light preparation.
        auto& front=shadow_only?shadow_front:shared_front;
        auto& front_metadata=shadow_only?shadow_metadata:shared_metadata;
        Submission::Key identity={caster_signature,renderer.device_generation,renderer.content_revision,
            renderer.geometry_vertex_buffers.revision(),++submission_generation,0,0,unsigned(shadow_only)};
        auto matches=[&](Submission::Generation const& candidate){
            if(candidate.identity[1]!=renderer.device_generation || candidate.identity[2]!=renderer.content_revision || candidate.identity[7]!=unsigned(shadow_only))return false;
            if(!shadow_only&&!(checked==&candidate?covered:body_placements_covered(candidate,inputs)))return false;
            if(shadow_only)for(auto const& caster:casters)if(caster.instances && !caster.instances->empty()){
                auto range=candidate.find(shared_caster_placement_key(caster));if(range.count!=caster.instances->size())return false;
            }return true;
        };
        auto reused=renderer.shared_instances.valid(front)&&matches(*front)?front:renderer.shared_instances.find_covering(matches);
        if(reused&&!shadow_only){shared_front=std::move(reused);++body_placement_reuses;return true;}
        // Completed pass selections pin the old placement generation.
        // Retire them before charging a replacement, while preserving
        // the active union and any independently held consumer leases.
        if(!reused&&!shadow_only)retire_completed_plans();
        if(!renderer.shared_instances.reserve_index_scratch())return refused("index_scratch");
        front_metadata.reset();std::size_t parts=0;
        if(shadow_only)for(auto const& caster:casters)if(caster.instances && !caster.instances->empty())++parts;
        auto metadata=renderer.shared_instances.retain_metadata(parts*(sizeof(InstanceGroup)*2+sizeof(InstanceGroup::Part)*2+sizeof(std::array<std::uintptr_t,9>)+96u));
        if(!metadata)return refused("metadata");
        // Packed placements own no draw sources. The current resident/caster
        // publications below each pass already own the actually bound meshes.
        auto builder=reused?Submission::Builder{}:renderer.shared_instances.begin_retained(identity,{},true,renderer.shared_instances.valid(front)?front:Submission::Lease{});
        if(!reused && !builder)return refused("builder");
        if(!reused && !shadow_only && !append_body_placements(builder,inputs))return refused("body_append");
        // Only the required current membership owns source meshes. Reusable
        // placement ranges retain weak, epoch-protected identities instead.
        std::map<std::uint64_t,c3x_renderer::render_core::ContentHandle> source_handles;
        Submission::CpuLease source_handle_metadata;
        if(!reused&&shadow_only){
            using SourceHandle=decltype(source_handles)::value_type;
            source_handle_metadata=renderer.shared_instances.retain_metadata(sizeof(source_handles)+
                caster_lease->content.size()*(sizeof(SourceHandle)+64u));
            if(!source_handle_metadata)return refused("source_metadata");
            for(auto const& layer:caster_lease->records)for(auto const& record:layer)
                if(record.owner.generation && record.content().instances && !record.content().instances->empty())
                    source_handles.emplace(record.owner.generation,record.owner);
        }
        auto retain_source=[&](Submission::Key const& key,
                c3x_renderer::render_core::ContentHandle owner,void const* instances,float material){
            auto source=renderer.resident_content.lease(owner);
            if(!source)return true; // Dynamic/unproved ranges are never carried.
            Submission::RetainedSource proof;proof.source=source;proof.owner={owner.slot,owner.generation};
            proof.canonical_source=Submission::Generation::source_key(instances,material);
            auto canonical=builder->sources.find(proof.canonical_source);
            auto range=builder->find(key);
            proof.canonical=canonical!=builder->sources.end() && canonical->second.first==range.first;
            return renderer.shared_instances.retain_source(builder,key,std::move(proof));
        };
        if(!reused&&!shadow_only){bool retained=true;
            inputs.visit([&](unsigned,Submission::Key const& key,GeometryDrawRecord const& record,unsigned){
                auto draw=GeometryDrawReference(record);auto const& mesh=draw.content();if(!retained || !draw.occurrence)return;
                retained=retain_source(key,draw.occurrence->owner,
                    mesh.instances.get(),mesh.instance_material);
            });if(!retained)return refused("body_source");
        }
        std::map<std::array<std::uintptr_t,9>,std::size_t> lookup;
        std::size_t total=0;
        if(shadow_only)for(auto const& caster:casters){
            if(!caster.instances || caster.instances->empty())continue;
            std::array<std::uintptr_t,9> key={reinterpret_cast<std::uintptr_t>(caster.vertices),
                reinterpret_cast<std::uintptr_t>(caster.indices),caster.vertex_offset,caster.index_offset,caster.count,
                std::uintptr_t(caster.index_format),caster.binding==0xffffffffu?caster.layer:caster.binding,
                std::uintptr_t(caster.rigid),caster.stride};
            auto found=lookup.find(key);
            if(found==lookup.end()){
                found=lookup.emplace(key,instance_groups.size()).first;instance_groups.emplace_back();instance_groups.back().source=caster;
            }
            auto& group=instance_groups[found->second];
            auto placement_key=shared_caster_placement_key(caster);
            auto range=reused?reused->find(placement_key):Submission::Range{};
            if(!reused && caster.content_generation){
                auto owner=source_handles.find(caster.content_generation);
                if(owner!=source_handles.end()){
                    auto source=renderer.resident_content.lease(owner->second);auto cached=renderer.resident_content.resolve(owner->second);
                    if(source && cached && cached->mesh && cached->mesh.get()==source.get() && cached->mesh->proof &&
                            renderer.raster_content_valid(*cached->mesh->proof)){
                        Submission::RetainedSource proof;proof.source=source;proof.owner={owner->second.slot,owner->second.generation};
                        proof.canonical_source=Submission::Generation::source_key(caster.instances,caster.instance_material);
                        renderer.shared_instances.reuse_range(builder,placement_key,unsigned(caster.instances->size()),proof,range);
                        if(builder->complete)return refused("source_reuse");
                    }
                }
            }
            if(!reused && !range && !renderer.shared_instances.append(builder,placement_key,caster.instances,caster.instances->data(),
                unsigned(caster.instances->size()),caster.instances->front().projection,
                caster.offset[0],caster.offset[1],caster.offset[2],caster.instance_material,range))return refused("shadow_append");
            if(!reused && caster.content_generation){
                auto owner=source_handles.find(caster.content_generation);
                if(owner!=source_handles.end() && !retain_source(placement_key,owner->second,caster.instances,caster.instance_material))return refused("shadow_source");
            }
            InstanceGroup::Part part;part.first=range.first;part.count=range.count;
            for(int axis=0;axis<3;++axis){part.bounds.low[axis]=caster.bounds.low[axis]+caster.offset[axis];
                part.bounds.high[axis]=caster.bounds.high[axis]+caster.offset[axis];}
            group.parts.push_back(part);total+=range.count;
        }
        if(!reused && renderer.shared_instances.valid(front) && !renderer.shared_instances.carry_forward(builder,[&](Submission::RetainedSource const& proof){
            auto source=proof.source.lock();
            auto cached=renderer.resident_content.resolve({std::size_t(proof.owner[0]),proof.owner[1]});
            return source && cached && cached->mesh && cached->mesh->proof && cached->mesh.get()==source.get() &&
                renderer.raster_content_valid(*cached->mesh->proof);
        }))return refused("carry_forward");
        auto uploads=renderer.shared_instances.uploads;auto bytes=renderer.shared_instances.uploaded_bytes;
        front=reused?reused:renderer.shared_instances.upload(builder,renderer.device,renderer.context);
        if(!front)return refused("upload");
        front_metadata=std::move(metadata);
        if(shadow_only)++group_builds;else ++body_placement_builds;
        renderer.frame_content_uploads+=renderer.shared_instances.uploads-uploads;
        renderer.frame_upload_bytes+=renderer.shared_instances.uploaded_bytes-bytes;
        work->upload(renderer.shared_instances.uploaded_bytes-bytes);
        std::printf("SANDBOX_SHADOW_INSTANCES casters=%zu groups=%zu instances=%zu bytes=%zu\n",
            casters.size(),instance_groups.size(),total,total*sizeof(unsigned));std::fflush(stdout);return true;
    }
    template<class BodyInputs,class RetireCompletedPlans> bool prepare_required_instances(BodyInputs const& inputs,
            RetireCompletedPlans const& retire_completed_plans,bool shadow_only=false) {
        if(prepare_instances(inputs,retire_completed_plans,nullptr,false,shadow_only))return true;
        // Optional shadow reuse must yield to the current view's placements.
        // Retry once under the same joint cap; keep completed placement leases
        // alive until their replacement has uploaded successfully.
        renderer.trace.write("shadow-storage-pressure","retiring optional shadow caches before placement retry",true);
        terrain_batches.clear();terrain_metadata.reset();
        page_contents.clear();atlas_inputs.clear();page_metadata.reset();
        proof_membership_signature=~std::uint64_t(0);atlas_complete=false;
        renderer.fresh_shadow_working_bytes=bytes();
        if(shadow_only)std::vector<InstanceGroup>().swap(instance_groups);
        return prepare_instances(inputs,retire_completed_plans,nullptr,false,shadow_only);
    }
    void batch_terrain_casters() {
        auto terrain=[](unsigned layer){return layer==geometry_land || layer==geometry_natural_terrain || layer==geometry_natural_mountain;};
        using RegionKey=std::tuple<int,int,unsigned>;
        // Admit both the old active metadata and conservative new grouping
        // scratch before constructing a map or growing a temporary vector.
        std::size_t metadata=sizeof(terrain_batches)+casters.size()*sizeof(Shadow::Caster)*2;
        for(auto const& batch:terrain_batches)metadata+=sizeof(TerrainBatch)+96u+
            batch.second->keys.capacity()*sizeof(AtlasInputs::Key)+batch.second->draws.capacity()*sizeof(Shadow::Caster);
        for(auto const& caster:casters)if(terrain(caster.layer))metadata+=sizeof(TerrainBatch)+96u+
            (sizeof(AtlasInputs::Key)+sizeof(Shadow::Caster))*4+160u;
        if(metadata>AtlasInputs::limit || (terrain_metadata?!renderer.shared_instances.resize_metadata(terrain_metadata,metadata):
                !(terrain_metadata=renderer.shared_instances.retain_metadata(metadata)))){
            terrain_batches.clear();terrain_metadata.reset();renderer.sandbox_shadow_meshes.clear();renderer.fresh_shadow_working_bytes=bytes();return;
        }
        std::map<RegionKey,std::vector<Shadow::Caster>> selected;
        std::vector<Shadow::Caster> other;
        for(auto const& caster:casters){
            if(!terrain(caster.layer)){other.push_back(caster);continue;}
            auto mesh=std::static_pointer_cast<CachedMeshGeneration>(caster_lease->content.get({0,caster.content_generation}));
            auto tile=mesh&&mesh->proof?renderer.topology_cache.retained(mesh->proof->tile):nullptr;
            if(!tile || !renderer.raster_content_valid(*mesh->proof)){other.push_back(caster);continue;}
            selected[{tile->appearance.tile_x/8,tile->appearance.tile_y/8,caster.layer}].push_back(caster);
        }
        // No leaving camera generation remains strongly owned. Each active
        // region is compared separately, including its exact wrapped ranges.
        for(auto it=terrain_batches.begin();it!=terrain_batches.end();)
            if(selected.find(it->first)==selected.end())it=terrain_batches.erase(it);else ++it;
        std::array<std::uint64_t,3> context={renderer.topology_cache.scope_sequence(),renderer.content_revision,renderer.device_generation};
        auto& captured=renderer.sandbox_shadow_meshes;
        // Optional CPU observations are not content proofs. Before first
        // merging a region, prove its bytes against the actual resident ranges.
        auto equal_buffer=[&](ID3D11Buffer* source,unsigned offset,std::vector<std::uint8_t> const& bytes){
            if(!source)return false;D3D11_BUFFER_DESC desc{};source->GetDesc(&desc);
            if(offset>desc.ByteWidth || bytes.size()>desc.ByteWidth-offset)return false;
            desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=desc.MiscFlags=desc.StructureByteStride=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            ID3D11Buffer* staging=nullptr;
            if(FAILED(renderer.device->CreateBuffer(&desc,nullptr,&staging)))return false;
            renderer.context->CopyResource(staging,source);++work->calls.copies;D3D11_MAPPED_SUBRESOURCE mapped{};
            bool equal=false;if(SUCCEEDED(renderer.context->Map(staging,0,D3D11_MAP_READ,0,&mapped))){
                equal=!std::memcmp(static_cast<std::uint8_t const*>(mapped.pData)+offset,bytes.data(),bytes.size());
                renderer.context->Unmap(staging,0);
            }staging->Release();return equal;
        };
        for(auto const& entry:selected){
            std::vector<AtlasInputs::Key> keys;keys.reserve(entry.second.size());
            for(auto const& caster:entry.second)keys.push_back(caster_key(caster));std::sort(keys.begin(),keys.end());
            auto previous=terrain_batches.find(entry.first);
            if(previous!=terrain_batches.end() && previous->second->context==context && previous->second->keys==keys){
                other.insert(other.end(),previous->second->draws.begin(),previous->second->draws.end());++terrain_batch_reuses;continue;
            }
            if(previous!=terrain_batches.end())terrain_batches.erase(previous);
            auto batch=std::make_unique<TerrainBatch>();batch->keys=std::move(keys);batch->context=context;batch->draws=entry.second;
            std::map<AtlasInputs::Key,Shadow::Caster> canonical;
            std::set<std::array<float,3>> offsets;
            for(auto caster:entry.second){offsets.insert({caster.offset[0],caster.offset[1],caster.offset[2]});
                std::fill(std::begin(caster.offset),std::end(caster.offset),0.f);canonical.emplace(caster_key(caster),caster);}
            std::vector<std::uint8_t> vertices;std::vector<std::uint32_t> indices;Shadow::Bounds bounds{};
            unsigned stride=0;bool merge=!captured.empty(),filled=false;
            std::size_t vertex_bytes=0,index_count=0,staging_bytes=0;
            for(auto const& source:canonical){if(!merge)break;auto const& caster=source.second;
                if(!caster.vertices || !caster.indices){merge=false;break;}
                auto mesh=std::static_pointer_cast<CachedMeshGeneration>(caster_lease->content.get({0,caster.content_generation}));
                auto tile=mesh&&mesh->proof?renderer.topology_cache.retained(mesh->proof->tile):nullptr;
                auto cpu=tile?captured.find({tile->appearance.tile_x,tile->appearance.tile_y,caster.layer}):captured.end();
                if(cpu==captured.end()){merge=false;break;}
                if(cpu->second.vertices.size()>Submission::budget-vertex_bytes || cpu->second.index_count>Submission::budget/4-index_count){merge=false;break;}
                vertex_bytes+=cpu->second.vertices.size();index_count+=cpu->second.index_count;
                D3D11_BUFFER_DESC desc{};caster.vertices->GetDesc(&desc);staging_bytes=std::max(staging_bytes,std::size_t(desc.ByteWidth));
                caster.indices->GetDesc(&desc);staging_bytes=std::max(staging_bytes,std::size_t(desc.ByteWidth));
            }
            auto merged_bytes=vertex_bytes+index_count*4u;
            Submission::CpuLease scratch;
            if(merge){
                scratch=renderer.shared_instances.retain_metadata(merged_bytes,staging_bytes);
                if(!scratch)merge=false;
                else try{vertices.reserve(vertex_bytes);indices.reserve(index_count);}catch(...){merge=false;}
            }
            for(auto const& source:canonical){
                if(!merge)break;auto const& caster=source.second;
                auto mesh=std::static_pointer_cast<CachedMeshGeneration>(caster_lease->content.get({0,caster.content_generation}));
                auto tile=mesh&&mesh->proof?renderer.topology_cache.retained(mesh->proof->tile):nullptr;
                auto cpu=tile?captured.find({tile->appearance.tile_x,tile->appearance.tile_y,caster.layer}):captured.end();
                if(cpu==captured.end()){merge=false;break;}auto const& content=cpu->second;
                if(!content.vertex_stride || content.vertex_stride!=caster.stride || (stride&&stride!=caster.stride) ||
                        (content.index_stride!=2 && content.index_stride!=4) || content.vertices.size()%content.vertex_stride ||
                        content.index_count!=caster.count || content.indices.size()!=std::size_t(caster.count)*content.index_stride ||
                        (caster.index_format==DXGI_FORMAT_R16_UINT?2u:caster.index_format==DXGI_FORMAT_R32_UINT?4u:0u)!=content.index_stride ||
                        !equal_buffer(caster.vertices,caster.vertex_offset,content.vertices) || !equal_buffer(caster.indices,caster.index_offset,content.indices)){
                    merge=false;break;
                }
                stride=caster.stride;auto base=vertices.size()/stride;
                vertices.insert(vertices.end(),content.vertices.begin(),content.vertices.end());
                for(unsigned i=0;i<content.index_count;++i){std::uint32_t index=0;
                    std::memcpy(&index,content.indices.data()+std::size_t(i)*content.index_stride,content.index_stride);
                    if(index>=content.vertices.size()/stride || base+index>std::numeric_limits<std::uint32_t>::max()){merge=false;break;}
                    indices.push_back(std::uint32_t(base+index));
                }
                for(int axis=0;axis<3;++axis){bounds.low[axis]=filled?std::min(bounds.low[axis],caster.bounds.low[axis]):caster.bounds.low[axis];
                    bounds.high[axis]=filled?std::max(bounds.high[axis],caster.bounds.high[axis]):caster.bounds.high[axis];}filled=true;
            }
            if(merge && !vertices.empty() && !indices.empty() && vertices.size()<=std::numeric_limits<UINT>::max() && indices.size()<=std::numeric_limits<UINT>::max()/4u){
                batch->gpu_charge=renderer.shared_instances.retain_metadata(0,merged_bytes);
                if(batch->gpu_charge){
                D3D11_BUFFER_DESC desc{};desc.Usage=D3D11_USAGE_IMMUTABLE;desc.BindFlags=D3D11_BIND_VERTEX_BUFFER;desc.ByteWidth=UINT(vertices.size());
                D3D11_SUBRESOURCE_DATA data{vertices.data(),0,0};
                if(SUCCEEDED(renderer.device->CreateBuffer(&desc,&data,&batch->vertices))){
                    ++renderer.frame_content_uploads;renderer.frame_upload_bytes+=vertices.size();work->upload(vertices.size(),std::get<2>(entry.first));
                    desc.BindFlags=D3D11_BIND_INDEX_BUFFER;desc.ByteWidth=UINT(indices.size()*4u);data.pSysMem=indices.data();
                    if(SUCCEEDED(renderer.device->CreateBuffer(&desc,&data,&batch->indices))){
                        batch->draws.clear();batch->gpu_bytes=vertices.size()+indices.size()*4u;
                        ++renderer.frame_content_uploads;renderer.frame_upload_bytes+=indices.size()*4u;work->upload(indices.size()*4u,std::get<2>(entry.first));
                        for(auto const& offset:offsets){Shadow::Caster draw;draw.vertices=batch->vertices;draw.indices=batch->indices;
                            draw.count=UINT(indices.size());draw.stride=stride;draw.layer=std::get<2>(entry.first);draw.index_format=DXGI_FORMAT_R32_UINT;
                            draw.bounds=bounds;std::copy(offset.begin(),offset.end(),draw.offset);batch->draws.push_back(draw);}
                    }else drop(batch->vertices);
                }
                }
            }
            if(!batch->gpu_bytes)batch->gpu_charge.reset();
            other.insert(other.end(),batch->draws.begin(),batch->draws.end());terrain_batches.emplace(entry.first,std::move(batch));++terrain_batch_builds;
        }
        captured.clear();casters=std::move(other);
        renderer.fresh_shadow_working_bytes=bytes();
    }
    bool bind_cutout(unsigned layer) {
        auto* context=renderer.context;
        if (layer>=10000) {
            auto mask=renderer.cities.materials[layer-10000][6];
            context->PSSetShaderResources(34,1,&mask);return mask!=nullptr;
        }
        if (layer==geometry_land) return false;
        if (layer==geometry_natural_terrain || layer==geometry_natural_mountain) return true;
        if (layer>=geometry_natural_forest0) {
            auto const& material=renderer.natural.materials[
                renderer.natural.bodies[layer-geometry_natural_forest0].material];
            ID3D11ShaderResourceView* mask=material.channels[6]==0xffffffffu?nullptr:
                renderer.natural.textures[material.channels[6]];
            context->PSSetShaderResources(33,1,&mask);return mask!=nullptr;
        }
        std::array<ID3D11ShaderResourceView*,33> views{};
        std::copy(renderer.feature_texture_views.begin(),renderer.feature_texture_views.end(),views.begin());
        std::copy(renderer.river_rock_texture_views.begin(),renderer.river_rock_texture_views.end(),views.begin()+8);
        std::copy(renderer.bridge_texture_views.begin(),renderer.bridge_texture_views.end(),views.begin()+13);
        std::copy(renderer.resource_texture_views.begin(),renderer.resource_texture_views.end(),views.begin()+21);
        std::copy(renderer.city_base_views.begin(),renderer.city_base_views.end(),views.begin()+29);
        if (layer==geometry_wall) std::copy(renderer.wall_texture_views.begin(),renderer.wall_texture_views.end(),views.begin()+29);
        if (layer==geometry_site) std::copy(renderer.site_views.begin(),renderer.site_views.end(),views.begin()+21);
        if (layer==geometry_mine) std::copy(renderer.mine_base_views.begin(),renderer.mine_base_views.end(),views.begin()+21);
        if (layer==geometry_farm) std::copy(renderer.farm_base_views.begin(),renderer.farm_base_views.end(),views.begin()+21);
        if (layer>=geometry_cliff0 && layer<geometry_natural_terrain)
            views[0]=renderer.cliff_views[renderer.cliff_bundle.assets[layer-geometry_cliff0].texture_index];
        context->PSSetShaderResources(0,33,views.data());return true;
    }
    bool shadow_metadata_admit(std::size_t proofs,std::size_t pages){
        if(proofs>AtlasInputs::limit || pages>AtlasInputs::limit-proofs)return false;
        auto bytes=proofs+pages;
        if(page_metadata)return renderer.shared_instances.resize_metadata(page_metadata,bytes);
        page_metadata=renderer.shared_instances.retain_metadata(bytes);return bool(page_metadata);
    }
    AtlasInputs::Key caster_key(Shadow::Caster const& caster)const{
        auto bits=[](float value){std::uint32_t result;std::memcpy(&result,&value,sizeof(result));return std::uint64_t(result);};
        return {caster.content_generation,caster.version,(std::uint64_t(caster.layer)<<32)|caster.binding,
            bits(caster.bounds.low[0]),bits(caster.bounds.low[1]),bits(caster.bounds.low[2]),
            bits(caster.bounds.high[0]),bits(caster.bounds.high[1]),bits(caster.bounds.high[2]),
            bits(caster.offset[0]),bits(caster.offset[1]),bits(caster.offset[2]),
            (std::uint64_t(caster.count)<<32)|caster.vertex_offset,std::uint64_t(reinterpret_cast<std::uintptr_t>(caster.vertices)),
            std::uint64_t(reinterpret_cast<std::uintptr_t>(caster.indices)),
            (std::uint64_t(caster.index_offset)<<32)|unsigned(caster.index_format),caster.stride,
            // Instances belong to the immutable content generation held by
            // caster_lease. Include their identity and every draw variant;
            // sharing a vertex buffer never proves equal index/instance work.
            std::uint64_t(reinterpret_cast<std::uintptr_t>(caster.instances)),
            bits(caster.instance_material),std::uint64_t(caster.rigid)};
    }
    bool atlas_dependencies(bool append){
#ifdef C3X_RENDERER64_FRESH
        std::array<std::uint64_t,4> context={renderer.topology_cache.scope_sequence(),renderer.content_revision,
            renderer.device_generation,unsigned(renderer.geometry_canonical_world)};
        if(append && (proof_membership_signature!=caster_signature || atlas_inputs.context!=context || !atlas_inputs.complete || !atlas_inputs.valid_all)){
            if(atlas_inputs.context!=context)page_contents.clear();
            auto begin=std::chrono::steady_clock::now();
            try{
                atlas_inputs.begin(context,renderer.raster_dependency_revisions,[&](auto bytes){return shadow_metadata_admit(bytes,page_contents.bytes());});
                for(auto const& caster:caster_inputs)atlas_inputs.mark(caster.content_generation);
                atlas_inputs.finish(false); // retire leaving producers; share charged input registrations with entering producers
                for(auto const& caster:caster_inputs){
                    auto mesh=std::static_pointer_cast<CachedMeshGeneration>(caster_lease->content.get({0,caster.content_generation}));
                    auto tile=mesh&&mesh->proof?mesh->proof->tile:0;auto observed=renderer.topology_cache.retained(tile);
                    if(!atlas_inputs.add(caster.content_generation,mesh?mesh->proof:nullptr,tile,observed?observed->visibility_revision:0,
                        [&](auto const& proof,auto& owner){return renderer.watch_raster_dependencies(proof,owner);},
                        [&](auto const& proof){return renderer.raster_content_valid(proof);}))break;
                }
                atlas_inputs.finish();proof_membership_signature=caster_signature;
            }catch(...){atlas_inputs.complete=false;}
            atlas_inputs.validation_counts.append_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        }
        if(proof_membership_signature!=caster_signature || atlas_inputs.context!=context)return false;
        return atlas_inputs.validate(renderer.raster_dependency_revisions,
            [&](auto const& proof){return renderer.raster_content_valid(proof);},
            [&](auto tile){auto record=renderer.topology_cache.retained(tile);return record?record->visibility_revision:0;});
#else
        return true;
#endif
    }
    std::array<std::uint64_t,9> receiver_identity(std::uint64_t revision,std::uint64_t scene)const{
        auto dims=renderer.world_coast.world().dimensions();
        return {revision,scene,renderer.content_revision,renderer.device_generation,unsigned(renderer.geometry_canonical_world),
            std::uint64_t(dims.width),std::uint64_t(dims.height),unsigned(dims.wrap_x),unsigned(dims.wrap_y)};
    }
    // A complete atlas for exactly this scene and caster set is reused when its
    // inputs are still current. While a camera job runs, ambient frames draw
    // the borrowed completed view: an immutable snapshot whose casters the
    // atlas was built from. The job has already applied newer topology, so
    // validating against it failed and redrew every page on every frame
    // after a unit move (25 pages, ~20 ms) until the job completed.
    bool atlas_reusable(unsigned reuse_failures){
        return !reuse_failures && (renderer.borrowed_scene_frame || atlas_dependencies(false));
    }
    template<class BodyInputs,class RetireCompletedPlans> bool render(GeometryDrawView::Records const& receivers,std::uint64_t scene,std::uint64_t revision,std::uint64_t membership,BodyInputs const& inputs,RetireCompletedPlans const& retire_completed_plans) {
        SandboxPassWorkload::Scope pass(*work,SandboxPassWorkload::shadow);
        // Profiling laps: refresh, receivers, body placements, page proofs,
        // caster selection/placements, terrain batching, bounds, page draws.
        std::array<double,8> laps{};auto lap_clock=std::chrono::steady_clock::now();
        auto lap=[&](unsigned slot){auto now=std::chrono::steady_clock::now();
            laps[slot]+=std::chrono::duration<double,std::milli>(now-lap_clock).count();lap_clock=now;};
        if(!renderer.borrowed_scene_frame)renderer.borrowed_scene_stale=true;
        if (!ensure() || !refresh_casters(membership)) return false;
        lap(0);
        float needed[4]={std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),
            -std::numeric_limits<float>::max(),-std::numeric_limits<float>::max()};
        bool any=false;
        float zero[3]={};
        std::array<float,4> wrap_query{};
        Grid query_grid;
        auto next_receiver_key=receiver_identity(revision,scene);
        bool receiver_reused=receiver_grid_valid&&receiver_key==next_receiver_key&&receiver_light==renderer.shadow_basis;
        if(receiver_reused){query_grid=receiver_grid;wrap_query=receiver_wrap;++receiver_reuses;}
        else{
        if(renderer.geometry_canonical_world){
            auto dims=renderer.world_coast.world().dimensions();
            float low[2]={1e9f,1e9f},high[2]={-1e9f,-1e9f};
            float occurrence_low[2]={1e9f,1e9f},occurrence_high[2]={-1e9f,-1e9f};
            for(unsigned layer=0;layer<geometry_layer_count;++layer)for(auto const& record:receivers[layer]){if(layer==geometry_shadow)continue;++receiver_visits;auto const& b=record.content().world_bounds;
                low[0]=std::min(low[0],b.low[0]+b.low[1]);high[0]=std::max(high[0],b.high[0]+b.high[1]);
                low[1]=std::min(low[1],b.low[0]-b.high[1]);high[1]=std::max(high[1],b.high[0]-b.low[1]);
                occurrence_low[0]=std::min(occurrence_low[0],float(record.tile_x));occurrence_high[0]=std::max(occurrence_high[0],float(record.tile_x));
                occurrence_low[1]=std::min(occurrence_low[1],float(record.tile_y));occurrence_high[1]=std::max(occurrence_high[1],float(record.tile_y));}
            // A viewport narrower than one world can straddle its canonical
            // cut. Its captured occurrence center selects the nearby continuous
            // basis; a fixed seam origin could fold a wide ordinary viewport.
            auto center=[&](unsigned axis,int period){float value=(occurrence_low[axis]+occurrence_high[axis])*.5f;
                return value-std::floor(value/period)*period;};
            if(dims.wrap_x && high[0]-low[0]>dims.width*.5f){wrap_query[0]=center(0,dims.width);wrap_query[2]=float(dims.width);}
            if(dims.wrap_y && high[1]-low[1]>dims.height*.5f){wrap_query[1]=center(1,dims.height);wrap_query[3]=float(dims.height);}
        }
        for (unsigned layer=0;layer<geometry_layer_count;++layer)
            for (auto const& record:receivers[layer]) {
                if (layer==geometry_shadow) continue;
                ++receiver_visits;
                auto const& b=record.content().world_bounds;float offset[3]={};
                float u=(b.low[0]+b.high[0])*.5f,v=(b.low[1]+b.high[1])*.5f;
                if(wrap_query[2]){auto shift=std::floor((u+v-wrap_query[0]+wrap_query[2]*.5f)/wrap_query[2])*wrap_query[2]*.5f;offset[0]-=shift;offset[1]-=shift;}
                if(wrap_query[3]){auto shift=std::floor((u-v-wrap_query[1]+wrap_query[3]*.5f)/wrap_query[3])*wrap_query[3]*.5f;offset[0]-=shift;offset[1]+=shift;}
                auto p=Shadow::project(b,offset,renderer.shadow_basis);
                needed[0]=std::min(needed[0],p[0]);needed[1]=std::min(needed[1],p[1]);
                needed[2]=std::max(needed[2],p[2]);needed[3]=std::max(needed[3],p[3]);
                any=true;
            }
        // An unrevealed viewport has no receivers. Keep a finite neutral
        // extent so empty placement/shadow submissions remain valid.
        if (!any) { needed[0]=needed[1]=0;needed[2]=needed[3]=1; }
        if(!configure_stable(query_grid,needed))return false;
        receiver_grid=query_grid;receiver_wrap=wrap_query;receiver_key=next_receiver_key;receiver_light=renderer.shadow_basis;
        receiver_grid_valid=true;++receiver_builds;
        }
        receiver_revision=revision;lap(1);
        bool body_covered=renderer.shared_instances.valid(shared_front) && inputs.covers(shared_front);
        if(body_covered)++body_placement_reuses;
        if(!body_covered&&!prepare_required_instances(inputs,retire_completed_plans))return false;
        // Which reuse conditions failed (bit set = failed), for the trace below.
        unsigned reuse_failures=unsigned(!atlas_complete)|unsigned(!renderer.shared_instances.valid(shadow_front))<<1|
            unsigned(prepared_signature!=membership)<<2|unsigned(signature!=scene)<<3|unsigned(wrap_basis!=wrap_query)<<4|
            unsigned(light_basis!=renderer.shadow_basis)<<5|unsigned(!sampling_grid.covers(query_grid))<<6;
        if (atlas_reusable(reuse_failures)){if(work->enabled)++work->row().reuses;return true;}
        if(!reuse_failures)reuse_failures=1u<<7;
        if(work->enabled)++work->row().rebuilds;
        lap(2);
        atlas_complete=false;
        if(!sampling_grid.covers(query_grid))sampling_grid=query_grid;
        auto coverage=sampling_grid.coverage();
        std::copy(coverage.begin(),coverage.end(),box);
        bool dependency_proved=atlas_dependencies(true);
        bool proved=dependency_proved;unsigned page_proof_stage=dependency_proved?0u:1u;
        Pages::Context page_context={renderer.topology_cache.scope_sequence(),renderer.content_revision,
            renderer.device_generation,unsigned(renderer.geometry_canonical_world)};
        auto bits=[](float value){std::uint32_t word=0;std::memcpy(&word,&value,4);return std::uint64_t(word);};
        page_context[5]=bits(sampling_grid.quality_span[0]);page_context[6]=bits(sampling_grid.quality_span[1]);
        for(unsigned i=0;i<renderer.shadow_basis.size();++i)page_context[7+i]=bits(renderer.shadow_basis[i]);
        for(unsigned i=0;i<wrap_query.size();++i)page_context[19+i]=bits(wrap_query[i]);
        if(input_keys.size()!=caster_inputs.size()){input_keys.clear();input_sources.clear();
            for(auto const& caster:caster_inputs){input_keys.push_back(caster_key(caster));input_sources.push_back(caster.source);}}
        if(proved)try{
            page_contents.begin_incremental(sampling_grid,page_context,renderer.shadow_basis);
            for(auto const& key:input_keys)page_contents.mark(key);
            proved=page_contents.retire_missing([&](auto bytes){return shadow_metadata_admit(atlas_inputs.bytes(),bytes);});
            if(!proved)page_proof_stage=2;
            for(std::size_t input=0;input<caster_inputs.size();++input){auto const& caster=caster_inputs[input];auto const& key=input_keys[input];
                if(!page_contents.update(key,sampling_grid,[&]{return Shadow::project(caster.bounds,caster.offset,renderer.shadow_basis);},
                    [&](auto bytes){return shadow_metadata_admit(atlas_inputs.bytes(),bytes);})){proved=false;page_proof_stage=3;break;}}
            if(proved){proved=page_contents.finish_incremental(sampling_grid,true,
                [&](auto bytes){return shadow_metadata_admit(atlas_inputs.bytes(),bytes);});if(!proved)page_proof_stage=4;}
        }catch(...){proved=false;}
        if(proved && !shadow_metadata_admit(atlas_inputs.bytes(),page_contents.bytes()))proved=false;
        if(!proved){page_contents.clear();
            // Optional projected page membership can fail while the exact
            // current source dependencies remain fully validated. Keep those
            // bounded registrations charged through the ordinary full redraw.
            if(!dependency_proved || !shadow_metadata_admit(atlas_inputs.bytes(),page_contents.bytes())){
                atlas_inputs.clear();proof_membership_signature=~std::uint64_t(0);page_metadata.reset();
            }
            ++page_contents.refused;
            Pages::Inputs empty;page_contents.select(sampling_grid,page_context,empty,false);}
        casters.clear();lap(3);
        auto coverage_box=std::array<float,4>{box[0],box[1],box[2],box[3]};
        ++projection_serial;
        for(std::size_t input=0;input<caster_inputs.size();++input){auto const& caster=caster_inputs[input];
            auto cached=page_contents.projected(input_keys[input]);
            auto p=cached?*cached:Shadow::project(caster.bounds,caster.offset,renderer.shadow_basis);
            if(Pages::intersects(p,coverage_box)){casters.push_back(caster);casters.back().projected=p;casters.back().projected_serial=projection_serial;}
        }
        std::vector<Submission::Key> placement_keys;
        for(auto const& caster:casters)if(caster.instances&&!caster.instances->empty())placement_keys.push_back(shared_caster_placement_key(caster));
        std::sort(placement_keys.begin(),placement_keys.end());
        if(!renderer.shared_instances.valid(shadow_front) || shadow_front->identity[1]!=renderer.device_generation ||
                shadow_front->identity[2]!=renderer.content_revision || placement_keys!=prepared_instances ||
                (instance_groups.empty() && !placement_keys.empty())){
            std::vector<InstanceGroup>().swap(instance_groups);
            if(!prepare_required_instances(inputs,retire_completed_plans,true))return false;
            prepared_instances=std::move(placement_keys);
        }else ++group_reuses;
        if(proved && !page_metadata){
            // Placement admission retired the optional page proof. Draw every
            // required page from its authoritative casters this time.
            proved=false;Pages::Inputs empty;page_contents.select(sampling_grid,page_context,empty,false);
        }
        ID3D11Buffer* empty_streams[]={nullptr,nullptr};UINT empty_strides[2]={};
        renderer.context->IASetVertexBuffers(0,2,empty_streams,empty_strides,empty_strides);
        renderer.context->IASetIndexBuffer(nullptr,DXGI_FORMAT_UNKNOWN,0);
        lap(4);
        batch_terrain_casters();
        lap(5);
        auto* context=renderer.context;
        std::array<ID3D11ShaderResourceView*,128> empty{};
        context->PSSetShaderResources(0,128,empty.data());
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetConstantBuffers(0,1,&constants);
        context->RSSetState(raster);
        D3D11_VIEWPORT viewport={0,0,float(resolution),float(resolution),0,1};context->RSSetViewports(1,&viewport);
        context->OMSetDepthStencilState(nullptr,0);
        context->OMSetBlendState(maximum,nullptr,0xffffffffu);
        bool needs_draw=!std::all_of(page_contents.reused.begin(),page_contents.reused.begin()+sampling_grid.pages(),[](bool reused){return reused;});
        auto draw_bounds_charge=needs_draw?renderer.shared_instances.retain_metadata(casters.size()*sizeof(std::array<float,4>)):Submission::CpuLease{};
        std::vector<std::array<float,4>> draw_bounds;
        if(draw_bounds_charge)try{
            draw_bounds.reserve(casters.size());
            for(auto const& caster:casters){
                if(caster.projected_serial==projection_serial){draw_bounds.push_back(caster.projected);continue;}
                auto cached=page_contents.projected(caster_key(caster));
                draw_bounds.push_back(cached?*cached:Shadow::project(caster.bounds,caster.offset,renderer.shadow_basis));if(!cached)++draw_projections;}
        }catch(...){std::vector<std::array<float,4>>().swap(draw_bounds);draw_bounds_charge.reset();}
        if(needs_draw && (!part_projection_valid || part_group_builds!=group_builds || std::memcmp(part_light_basis.data(),renderer.shadow_basis.data(),sizeof(part_light_basis)))){
            for(auto& group:instance_groups)for(auto& part:group.parts){part.projected=Shadow::project(part.bounds,zero,renderer.shadow_basis);++instance_projections;}
            part_projection_valid=true;part_group_builds=group_builds;part_light_basis=renderer.shadow_basis;
        }
        auto drawn_pages=unsigned(std::count(page_contents.reused.begin(),page_contents.reused.begin()+sampling_grid.pages(),false));
        lap(6);
        float clear[4]={-1e6f,-1e6f,-1e6f,-1e6f};
        draws=0;
        // A published terrain pixel must sample the completed shadow field.
        // Exposing only two rebuilt pages made unchanged forests briefly lose
        // their shadows after a reveal. Exact page proofs still reuse every
        // unchanged page; finish the bounded dirty set before shading receivers.
        std::array<unsigned,Grid::max_pages> page_order{};
        for(unsigned i=0;i<sampling_grid.pages();++i)page_order[i]=i;
        auto distance=[&](unsigned slot){auto p=sampling_grid.page(slot);
            int dx=2*(p[0]-sampling_grid.low[0])+1-int(sampling_grid.count[0]);
            int dy=2*(p[1]-sampling_grid.low[1])+1-int(sampling_grid.count[1]);return dx*dx+dy*dy;};
        std::stable_sort(page_order.begin(),page_order.begin()+sampling_grid.pages(),[&](unsigned a,unsigned b){return distance(a)<distance(b);});
        for(unsigned ordered=0;ordered<sampling_grid.pages();++ordered){
        unsigned page_slot=page_order[ordered];
        if(page_contents.reused[page_slot])continue;
        auto* target=targets[page_contents.slots[page_slot]];
        context->ClearRenderTargetView(target,clear);work->clear(target);
        context->OMSetRenderTargets(1,&target,nullptr);
        auto page_box=sampling_grid.page_box(page_slot);
        auto page= sampling_grid.page(page_slot);
        // Consecutive casters share the page constants (the wrap offset is
        // zero for all but wrapped copies) and usually their shaders, layout
        // and buffers; re-issuing every state per draw dominated page rebuilds.
        bool constants_set=false;std::array<float,3> last_offset{};
        unsigned last_binding=~0u;bool last_cutout=false;ID3D11PixelShader* last_ps=nullptr;
        ID3D11InputLayout* last_layout=nullptr;ID3D11Buffer* last_vertices=nullptr;UINT last_stride=0,last_vertex_offset=0;
        ID3D11Buffer* last_indices=nullptr;DXGI_FORMAT last_format=DXGI_FORMAT_UNKNOWN;UINT last_index_offset=0;
        context->VSSetShader(vertex,nullptr,0);
        for (std::size_t caster_index=0;caster_index<casters.size();++caster_index) {
            auto const& caster=casters[caster_index];if(caster.instances)continue;
            if(work->enabled)++work->row(caster.layer).tested_records;
            auto bounds=draw_bounds.empty()?Shadow::project(caster.bounds,caster.offset,renderer.shadow_basis):draw_bounds[caster_index];if(draw_bounds.empty())++draw_projections;
            if(bounds[2]<page_box[0] || bounds[0]>page_box[0]+page_box[2] ||
               bounds[3]<page_box[1] || bounds[1]>page_box[1]+page_box[3])continue;
            if(work->enabled)++work->row(caster.layer).accepted_records;
            std::array<float,3> caster_offset{caster.offset[0],caster.offset[1],caster.offset[2]};
            if(!constants_set || caster_offset!=last_offset){
                float settings[20]={};
                std::copy(renderer.shadow_basis.begin(),renderer.shadow_basis.end(),settings);
                for (int i=0;i<3;++i) {
                    settings[i]*=6/page_box[2];settings[4+i]*=6/page_box[3];
                    settings[16+i]=caster.offset[i];
                }
                settings[12]=float(page[0]);settings[13]=float(page[1]);
                context->UpdateSubresource(constants,0,nullptr,settings,0,0);work->upload_buffer(constants);
                constants_set=true;last_offset=caster_offset;
            }
            unsigned binding=caster.binding==0xffffffffu?caster.layer:caster.binding;
            if(binding!=last_binding){last_cutout=bind_cutout(binding);last_binding=binding;}
            auto* pixel=last_cutout?cutout:opaque;
            if(pixel!=last_ps){context->PSSetShader(pixel,nullptr,0);last_ps=pixel;}
            auto* input=caster.stride==88?city_layout:caster.stride==92?natural_layout:caster.stride==48?feature_layout:layout;
            if(input!=last_layout){context->IASetInputLayout(input);last_layout=input;}
            if(caster.vertices!=last_vertices || caster.stride!=last_stride || caster.vertex_offset!=last_vertex_offset){
                UINT stride=caster.stride,offset=caster.vertex_offset;
                context->IASetVertexBuffers(0,1,&caster.vertices,&stride,&offset);
                last_vertices=caster.vertices;last_stride=stride;last_vertex_offset=offset;
            }
            if(caster.indices!=last_indices || caster.index_format!=last_format || caster.index_offset!=last_index_offset){
                context->IASetIndexBuffer(caster.indices,caster.index_format,caster.index_offset);
                last_indices=caster.indices;last_format=caster.index_format;last_index_offset=caster.index_offset;
            }
            context->DrawIndexed(caster.count,0,0);work->draw(caster.count,1,caster.layer);
            ++draws;
        }
        std::vector<unsigned> selected;
        for(auto& group:instance_groups){
            selected.clear();
            for(auto const& part:group.parts){
                if(work->enabled){++work->row(group.source.layer).tested_records;work->row(group.source.layer).tested_instances+=part.count;}
                auto const& bounds=part.projected;
                if(bounds[2]<page_box[0] || bounds[0]>page_box[0]+page_box[2] ||
                   bounds[3]<page_box[1] || bounds[1]>page_box[1]+page_box[3])continue;
                if(work->enabled){++work->row(group.source.layer).accepted_records;work->row(group.source.layer).accepted_instances+=part.count;}
                for(unsigned i=0;i<part.count;++i)selected.push_back(part.first+i);
            }
            if(selected.empty())continue;
            auto const& caster=group.source;
            float settings[20]={};
            std::copy(renderer.shadow_basis.begin(),renderer.shadow_basis.end(),settings);
            for(int axis=0;axis<3;++axis){
                settings[axis]*=6/page_box[2];settings[4+axis]*=6/page_box[3];
            }
            settings[12]=float(page[0]);settings[13]=float(page[1]);
            context->UpdateSubresource(constants,0,nullptr,settings,0,0);work->upload_buffer(constants);
            context->PSSetShader(bind_cutout(caster.binding==0xffffffffu?
                caster.layer:caster.binding)?cutout:opaque,nullptr,0);
            context->VSSetShader(caster.rigid?rigid_vertex:instance_vertex,nullptr,0);
            context->IASetInputLayout(instance_layout);
            context->VSSetShaderResources(15,1,&shadow_front->view);
            context->IASetIndexBuffer(caster.indices,caster.index_format,caster.index_offset);
            for(std::size_t begin=0;begin<selected.size();begin+=Submission::record_limit){
                auto count=std::min(selected.size()-begin,std::size_t(Submission::record_limit));
                if(!renderer.shared_instances.select_indices(renderer.device,context,shadow_front,selected.data()+begin,unsigned(count)))return false;
                ID3D11Buffer* streams[]={caster.vertices,renderer.shared_instances.selection_buffer};
                UINT strides[]={32,4},offsets[]={caster.vertex_offset,renderer.shared_instances.selection_offset};
                context->IASetVertexBuffers(0,2,streams,strides,offsets);
                context->DrawIndexedInstanced(caster.count,UINT(count),0,0,0);work->draw(caster.count,count,caster.layer);
                work->upload(count*sizeof(unsigned),caster.layer);
                ++draws;
            }
        }
        if(!page_contents.complete_incremental(page_slot,proved))return false;
        page_contents.reused[page_slot]=true;
        }
        context->OMSetRenderTargets(0,nullptr,nullptr);
        std::array<std::array<float,4>,64> table{};
        table[0]={float(sampling_grid.low[0]),float(sampling_grid.low[1]),
            float(sampling_grid.count[0]),float(sampling_grid.count[1])};
        table[2]={sampling_grid.inverse_pitch(0),sampling_grid.inverse_pitch(1),
            sampling_grid.page_span(0),sampling_grid.page_span(1)};
        table[1]=wrap_query;
        for(unsigned slot=0;slot<sampling_grid.pages();++slot)
            table[3+slot][0]=page_contents.reused[slot]?float(page_contents.slots[slot]):-1.f;
        wrap_basis=wrap_query;
        context->UpdateSubresource(renderer.source_shadow.table,0,nullptr,table.data(),0,0);work->upload_buffer(renderer.source_shadow.table);
        signature=scene;
        light_basis=renderer.shadow_basis;
        prepared_signature=membership;prepared_box={box[0],box[1],box[2],box[3]};prepared_light=renderer.shadow_basis;
        atlas_complete=std::all_of(page_contents.reused.begin(),page_contents.reused.begin()+sampling_grid.pages(),[](bool complete){return complete;});
        ++builds;
        update_sampling_identity(scene,wrap_query);
        lap(7);
        char detail[384];sprintf_s(detail,"refresh_ms=%.2f receivers_ms=%.2f body_ms=%.2f proofs_ms=%.2f casters_ms=%.2f terrain_ms=%.2f bounds_ms=%.2f draw_ms=%.2f casters=%zu inputs=%zu pages=%u draws=%llu proved=%u reuse_failures=%u proof_stage=%u borrowed=%u",
            laps[0],laps[1],laps[2],laps[3],laps[4],laps[5],laps[6],laps[7],casters.size(),caster_inputs.size(),drawn_pages,
            static_cast<unsigned long long>(draws),unsigned(proved),reuse_failures,page_proof_stage,unsigned(renderer.borrowed_scene_frame));
        renderer.trace.write("fresh-shadow-build",detail);
        return true;
    }
};

struct SandboxMirrorTarget {
    ID3D11Texture2D* color=nullptr;
    ID3D11Texture2D* depth_texture=nullptr;
    ID3D11RenderTargetView* target=nullptr;
    ID3D11DepthStencilView* depth=nullptr;
    ID3D11ShaderResourceView* view=nullptr;
    unsigned width=0,height=0;
    template<class T> static void drop(T*& p) {if(p)p->Release();p=nullptr;}
    ~SandboxMirrorTarget() {drop(view);drop(depth);drop(target);drop(depth_texture);drop(color);}
    bool ensure(ID3D11Device* device,unsigned w,unsigned h) {
        if (target && width==w && height==h) return true;
        drop(view);drop(depth);drop(target);drop(depth_texture);drop(color);
        D3D11_TEXTURE2D_DESC d={};d.Width=w;d.Height=h;
        d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
        d.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;
        d.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
        HRESULT hr=device->CreateTexture2D(&d,nullptr,&color);
        if (SUCCEEDED(hr)) hr=device->CreateRenderTargetView(color,nullptr,&target);
        if (SUCCEEDED(hr)) hr=device->CreateShaderResourceView(color,nullptr,&view);
        d.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;
        d.BindFlags=D3D11_BIND_DEPTH_STENCIL;
        if (SUCCEEDED(hr)) hr=device->CreateTexture2D(&d,nullptr,&depth_texture);
        if (SUCCEEDED(hr)) hr=device->CreateDepthStencilView(depth_texture,nullptr,&depth);
        if (FAILED(hr)) return false;
        width=w;height=h;return true;
    }
    std::size_t bytes() const {return std::size_t(width)*height*12;}
};

struct SandboxMaterialChannel {
    ID3D11Texture2D* texture=nullptr;
    ID3D11RenderTargetView* target=nullptr;
    ID3D11ShaderResourceView* view=nullptr;
    unsigned width=0,height=0;
    DXGI_FORMAT format=DXGI_FORMAT_UNKNOWN;
    template<class T> static void drop(T*& p) {if(p)p->Release();p=nullptr;}
    ~SandboxMaterialChannel(){reset();}
    void reset(){drop(view);drop(target);drop(texture);width=height=0;format=DXGI_FORMAT_UNKNOWN;}
    bool ensure(ID3D11Device* device,unsigned w,unsigned h,DXGI_FORMAT next_format){
        if(target && width==w && height==h && format==next_format)return true;
        reset();
        D3D11_TEXTURE2D_DESC desc={};desc.Width=w;desc.Height=h;
        desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=next_format;
        desc.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
        HRESULT hr=device->CreateTexture2D(&desc,nullptr,&texture);
        if(SUCCEEDED(hr))hr=device->CreateRenderTargetView(texture,nullptr,&target);
        if(SUCCEEDED(hr))hr=device->CreateShaderResourceView(texture,nullptr,&view);
        if(FAILED(hr)){reset();return false;}
        width=w;height=h;format=next_format;return true;
    }
};

struct SandboxFreshPipeline {
    SandboxPassWorkload work;
    struct PhaseConstantCounts {
        std::uint64_t water_records=0,water_updates=0,water_hits=0;
        std::uint64_t wave_records=0,wave_updates=0,wave_hits=0;
    } phase_constant_counts;
    SandboxVisualShaders visual;
    SandboxSceneShadow shadow;
    c3x_renderer::city_fidelity::Glow glow;
    c3x_renderer::GpuCitySiteOverlay city_site_overlay;
    SandboxBloom bloom;
    c3x_renderer::render_core::LinearTarget static_cache;
    using StaticRasters=c3x_renderer::render_core::StaticRasterStates<c3x_renderer::render_core::LinearTarget>;
    StaticRasters static_rasters;
    // Near-water static overlays (roads, improvements, features, cliffs and
    // their shadows) retained beside each static slot; see
    // write_overlay_strip. A layer is usable only while `revision` equals its
    // slot's revision, so every slot mutation either updates it or retires it.
    struct OverlaySlot {c3x_renderer::render_core::LinearTarget layer;std::uint64_t revision=~0ull;};
    std::array<OverlaySlot,StaticRasters::slot_count> overlay_slots;
    c3x_renderer::render_core::OverlayComposite overlay_composite;
    ID3D11BlendState* overlay_depth_only=nullptr;
    std::uint64_t overlay_device=~0ull,overlay_hits=0,overlay_misses=0,overlay_strips=0,overlay_refusals=0;
    // Diagnostics: why a frame used the per-frame path, and why a layer retired.
    std::uint64_t overlay_miss_unshifted=0,overlay_miss_revision=0,overlay_miss_layer=0,
        overlay_retired_mismatch=0,overlay_retired_failure=0,overlay_preview_hits=0;unsigned overlay_miss_last_layer=~0u;
    struct OverlayFrame {bool valid=false;unsigned slot=0;int move_x=0,move_y=0;float depth_shift=0;} overlay_frame;
    // Zoom previews reconstruct static depth from a raster at another
    // projection. Its small error at tile creases made live water fail the
    // depth test along every tile row (dashed seabed lines mid-zoom), so the
    // water pass is biased toward the camera while the static layer is a
    // preview. Exact (restored) frames never use it.
    bool static_preview=false;ID3D11RasterizerState* preview_water_raster=nullptr;std::uint64_t preview_raster_device=~0ull;
    // Slot selection is fixed for this synchronous draw. A resumable owner must
    // capture this slot address before its batches and hold it through retirement.
    auto& current_raster(){return static_rasters.current();}
    auto& static_region(){return current_raster().region;}
    c3x_renderer::render_core::LinearTarget material_albedo;
    SandboxMaterialChannel material_normal,material_world;
    c3x_renderer::render_core::LinearTarget terrain_albedo;
    SandboxMaterialChannel terrain_normal,terrain_world,terrain_properties;
    c3x_renderer::render_core::LinearTarget reflected_terrain_albedo;
    SandboxMaterialChannel reflected_terrain_normal,reflected_terrain_world,
        reflected_terrain_properties;
    SandboxMaterialChannel aquatic_scene;
    SandboxMaterialChannel water_lighting;
    bool capture_water_lighting=false,water_lighting_valid=false;
    std::array<std::uint64_t,13> water_lighting_key{};
    std::uint64_t water_lighting_builds=0,water_lighting_reuses=0;
    ID3D11Buffer* aquatic_bounds_buffer=nullptr;
    ID3D11BlendState* terrain_material_blend=nullptr;
    ID3D11DepthStencilState* aquatic_depth=nullptr;
    c3x_renderer::render_core::LinearRestore static_restore;
    c3x_renderer::render_core::DrawParameterStream parameters;
    using PreparedParameters=c3x_renderer::render_core::PreparedDrawParameters<ViewportShaderSettings,geometry_layer_count>;
    PreparedParameters water_parameters;
    // Batched non-rigid records (water, river, bed, shore shadow, ground).
    c3x_renderer::render_core::PulledMeshPages pulled_pages;
    ID3D11Buffer* pulled_batch=nullptr;
    std::uint64_t pulled_device=~0ull,pulled_draws=0,pulled_records=0,pulled_fallbacks=0;
    std::vector<c3x_renderer::render_core::PulledMeshPages::Source> pulled_sources;
    std::vector<c3x_renderer::render_core::PulledMeshPages::Range> pulled_ranges;
    c3x_renderer::render_core::SubmissionCensus<std::array<std::uint64_t,12>> submission_census;
    SandboxMirrorTarget reflection,reflection_static;
    GeometryDrawView::Records resident, static_visible, water_visible,
        reflection_visible, all_visible, effect_visible;
    using Membership=c3x_renderer::render_core::SceneMembership<CachedVertexChunk,geometry_layer_count>;
    Membership::Lease resident_lease;
    using RasterInputs=c3x_renderer::render_core::RasterContributors<CachedGeometryProof>;
    std::array<RasterInputs,4> raster_inputs;
    float resident_basis_x=0,resident_basis_y=0;
    // Tiles with records in resident_lease (contributor key word 2).
    std::unordered_set<std::uint64_t> resident_tiles;
    bool tile_resident(RasterInputs::Key const& key)const{return resident_tiles.count(key[2])!=0;}
    std::vector<RasterInputs::Key> reflection_inputs;
    std::uint64_t static_receiver_revision=0;
    std::uint64_t reflection_revision=0;
    std::uint64_t resident_signature=0;
    unsigned resident_builds=0;
    int wrap_pixels=0;
    bool visibility_valid=false;
    std::uint64_t visibility_revision=0;
    std::array<std::uint64_t,6> visibility_scene_key{};
    std::array<float,6> visibility_view_key{};
    ID3D11ShaderResourceView* production_reflection=nullptr;
    unsigned production_reflection_width=0,production_reflection_height=0;
    int camera_x=0,camera_y=0;
    unsigned visible=0,culled=0;
    unsigned reflection_count=0;
    unsigned scene_scale=1,scene_samples=2;
    static constexpr int region_margin_x=320,region_margin_y=192;
    // Zoom changes raster projection; completed scene pixels stay at 1:1.
    float display_zoom=1.f,projection_zoom=1.f;
    float reflection_scale=1;
    // --- Retained static layer: preview, refinement and region of interest ---
    using StaticState=c3x_renderer::render_core::StaticRasterState<c3x_renderer::render_core::LinearTarget>;
    using StaticRect=StaticState::Rect;
    // Low-resolution first image per lane, used only until refinement completes.
    std::array<StaticState,2> bootstrap;
    // A zoom-out bootstrap drawn only around the displayed raster it extends
    // (render_bootstrap); usable only beside that exact raster revision.
    std::array<StaticState const*,2> bootstrap_ring{};std::array<std::uint64_t,2> bootstrap_ring_revision{};
    // Bootstraps are transient previews: any world, residency or asset change
    // redraws them. Their draw costs 20-60 ms on a busy map; exact dependency
    // registration for a whole view cost 170-640 ms.
    std::array<std::array<std::uint64_t,6>,2> bootstrap_stamp{};
    std::array<std::uint64_t,6> bootstrap_identity()const{
#ifdef C3X_RENDERER64_FRESH
        auto revision=renderer.raster_dependency_revisions.checkpoint();
        return {std::uint64_t(reinterpret_cast<std::uintptr_t>(revision.owner)),revision.sequence,view_revision(),raster_scope(),
            renderer.topology_cache.visibility_sequence(),std::uint64_t(renderer.device_generation)|(shadow.change_serial<<32)};
#else
        return {view_revision(),raster_scope(),0,0,0,0};
#endif
    }
    struct RasterOverride {ID3D11RenderTargetView* target=nullptr;unsigned width=0,height=0;float margin_x=0,margin_y=0;};
    RasterOverride raster_override;
    c3x_renderer::render_core::LinearResample static_resample;
    std::array<std::uint64_t,12> restore_key{};
    // A preview whose sources' near-water overlays are already in static_cache
    // (composited retained layers, or baked into a bootstrap); the water pass
    // then draws only the animated water.
    bool restore_overlays=false,preview_overlays=false;
    unsigned region_width_px=0,region_height_px=0;
    double refine_pixels=600000;bool refine_worked=false;LARGE_INTEGER refine_last{};
    double last_static_ms=0; // static raster work (strips, previews, restore) in the previous frame
    // Per-frame static composition decisions (profiling trace "static-compose").
    struct StaticDecision {unsigned lane=0;int reusable=-1,recenter=0,shifted=0,refine=0,sync=0,preview=0,
        front_cover=0,home_cover=0,boot_cover=0;long long missing=0;int camera_x=0,camera_y=0,slot_x=0,slot_y=0;
        // entry: 1 valid, 2 stale, 4 key mismatch, 8 proven, 16 repaired, 32 inputs complete, 64 layout reset since last frame.
        unsigned entry=0,home_entry=0;std::size_t input_bytes=0;
        // repair: 1 incomplete, 2 overflow, 3 order, 4 unlocated, 5 too broad, 6 draw; dirty rects/area %.
        unsigned repair=0,repair_rects=0,repair_percent=0,repair_changed=0,repair_reordered=0;
        double boot_draw_ms=0,boot_deps_ms=0;long long boot_area=0;} static_decision;
    bool layout_reset=false;
    // Per-lane zoom motion: transactions draw lane 0 at 1x between visual frames.
    unsigned reflection_skips=0;std::array<float,2> lane_projection{};std::array<unsigned,2> lane_still{};
    std::uint64_t draw_serial=0;std::array<std::uint64_t,2> lane_drawn{};
    std::uint64_t preview_frames=0,refine_slices=0,refine_promotions=0,bootstrap_draws=0,recenter_copies=0;
    // Consecutive refinement restarts without a promotion (content changing
    // every frame); past a limit the refinement completes synchronously.
    unsigned refine_restarts=0;
    GeometryDrawView::Records roi_records,roi_shadow_records;
    // Source-pixel rectangle whose receivers the shadow atlas covers.
    StaticRect shadow_field{INT_MIN/4,INT_MIN/4,INT_MAX/4,INT_MAX/4};
    std::array<std::int64_t,11> roi_key{};
    std::uint64_t roi_revision=1,roi_receiver_check=0;
    static constexpr int roi_quantum=128;
    struct ZoomScope {
        SandboxFreshPipeline& owner;float previous;
        ZoomScope(SandboxFreshPipeline& o,float zoom):owner(o),previous(o.projection_zoom){o.projection_zoom=zoom;}
        ~ZoomScope(){owner.projection_zoom=previous;}
        ZoomScope(ZoomScope const&)=delete;ZoomScope& operator=(ZoomScope const&)=delete;
    };
    struct TargetScope {
        SandboxFreshPipeline& owner;RasterOverride previous;
        TargetScope(SandboxFreshPipeline& o,RasterOverride next):owner(o),previous(o.raster_override){o.raster_override=next;}
        ~TargetScope(){owner.raster_override=previous;}
        TargetScope(TargetScope const&)=delete;TargetScope& operator=(TargetScope const&)=delete;
    };
    unsigned depth_copies=0,cache_scrolls=0,cache_full_draws=0;
    c3x_renderer::TerritoryBorders territory_borders;
    unsigned reflection_reuses=0,reflection_draws=0;
    float visual_hour=12.f,visual_sun_intensity=0.f,previous_hour=-1.f;
    int previous_season=INT_MIN;
    std::uint64_t lighting_revision=0;
    std::vector<c3x_renderer::city_fidelity::Lighting const*> selected_lighting;
    bool resident_water_scene=false;
    c3x_renderer::render_core::RasterScratchIdentity reflection_key,reflected_terrain_key;
    bool reflection_dynamic_written=false;
    // Mirror reuse while the camera moves: the water pass samples the mirror
    // at screen + NativeReflectionTarget.zw, so a camera-only change is a pure
    // offset. The exact mirror is redrawn once the camera settles.
    int reflection_camera_x=0,reflection_camera_y=0;
    float reflection_shift[2]={};
    int moved_camera_x=INT_MIN,moved_camera_y=INT_MIN;
    LARGE_INTEGER camera_moved_at{};
    std::uint64_t reflection_shift_reuses=0;
    static bool mirror_camera_only(c3x_renderer::render_core::RasterScratchIdentity const& a,
            c3x_renderer::render_core::RasterScratchIdentity const& b){
        for(unsigned i=0;i<a.scene.size();++i)if(i!=5 && i!=6 && a.scene[i]!=b.scene[i])return false;
        for(unsigned i=0;i<a.view.size();++i)if(i!=2 && i!=3 && a.view[i]!=b.view[i])return false;
        return true;
    }
    bool reflected_terrain_material_valid=false;
    bool mirror_mountain_coverage=false;
    bool reflection_valid=false;
    using Submission=c3x_renderer::render_core::SharedInstanceSubmission;
    Submission::Lease shared_front;
    c3x_renderer::render_core::BodyPlacementRequirements<CachedVertexChunk> body_requirements;
    std::array<std::uint64_t,4> body_requirement_scene{};
    std::array<float,7> body_requirement_view{};
    bool body_requirements_valid=false;
    unsigned body_requirement_builds=0,body_requirement_reuses=0,body_requirement_visits=0;
    double body_requirement_ms=0;
    struct StaticValidationCounts {
        RasterInputs::ValidationCounts raster;
        SandboxSceneShadow::AtlasInputs::ValidationCounts atlas;
        std::uint64_t receiver_visits=0,receiver_builds=0,receiver_reuses=0,placement_probes=0,placement_reuses=0;
        std::uint64_t caster_collections=0,caster_reuses=0,shadow_group_builds=0,shadow_group_reuses=0;
        std::uint64_t body_placement_builds=0,body_placement_reuses=0,terrain_batch_builds=0,terrain_batch_reuses=0;
        std::uint64_t shadow_page_reuses=0,shadow_page_rebuilds=0,shadow_page_refused=0,shadow_draws=0;
        std::uint64_t shadow_projections=0,shadow_projection_reuses=0,shadow_page_tests=0,shadow_contributor_edits=0,shadow_instance_projections=0,shadow_page_sorts=0,shadow_draw_projections=0;
        std::size_t shadow_proof_bytes=0,shadow_page_bytes=0,shadow_producers=0,shadow_dependency_sources=0;
    };
    StaticValidationCounts static_validation_counts()const{
        StaticValidationCounts result;
        for(auto const& inputs:raster_inputs){auto const& counts=inputs.validation_counts;
            result.raster.full+=counts.full;result.raster.content+=counts.content;result.raster.visibility+=counts.visibility;
            result.raster.membership+=counts.membership;result.raster.regions+=counts.regions;result.raster.reused+=counts.reused;result.raster.changes+=counts.changes;
            result.raster.visibility_rejects+=counts.visibility_rejects;
            result.raster.proof_registrations+=counts.proof_registrations;result.raster.dependency_watch_calls+=counts.dependency_watch_calls;
            result.raster.source_expansions+=counts.source_expansions;result.raster.source_reuses+=counts.source_reuses;result.raster.append_ms+=counts.append_ms;
        }
        result.atlas=shadow.atlas_inputs.validation_counts;result.receiver_visits=shadow.receiver_visits;
        result.receiver_builds=shadow.receiver_builds;result.receiver_reuses=shadow.receiver_reuses;
        result.placement_probes=body_requirements.coverage_total_probes;result.placement_reuses=body_requirements.coverage_total_reuses;
        result.caster_collections=shadow.caster_collections;result.caster_reuses=shadow.caster_reuses;
        result.shadow_group_builds=shadow.group_builds;result.shadow_group_reuses=shadow.group_reuses;
        result.body_placement_builds=shadow.body_placement_builds;result.body_placement_reuses=shadow.body_placement_reuses;
        result.terrain_batch_builds=shadow.terrain_batch_builds;result.terrain_batch_reuses=shadow.terrain_batch_reuses;
        result.shadow_page_reuses=shadow.page_contents.hits;result.shadow_page_rebuilds=shadow.page_contents.rebuilt;
        result.shadow_page_refused=shadow.page_contents.refused;result.shadow_draws=shadow.draws;
        result.shadow_projections=shadow.page_contents.projections;result.shadow_projection_reuses=shadow.page_contents.projection_reuses;
        result.shadow_page_tests=shadow.page_contents.page_tests;result.shadow_contributor_edits=shadow.page_contents.contributor_edits;
        result.shadow_page_sorts=shadow.page_contents.page_sorts;result.shadow_draw_projections=shadow.draw_projections;
        result.shadow_instance_projections=shadow.instance_projections;result.shadow_proof_bytes=shadow.atlas_inputs.bytes();
        result.shadow_page_bytes=shadow.page_contents.bytes();result.shadow_producers=shadow.atlas_inputs.producers.size();
        result.shadow_dependency_sources=shadow.atlas_inputs.sources.size();
        return result;
    }
    enum PrepareSpan {prepare_setup_resources,prepare_capture,prepare_raster_proof,
        prepare_body_requirements,prepare_city_shadow,prepare_unit_selection,prepare_unit_pose,prepare_span_count};
    std::array<double,prepare_span_count> prepare_subspans{};
    enum DynamicSpan {dynamic_depth_setup,dynamic_aquatic,dynamic_water_scene,dynamic_resources,dynamic_waves,dynamic_span_count};
    std::array<double,dynamic_span_count> dynamic_subspans{};
    std::array<SandboxPassWorkload::CallCounts,dynamic_span_count> dynamic_calls{};
    unsigned prepare_unit_plan_reused=0,prepare_unit_reselected=0;
    double phases[6]={};
#ifdef C3X_RENDERER64_FRESH

    struct InstancePlan {Submission::SelectionLease selection;std::size_t bytes=0;unsigned count=0;};
    c3x_renderer::render_core::FrameSampleCache<std::vector<std::uint64_t>,InstancePlan,128> instance_plans;
    std::size_t instance_plan_bytes=0;
    unsigned instance_plan_builds=0,instance_plan_reuses=0;
    using GpuPhases=c3x_renderer::render_core::GpuAnimationTelemetry;
    GpuPhases gpu_phases;
#endif
    bool active=false;
    std::uint64_t scene_revision() const {
#ifdef C3X_RENDERER64_FRESH
        return renderer.topology_cache.appearance_sequence() ^
            (renderer.topology_cache.scope_sequence()*0x9e3779b97f4a7c15ull) ^ renderer.content_revision;
#else
        return renderer.cached_signature.complete;
#endif
    }
    std::uint64_t view_revision()const{
#ifdef C3X_RENDERER64_FRESH
        return renderer.geometry_vertex_buffers.revision();
#else
        return scene_revision();
#endif
    }
    std::uint64_t raster_scope()const{
#ifdef C3X_RENDERER64_FRESH
        return renderer.topology_cache.scope_sequence() ^ (std::uint64_t(renderer.content_revision)<<32);
#else
        return scene_revision();
#endif
    }
    ~SandboxFreshPipeline() {
        if(preview_water_raster)preview_water_raster->Release();
        if(overlay_depth_only)overlay_depth_only->Release();
        if(pulled_batch)pulled_batch->Release();
        if(aquatic_bounds_buffer)aquatic_bounds_buffer->Release();
        if(terrain_material_blend)terrain_material_blend->Release();
        if(aquatic_depth)aquatic_depth->Release();
#ifndef C3X_RENDERER64_FRESH
        if (production_reflection) {
            renderer.scene_reflection_view=production_reflection;
            renderer.scene_reflection_width=production_reflection_width;
            renderer.scene_reflection_height=production_reflection_height;
        }
#endif
    }
    void retire_geometry_selection(){
        // Pass records borrow meshes; published images and raster proofs do not.
        // Clear borrowers before releasing their exact membership generations.
        resident={};static_visible={};water_visible={};reflection_visible={};all_visible={};effect_visible={};
        roi_records={};roi_shadow_records={};selected_lighting.clear();
        body_requirements.clear();body_requirements_valid=false;
        // The instance groups are the forests' only shadow submission. Drop
        // their placement keys too: unchanged keys otherwise reused the empty
        // groups, and every page drawn then lacked forest shadows while its
        // proof still certified it (forests regained shadows only near later
        // repairs, such as a selected unit).
        shadow.casters.clear();shadow.caster_inputs.clear();shadow.instance_groups.clear();shadow.prepared_instances.clear();
        shadow.caster_signature=~std::uint64_t(0);shadow.prepared_signature=~std::uint64_t(0);
        shadow.caster_lease.reset();resident_lease.reset();
        resident_signature=0;visibility_valid=false;
    }
    bool capture(ViewportShaderSettings const& settings,
            ViewportShaderSettings const& reflected,int width,int height,int next_wrap_pixels) {
        // Raster proofs include the relative draw order of their contributors.
        // Reassembling/sorting an entering strip must not discard the entire
        // full-resolution viewport before those local proofs can be repaired.
        if (resident_signature!=view_revision() ||
                wrap_pixels!=next_wrap_pixels) {
#ifndef C3X_RENDERER64_FRESH
            static_rasters.invalidate_all(c3x_renderer::render_core::raster_scene);
            for(auto& image:bootstrap)image.valid=false;
#endif
            visibility_valid=false;
#ifndef C3X_RENDERER64_FRESH
            reflection_valid=false;reflected_terrain_material_valid=false;
#endif
            wrap_pixels=next_wrap_pixels;
#ifdef C3X_RENDERER64_FRESH
            // Borrow the production selection and its immutable content lease.
            // Wrapped occurrences are transformed on traversal; no second full
            // assembled array or per-occurrence COM/resource leases exist.
            resident_lease=renderer.geometry_vertex_buffers.publish();
            resident_tiles.clear();
            if(resident_lease)for(auto const& records:resident_lease->records)for(auto const& record:records)
                resident_tiles.insert((std::uint64_t(std::uint32_t(record.tile_x))<<32)|std::uint32_t(record.tile_y));
            resident_basis_x=renderer.geometry_viewport_settings.translation[0]-camera_x;
            resident_basis_y=renderer.geometry_viewport_settings.translation[1]-camera_y;
            renderer.region_contributors.clear();
            renderer.prepare_region_contributors(renderer.geometry_vertex_buffers);
#else
            resident={};
            for(unsigned layer=0;layer<geometry_layer_count;++layer)
                for(auto const& record:renderer.geometry_vertex_buffers[layer])resident[layer].push_back(record);
#endif
            resident_signature=view_revision();
            ++resident_builds;
        }
        if(resident_water_scene!=renderer.water_scene_active){
            static_rasters.invalidate_all(c3x_renderer::render_core::raster_classification);
            resident_water_scene=renderer.water_scene_active;
            reflection_valid=false;reflected_terrain_material_valid=false;
        }
        std::array<std::uint64_t,6> scene_key={view_revision(),
            std::uint64_t(next_wrap_pixels),std::uint64_t(width),
            std::uint64_t(height),std::uint64_t(renderer.water_scene_active),0};
#ifdef C3X_RENDERER64_FRESH
        scene_key[5]=renderer.topology_cache.visibility_sequence();
#endif
        std::array<float,6> view_key={settings.translation[0],settings.translation[1],
            reflected.translation[0],reflected.translation[1],
            renderer.reflection.height_pixels,projection_zoom};
        if(visibility_valid && scene_key==visibility_scene_key &&
                view_key==visibility_view_key){if(work.enabled)++work.counts[SandboxPassWorkload::selection][SandboxPassWorkload::screen].reuses;return true;}
        if(work.enabled)++work.counts[SandboxPassWorkload::selection][SandboxPassWorkload::screen].rebuilds;
        static_visible={};water_visible={};reflection_visible={};effect_visible={};
        auto prior_receivers=std::move(all_visible);all_visible={};
        visible=culled=reflection_count=0;
        D3D11_RECT rect=source_bounds(settings,{0,0,width,height},false);
        contributors(settings,rect,false,[&](unsigned layer,auto const& record) {
                if(work.enabled)++work.counts[SandboxPassWorkload::selection][layer].tested_records;
                if (!renderer.chunk_intersects_region(GeometryDrawReference(record),settings,rect,false)) {
                    ++culled;return;
                }
                // Attached city effects follow the visual clock: they leave
                // the retained static layer and redraw with the dynamic pass,
                // after water and waves. A smoke quad overhanging a river is
                // water-dependent, but must not join the water scene, whose
                // retained-overlay and scene paths would draw it differently
                // while the camera moves.
                auto& output=record.content().city_effect?effect_visible:
                    renderer.water_scene_active && record.water_dependent?water_visible:static_visible;
                if(work.enabled)++work.counts[SandboxPassWorkload::selection][layer].accepted_records;
                output[layer].push_back(record);
                all_visible[layer].push_back(record);++visible;
            });
        // Visible attached effects keep visual frames coming (ambient_count).
        renderer.visible_city_effects=unsigned(std::count_if(all_visible[geometry_city].begin(),
            all_visible[geometry_city].end(),[](auto const& record){return record.content().city_effect;}));
        std::vector<D3D11_RECT> water_receivers;
        auto mirror=source_bounds(reflected,reflected_water_bounds(settings,width,height,&water_receivers),true);
        index_mirror_receivers(reflected,mirror,water_receivers);
        contributors(reflected,mirror,true,[&](unsigned layer,auto const& record) {
            if (layer==geometry_underlay || layer==geometry_bed ||
                layer==geometry_water || layer==geometry_river ||
                layer==geometry_route || layer==geometry_shadow || layer==geometry_wave)
                return;
                // Attached effects are screen-facing quads; water does not mirror them.
                if (record.content().animation_texture || record.content().city_effect ||
                    !renderer.chunk_intersects_region(GeometryDrawReference(record),
                        reflected,mirror,true) ||
                    !reflection_reaches_water(GeometryDrawReference(record),reflected)) return;
                reflection_visible[layer].push_back(record);
                all_visible[layer].push_back(record);
                ++reflection_count;
            });
        std::vector<RasterInputs::Key> next_reflection;
        for(unsigned layer=0;layer<geometry_layer_count;++layer)
            for(auto const& record:reflection_visible[layer])next_reflection.push_back(contributor_key(layer,record));
        if(next_reflection!=reflection_inputs){reflection_inputs=std::move(next_reflection);++reflection_revision;}
        bool same_receivers=true;
        for(unsigned layer=0;same_receivers&&layer<geometry_layer_count;++layer)if(layer!=geometry_shadow){
            if(prior_receivers[layer].size()!=all_visible[layer].size()){same_receivers=false;break;}
            for(std::size_t i=0;i<all_visible[layer].size();++i)
                if(contributor_key(layer,prior_receivers[layer][i])!=contributor_key(layer,all_visible[layer][i])){same_receivers=false;break;}
        }
        if(!same_receivers){
#ifdef C3X_RENDERER64_FRESH
            renderer.retire_ordered_rigid_packets(all_visible);
#endif
            ++static_receiver_revision;
        }
        visibility_scene_key=scene_key;visibility_view_key=view_key;
        visibility_valid=true;++visibility_revision;
        // A zoomed or edge viewport may contain no revealed geometry. That is
        // a valid empty selection, not a failure of the camera/animation sampler.
        return true;
    }
    template<class Visit>void contributors(ViewportShaderSettings const& settings,D3D11_RECT clip,bool mirrored,Visit visit)const{
        if(clip.left>=clip.right || clip.top>=clip.bottom)return;
#ifdef C3X_RENDERER64_FRESH
        std::vector<std::array<unsigned,3>> plan;
        std::vector<std::pair<unsigned,unsigned>> candidates;
        bool indexed=resident_lease && clip.left<clip.right && clip.top<clip.bottom;
        for(unsigned turn=0;indexed && turn<(wrap_pixels?3u:1u);++turn){
            int wrap=turn==0?0:turn==1?-wrap_pixels:wrap_pixels;
            indexed=renderer.query_region_inputs(mirrored?1u:0u,
                int(std::floor(clip.left-settings.translation[0]-resident_basis_x-wrap)),
                int(std::floor(clip.top-settings.translation[1]-resident_basis_y)),
                unsigned(clip.right-clip.left+2),unsigned(clip.bottom-clip.top+2),candidates);
            for(auto item:candidates)plan.push_back({item.first,item.second,turn});
        }
        if(indexed){
            // Preserve original layer/occurrence/wrap order, including alpha.
            std::sort(plan.begin(),plan.end());
            for(auto item:plan){if(item[0]==geometry_wave)continue;
                auto record=resident_lease->records[item[0]][item[1]];
                record.translation_x+=int(resident_basis_x)+(item[2]==0?0:item[2]==1?-wrap_pixels:wrap_pixels);
                record.translation_y+=int(resident_basis_y);visit(item[0],record);
            }return;
        }
#endif
        for(unsigned layer=0;layer<geometry_layer_count;++layer)
            each_resident(layer,[&](auto const& record){visit(layer,record);});
    }
    RasterInputs::ValidationKey raster_validation_key(ViewportShaderSettings const& settings,D3D11_RECT rect)const{
        auto bits=[](float value){std::uint32_t result;std::memcpy(&result,&value,sizeof(result));return std::uint64_t(result);};
        return {view_revision(),renderer.topology_cache.scope_sequence(),renderer.content_revision,renderer.device_generation,
            unsigned(renderer.water_scene_active),std::uint64_t(wrap_pixels),std::uint64_t(rect.left),std::uint64_t(rect.top),
            std::uint64_t(rect.right),std::uint64_t(rect.bottom),bits(projection_zoom),bits(settings.translation[0]),bits(settings.translation[1]),
            bits(settings.depth_translation),bits(settings.inverse_size[0]),bits(settings.inverse_size[1]),bits(settings.natural_projection[0]),
            bits(settings.natural_projection[1]),bits(settings.natural_projection[2]),bits(settings.natural_projection[3]),
            std::uint64_t(renderer.scene_depth_origin),bits(resident_basis_x),bits(resident_basis_y),unsigned(renderer.geometry_canonical_world)};
    }
    RasterInputs::Key contributor_key(unsigned layer,GeometryDrawRecord const& record)const{
        auto bits=[](float value){std::uint32_t result;std::memcpy(&result,&value,sizeof(result));return std::uint64_t(result);};
        auto const& b=record.bounds;
        return {record.owner.generation,(std::uint64_t(layer)<<32)|record.ordinal,
            (std::uint64_t(std::uint32_t(record.tile_x))<<32)|std::uint32_t(record.tile_y),
            std::uint64_t(record.translation_x),std::uint64_t(record.translation_y),
            bits(record.natural_projection[0]),bits(record.natural_projection[1]),
            bits(record.natural_projection[2]),bits(record.natural_projection[3]),
            std::uint64_t(b.left),std::uint64_t(b.top),std::uint64_t(b.right),std::uint64_t(b.bottom),
            (std::uint64_t(record.territory_rgb)<<32)|record.territory_edges|
                (std::uint64_t(record.water_dependent)<<16)|(std::uint64_t(record.water_visible)<<17)};
    }
    bool raster_dependencies(RasterInputs& inputs,ViewportShaderSettings const& settings,D3D11_RECT rect,bool append)const{
#ifdef C3X_RENDERER64_FRESH
        auto exact=[&](){
        if(!append && !inputs.valid([&](auto const& proof){return renderer.raster_content_valid(proof);},
                [&](auto tile){auto record=renderer.topology_cache.retained(tile);return record?record->visibility_revision:0;}))return false;
        bool valid=true;if(append)inputs.begin_append();else inputs.begin_membership();auto clip=source_bounds(settings,rect,false);
        // Retained overlays make water-dependent records raster content.
        bool const retain_overlays=overlay_enabled();
        contributors(settings,clip,false,[&](unsigned layer,auto const& record){
            ++inputs.validation_counts.membership;
            if(renderer.water_scene_active && record.water_dependent && !retain_overlays)return;
            if(!renderer.chunk_intersects_region(GeometryDrawReference(record),settings,clip,false))return;
            auto key=contributor_key(layer,record);
            if(!append){if(!inputs.visit_membership(key)){valid=false;++renderer.raster_proof_rejections[5];}return;}
            auto mesh=std::static_pointer_cast<CachedMeshGeneration>(resident_lease->content.get(record.owner));
            auto tile=renderer.topology_cache.key(record.tile_x,record.tile_y);
            auto observed=renderer.topology_cache.retained(tile);
            // No per-input watches: each new proof expanded ~3000 keys (river
            // inputs per tile), so registering a whole view cost 150-650 ms.
            // Proofs and visibility revisions are still checked exactly.
            valid=inputs.add(key,mesh?mesh->proof:nullptr,tile,observed?observed->visibility_revision:0) && valid;
        });
        if(append)inputs.finish_dependencies();
        else if(!inputs.exact_membership() && !inputs.resident_membership([&](auto const& key){return tile_resident(key);})){
            valid=false;++renderer.raster_proof_rejections[5];}
        return valid;
        };
        if(append){auto begin=std::chrono::steady_clock::now();bool valid=exact();
            inputs.validation_counts.append_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();return valid;}
        return inputs.validate(renderer.raster_dependency_revisions,raster_validation_key(settings,rect),exact);
#else
        return true;
#endif
    }
    template<class Visit>void each_resident(unsigned layer,Visit visit)const{
#ifdef C3X_RENDERER64_FRESH
        if(!resident_lease || layer==geometry_wave)return;
        auto const& records=resident_lease->records[layer];
#else
        auto const& records=resident[layer];
#endif
        for(auto const& record:records){
            auto base=record;
#ifdef C3X_RENDERER64_FRESH
            base.translation_x+=int(resident_basis_x);base.translation_y+=int(resident_basis_y);
#endif
            visit(base);
            if(wrap_pixels)for(int direction:{-1,1}){
                auto occurrence=base;occurrence.translation_x+=direction*wrap_pixels;visit(occurrence);
            }
        }
    }
    bool draw_features(GeometryDrawView::Records const& features,
            ViewportShaderSettings const& settings,D3D11_RECT rect,
            ID3D11RenderTargetView* target,ID3D11DepthStencilView* depth,
            bool mirrored,float scale,bool submerged=false) {
        if (features[geometry_feature].empty()) return true;
        auto* context=renderer.context;
        context->OMSetRenderTargets(1,&target,depth);
        context->OMSetDepthStencilState(submerged?
            aquatic_depth:renderer.depth_state,0);
        context->OMSetBlendState(renderer.blend_state,nullptr,0xffffffffu);
        context->RSSetState(renderer.rasterizer_state);
        D3D11_VIEWPORT viewport=target_viewport(target,mirrored,scale);
        context->RSSetViewports(1,&viewport);
        D3D11_RECT scissor={LONG(rect.left*scale),LONG(rect.top*scale),
            LONG(rect.right*scale),LONG(rect.bottom*scale)};
        context->RSSetScissorRects(1,&scissor);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->IASetInputLayout(renderer.feature_input_layout);
        auto& active_reflection=sandbox_active_reflection();
        context->VSSetShader(mirrored?active_reflection.vs[1]:
            renderer.feature_vertex_shader,nullptr,0);
        context->PSSetShader(mirrored?active_reflection.ps[1]:
            submerged?visual.aquatic_feature:renderer.feature_pixel_shader,nullptr,0);
        context->PSSetConstantBuffers(0,1,&renderer.terrain_settings_buffer);
        context->PSSetConstantBuffers(2,1,&renderer.shadow_settings_buffer);
        context->PSSetConstantBuffers(3,1,&renderer.world_settings_buffer);
        context->PSSetConstantBuffers(4,1,&renderer.source_shadow.table);
        context->VSSetConstantBuffers(1,1,&renderer.viewport_settings_buffer);
        auto const& materials=renderer.compiled_material_views();
        context->PSSetShaderResources(0,UINT(materials.size()),materials.data());
        context->PSSetShaderResources(17,1,&renderer.source_shadow.sampled_view());
        // Terrain uses t25 for shadow depth; feature materials reuse t25-t28.
        // Bind them for this pass even when no cliff draw preceded the feature.
        context->PSSetShaderResources(25,4,renderer.feature_texture_views.data());
        context->PSSetShaderResources(94,4,renderer.feature_texture_views.data()+4);
        ID3D11SamplerState* samplers[]={renderer.terrain_sampler,renderer.decal_sampler};
        context->PSSetSamplers(0,2,samplers);
        return issue_records(features,geometry_feature,settings,rect,mirrored);
    }
    bool draw_resource_poses(GeometryDrawView::Records const& poses,
            ViewportShaderSettings const& settings,D3D11_RECT rect,
            ID3D11RenderTargetView* target,ID3D11DepthStencilView* depth,
            bool submerged=false) {
        if(poses[geometry_feature].empty())return true;
#ifdef C3X_RENDERER64_FRESH
        auto projected=poses;
        for(auto& layer:projected)for(auto& record:layer){record.translation_x+=int(resident_basis_x);record.translation_y+=int(resident_basis_y);}
        auto const& inputs=projected;
#else
        auto const& inputs=poses;
#endif
        if(!submerged){
            bind_common(settings,rect,target,depth,false,float(scene_scale));
            auto* context=renderer.context;
            context->PSSetShader(renderer.resource_shadow_shader,nullptr,0);
            context->PSSetShaderResources(17,1,&renderer.source_shadow.sampled_view());
            context->PSSetShaderResources(25,1,&renderer.source_shadow.sampled_view());
            context->PSSetShaderResources(116,8,renderer.resource_texture_views.data());
            ID3D11SamplerState* shadow_samplers[]={renderer.natural_wrap,renderer.natural_clamp};
            context->PSSetSamplers(0,2,shadow_samplers);
            context->OMSetDepthStencilState(renderer.natural.decal_depth,0);
            if(!issue_records(inputs,geometry_shadow,settings,rect,false)){
                std::printf("SANDBOX_RESOURCE_DRAW_ERROR pass=shadow records=%zu\n",
                    poses[geometry_shadow].size());std::fflush(stdout);return false;
            }
        }
        if(!draw_features(inputs,settings,rect,target,depth,false,float(scene_scale),
                submerged)){
            std::printf("SANDBOX_RESOURCE_DRAW_ERROR pass=body records=%zu\n",
                poses[geometry_feature].size());std::fflush(stdout);return false;
        }
        return true;
    }
    bool region_target(ID3D11RenderTargetView* target)const{
        if(!target)return false;
        for(auto const& state:static_rasters.states)if(state.region.target==target)return true;
        return target==material_albedo.target || target==terrain_albedo.target;
    }
    // Viewport of a static/scene target. Region slots carry the guard margins;
    // an explicit override describes slots drawn at another raster scale.
    D3D11_VIEWPORT target_viewport(ID3D11RenderTargetView* target,bool mirrored,float scale)const{
        float width=0,height=0,margin_x=0,margin_y=0;
        if(mirrored){width=float(reflection.width);height=float(reflection.height);}
        else if(raster_override.target && target==raster_override.target){
            width=float(raster_override.width);height=float(raster_override.height);
            margin_x=raster_override.margin_x;margin_y=raster_override.margin_y;
        }else if(region_target(target)){
            width=float(region_width_px);height=float(region_height_px);
            margin_x=float(region_margin_x);margin_y=float(region_margin_y);
        }else{width=float(glow.linear.width);height=float(glow.linear.height);}
        D3D11_VIEWPORT viewport={0,0,width,height,0,1};
        c3x_renderer::SceneProjection(renderer.content_view_width,renderer.content_view_height,projection_zoom)
            .viewport(viewport,mirrored?8.f:4.f,margin_x,margin_y,scale);
        return viewport;
    }
    // Like CPatchRData::RenderBases, visible records are submitted from resident
    // buffers after the pass has bound its material variant. The producer owns
    // meshes and materials; this pipeline owns per-camera selection and draws.
    D3D11_RECT source_bounds(ViewportShaderSettings const& settings,D3D11_RECT rect,bool mirrored)const {
        float guard=mirrored?8.f:4.f;
        return c3x_renderer::SceneProjection(renderer.content_view_width,
            renderer.content_view_height,projection_zoom).source_rect(rect,guard,
                (1.f/settings.inverse_size[0]-renderer.content_view_width-2*guard)*.5f,
                (1.f/settings.inverse_size[1]-renderer.content_view_height-2*guard)*.5f);
    }
#ifdef C3X_RENDERER64_FRESH
    SandboxDirectUnits::ContributionPlan unit_contribution_plan;
    std::vector<SandboxDirectUnits::ScenePose> unit_contribution_candidates,unit_contribution_required;
    std::uint64_t unit_contribution_revision=1,unit_contribution_scope=0;
    int unit_plan_x=0,unit_plan_y=0,unit_plan_season=0;float unit_plan_zoom=0,unit_plan_hour=-1;
    std::uint64_t unit_plan_bounds=0;bool unit_plan_reflection=false;
    static bool same_unit_inputs(std::vector<SandboxDirectUnits::ScenePose> const& a,
            std::vector<SandboxDirectUnits::ScenePose> const& b){
        return a.size()==b.size() && std::equal(a.begin(),a.end(),b.begin(),[](auto const& left,auto const& right){
            return !std::memcmp(&left.draw,&right.draw,sizeof(left.draw)) && left.pose_identity==right.pose_identity &&
                left.tile_x==right.tile_x && left.tile_y==right.tile_y && left.unit==right.unit && left.action==right.action &&
                left.cursor==right.cursor && left.owner_ring==right.owner_ring && left.predict==right.predict && left.animated==right.animated && left.travelling==right.travelling;
        });
    }
    bool select_unit_contributors(c3x_renderer_frame_v1 const& frame,
            std::vector<SandboxDirectUnits::ScenePose> const& candidates,
            std::vector<SandboxDirectUnits::ScenePose>& required,float next_zoom){
        if(!std::isfinite(next_zoom))return false;
        auto old_zoom=projection_zoom;auto old_x=camera_x,old_y=camera_y;
        struct Restore {SandboxFreshPipeline& owner;float zoom;int x,y;
            ~Restore(){owner.projection_zoom=zoom;owner.camera_x=x;owner.camera_y=y;}} restore{*this,old_zoom,old_x,old_y};
        projection_zoom=std::clamp(next_zoom,c3x_renderer::SceneProjection::minimum,c3x_renderer::SceneProjection::maximum);
        camera_x=camera_y=0;
        if(frame.tile_count && frame.tiles){auto const& first=frame.tiles[0];
            camera_x=first.anchor_x-first.tile_x*frame.tile_width/2;
            camera_y=first.anchor_y-first.tile_y*frame.tile_height/2;}
        int w=renderer.content_view_width+8,h=renderer.content_view_height+8;
        auto settings=renderer.geometry_viewport_settings,reflected=settings;
        settings.translation[0]=float(camera_x)+4;settings.translation[1]=float(camera_y)+4;
        reflected.translation[0]=float(camera_x)+8;reflected.translation[1]=float(camera_y)+8;
        settings.depth_translation=reflected.depth_translation=-float(renderer.scene_depth_origin);
        settings.inverse_size[0]=1.f/w;settings.inverse_size[1]=1.f/h;
        reflected.inverse_size[0]=1.f/(w+8);reflected.inverse_size[1]=1.f/(h+8);
        if(!capture(settings,reflected,w,h,frame.world_wrap_x?frame.world_width_tiles*frame.tile_width/2:0))return false;
        std::vector<D3D11_RECT> receivers;reflected_water_bounds(settings,w,h,&receivers);
        // Level of detail below 0.8x: a unit is at most a few dozen pixels and
        // its mirror image a few pixels, so units contribute no reflections
        // there (no reflection preparation, draws or per-frame mirror copy).
        // The static mirror of terrain, cities and features is unchanged.
        if(projection_zoom<.8f)receivers.clear();
        char cycle[8]{};bool day_night=c3x_renderer::render_core::cached_environment("C3X_SANDBOX_DAY_NIGHT",cycle,sizeof(cycle)) && cycle[0]=='1';
        float hour=day_night?12.f+24.f*float(frame.presentation_time_ticks)/float(std::max<c3x_renderer_i64>(1,frame.presentation_frequency))/30.f:float(frame.hour);
        SandboxDirectUnits::ContributionPlan next;
        if(!sandbox_direct_units.select_real(frame,candidates,hour,projection_zoom,receivers,next))return false;
        bool same=next.entries.size()==unit_contribution_plan.entries.size() &&
            std::equal(next.entries.begin(),next.entries.end(),unit_contribution_plan.entries.begin(),[](auto const& a,auto const& b){return a.candidate==b.candidate && a.mask==b.mask;});
        if(!same)++unit_contribution_revision;
        unit_contribution_plan=std::move(next);unit_contribution_candidates=candidates;
        required=unit_contribution_plan.required(candidates);unit_contribution_required=required;
        unit_contribution_scope=view_revision();unit_plan_x=camera_x;unit_plan_y=camera_y;unit_plan_zoom=projection_zoom;unit_plan_hour=hour;unit_plan_season=frame.season;
        unit_plan_bounds=renderer.unit_bodies.contribution_sequence;unit_plan_reflection=renderer.reflection.enabled;return true;
    }
#endif
    // The mirror is sampled only by water receivers (each already guarded for
    // normal distortion and filtering). On a coastal view their union spans
    // the screen, so a record is kept only when its reflected bounds share a
    // coarse cell with some receiver; inland reflections are never sampled.
    c3x_renderer::render_core::ReceiverCells mirror_cells;
    void index_mirror_receivers(ViewportShaderSettings const& reflected,D3D11_RECT mirror,
            std::vector<D3D11_RECT> const& receivers){
        std::vector<std::array<int,4>> rects;rects.reserve(receivers.size());
        for(auto const& receiver:receivers){auto r=source_bounds(reflected,receiver,true);rects.push_back({r.left,r.top,r.right,r.bottom});}
        mirror_cells.build({mirror.left,mirror.top,mirror.right,mirror.bottom},rects);
    }
    bool reflection_reaches_water(GeometryDrawReference const& chunk,ViewportShaderSettings const& reflected)const{
        // The same reflected extent as chunk_intersects_region(..., true).
        int dx=chunk.translation_x()+int(reflected.translation[0]),dy=chunk.translation_y()+int(reflected.translation[1]);
        auto const& world=chunk.content().world_bounds;auto const& b=chunk.bounds();
        float low=2*renderer.reflection.height_pixels*std::max(0.f,world.low[2]-2.5f/112.f);
        float high=2*renderer.reflection.height_pixels*std::max(0.f,world.high[2]-2.5f/112.f);
        return mirror_cells.any({int(b.left)+dx,int(std::floor(float(b.top+dy)+low)),
            int(b.right)+dx,int(std::ceil(float(b.bottom+dy)+high))});
    }
    D3D11_RECT reflected_water_bounds(ViewportShaderSettings const& settings,int width,int height,
            std::vector<D3D11_RECT>* receivers=nullptr)const {
        D3D11_RECT result={width+8,height+8,0,0};
        c3x_renderer::SceneProjection projection(renderer.content_view_width,
            renderer.content_view_height,projection_zoom);
        for(auto layer:{geometry_water,geometry_river})for(auto const& record:water_visible[layer]){
            auto const& b=GeometryDrawReference(record).bounds();
            float dx=settings.translation[0]+record.translation_x-4;
            float dy=settings.translation[1]+record.translation_y-4;
            int left=std::max(0,int(std::floor(projection.x(b.left+dx)+4)));
            int top=std::max(0,int(std::floor(projection.y(b.top+dy)+4)));
            int right=std::min(width,int(std::ceil(projection.x(b.right+dx)+4)));
            int bottom=std::min(height,int(std::ceil(projection.y(b.bottom+dy)+4)));
            if(left>=right || top>=bottom)continue;
            // Water uses a four-pixel reflection guard. Its bounded normal
            // distortion reaches at most 60x30 display pixels; retain another
            // four/six pixels for filtering at the .375 reflection scale.
            if(receivers)receivers->push_back({std::max(0,left+4-64),std::max(0,top+4-36),
                std::min(width+8,right+4+64),std::min(height+8,bottom+4+36)});
            result.left=std::min<LONG>(result.left,std::max(0,left+4-64));
            result.top=std::min<LONG>(result.top,std::max(0,top+4-36));
            result.right=std::max<LONG>(result.right,std::min(width+8,right+4+64));
            result.bottom=std::max<LONG>(result.bottom,std::min(height+8,bottom+4+36));
        }
        return result;
    }
    ViewportShaderSettings clip_settings(ViewportShaderSettings settings,bool mirrored)const {
        float guard=mirrored?8.f:4.f;
        c3x_renderer::SceneProjection(renderer.content_view_width,renderer.content_view_height,projection_zoom)
            .clip_transform(settings.translation,settings.inverse_size,guard,
                (1.f/settings.inverse_size[0]-renderer.content_view_width-2*guard)*.5f,
                (1.f/settings.inverse_size[1]-renderer.content_view_height-2*guard)*.5f);
        return settings;
    }
    // One vertex-pulled draw per resident page range for ordered non-rigid
    // records. Per-occurrence translation, natural projection and kind come
    // from page records; the shared viewport terms stay in b1. Returns false
    // (ordinary per-record path) on any refusal.
    bool draw_pulled(GeometryDrawReference const* records,unsigned count,GeometryLayer layer,
            ViewportShaderSettings const& viewport,bool mirrored,bool feature){
        auto* context=renderer.context;
        if(!pulled_batch){
            D3D11_BUFFER_DESC desc{};desc.ByteWidth=32;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(FAILED(renderer.device->CreateBuffer(&desc,nullptr,&pulled_batch))){++pulled_fallbacks;return false;}
        }
        pulled_sources.resize(count);
        for(unsigned n=0;n<count;++n){auto const& r=records[n];auto const& m=r.content();auto& source=pulled_sources[n];
            source.buffer=m.buffer;source.indices=m.indices;source.version=m.version;
            source.vertex_offset=m.vertex_offset;source.index_offset=m.index_offset;source.index_count=m.index_count;
            source.vertex_count=m.vertex_count;source.stride=m.vertex_stride;source.index16=m.index_format==DXGI_FORMAT_R16_UINT;
            source.tx=r.translation_x();source.ty=r.translation_y();source.kind=m.projection_kind;
            std::copy(std::begin(r.natural_projection()),std::end(r.natural_projection()),source.projection);}
        if(!pulled_pages.resolve(renderer.device,context,pulled_sources.data(),count,pulled_ranges)){++pulled_fallbacks;return false;}
        auto base=viewport;base.padding=0;
        if(renderer.pickup_profile)base.reserved[1]=layer==geometry_underlay?.5f:layer==geometry_bed?4.f:layer==geometry_water?5.f:0.f;
        float depth_base=renderer.city_profile?viewport.depth_translation:viewport.translation[1];
        base=clip_settings(base,mirrored);
        context->UpdateSubresource(renderer.viewport_settings_buffer,0,nullptr,&base,0,0);work.upload_buffer(renderer.viewport_settings_buffer);
        context->VSSetConstantBuffers(1,1,&renderer.viewport_settings_buffer);
        context->IASetInputLayout(nullptr);
        ID3D11Buffer* no_buffer=nullptr;UINT zero=0;context->IASetVertexBuffers(0,1,&no_buffer,&zero,&zero);
        context->IASetIndexBuffer(nullptr,DXGI_FORMAT_R32_UINT,0);
        context->VSSetShader(feature?visual.pulled_feature:visual.pulled_vertex,nullptr,0);
        context->VSSetConstantBuffers(12,1,&pulled_batch);
        for(auto const& range:pulled_ranges){
            struct {float translation[2];float depth;std::uint32_t vertex_base,first,last,pad2[2];} batch{
                {base.translation[0],base.translation[1]},depth_base,range.first_vertex,range.first_record,range.first_record+range.records-1,{0,0}};
            context->UpdateSubresource(pulled_batch,0,nullptr,&batch,0,0);
            ID3D11ShaderResourceView* views[]={range.page->record_view.Get(),range.page->vertex_view.Get(),range.page->index_view.Get()};
            context->VSSetShaderResources(60,3,views);
            context->Draw(range.vertices,0);
            work.draw(range.vertices,1,unsigned(layer));++renderer.frame_draw_calls;
            if(work.enabled)work.row(unsigned(layer)).submitted_instances+=range.records-1;
            ++pulled_draws;pulled_records+=range.records;
        }
        ID3D11ShaderResourceView* no_views[3]={};context->VSSetShaderResources(60,3,no_views);
        context->IASetInputLayout(feature?renderer.feature_input_layout:renderer.input_layout);
        context->VSSetShader(feature?renderer.feature_vertex_shader:renderer.vertex_shader,nullptr,0);
        return true;
    }
    bool issue_records(GeometryDrawView::Records const& records,GeometryLayer layer,
            ViewportShaderSettings const& viewport,D3D11_RECT rect,bool mirrored) {
        auto* context=renderer.context;
        bool streamed=parameters.available(renderer.device,context);
        PreparedParameters::Entry* prepared_entry=nullptr;
        // Only this immutable captured membership has a stable visibility epoch.
        // Animation/resource passes and reflection retain their existing path.
#ifdef C3X_RENDERER64_FRESH
        if(streamed && !mirrored && &records==&water_visible){
            PreparedParameters::Key key;
            key.scope={view_revision(),visibility_revision,renderer.device_generation,renderer.content_revision};
            key.scene={unsigned(records[layer].size()),unsigned(renderer.content_view_width),
                unsigned(renderer.content_view_height),unsigned(renderer.city_profile)*2+unsigned(renderer.pickup_profile)};
            key.viewport=viewport;key.rect={rect.left,rect.top,rect.right,rect.bottom};key.view={projection_zoom,renderer.reflection.height_pixels};
            prepared_entry=water_parameters.begin(unsigned(layer),key);
        }
#endif
        bool prepared_hit=prepared_entry && prepared_entry->complete;
        // Integrated-layout layers (168-byte vertices) and packed feature
        // layers (48 bytes: features, mines, farms, sites, walls, cliffs)
        // bind exactly one ordinary vertex shader per layer; cities keep
        // per-record material state.
        bool const pulled_feature_layer=layer!=geometry_city &&
            ((layer>=geometry_feature && layer<=geometry_site) || (layer>=geometry_cliff0 && layer<=geometry_cliff7));
        bool const pulled_layer=!mirrored && (pulled_feature_layer?visual.pulled_feature!=nullptr:
            visual.pulled_vertex && (layer==geometry_underlay || layer==geometry_land || layer==geometry_bed ||
            layer==geometry_water || layer==geometry_river || layer==geometry_shadow || layer==geometry_route));
        unsigned const pulled_stride=pulled_feature_layer?48u:168u;
        auto pulled_eligible=[pulled_stride](GeometryDrawReference const& record){auto const& m=record.content();
            return !m.rigid_source && !m.animation_texture && !m.resource_instance && m.city_material==0xffffffffu &&
                m.vertex_stride==pulled_stride && m.vertex_count && m.buffer && m.indices &&
                (m.index_format==DXGI_FORMAT_R16_UINT || m.index_format==DXGI_FORMAT_R32_UINT);};
        char emission_control[8]={};
        bool original_emission=c3x_renderer::render_core::cached_environment("C3X_SANDBOX_CITY_SUBMISSION_REFERENCE",emission_control,sizeof(emission_control)) && emission_control[0]=='1';
        ViewportShaderSettings previous{};
        bool previous_valid=false;
        // Each layer invocation owns its last upload across parameter flushes.
        // Other passes can change these buffers, so no value survives the call.
        using WaterSample=c3x_renderer::render_core::WaterMaterialFrame;
        WaterSample last_water{};
        std::array<float,4> last_wave{};
        bool last_water_valid=false,last_wave_valid=false,water_bound=false;
        auto same_float=[](float a,float b){return !std::memcmp(&a,&b,sizeof(float));};
        auto same_water=[&](WaterSample const& a,WaterSample const& b){
            if(!same_float(a.time,b.time))return false;
            for(unsigned i=0;i<3;++i)if(!same_float(a.drift[i],b.drift[i]))return false;
            for(unsigned i=0;i<4;++i)if(!same_float(a.camera[i],b.camera[i]))return false;
            return true;
        };
        std::array<ViewportShaderSettings,c3x_renderer::render_core::DrawParameterStream::limit> generated_values{};
        std::vector<GeometryDrawReference> selected;
        selected.reserve(generated_values.size());
        auto flush=[&](PreparedParameters::Batch const* prepared=nullptr) {
            if(selected.empty())return true;
            auto const& values=prepared?prepared->values:generated_values;
            if(!prepared){
                auto opaque=[](GeometryDrawReference const& draw){auto const& mesh=draw.content();
                    return mesh.rigid_source && !mesh.animation_texture && !mesh.resource_instance && mesh.city_material==0xffffffffu &&
                        c3x_renderer::render_core::opaque_rigid_material(mesh.instance_material);};
                auto compatible=[](GeometryDrawReference const& a,GeometryDrawReference const& b){auto const& x=a.content();auto const& y=b.content();
                    return x.buffer==y.buffer && x.indices==y.indices && x.vertex_offset==y.vertex_offset && x.index_offset==y.index_offset &&
                        x.index_count==y.index_count && x.index_format==y.index_format && x.projection_kind==y.projection_kind &&
                        x.instance_material==y.instance_material;};
                auto conflict=[&](GeometryDrawReference const& a,GeometryDrawReference const& b){
                    auto coverage=[&](GeometryDrawReference const& draw,bool reflected){auto const& bounds=draw.bounds();auto const& world=draw.content().world_bounds;
                        float low=reflected?2*renderer.reflection.height_pixels*std::max(0.f,world.low[2]-2.5f/112.f):0;
                        float high=reflected?2*renderer.reflection.height_pixels*std::max(0.f,world.high[2]-2.5f/112.f):0;
                        return std::array<double,4>{double(bounds.left)+draw.translation_x()-4,double(bounds.top)+draw.translation_y()+low-4,
                            double(bounds.right)+draw.translation_x()+4,double(bounds.bottom)+draw.translation_y()+high+4};};
                    for(bool reflected:{false,true}){auto x=coverage(a,reflected),y=coverage(b,reflected);
                        if(!(x[2]<y[0] || x[0]>y[2] || x[3]<y[1] || x[1]>y[3]))return true;}return false;};
                c3x_renderer::render_core::group_independent_opaque(selected,compatible,opaque,conflict);
                for(unsigned i=0;i<selected.size();++i){auto const& chunk=selected[i];auto settings=viewport;
                    std::copy(std::begin(chunk.natural_projection()),std::end(chunk.natural_projection()),settings.natural_projection);
                    settings.padding=float(chunk.content().projection_kind);
                    if(renderer.pickup_profile)settings.reserved[1]=layer==geometry_underlay?.5f:layer==geometry_bed?4.f:layer==geometry_water?5.f:0.f;
                    settings.translation[0]+=float(chunk.translation_x());settings.translation[1]+=float(chunk.translation_y());
                    settings.depth_translation=renderer.city_profile?viewport.depth_translation+float(chunk.translation_y()):settings.translation[1];generated_values[i]=clip_settings(settings,mirrored);
                }
            }
            std::array<unsigned,c3x_renderer::render_core::DrawParameterStream::limit> parameter_index{};
            std::array<ViewportShaderSettings,c3x_renderer::render_core::DrawParameterStream::limit> nonrigid_parameters{};unsigned parameter_count=0;
            if(prepared){parameter_index=prepared->parameter_index;parameter_count=prepared->parameter_count;}
            else {
                for(unsigned i=0;i<selected.size();++i)if(!selected[i].content().rigid_source){parameter_index[i]=parameter_count;nonrigid_parameters[parameter_count++]=values[i];}
                if(prepared_entry){
                    std::array<unsigned,PreparedParameters::limit> order{};
                    for(unsigned i=0;i<selected.size();++i)order[i]=unsigned(selected[i].occurrence-records[layer].data());
                    prepared=water_parameters.append(*prepared_entry,renderer.device,order,unsigned(selected.size()),
                        values,parameter_index,nonrigid_parameters.data(),parameter_count);
                    if(!prepared)prepared_entry=nullptr; // Bounded admission/allocation failure keeps the streaming path.
                    else work.upload(parameter_count*PreparedParameters::stride,unsigned(layer));
                }
            }
            if(!prepared && streamed && parameter_count && !parameters.upload(nonrigid_parameters.data(),parameter_count)){
                std::printf("SANDBOX_RECORD_UPLOAD_ERROR layer=%u records=%zu reason=0x%08lx\n",
                    unsigned(layer),selected.size(),renderer.device->GetDeviceRemovedReason());
                std::fflush(stdout);return false;
            }
            if(!prepared && streamed)work.upload(parameter_count*((sizeof(ViewportShaderSettings)+255)/256)*256,unsigned(layer));
            std::array<decltype(renderer.ordered_rigid_packets)::Range,c3x_renderer::render_core::DrawParameterStream::limit> packets{};
            auto packet_bytes=renderer.ordered_rigid_packets.uploaded_bytes;
            auto packet_copies=renderer.ordered_rigid_packets.placement_copies;
            renderer.prepare_ordered_rigid_packets(unsigned(layer),selected,shared_front,packets);
            auto uploaded=renderer.ordered_rigid_packets.uploaded_bytes-packet_bytes;
            if(uploaded)work.upload(unsigned(uploaded),unsigned(layer));
            work.calls.copies+=renderer.ordered_rigid_packets.placement_copies-packet_copies;
            std::vector<unsigned> rigid;
            std::array<unsigned,c3x_renderer::render_core::DrawParameterStream::limit> rigid_offset{};
            for(unsigned i=0;i<selected.size();++i){auto const& chunk=selected[i];if(!chunk.content().rigid_source || packets[i])continue;
                auto range=shared_front->find(renderer.shared_instance_draw_key(unsigned(layer),chunk));
                if(!range || range.count!=1)return renderer.reject_shared_instance_range(unsigned(layer),chunk,shared_front,range);
                rigid_offset[i]=unsigned(rigid.size());rigid.push_back(range.first);}
            if(!rigid.empty()){
                if(!renderer.shared_instances.select_indices(renderer.device,context,shared_front,rigid.data(),unsigned(rigid.size())))return false;
                work.upload(rigid.size()*sizeof(unsigned),unsigned(layer));context->VSSetShaderResources(15,1,&shared_front->view);
            }
            auto& mirror=sandbox_active_reflection();
            for(unsigned i=0;i<selected.size();){
                auto const& chunk=selected[i];
                auto const& mesh=chunk.content();
                auto const& settings=values[i];
                if(mesh.rigid_source){
                    if(!previous_valid || std::memcmp(&previous,&viewport,sizeof(viewport))){
                        auto projected=clip_settings(viewport,mirrored);
                        context->UpdateSubresource(renderer.viewport_settings_buffer,0,nullptr,&projected,0,0);work.upload_buffer(renderer.viewport_settings_buffer);
                        previous=viewport;previous_valid=true;
                    }context->VSSetConstantBuffers(1,1,&renderer.viewport_settings_buffer);
                }else if(prepared)prepared->bind(parameters.context,1,i);
                else if(streamed)parameters.bind(1,parameter_index[i]);
                else if(!previous_valid || std::memcmp(&previous,&settings,sizeof(settings))){
                    context->UpdateSubresource(renderer.viewport_settings_buffer,0,nullptr,&settings,0,0);work.upload_buffer(renderer.viewport_settings_buffer);
                    previous=settings;previous_valid=true;
                }
                if(packets[i]){
                    // Generic fallback city pieces may follow a native city
                    // material, which changes the pixel shader and samplers.
                    if(layer==geometry_city){
                        context->PSSetShader(mirrored?mirror.ps[1]:renderer.feature_pixel_shader,nullptr,0);
                        context->PSSetShaderResources(116,4,renderer.city_emissive_views.data());
                        context->PSSetShaderResources(124,4,renderer.city_base_views.data());
                        ID3D11SamplerState* samplers[]={renderer.terrain_sampler,renderer.decal_sampler};
                        context->PSSetSamplers(0,2,samplers);
                    }
                    unsigned end=i+1;while(end<selected.size() && packets[end-1].contiguous(packets[end]))++end;
                    renderer.ordered_rigid_packets.issue(context,renderer.ordered_rigid_layout,
                        renderer.rigid_sources.resident_vertex[mirrored?1:0],packets.data()+i,end-i,
                        [&](unsigned indices,unsigned records){
                            work.draw(indices,1,unsigned(layer));
                            if(work.enabled)work.row(unsigned(layer)).submitted_instances+=records-1;
                            ++renderer.frame_draw_calls;
                        });
                    context->VSSetShaderResources(15,1,&shared_front->view);
                    context->IASetInputLayout(renderer.feature_input_layout);
                    context->VSSetShader(mirrored?mirror.vs[1]:renderer.feature_vertex_shader,nullptr,0);
                    i=end;continue;
                }
                if(pulled_layer && pulled_eligible(selected[i])){
                    // Contiguous eligible records share one vertex-pulled draw
                    // per page range; fog-frozen water splits on its sample.
                    unsigned end=i+1;auto water_sample=[&](GeometryDrawReference const& record){
                        auto sample=renderer.water_material;
                        if(!renderer.water_scene_active || !record.water_visible() || record.content().visual_time>=0){
                            sample.time=0;sample.drift[0]=sample.drift[1]=sample.drift[2]=0;}
                        return sample;};
                    bool water_layer=renderer.environment_profile && (layer==geometry_water || layer==geometry_river);
                    auto first_sample=water_sample(chunk);
                    while(end<selected.size() && !packets[end] && pulled_eligible(selected[end]) &&
                          (!water_layer || same_water(water_sample(selected[end]),first_sample)))++end;
                    // A pulled run costs a fixed state switch; short runs are
                    // cheaper as ordinary indexed draws.
                    if(end-i>=8){
                    if(water_layer){
                        if(!last_water_valid || !same_water(first_sample,last_water)){
                            context->UpdateSubresource(renderer.water_frame,0,nullptr,&first_sample,0,0);work.upload_buffer(renderer.water_frame);
                            last_water=first_sample;last_water_valid=true;++phase_constant_counts.water_updates;
                        }else ++phase_constant_counts.water_hits;
                        if(!water_bound){context->PSSetConstantBuffers(10,1,&renderer.water_frame);water_bound=true;}
                    }
                    if(draw_pulled(selected.data()+i,end-i,layer,viewport,mirrored,pulled_feature_layer)){
                        if(water_layer){phase_constant_counts.water_records+=end-i;phase_constant_counts.water_hits+=end-i-1;}
                        previous_valid=false;i=end;continue;
                    }
                    }
                }
                UINT stride=mesh.vertex_stride,offset=mesh.vertex_offset;
                context->IASetVertexBuffers(0,1,&mesh.buffer,&stride,&offset);
                context->IASetIndexBuffer(mesh.indices,mesh.index_format,mesh.index_offset);
                if(mesh.rigid_source){
                    unsigned end=i+1;
                    for(;end<selected.size();++end){
                        auto const& next=selected[end].content();
                        if(packets[end] || !next.rigid_source || next.buffer!=mesh.buffer ||
                            next.indices!=mesh.indices || next.vertex_offset!=mesh.vertex_offset ||
                            next.index_offset!=mesh.index_offset || next.index_count!=mesh.index_count ||
                            next.index_format!=mesh.index_format ||
                            next.projection_kind!=mesh.projection_kind)break;
                    }
                    context->IASetInputLayout(renderer.rigid_sources.resident_layout);
                    context->VSSetShader(renderer.rigid_sources.resident_vertex[mirrored?1:0],nullptr,0);
                    if(layer==geometry_city){
                        context->PSSetShader(mirrored?mirror.ps[1]:renderer.feature_pixel_shader,nullptr,0);
                        context->PSSetShaderResources(116,4,renderer.city_emissive_views.data());
                        context->PSSetShaderResources(124,4,renderer.city_base_views.data());
                        ID3D11SamplerState* samplers[]={renderer.terrain_sampler,renderer.decal_sampler};
                        context->PSSetSamplers(0,2,samplers);
                    }
                    ID3D11Buffer* streams[]={mesh.buffer,renderer.shared_instances.selection_buffer};
                    UINT strides[]={32,4};UINT offsets[]={mesh.vertex_offset,renderer.shared_instances.selection_offset+rigid_offset[i]*4};
                    context->IASetVertexBuffers(0,2,streams,strides,offsets);
                    context->DrawIndexedInstanced(mesh.index_count,end-i,0,0,0);
                    work.draw(mesh.index_count,end-i,unsigned(layer));
                    ++renderer.frame_draw_calls;
                    context->IASetInputLayout(renderer.feature_input_layout);
                    context->VSSetShader(mirrored?mirror.vs[1]:renderer.feature_vertex_shader,nullptr,0);
                    i=end;continue;
                }
                // Native city emission interleaves a second pass after each
                // body. Keep that boundary; all other compatible adjacent
                // ranges can share one draw without changing primitive order.
                bool city_emission=mesh.city_material!=0xffffffffu &&
                    !renderer.cities.library.materials[mesh.city_material].ground &&
                    (original_emission || renderer.cities.emits(mesh.city_material));
                unsigned end=i+1,index_count=mesh.index_count;
                auto index_stride=mesh.index_format==DXGI_FORMAT_R16_UINT?2u:4u;
                if(!city_emission)for(;end<selected.size();++end){
                    auto const& next=selected[end].content();
                    if(packets[end] || next.index_count>UINT_MAX-index_count ||
                            !c3x_renderer::render_core::compatible_ordered_index_range(
                                selected[end-1],selected[end],values[end-1],values[end],index_stride))break;
                    index_count+=next.index_count;
                }
                if(mesh.city_material!=0xffffffffu){
                    ID3D11SamplerState* samplers[]={renderer.natural_wrap,renderer.natural_clamp};
                    context->PSSetSamplers(0,2,samplers);
                    renderer.cities.bind(context,mesh.city_material,mesh.city_environment,
                        mesh.city_atlas,mirrored,false,stride==88);
                    context->DrawIndexed(index_count,0,0);work.draw(index_count,1,unsigned(layer));++renderer.frame_draw_calls;
                    if(work.enabled)work.row(unsigned(layer)).submitted_instances+=end-i-1;
                    if(city_emission){
                        if(original_emission)renderer.cities.bind(context,mesh.city_material,mesh.city_environment,
                            mesh.city_atlas,mirrored,true,stride==88);
                        else renderer.cities.bind_emission(context,mirrored);
                        context->DrawIndexed(mesh.index_count,0,0);work.draw(mesh.index_count,1,unsigned(layer));++renderer.frame_draw_calls;
                    }
                    context->OMSetBlendState(renderer.blend_state,nullptr,0xffffffffu);
                    context->OMSetDepthStencilState(renderer.depth_state,0);
                    i=end;continue;
                }
                if(renderer.city_profile && layer==geometry_city){
                    context->IASetInputLayout(renderer.feature_input_layout);
                    context->VSSetShader(mirrored?mirror.vs[1]:renderer.feature_vertex_shader,nullptr,0);
                    context->PSSetShader(mirrored?mirror.ps[1]:renderer.feature_pixel_shader,nullptr,0);
                    context->PSSetShaderResources(116,4,renderer.city_emissive_views.data());
                    context->PSSetShaderResources(124,4,renderer.city_base_views.data());
                    ID3D11SamplerState* samplers[]={renderer.terrain_sampler,renderer.decal_sampler};
                    context->PSSetSamplers(0,2,samplers);
                }
                if(mesh.animation_texture)context->PSSetShaderResources(116,1,&mesh.animation_texture);
                if(mesh.resource_instance){
                    context->IASetInputLayout(renderer.resource_input_layout);
                    context->VSSetConstantBuffers(8,1,&mesh.resource_instance);
                    context->VSSetConstantBuffers(2,1,&renderer.shadow_settings_buffer);
                    context->VSSetShader(layer==geometry_shadow?renderer.resource_shadow_vertex_shader:
                        renderer.resource_body_vertex_shader,nullptr,0);
                }
                if(renderer.environment_profile && (layer==geometry_water || layer==geometry_river)){
                    // Occurrence eligibility includes explored fog; only unknown
                    // coverage and explicit still controls use phase zero.
                    auto sample=renderer.water_material;
                    if(!renderer.water_scene_active || !chunk.water_visible() || mesh.visual_time>=0){
                        sample.time=0;sample.drift[0]=sample.drift[1]=sample.drift[2]=0;
                    }
                    phase_constant_counts.water_records+=end-i;
                    if(!last_water_valid || !same_water(sample,last_water)){
                        context->UpdateSubresource(renderer.water_frame,0,nullptr,&sample,0,0);work.upload_buffer(renderer.water_frame);
                        last_water=sample;last_water_valid=true;++phase_constant_counts.water_updates;
                    }else ++phase_constant_counts.water_hits;
                    phase_constant_counts.water_hits+=end-i-1;
                    if(!water_bound){context->PSSetConstantBuffers(10,1,&renderer.water_frame);water_bound=true;}
                }
                if(layer==geometry_wave){
                    std::array<float,4> sample={mesh.visual_time<0?renderer.wave_time_seconds:mesh.visual_time,0,0,0};
                    bool same=last_wave_valid;
                    for(unsigned j=0;same && j<sample.size();++j)same=same_float(sample[j],last_wave[j]);
                    phase_constant_counts.wave_records+=end-i;
                    if(!same){
                        context->UpdateSubresource(renderer.wave_frame,0,nullptr,sample.data(),0,0);work.upload_buffer(renderer.wave_frame);
                        last_wave=sample;last_wave_valid=true;++phase_constant_counts.wave_updates;
                    }else ++phase_constant_counts.wave_hits;
                    phase_constant_counts.wave_hits+=end-i-1;
                }
                context->DrawIndexed(index_count,0,0);work.draw(index_count,1,unsigned(layer));++renderer.frame_draw_calls;
                if(work.enabled)work.row(unsigned(layer)).submitted_instances+=end-i-1;
                if(mesh.resource_instance){
                    context->IASetInputLayout(layer==geometry_shadow?
                        renderer.input_layout:renderer.feature_input_layout);
                    context->VSSetShader(layer==geometry_shadow?
                        renderer.vertex_shader:renderer.feature_vertex_shader,nullptr,0);
                }
                if(mesh.animation_texture)context->PSSetShaderResources(116,1,
                    renderer.resource_texture_views.data());
                i=end;
            }
            selected.clear();return true;
        };
        if(prepared_hit){
            for(auto const& batch:prepared_entry->batches){
                // Resolve only after the exact generation/view check. Cached
                // order contains indices, never pointers into retired records.
                for(unsigned i=0;i<batch.count;++i)selected.emplace_back(records[layer][batch.order[i]]);
                water_parameters.reused_records+=batch.count;
                if(work.enabled)work.row(unsigned(layer)).accepted_records+=batch.count;
                if(!flush(&batch))return false;
            }
        }else{
            auto clip=source_bounds(viewport,rect,mirrored);
            for(auto const& record:records[layer]){
                if(work.enabled)++work.row(unsigned(layer)).tested_records;
                GeometryDrawReference chunk(record);
                if(mirrored && chunk.content().animation_texture)continue;
                if(!renderer.chunk_intersects_region(chunk,viewport,clip,mirrored))continue;
                if(work.enabled)++work.row(unsigned(layer)).accepted_records;
                selected.push_back(chunk);
                if(selected.size()==generated_values.size() && !flush())return false;
            }
            if(!flush())return false;
            if(prepared_entry)prepared_entry->complete=true;
        }
        if(streamed)context->VSSetConstantBuffers(1,1,&renderer.viewport_settings_buffer);
        return true;
    }
    void bind_common(ViewportShaderSettings const& settings,D3D11_RECT rect,
            ID3D11RenderTargetView* target,ID3D11DepthStencilView* depth,
            bool mirrored,float scale,bool default_materials=true) {
        auto* context=renderer.context;
        context->PSSetSamplers(15,1,&visual.shadow_comparison);
        context->OMSetRenderTargets(1,&target,depth);
        context->OMSetDepthStencilState(renderer.depth_state,0);
        context->OMSetBlendState(renderer.blend_state,nullptr,0xffffffffu);
        context->RSSetState(renderer.rasterizer_state);
        D3D11_VIEWPORT viewport=target_viewport(target,mirrored,scale);
        context->RSSetViewports(1,&viewport);
        D3D11_RECT scissor={LONG(rect.left*scale),LONG(rect.top*scale),
            LONG(rect.right*scale),LONG(rect.bottom*scale)};
        context->RSSetScissorRects(1,&scissor);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->IASetInputLayout(renderer.input_layout);
        auto& mirror=sandbox_active_reflection();
        context->VSSetShader(mirrored?mirror.vs[0]:renderer.vertex_shader,nullptr,0);
        context->PSSetShader(mirrored?mirror.ps[0]:renderer.pixel_shader,nullptr,0);
        context->PSSetConstantBuffers(0,1,&renderer.terrain_settings_buffer);
        context->VSSetConstantBuffers(1,1,&renderer.viewport_settings_buffer);
        context->PSSetConstantBuffers(2,1,&renderer.shadow_settings_buffer);
        context->PSSetConstantBuffers(3,1,&renderer.world_settings_buffer);
        context->PSSetConstantBuffers(4,1,&renderer.source_shadow.table);
        ID3D11SamplerState* samplers[]={renderer.natural_wrap,renderer.natural_clamp};
        context->PSSetSamplers(0,2,samplers);
        auto const& materials=renderer.compiled_material_views();
        // Natural providers immediately bind their complete material closure;
        // rebinding the generic 128-slot table first adds unused descriptors
        // for every tree body. Their shared/extended inputs are bound below.
        if(default_materials){
            context->PSSetShaderResources(0,UINT(materials.size()),materials.data());
            context->PSSetShaderResources(25,1,&renderer.source_shadow.sampled_view());
        }
        ID3D11ShaderResourceView* reflection_view=mirrored?nullptr:reflection.view;
        context->PSSetShaderResources(121,1,&reflection_view);
        mirror.bind(context,unsigned(reflection.width/reflection_scale),
            unsigned(reflection.height/reflection_scale),4+(mirrored?0.f:reflection_shift[0]),
            4+(mirrored?0.f:reflection_shift[1]));
        (void)settings;
    }
    bool draw_vegetation_instances(GeometryDrawView::Records const& records,
            GeometryLayer layer,ViewportShaderSettings const& settings,D3D11_RECT rect,bool mirrored) {
        using Stream=c3x_renderer::render_core::InstanceStream;
        auto& group=records[layer];
        if(group.empty())return true;
        auto const& mesh=group.front().content();
        if(!mesh.instances)return false;
        auto* context=renderer.context;
        auto projected=clip_settings(settings,mirrored);
        context->UpdateSubresource(renderer.viewport_settings_buffer,0,nullptr,&projected,0,0);work.upload_buffer(renderer.viewport_settings_buffer);
        renderer.natural.bind_instances(context,unsigned(layer-geometry_natural_forest0));
        if(mirrored)context->VSSetShader(renderer.reflection.instance_vs,nullptr,0);
#ifdef C3X_RENDERER64_FRESH
        // Select the exact contributing order before packing or uploading. The
        // bounded cache owns GPU submissions, never another world/mesh owner.
        auto immutable_clip=source_bounds(settings,rect,mirrored);
        std::vector<GeometryDrawRecord const*> selected;
        std::vector<std::uint64_t> key;std::size_t count=0;bool admissible=true;
        key.reserve(std::min(group.size(),std::size_t(8192/14))*14);
        for(auto const& record:group){
            if(work.enabled)++work.row(unsigned(layer)).tested_records;
            if(!renderer.chunk_intersects_region(GeometryDrawReference(record),settings,immutable_clip,mirrored))continue;
            auto const& content=record.content();if(!content.instances || content.buffer!=mesh.buffer)return false;
            if(work.enabled){++work.row(unsigned(layer)).accepted_records;
                work.row(unsigned(layer)).tested_instances+=content.instances->size();
                work.row(unsigned(layer)).accepted_instances+=content.instances->size();}
            count+=content.instances->size();
            if(admissible){
                if(key.size()>8192-14 || count>Stream::limit){admissible=false;selected.clear();key.clear();}
                else {selected.push_back(&record);auto input=contributor_key(unsigned(layer),record);
                    key.insert(key.end(),input.begin(),input.end());}
            }
        }
        if(!count)return true;
        auto slot=admissible?instance_plans.select(key):UINT_MAX;
        if(slot!=UINT_MAX){
            auto& cached=instance_plans[slot];auto& plan=cached.value;
            if(!cached.prepared){
                // FrameSampleCache replaces the exact key but leaves its value
                // alive. Retire that old index/front lease before admission.
                instance_plan_bytes-=plan.bytes;plan={};cached.prepared=true;
            }
            if(!cached.valid || !renderer.shared_instances.valid(plan.selection) || plan.selection->content!=shared_front){
                instance_plan_bytes-=plan.bytes;plan={};cached.valid=false;
                auto metadata=key.capacity()*sizeof(key[0])+sizeof(typename decltype(instance_plans)::Entry);
                // Unchanged capacity refusal goes directly to bounded index
                // scratch. Do not repack/allocate a doomed immutable plan on
                // every animation tick; released readers reopen admission.
                if(renderer.shared_instances.can_prepare_selection(unsigned(count),metadata)){
                    std::vector<unsigned> prepared;prepared.reserve(count);
                    for(auto const* record:selected){auto draw=GeometryDrawReference(*record);auto range=shared_front->find(renderer.shared_instance_draw_key(unsigned(layer),draw));
                        if(!range || range.count!=record->content().instances->size())return renderer.reject_shared_instance_range(unsigned(layer),draw,shared_front,range);
                        for(unsigned i=0;i<range.count;++i)prepared.push_back(range.first+i);}
                    auto candidate=renderer.shared_instances.prepare_selection(renderer.device,shared_front,prepared.data(),unsigned(prepared.size()),metadata);
                    if(candidate){instance_plan_bytes+=candidate->bytes();plan.selection=std::move(candidate);
                        plan.bytes=plan.selection->bytes();plan.count=unsigned(count);cached.valid=true;++instance_plan_builds;work.upload(count*sizeof(unsigned),unsigned(layer));}
                }
            }else ++instance_plan_reuses;
            if(cached.valid){
                context->IASetInputLayout(renderer.natural.resident_instance_layout);
                context->VSSetShader(visual.vegetation_instances[mirrored?1:0],nullptr,0);context->VSSetShaderResources(15,1,&shared_front->view);
                ID3D11Buffer* streams[]={mesh.buffer,plan.selection->buffer};UINT strides[]={32,4},offsets[]={0,0};
                context->IASetVertexBuffers(0,2,streams,strides,offsets);context->IASetIndexBuffer(mesh.indices,mesh.index_format,0);
                char skip_depth[8]{};
                if(!(c3x_renderer::render_core::cached_environment("C3X_SANDBOX_SKIP_VEGETATION_DEPTH",skip_depth,sizeof(skip_depth)) && skip_depth[0]=='1')){
                    context->PSSetShader(visual.vegetation_depth[mirrored?1:0],nullptr,0);
                    context->DrawIndexedInstanced(mesh.index_count,plan.count,0,0,0);work.draw(mesh.index_count,plan.count,unsigned(layer));++renderer.frame_draw_calls;
                    context->PSSetShader(mirrored?sandbox_active_reflection().ps[4]:renderer.natural.ps[2],nullptr,0);
                }
                context->DrawIndexedInstanced(mesh.index_count,plan.count,0,0,0);work.draw(mesh.index_count,plan.count,unsigned(layer));++renderer.frame_draw_calls;return true;
            }
        }
        // Admission pressure still submits identical selected indices through
        // the bounded scratch stream; no source placements are repacked.
#endif
        context->IASetInputLayout(renderer.natural.resident_instance_layout);
        context->VSSetShader(mirrored?renderer.reflection.resident_instance_vs:renderer.natural.resident_instance_vs,nullptr,0);
        context->VSSetShaderResources(15,1,&shared_front->view);
        std::vector<unsigned> batch;batch.reserve(Submission::record_limit);
        auto flush=[&](){if(batch.empty())return true;
            if(!renderer.shared_instances.select_indices(renderer.device,context,shared_front,batch.data(),unsigned(batch.size())))return false;
            work.upload(batch.size()*sizeof(unsigned),unsigned(layer));
            ID3D11Buffer* streams[]={mesh.buffer,renderer.shared_instances.selection_buffer};UINT strides[]={32,4},offsets[]={0,renderer.shared_instances.selection_offset};
            context->IASetVertexBuffers(0,2,streams,strides,offsets);context->IASetIndexBuffer(mesh.indices,mesh.index_format,0);
            char skip_depth[8]{};
            if(!(c3x_renderer::render_core::cached_environment("C3X_SANDBOX_SKIP_VEGETATION_DEPTH",skip_depth,sizeof(skip_depth)) && skip_depth[0]=='1')){
                context->PSSetShader(visual.vegetation_depth[mirrored?1:0],nullptr,0);
                context->DrawIndexedInstanced(mesh.index_count,UINT(batch.size()),0,0,0);work.draw(mesh.index_count,batch.size(),unsigned(layer));++renderer.frame_draw_calls;
                context->PSSetShader(mirrored?sandbox_active_reflection().ps[4]:renderer.natural.ps[2],nullptr,0);
            }
            context->DrawIndexedInstanced(mesh.index_count,UINT(batch.size()),0,0,0);work.draw(mesh.index_count,batch.size(),unsigned(layer));++renderer.frame_draw_calls;batch.clear();return true;};
        auto clip=source_bounds(settings,rect,mirrored);
        for(auto const& record:group){
            if(!renderer.chunk_intersects_region(GeometryDrawReference(record),settings,clip,mirrored))continue;
            auto draw=GeometryDrawReference(record);auto range=shared_front->find(renderer.shared_instance_draw_key(unsigned(layer),draw));
            if(!range || range.count!=record.content().instances->size())return renderer.reject_shared_instance_range(unsigned(layer),draw,shared_front,range);
            for(unsigned i=0;i<range.count;++i){if(batch.size()==Submission::record_limit && !flush())return false;batch.push_back(range.first+i);}
        }
        return flush();
    }
    bool draw_layer(GeometryDrawView::Records const& records,GeometryLayer layer,
            ViewportShaderSettings const& settings,D3D11_RECT rect,
            ID3D11RenderTargetView* target,ID3D11DepthStencilView* depth,
            bool mirrored,float scale,bool capture_terrain_material=false) {
        if (records[layer].empty()) return true;
        char skip_vegetation[8]={};
        if(layer>=geometry_natural_forest0 && c3x_renderer::render_core::cached_environment("C3X_SANDBOX_SKIP_VEGETATION",skip_vegetation,sizeof(skip_vegetation)) && skip_vegetation[0]=='1')return true;
        // Opt-in attribution only: virtual adapters may return retrieval-clock
        // timestamps. A bounded completion probe includes driver submission and
        // GPU work, and deliberately serializes passes. Never use for FPS claims.
        struct CompletionProbe {
            ID3D11DeviceContext* context=nullptr;
            Microsoft::WRL::ComPtr<ID3D11Query> query;
            LARGE_INTEGER start{},frequency{};
            unsigned layer;bool mirrored;
            bool wait() {
                context->End(query.Get());context->Flush();
                ULONGLONG deadline=GetTickCount64()+2000;
                HRESULT result;
                while((result=context->GetData(query.Get(),nullptr,0,
                        D3D11_ASYNC_GETDATA_DONOTFLUSH))==S_FALSE && GetTickCount64()<deadline)Sleep(0);
                return result==S_OK;
            }
            CompletionProbe(unsigned l,bool m):layer(l),mirrored(m) {
                char option[8]{};
                if(!c3x_renderer::render_core::cached_environment("C3X_SANDBOX_PROFILE_COMPLETION",option,sizeof(option)) || option[0]!='1')return;
                D3D11_QUERY_DESC desc={D3D11_QUERY_EVENT,0};
                if(FAILED(renderer.device->CreateQuery(&desc,&query)))return;
                context=renderer.context;
                if(!wait()){context=nullptr;return;}
                QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&start);
            }
            ~CompletionProbe(){if(context){
                bool valid=wait();LARGE_INTEGER end{};QueryPerformanceCounter(&end);
                double ms=1000.*double(end.QuadPart-start.QuadPart)/frequency.QuadPart;
                if(ms>1 || !valid){
                    std::printf("SANDBOX_COMPLETION layer=%u mirror=%u valid=%u ms=%.3f\n",layer,unsigned(mirrored),unsigned(valid),ms);
                    std::fflush(stdout);
                }
            }}
        } completion_probe(unsigned(layer),mirrored);
        bind_common(settings,rect,target,depth,mirrored,scale,layer<geometry_natural_terrain);
        auto* context=renderer.context;
        auto& mirror=sandbox_active_reflection();
        // Mirrored ground (terrain, hills, decals, farm fields) adds color but
        // no coverage: rivers reflect only the standing objects and mountains
        // that do. The cached-terrain mirror adds mountain coverage separately.
        if(mirrored && mirror_mountain_coverage && renderer.reflection_coverage_blend)
            context->OMSetBlendState(renderer.reflection_coverage_blend,
                renderer.mountain_mirror_coverage,0xffffffffu);
        else if(mirrored && renderer.reflection_terrain_blend &&
           (layer==geometry_farm || layer==geometry_natural_terrain ||
            layer==geometry_natural_decal))
            context->OMSetBlendState(renderer.reflection_terrain_blend,nullptr,0xffffffffu);
        if (layer>=geometry_natural_terrain) {
            unsigned provider=layer==geometry_natural_mountain?1:
                layer>=geometry_natural_forest0?2:0;
            unsigned body=layer>=geometry_natural_forest0?
                unsigned(layer-geometry_natural_forest0):0;
            context->OMSetDepthStencilState(layer==geometry_natural_decal?
                renderer.natural.decal_depth:renderer.depth_state,0);
            renderer.natural.bind(context,provider,body);
            auto const& materials=renderer.compiled_material_views();
            context->PSSetShaderResources(69,1,materials.data()+69);
            context->PSSetShaderResources(71,1,materials.data()+71);
            if(provider==0) {
                ID3D11ShaderResourceView* cliff[]={renderer.cliff_views[0],materials[15]};
                context->PSSetShaderResources(31,2,cliff);
            }
        } else if (layer>=geometry_cliff0 && layer<geometry_natural_terrain) {
            context->IASetInputLayout(renderer.feature_input_layout);
            context->VSSetShader(renderer.feature_vertex_shader,nullptr,0);
            context->PSSetShader(renderer.feature_pixel_shader,nullptr,0);
            context->PSSetSamplers(0,1,&renderer.natural_clamp);
            unsigned index=renderer.cliff_bundle.assets[layer-geometry_cliff0].texture_index;
            context->PSSetShaderResources(25,4,renderer.cliff_views.data()+index);
        } else if (layer>=geometry_feature && layer<=geometry_site) {
            context->IASetInputLayout(renderer.feature_input_layout);
            context->VSSetShader(renderer.feature_vertex_shader,nullptr,0);
            context->PSSetShader(renderer.feature_pixel_shader,nullptr,0);
            context->PSSetShaderResources(25,4,renderer.feature_texture_views.data());
            context->PSSetShaderResources(94,4,renderer.feature_texture_views.data()+4);
            if(layer==geometry_site)context->PSSetShaderResources(116,8,renderer.site_views.data());
            if(layer==geometry_mine){
                context->PSSetShaderResources(116,6,renderer.mine_base_views.data());
                context->PSSetShaderResources(124,2,renderer.mine_emissive_views.data());
            }
            if(layer==geometry_farm){
                context->PSSetShaderResources(116,6,renderer.farm_base_views.data());
                context->PSSetShaderResources(124,2,renderer.farm_emissive_views.data());
            }
            if(layer==geometry_city){
                context->PSSetShaderResources(116,4,renderer.city_emissive_views.data());
                context->PSSetShaderResources(124,4,renderer.city_base_views.data());
            }
            if(layer==geometry_wall){
                std::array<ID3D11ShaderResourceView*,4> empty{};
                context->PSSetShaderResources(116,4,empty.data());
                context->PSSetShaderResources(124,4,renderer.wall_texture_views.data());
            }
        } else if (layer==geometry_wave) {
            context->PSSetShader(renderer.wave_shader,nullptr,0);
            context->PSSetShaderResources(0,3,renderer.wave_views.data());
            context->PSSetConstantBuffers(7,1,&renderer.wave_frame);
            context->OMSetDepthStencilState(renderer.natural.decal_depth,0);
        } else if ((layer==geometry_route || layer==geometry_river) && renderer.natural.decal_depth) {
            // Routes are a surface decal: a translucent fringe must not write
            // depth and reject another path's core where strips overlap.
            // The river is one too: its bank band tests against the ground
            // but must not reject the bridges, farms and routes drawn after it.
            context->OMSetDepthStencilState(renderer.natural.decal_depth,0);
        }
        if(mirrored) {
            unsigned provider=layer<geometry_feature?0:layer<geometry_natural_terrain?1:
                layer<=geometry_natural_decal?2:layer==geometry_natural_mountain?3:4;
            context->VSSetShader(mirror.vs[provider],nullptr,0);
            context->PSSetShader(mirror.ps[provider],nullptr,0);
        }
        // t17 is the terrain shader's shadow atlas, but the hydrology shader's
        // shallow-bed color. Never carry the shadow binding into water.
        if(layer>=geometry_natural_terrain ||
            (layer>=geometry_cliff0 && layer<geometry_natural_terrain) ||
            (layer>=geometry_feature && layer<=geometry_site))
            context->PSSetShaderResources(17,1,&renderer.source_shadow.sampled_view());
        if(renderer.city_profile)renderer.cities.scene_lights.bind(context);
        if(layer>=geometry_natural_forest0 && records[layer].front().content().instances)
            return draw_vegetation_instances(records,layer,settings,rect,mirrored);
        if(!mirrored && layer==geometry_water){
            char full_water[8]{};
            if(!(c3x_renderer::render_core::cached_environment("C3X_SANDBOX_WATER_FULL_SHADER",
                    full_water,sizeof(full_water)) && std::strcmp(full_water,"1")==0))
                context->PSSetShader(visual.water_surface,nullptr,0);
            if(capture_water_lighting){
                context->PSSetShader(visual.water_lighting,nullptr,0);
                context->OMSetBlendState(nullptr,nullptr,0xffffffffu);
            }
            ID3D11ShaderResourceView* lighting=capture_water_lighting?nullptr:water_lighting.view;
            context->PSSetShaderResources(122,1,&lighting);
            // This source-material slot is unused by water. Its water variant
            // samples the animated marine layer beneath the surface.
            ID3D11ShaderResourceView* aquatic=
                renderer.sandbox_aquatic_resource_poses[geometry_feature].empty()?
                    nullptr:aquatic_scene.view;
            context->PSSetShaderResources(123,1,&aquatic);
            context->PSSetConstantBuffers(11,1,&aquatic_bounds_buffer);
        }
        if(!mirrored && layer==geometry_river)
            context->PSSetShader(visual.river_surface,nullptr,0);
        if(!mirrored && layer==geometry_underlay)
            context->PSSetShader(visual.underlay,nullptr,0);
        if(capture_terrain_material) {
            auto& albedo=mirrored?reflected_terrain_albedo:terrain_albedo;
            auto& normal=mirrored?reflected_terrain_normal:terrain_normal;
            auto& world=mirrored?reflected_terrain_world:terrain_world;
            auto& properties=mirrored?reflected_terrain_properties:terrain_properties;
            ID3D11RenderTargetView* targets[]={albedo.target,
                normal.target,world.target,properties.target};
            context->OMSetRenderTargets(4,targets,albedo.depth);
            context->OMSetBlendState(terrain_material_blend,nullptr,0xffffffffu);
            context->PSSetShader(mirrored?
                (layer==geometry_natural_mountain?visual.reflected_mountain_material:
                    visual.reflected_terrain_material):
                (layer==geometry_natural_mountain?visual.mountain_material:
                    visual.terrain_material),nullptr,0);
        }
        return issue_records(records,layer,settings,rect,mirrored);
    }
    bool draw_scene(GeometryDrawView::Records const& records,
            ViewportShaderSettings const& settings,D3D11_RECT rect,
            ID3D11RenderTargetView* target,ID3D11DepthStencilView* depth,
            bool mirrored,float scale,bool skip_underlay=false,
            bool cached_terrain=false) {
        SandboxPassWorkload::Scope pass(work,mirrored?SandboxPassWorkload::reflection_scene:work.pass);
        auto draw=[&](GeometryLayer layer) {
            if(draw_layer(records,layer,settings,rect,target,depth,mirrored,scale))return true;
            char detail[128];sprintf_s(detail,"layer=%u mirrored=%u records=%zu",unsigned(layer),unsigned(mirrored),records[layer].size());
            renderer.trace.write("fresh-layer-draw-failed",detail,true);return false;
        };
        if(!mirrored && ((!skip_underlay && !draw(geometry_underlay)) ||
                !draw(geometry_land)))return false;
        if(cached_terrain){
            if(!mirrored)relight_terrain(rect);
            else if(renderer.reflection_coverage_blend){
                // The relit mirror terrain is color only. Mountains then add
                // their coverage (alpha only, at the same depth) so rivers
                // reflect peaks but not the hills and ground around them.
                mirror_mountain_coverage=true;
                bool covered=draw(geometry_natural_mountain);
                mirror_mountain_coverage=false;
                if(!covered)return false;
            }
        }else for(auto layer:{geometry_natural_terrain,geometry_natural_mountain,
                              geometry_natural_decal})
            if(!draw(layer))return false;
        for(unsigned layer=geometry_natural_forest0;layer<geometry_layer_count;++layer)
            if(!draw(static_cast<GeometryLayer>(layer)))return false;
        if(!mirrored) {
            if(!draw(geometry_bed))return false;
            if(!draw(geometry_water))return false;
            // Farm fields are ground: they draw before the route strips, which
            // write no depth, so a road or railroad always paints over them.
            if(!draw(geometry_river) || !draw(geometry_shadow) ||
                    !draw(geometry_farm) || !draw(geometry_route))return false;
        }
        for(unsigned i=0;i<renderer.cliff_bundle.assets.size();++i)
            if(!draw(static_cast<GeometryLayer>(geometry_cliff0+i)))return false;
        if(!draw_features(records,settings,rect,target,depth,mirrored,scale)){
            char detail[128];sprintf_s(detail,"layer=%u mirrored=%u records=%zu",unsigned(geometry_feature),unsigned(mirrored),records[geometry_feature].size());
            renderer.trace.write("fresh-layer-draw-failed",detail,true);return false;
        }
        for(auto layer:{geometry_site,geometry_mine,geometry_farm,geometry_city,geometry_wall})
            if((layer!=geometry_farm || mirrored) && !draw(layer))return false;
        char wave_diagnostic[8]{};
        bool skip_wave=c3x_renderer::render_core::cached_environment("C3X_SANDBOX_SKIP_WAVE",
            wave_diagnostic,sizeof(wave_diagnostic)) &&
            std::strcmp(wave_diagnostic,"1")==0;
        if(!mirrored && !skip_wave && !draw(geometry_wave))return false;
        return true;
    }
    bool ensure_glow(unsigned width,unsigned height) {
        if(glow.linear.color && glow.native_extent==width &&
            glow.native_height==height && glow.linear.width==width*scene_scale &&
            glow.linear.height==height*scene_scale && glow.linear.sample_count==scene_samples && glow.linear.samples &&
            bloom.width==(width+1)/2 && bloom.height==(height+1)/2)return true;
        glow.reset();glow.native_extent=width;glow.native_height=height;
        if(!ensure_linear_target(glow.linear,width*scene_scale,
                height*scene_scale,scene_samples,true) || !bloom.ensure(width,height)){
            glow.reset();return false;
        }
        return true;
    }
    void update_environment(c3x_renderer_frame_v1 const& frame) {
        char cycle[8]{};
        bool day_night=c3x_renderer::render_core::cached_environment("C3X_SANDBOX_DAY_NIGHT",cycle,
            sizeof(cycle)) && std::strcmp(cycle,"1")==0;
        float hour=day_night?12.f+24.f*float(frame.presentation_time_ticks)/
            float(std::max<c3x_renderer_i64>(1,frame.presentation_frequency))/30.f:
            float(frame.hour);
        visual_hour=hour;
        if(previous_hour==hour && previous_season==frame.season)return;
        static_rasters.invalidate_all(c3x_renderer::render_core::raster_environment);
        ++lighting_revision;reflection_valid=false;reflected_terrain_material_valid=false;
        previous_hour=hour;previous_season=frame.season;
        auto environment=c3x_renderer::evaluate_environment(hour,frame.season);
        visual_sun_intensity=environment.sun_intensity;
        auto light=c3x_renderer::lighting::key_light(environment);
        TerrainShaderSettings settings{};
        settings.height_texel[0]=settings.height_texel[1]=1.f/2048.f;
        settings.normal_strength=4.f;settings.exposure=1.f;
        std::copy(light.direction.begin(),light.direction.end(),settings.light_direction);
        settings.sun_intensity=environment.sun_intensity;
        std::copy(std::begin(environment.sun_color),std::end(environment.sun_color),settings.sun_color);
        settings.shadow_strength=environment.shadow_strength;
        std::copy(light.direction.begin(),light.direction.end(),settings.moon_direction);
        settings.moon_intensity=environment.moon_intensity;
        std::copy(std::begin(environment.moon_color),std::end(environment.moon_color),settings.moon_color);
        settings.night_activation=environment.night_activation;
        std::copy(std::begin(environment.ambient_color),std::end(environment.ambient_color),settings.ambient_color);
        settings.environment_exposure=environment.exposure;
        settings.water_fresnel=environment.water_fresnel;
        settings.water_specular=environment.water_specular;
        settings.emissive_scale=environment.emissive_scale;
        settings.hour=hour;
        renderer.context->UpdateSubresource(renderer.terrain_settings_buffer,0,nullptr,&settings,0,0);work.upload_buffer(renderer.terrain_settings_buffer);
        renderer.display_exposure=environment.exposure;
        renderer.shadow_basis=c3x_renderer::fidelity::light_frame(environment);
        float shadow_values[20]={};
        std::copy(renderer.shadow_basis.begin(),renderer.shadow_basis.end(),shadow_values);
        shadow_values[16]=renderer.fidelity_profile && renderer.fidelity_shadow_control?0.f:1.f;
        shadow_values[17]=1.f;
        renderer.context->UpdateSubresource(renderer.shadow_settings_buffer,0,nullptr,shadow_values,0,0);work.upload_buffer(renderer.shadow_settings_buffer);
        if(renderer.fidelity_profile)
            renderer.natural.update(renderer.context,environment,renderer.shadow_basis.data()+8);
        renderer.cities.night=environment.night_activation;
        renderer.cities.emissive_scale=environment.emissive_scale;
        glow.gain=renderer.city_glow.gain;
    }
    bool update_city_lights() {
        // Lights come from the stable region of interest (the 1x region plus
        // a light-reach margin), so every retained static pixel already saw
        // every light that can reach it. A camera move that changes the set
        // only adds/removes distant lights; it no longer invalidates rasters.
        std::vector<c3x_renderer::city_fidelity::Lighting const*> selected;
        auto add=[&](GeometryDrawView::Records const& records){
            for(auto const& record:records[geometry_city])
                if(record.content().city_lighting){
                    auto* light=record.content().city_lighting.get();
                    if(std::find(selected.begin(),selected.end(),light)==selected.end())
                        selected.push_back(light);
                }
        };
        add(roi_records);add(all_visible);
        if(sandbox_perf_options().legacy && selected!=selected_lighting){
            static_rasters.invalidate_all(c3x_renderer::render_core::raster_lights);
            ++lighting_revision;reflection_valid=false;reflected_terrain_material_valid=false;
        }
        // The CPU light field is rebuilt on every upload; skip it while the
        // selection, night factor and owner of the GPU field are unchanged.
        if(!sandbox_perf_options().legacy && lights_uploaded && selected==selected_lighting &&
                renderer.cities.light_uploads==lights_upload_serial &&
                renderer.cities.night==lights_night && renderer.cities.emissive_scale==lights_emission)return true;
        selected_lighting=selected;
        lights_uploaded=renderer.cities.lights(renderer.context,selected);
        lights_upload_serial=renderer.cities.light_uploads;
        lights_night=renderer.cities.night;lights_emission=renderer.cities.emissive_scale;
        return lights_uploaded;
    }
    bool lights_uploaded=false;std::uint64_t lights_upload_serial=0;float lights_night=-1,lights_emission=-1;
    bool ensure_linear_target(c3x_renderer::render_core::LinearTarget& target,
            unsigned width,unsigned height,unsigned samples,bool resolved) {
        char reference[8]={};
        bool copy_reference=c3x_renderer::render_core::cached_environment("C3X_SANDBOX_RESOLVE_COPY_REFERENCE",reference,sizeof(reference)) && reference[0]=='1';
        if(resolved && samples==1 && !copy_reference){
            // A single-sample color target is already the exact resolved image.
            // Keep both owned references for LinearTarget's ordinary reset/swap
            // contract, and sample only after the target has been unbound.
            if(!target.ensure(renderer.device,width,height,true,false,1))return false;
            if(target.resolved!=target.color){
                target.release(target.view);target.release(target.resolved);
                target.resolved=target.color;target.resolved->AddRef();
                target.view=target.samples;target.view->AddRef();
            }
            return true;
        }
        return target.ensure(renderer.device,width,height,true,resolved,samples);
    }
    unsigned configured_samples() const {
        unsigned samples=1;char option[8]={};
        if(c3x_renderer::render_core::cached_environment("C3X_SANDBOX_MSAA_2X",option,sizeof(option)) && option[0]=='1')samples=2;
        if(c3x_renderer::render_core::cached_environment("C3X_RENDERER_SCENE_SAMPLES",option,sizeof(option)))
            samples=option[0]=='4'?4u:option[0]=='2'?2u:1u;
        return samples;
    }
    bool prepare_assets() {
        return visual.install() && shadow.ensure() && bloom.ensure_shaders() &&
            static_restore.ensure(renderer.device,configured_samples());
    }
    bool ensure_targets(unsigned width,unsigned height) {
        char value[8]{};
        scene_scale=1;scene_samples=configured_samples();
        reflection_scale=c3x_renderer::render_core::cached_environment("C3X_SANDBOX_REFLECTION_FULL",value,sizeof(value)) &&
            std::strcmp(value,"1")==0?1.f:.375f;
        unsigned reflection_width=unsigned((width+8)*reflection_scale);
        unsigned reflection_height=unsigned((height+8)*reflection_scale);
        unsigned region_width=width+2*region_margin_x;
        unsigned region_height=height+2*region_margin_y;
        auto previous_layout=static_rasters.layout;
        static_rasters.set_layout(width,height,scene_samples);
        region_width_px=region_width*scene_scale;region_height_px=region_height*scene_scale;
        if(previous_layout!=static_rasters.layout){
            layout_reset=true;
            for(auto& retained:overlay_slots){retained.layer.reset();retained.revision=~0ull;}
            for(auto& image:bootstrap){image.region.reset();image.valid=false;image.covered={};}
            restore_key={};
        }
        if(static_cache.width!=width || static_cache.height!=height || static_cache.sample_count!=scene_samples)
            restore_key={};
        if(reflection.width!=reflection_width || reflection.height!=reflection_height){
            reflection_valid=false;
            reflected_terrain_material_valid=false;
        }
        if(!ensure_glow(width,height))return false;
        if(!aquatic_bounds_buffer){
            D3D11_BUFFER_DESC bounds={};
            bounds.ByteWidth=sizeof(float)*8;
            bounds.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(FAILED(renderer.device->CreateBuffer(&bounds,nullptr,&aquatic_bounds_buffer)))
                return false;
        }
        // Region slots are allocated on first use (see reset_slot/recenter).
        if(!ensure_linear_target(static_cache,width*scene_scale,
                height*scene_scale,scene_samples,true) ||
           !static_restore.ensure(renderer.device,scene_samples))
            return false;
        if(scene_samples==1 &&
           (!reflected_terrain_albedo.ensure(renderer.device,reflection_width,
                reflection_height,true,false,1) ||
            !reflected_terrain_normal.ensure(renderer.device,reflection_width,
                reflection_height,DXGI_FORMAT_R8G8B8A8_UNORM) ||
            !reflected_terrain_world.ensure(renderer.device,reflection_width,
                reflection_height,DXGI_FORMAT_R16G16B16A16_FLOAT) ||
            !reflected_terrain_properties.ensure(renderer.device,reflection_width,
                reflection_height,DXGI_FORMAT_R16G16B16A16_FLOAT)))
            return false;
        if(scene_samples==1 && !terrain_material_blend) {
            D3D11_BLEND_DESC blend={};blend.IndependentBlendEnable=TRUE;
            for(auto& target:blend.RenderTarget) {
                target.BlendEnable=TRUE;
                target.SrcBlend=D3D11_BLEND_ONE;
                target.DestBlend=D3D11_BLEND_INV_SRC_ALPHA;
                target.BlendOp=D3D11_BLEND_OP_ADD;
                target.SrcBlendAlpha=D3D11_BLEND_ONE;
                target.DestBlendAlpha=D3D11_BLEND_INV_SRC_ALPHA;
                target.BlendOpAlpha=D3D11_BLEND_OP_ADD;
                target.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;
            }
            if(FAILED(renderer.device->CreateBlendState(&blend,&terrain_material_blend)))return false;
        }
        if(!aquatic_depth) {
            D3D11_DEPTH_STENCIL_DESC depth={};
            depth.DepthEnable=FALSE;
            depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;
            if(FAILED(renderer.device->CreateDepthStencilState(&depth,&aquatic_depth)))return false;
        }
        return reflection.ensure(renderer.device,reflection_width,reflection_height) &&
            reflection_static.ensure(renderer.device,reflection_width,reflection_height);
    }
    c3x_renderer::render_core::RasterScratchIdentity reflection_writer(
            ViewportShaderSettings const& settings,D3D11_RECT rect)const{
        c3x_renderer::render_core::RasterScratchIdentity key;
        key.scene={reflection_revision,raster_scope(),
            std::uint64_t(renderer.scene_depth_origin),lighting_revision,shadow.sampling_identity,
            std::uint64_t(camera_x),std::uint64_t(camera_y),reflection.width,
            reflection.height,scene_samples};
        key.view={projection_zoom,reflection_scale,settings.translation[0],
            settings.translation[1],settings.depth_translation,renderer.reflection.height_pixels,
            settings.inverse_size[0],settings.inverse_size[1],visual_hour,float(previous_season)};
        key.area={rect.left,rect.top,rect.right,rect.bottom};return key;
    }
    bool prepare_reflected_terrain_material(ViewportShaderSettings const& settings,
            D3D11_RECT rect) {
        SandboxPassWorkload::Scope pass(work,SandboxPassWorkload::reflection_material);
        auto key=reflection_writer(settings,rect);
        if(reflected_terrain_material_valid && reflected_terrain_key==key)return true;
        reflected_terrain_material_valid=false;
        auto* context=renderer.context;
        float clear[4]={};
        context->ClearRenderTargetView(reflected_terrain_albedo.target,clear);work.clear(reflected_terrain_albedo.target);
        context->ClearRenderTargetView(reflected_terrain_normal.target,clear);work.clear(reflected_terrain_normal.target);
        context->ClearRenderTargetView(reflected_terrain_world.target,clear);work.clear(reflected_terrain_world.target);
        context->ClearRenderTargetView(reflected_terrain_properties.target,clear);work.clear(reflected_terrain_properties.target);
        context->ClearDepthStencilView(reflected_terrain_albedo.depth,
            D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);work.clear(reflected_terrain_albedo.depth);
        for(auto layer:{geometry_natural_terrain,geometry_natural_mountain,
                        geometry_natural_decal})
            if(!draw_layer(reflection_visible,layer,settings,rect,
                    reflected_terrain_albedo.target,reflected_terrain_albedo.depth,
                    true,reflection_scale,true))return false;
        context->OMSetRenderTargets(0,nullptr,nullptr);
        reflected_terrain_key=key;reflected_terrain_material_valid=true;
        return true;
    }
    void relight_reflected_terrain() {
        SandboxPassWorkload::Scope pass(work,SandboxPassWorkload::relight);
        auto* context=renderer.context;
        context->OMSetRenderTargets(1,&reflection_static.target,reflection_static.depth);
        // Mirrored terrain adds color only. The mirror's alpha then counts the
        // standing objects drawn next, which are all a river reflects; the sea
        // also treats this terrain color as coverage.
        context->OMSetBlendState(renderer.reflection_terrain_blend?renderer.reflection_terrain_blend:
            renderer.blend_state,nullptr,0xffffffffu);
        context->OMSetDepthStencilState(renderer.depth_state,0);
        context->RSSetState(renderer.rasterizer_state);
        D3D11_VIEWPORT viewport={0,0,float(reflection_static.width),
            float(reflection_static.height),0,1};
        context->RSSetViewports(1,&viewport);
        D3D11_RECT rect={0,0,LONG(reflection_static.width),LONG(reflection_static.height)};
        context->RSSetScissorRects(1,&rect);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(static_restore.vertex,nullptr,0);
        context->PSSetShader(visual.terrain_relight,nullptr,0);
        if(renderer.city_profile)renderer.cities.scene_lights.bind(context);
        context->PSSetConstantBuffers(0,1,&renderer.natural.frames[0]);
        context->PSSetConstantBuffers(2,1,&renderer.shadow_settings_buffer);
        context->PSSetConstantBuffers(4,1,&renderer.source_shadow.table);
        context->PSSetShaderResources(17,1,&renderer.source_shadow.sampled_view());
        ID3D11ShaderResourceView* inputs[]={reflected_terrain_albedo.samples,
            reflected_terrain_normal.view,reflected_terrain_world.view,
            reflected_terrain_properties.view,reflected_terrain_albedo.depth_samples};
        context->PSSetShaderResources(116,5,inputs);
        context->Draw(3,0);work.draw(3);
        ID3D11ShaderResourceView* empty[5]={};
        context->PSSetShaderResources(116,5,empty);
        context->OMSetRenderTargets(0,nullptr,nullptr);
    }
    void relight_terrain(D3D11_RECT rect) {
        SandboxPassWorkload::Scope pass(work,SandboxPassWorkload::relight);
        auto* context=renderer.context;
        context->OMSetRenderTargets(1,&static_region().target,static_region().depth);
        context->OMSetBlendState(renderer.blend_state,nullptr,0xffffffffu);
        context->OMSetDepthStencilState(renderer.depth_state,0);
        context->RSSetState(renderer.rasterizer_state);
        D3D11_VIEWPORT viewport={0,0,float(static_region().width),float(static_region().height),0,1};
        context->RSSetViewports(1,&viewport);
        context->RSSetScissorRects(1,&rect);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(static_restore.vertex,nullptr,0);
        context->PSSetShader(visual.terrain_relight,nullptr,0);
        if(renderer.city_profile)renderer.cities.scene_lights.bind(context);
        context->PSSetConstantBuffers(0,1,&renderer.natural.frames[0]);
        context->PSSetConstantBuffers(2,1,&renderer.shadow_settings_buffer);
        context->PSSetConstantBuffers(4,1,&renderer.source_shadow.table);
        context->PSSetShaderResources(17,1,&renderer.source_shadow.sampled_view());
        ID3D11ShaderResourceView* inputs[]={terrain_albedo.samples,
            terrain_normal.view,terrain_world.view,terrain_properties.view,
            terrain_albedo.depth_samples};
        context->PSSetShaderResources(116,5,inputs);
        context->Draw(3,0);work.draw(3);
        ID3D11ShaderResourceView* empty[5]={};
        context->PSSetShaderResources(116,5,empty);
        context->OMSetRenderTargets(0,nullptr,nullptr);
    }
    void relight_underlay(D3D11_RECT rect) {
        auto* context=renderer.context;
        context->OMSetRenderTargets(1,&static_region().target,nullptr);
        context->OMSetBlendState(nullptr,nullptr,0xffffffffu);
        context->OMSetDepthStencilState(nullptr,0);
        context->RSSetState(renderer.rasterizer_state);
        D3D11_VIEWPORT viewport={0,0,float(static_region().width),float(static_region().height),0,1};
        context->RSSetViewports(1,&viewport);
        context->RSSetScissorRects(1,&rect);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(static_restore.vertex,nullptr,0);
        context->PSSetShader(visual.material_relight,nullptr,0);
        if(renderer.city_profile)renderer.cities.scene_lights.bind(context);
        context->PSSetConstantBuffers(0,1,&renderer.terrain_settings_buffer);
        context->PSSetConstantBuffers(2,1,&renderer.shadow_settings_buffer);
        context->PSSetConstantBuffers(3,1,&renderer.world_settings_buffer);
        context->PSSetConstantBuffers(4,1,&renderer.source_shadow.table);
        ID3D11ShaderResourceView* inputs[]={material_albedo.samples,
            material_normal.view,material_world.view};
        context->PSSetShaderResources(116,3,inputs);
        context->PSSetShaderResources(25,1,&renderer.source_shadow.sampled_view());
        context->Draw(3,0);work.draw(3);
        ID3D11ShaderResourceView* empty[3]={};
        context->PSSetShaderResources(116,3,empty);
        context->OMSetRenderTargets(0,nullptr,nullptr);
        context->CopyResource(static_region().depth_texture,material_albedo.depth_texture);work.copy(static_region().depth_texture,true);
    }
    // =====================================================================
    // Retained static layer. Static (camera-independent) terrain, relief,
    // vegetation and cities are retained per zoom lane in world-anchored
    // regions. A frame never waits for a full-quality redraw while a usable
    // preview exists: whole-pixel translation first, then an affine
    // resample of the previous raster (zoom), then a low-resolution
    // bootstrap (jumps). Full quality is rebuilt in a back slot under a
    // per-frame pixel budget and swapped in once its visible area is done.
    // See docs/performance_overhaul_20261003.md.
    // =====================================================================
    void discard_static(){
        static_rasters.discard_all(c3x_renderer::render_core::raster_error);
        for(auto& image:bootstrap){image.valid=false;image.covered={};}
        restore_key={};
    }
    c3x_renderer::render_core::StaticRasterKey static_key()const{
        std::uint32_t hour=0;std::memcpy(&hour,&visual_hour,sizeof(hour));
        return {raster_scope(),lighting_revision,(std::uint64_t(hour)<<32)|std::uint32_t(previous_season),
            shadow.sampling_identity,std::uint64_t(renderer.water_scene_active),std::uint64_t(scene_samples)};
    }
    float zoom_destination()const{
        float value=c3x_renderer::zoom_destination_hint().load(std::memory_order_relaxed);
        if(!std::isfinite(value))return projection_zoom;
        return std::clamp(value,c3x_renderer::SceneProjection::minimum,c3x_renderer::SceneProjection::maximum);
    }
    // Lane 0 is hidden only while a non-1x zoom is the destination *and* the
    // display lane was drawn recently (a stale hint cannot starve lane 0).
    bool canonical_hidden()const{
        return zoom_destination()!=1.f && draw_serial-lane_drawn[1]<32;
    }
    bool static_refinement_pending(float zoom)const{
        if(!shadow.atlas_complete)return true;
        unsigned lane=StaticRasters::lane_of(zoom);
        if(static_rasters.lane_refining(lane))return true;
        auto const& shown=static_rasters.states[static_rasters.front_slot[lane]];
        // Journaled caster changes repair in the next drawn frame; do not let
        // an idle display wait for an unrelated event to draw it.
        return !shown.valid || shown.stale || shown.projection!=zoom || shown.shadow_serial<shadow.change_serial;
    }
    // Adaptive pixels-per-frame for refinement: shrink when frames run long,
    // grow while they fit in the vsync interval. The budget is never applied
    // to strips needed for the displayed image.
    double refinement_budget(unsigned lane){
        auto const& options=sandbox_perf_options();
        if(options.legacy)return -1.;
        // The canonical 1x raster is not displayed while another zoom is the
        // destination; it refines at a reduced budget (and skips its mirror).
        bool hidden=lane==0 && canonical_hidden();
        // Time-sliced refinement: steer the pixel budget by the static work
        // actually spent last frame. The frame interval it used before mixed in
        // GPU waits and ignored any frame over 100 ms, so a slow refinement
        // (zooming out on a busy map) never shrank its own budget and kept
        // frames at 2-10/s. C3X_RENDERER_STATIC_BUDGET_MS sets the target.
        static double const target=[]{char value[16]={};double ms=8.;
            if(c3x_renderer::render_core::cached_environment("C3X_RENDERER_STATIC_BUDGET_MS",value,sizeof(value))){
                double parsed=std::atof(value);if(std::isfinite(parsed)&&parsed>0)ms=parsed;}
            return ms;}();
        if(refine_worked && last_static_ms>0){
            if(last_static_ms>target)refine_pixels*=std::max(.35,target/last_static_ms);
            else if(last_static_ms<target*.6)refine_pixels*=1.25;
        }
        refine_pixels=std::clamp(refine_pixels,60000.,32000000.);
        refine_worked=false;
        double pixels=options.refine_pixels>0?options.refine_pixels:refine_pixels;
        // A hidden canonical raster still converges, at a quarter budget.
        return hidden?pixels*.25:pixels;
    }
    ViewportShaderSettings slot_settings(StaticState const& slot,ViewportShaderSettings const& screen)const{
        auto result=screen;
        result.translation[0]+=float(region_margin_x-(camera_x-slot.camera_x));
        result.translation[1]+=float(region_margin_y-(camera_y-slot.camera_y));
        result.depth_translation=slot.depth_translation;
        result.inverse_size[0]=1.f/float(region_width_px);result.inverse_size[1]=1.f/float(region_height_px);
        return result;
    }
    // Borders that animated water can cover (near water) stay in the per-frame
    // pass above it; all others are static and retained with the terrain.
    bool border_static(GeometryDrawRecord const& record)const{
        if(sandbox_perf_options().legacy)return false;
        if(!renderer.water_scene_active)return true;
        return !c3x_renderer::render_core::water_under_projection(renderer.world_coast.world(),
            GeometryDrawReference(record).content().world_bounds);
    }
    bool draw_static_borders(c3x_renderer::render_core::LinearTarget& target,ViewportShaderSettings const& view,
            D3D11_RECT rect,GeometryDrawView::Records const& records,float scale){
        auto clip=source_bounds(view,rect,false);
        D3D11_RECT scissor={LONG(rect.left*scale),LONG(rect.top*scale),LONG(std::ceil(rect.right*scale)),LONG(std::ceil(rect.bottom*scale))};
        for(unsigned layer:{unsigned(geometry_underlay),unsigned(geometry_natural_terrain),
                            unsigned(geometry_natural_mountain),unsigned(geometry_water)})
            if(!territory_borders.draw(renderer.device,renderer.context,records[layer],view,target,
                    renderer.content_view_width,renderer.content_view_height,projection_zoom,scale,
                    [&](auto const& r){return border_static(r) &&
                        renderer.chunk_intersects_region(GeometryDrawReference(r),view,clip,false);},
                    float(region_margin_x),float(region_margin_y),&scissor))return false;
        return true;
    }
    // Draw one rectangle (full-resolution region pixels) of a slot from the
    // resident world. `scale` < 1 rasterizes the same rectangle at low resolution.
    // Retained near-water overlays. Water must animate under these records,
    // so they used to be redrawn over it every frame (thousands of draws and
    // ~1M triangles at 1x on a busy map). They are static; draw them once per
    // static strip into a cleared premultiplied layer tested against the
    // strip's static and water depth, and composite that layer over the live
    // water. C3X_RENDERER_OVERLAY_CACHE=0 restores per-frame submission.
    bool overlay_enabled()const{
        char control[8]={};
        if(c3x_renderer::render_core::cached_environment("C3X_RENDERER_OVERLAY_CACHE",control,sizeof(control)) && control[0]=='0')
            return false;
        return renderer.water_scene_active && scene_samples==1 && scene_scale==1 && !sandbox_perf_options().legacy;
    }
    static bool overlay_layer(unsigned layer){
        return layer==geometry_shadow || layer==geometry_route || (layer>=geometry_feature && layer<=geometry_cliff7);
    }
    bool overlay_target(unsigned index){
        if(overlay_device!=renderer.device_generation){
            for(auto& slot:overlay_slots){slot.layer.reset();slot.revision=~0ull;}
            overlay_composite.reset();
            if(overlay_depth_only){overlay_depth_only->Release();overlay_depth_only=nullptr;}
            overlay_device=renderer.device_generation;
        }
        if(!overlay_depth_only){
            D3D11_BLEND_DESC desc={};desc.RenderTarget[0].RenderTargetWriteMask=0;
            if(FAILED(renderer.device->CreateBlendState(&desc,&overlay_depth_only)))return false;
        }
        return overlay_composite.ensure(renderer.device) &&
            ensure_linear_target(overlay_slots[index].layer,region_width_px,region_height_px,1,false);
    }
    // Draws one static strip's water-dependent records into the slot's layer:
    // static depth and a transparent color first, then water depth without
    // color (submerged parts stay hidden exactly as before), then the
    // overlays in their ordinary order. The river writes no depth, so it has
    // no part here. Animated content refuses the layer.
    bool write_overlay_strip(unsigned index,StaticState const& slot,ViewportShaderSettings const& view,D3D11_RECT rect,
            GeometryDrawView::Records& records,float scale){
        for(unsigned layer=0;layer<geometry_layer_count;++layer){
            if(records[layer].empty())continue;
            if(layer!=geometry_water && layer!=geometry_river && !overlay_layer(layer)){++overlay_refusals;return false;}
            for(auto const& record:records[layer]){auto const& m=GeometryDrawReference(record).content();
                if(m.animation_texture || m.resource_instance){++overlay_refusals;return false;}}
        }
        if(!overlay_target(index))return false;
        auto& target=overlay_slots[index].layer;
        if(!static_restore.draw(renderer.context,target,nullptr,slot.region.depth_samples,0,0,{},nullptr,
                slot.region.width,slot.region.height,false,false,0,&rect,1))return false;
        work.draw(3);
        TargetScope scope_target(*this,{target.target,target.width,target.height,
            float(region_margin_x),float(region_margin_y)});
        GeometryDrawView::Records water{};
        for(auto layer:{geometry_water,geometry_river}){water[layer].swap(records[layer]);}
        auto* blend=renderer.blend_state;renderer.blend_state=overlay_depth_only;
        bool drawn=draw_layer(water,geometry_water,view,rect,target.target,target.depth,false,scale);
        renderer.blend_state=blend;
        if(!drawn || !draw_scene(records,view,rect,target.target,target.depth,false,scale))return false;
        renderer.context->OMSetRenderTargets(0,nullptr,nullptr);
        ++overlay_strips;return true;
    }
    bool write_slot(unsigned index,StaticState& slot,ViewportShaderSettings const& screen,
            StaticRect strip,float scale,bool track_dependencies){
        if(strip.empty())return true;
        ZoomScope scope_zoom(*this,slot.projection);
        TargetScope scope_target(*this,{slot.region.target,slot.region.width,slot.region.height,
            float(region_margin_x),float(region_margin_y)});
        auto slot_view=slot_settings(slot,screen);
        D3D11_RECT rect={strip.left,strip.top,strip.right,strip.bottom};
        GeometryDrawView::Records chosen{},overlays{};
        // Bootstrap images reuse this writer with their own state and index 0;
        // only actual static slots own a retained overlay layer.
        bool const static_slot=index<overlay_slots.size() && &slot==&static_rasters.states[index];
        bool const retain_overlays=static_slot && overlay_enabled();
        auto clip=source_bounds(slot_view,rect,false);
        contributors(slot_view,clip,false,[&](unsigned layer,auto const& record){
            if(renderer.water_scene_active && record.water_dependent){
                if(retain_overlays && renderer.chunk_intersects_region(GeometryDrawReference(record),slot_view,clip,false))
                    overlays[layer].push_back(record);
                return;
            }
            if(renderer.chunk_intersects_region(GeometryDrawReference(record),slot_view,clip,false))
                chosen[layer].push_back(record);
        });
        auto before=slot.revision;++slot.revision;
        if(!draw_scene(chosen,slot_view,rect,slot.region.target,slot.region.depth,false,scale))return false;
        renderer.context->OMSetRenderTargets(0,nullptr,nullptr);
        if(!draw_static_borders(slot.region,slot_view,rect,chosen,scale))return false;
        if(static_slot){
            auto& retained=overlay_slots[index];
            bool matched=retain_overlays && retained.revision==before;
            bool kept=matched && write_overlay_strip(index,slot,slot_view,rect,overlays,scale);
            if(retain_overlays && !kept && retained.revision!=~0ull)++(matched?overlay_retired_failure:overlay_retired_mismatch);
            retained.revision=kept?slot.revision:~0ull;
        }
        ++slot.metrics.strip_fills;++slot.metrics.reasons[c3x_renderer::render_core::raster_strip_fills];
        if(static_slot && shadow_field.left!=INT_MIN/4){
            auto const& f=shadow_field;
            slot.unshadowed=slot.unshadowed.joined(strip.outside(source_region_rect(f.left,f.top,f.right,f.bottom,slot_view,0)));
        }
        if(track_dependencies && !raster_dependencies(raster_inputs[index],slot_view,rect,true))
            raster_inputs[index].complete=false;
        return true;
    }
    // Grow a slot's rectangular coverage toward `needed`. A negative budget
    // is unbounded; otherwise strips stop once the pixel budget is spent.
    bool extend_coverage(unsigned index,StaticState& slot,ViewportShaderSettings const& screen,
            StaticRect needed,double& budget,int ahead,float scale,bool track_dependencies){
        StaticRect limits={0,0,int(region_width_px),int(region_height_px)};
        needed={std::max(needed.left,limits.left),std::max(needed.top,limits.top),
            std::min(needed.right,limits.right),std::min(needed.bottom,limits.bottom)};
        if(needed.empty())return true;
        auto& area=slot.covered;
        auto spend=[&](StaticRect strip){
            if(!write_slot(index,slot,screen,strip,scale,track_dependencies))return false;
            if(budget>=0)budget=std::max(0.,budget-double(strip.area()));
            return true;
        };
        auto lines=[&](int extent){
            if(budget<0)return INT_MAX/4;
            return std::max(32,int(budget/double(std::max(1,extent))));
        };
        if(area.empty()){
            int rows=std::min(needed.bottom-needed.top,lines(needed.right-needed.left));
            StaticRect band={needed.left,needed.top,needed.right,needed.top+rows};
            if(!spend(band))return false;
            area=band;
        }
        for(int step=0;step<256 && budget!=0;++step){
            if(needed.bottom>area.bottom){
                int limit=std::min(limits.bottom,needed.bottom+ahead);
                StaticRect strip={area.left,area.bottom,area.right,std::min(limit,area.bottom+lines(area.right-area.left))};
                if(!spend(strip))return false;
                area.bottom=strip.bottom;continue;
            }
            if(needed.top<area.top){
                int limit=std::max(limits.top,needed.top-ahead);
                StaticRect strip={area.left,std::max(limit,area.top-lines(area.right-area.left)),area.right,area.top};
                if(!spend(strip))return false;
                area.top=strip.top;continue;
            }
            if(needed.left<area.left){
                int limit=std::max(limits.left,needed.left-ahead);
                StaticRect strip={std::max(limit,area.left-lines(area.bottom-area.top)),area.top,area.left,area.bottom};
                if(!spend(strip))return false;
                area.left=strip.left;continue;
            }
            if(needed.right>area.right){
                int limit=std::min(limits.right,needed.right+ahead);
                StaticRect strip={area.right,area.top,std::min(limit,area.right+lines(area.bottom-area.top)),area.bottom};
                if(!spend(strip))return false;
                area.right=strip.right;continue;
            }
            break;
        }
        return true;
    }
    // Dirty-rectangle repair: when a local world change (fog reveal, road,
    // city growth, ownership) invalidates the displayed raster, redraw only
    // the changed contributors' screen area instead of the whole region.
    // Returns false when the change is too broad; the caller then refines.
    std::uint64_t partial_repairs=0,partial_repair_pixels=0;
    // A source-pixel rectangle (occurrence translation applied) in a slot.
    StaticRect source_region_rect(std::int64_t left_px,std::int64_t top_px,std::int64_t right_px,std::int64_t bottom_px,
            ViewportShaderSettings const& view,int margin)const{
        float zoom=projection_zoom;
        float cx=float(renderer.content_view_width/2)+4+float(region_margin_x);
        float cy=float(renderer.content_view_height/2)+4+float(region_margin_y);
        float dx=float(std::int64_t(view.translation[0])),dy=float(std::int64_t(view.translation[1]));
        float left=float(left_px)+dx-float(margin),top=float(top_px)+dy-float(margin);
        float right=float(right_px)+dx+float(margin),bottom=float(bottom_px)+dy+float(margin);
        return {int(std::floor(cx+(left-cx)*zoom))-2,int(std::floor(cy+(top-cy)*zoom))-2,
                int(std::ceil(cx+(right-cx)*zoom))+2,int(std::ceil(cy+(bottom-cy)*zoom))+2};
    }
    StaticRect key_region_rect(RasterInputs::Key const& key,ViewportShaderSettings const& view,int margin)const{
        auto x=std::int64_t(key[3]),y=std::int64_t(key[4]);
        return source_region_rect(std::int64_t(key[9])+x,std::int64_t(key[10])+y,std::int64_t(key[11])+x,std::int64_t(key[12])+y,view,margin);
    }
    // Shadow footprints journaled since a slot was drawn (repair_front input).
    std::vector<std::array<int,4>> shadow_dirty;
    bool repair_front(unsigned index,StaticState& slot,ViewportShaderSettings const& screen,StaticRect ring=StaticRect()){
#ifdef C3X_RENDERER64_FRESH
        if(sandbox_perf_options().legacy || slot.covered.empty())return false;
        auto& inputs=raster_inputs[index];
        if(!inputs.complete || inputs.draws.empty()){static_decision.repair=1;return false;}
        ZoomScope scope_zoom(*this,slot.projection);
        auto view=slot_settings(slot,screen);
        auto const& c=slot.covered;
        D3D11_RECT covered={c.left,c.top,c.right,c.bottom};
        auto clip=source_bounds(view,covered,false);
        // Shadows and city light reach neighbors of a changed record.
        constexpr int reach=96;
        std::vector<StaticRect> dirty;
        std::unordered_set<RasterInputs::Key,RasterInputs::Hash> current;
        std::vector<RasterInputs::Key> stale;
        inputs.begin_membership();
        bool overflow=false;
        auto mark=[&](RasterInputs::Key const& key){
            if(inputs.contains(key))stale.push_back(key);
            if(overflow)return;
            auto r=key_region_rect(key,view,reach);
            r={std::max(r.left,c.left),std::max(r.top,c.top),std::min(r.right,c.right),std::min(r.bottom,c.bottom)};
            if(r.empty())return;
            dirty.push_back(r);if(dirty.size()>4096)overflow=true;
        };
        bool const retain_overlays=overlay_enabled();
        contributors(view,clip,false,[&](unsigned layer,auto const& record){
            if(overflow)return;
            if(renderer.water_scene_active && record.water_dependent && !retain_overlays)return;
            if(!renderer.chunk_intersects_region(GeometryDrawReference(record),view,clip,false))return;
            auto key=contributor_key(layer,record);current.insert(key);
            inputs.visit_membership(key);
            bool changed=!inputs.contains(key);
            if(!changed){
                auto tile=renderer.topology_cache.key(record.tile_x,record.tile_y);
                auto observed=renderer.topology_cache.retained(tile);
                auto seen=inputs.visibility.find(tile);
                changed=seen==inputs.visibility.end() || seen->second!=(observed?observed->visibility_revision:0);
            }
            if(!changed){
                auto proof=inputs.proofs.find(key[0]);
                changed=proof==inputs.proofs.end() || !proof->second || !renderer.raster_content_valid(*proof->second);
            }
            if(changed)mark(key);
        });
        for(auto const& draw:inputs.draws)if(!current.count(draw.first) && tile_resident(draw.first))mark(draw.first);
        // Lease order shifts as tiles move between the view and its halo;
        // order matters only where draws overlap, inside each draw's region.
        std::vector<std::uint32_t> reordered;
        if(!overflow && !inputs.remaining_order_preserved(&reordered)){static_decision.repair=3;return false;}
        if(!reordered.empty()){
            std::sort(reordered.begin(),reordered.end());
            for(auto const& draw:inputs.draws)
                if(std::binary_search(reordered.begin(),reordered.end(),draw.second.index))mark(draw.first);
        }
        // A caster's shadow lands beyond its own footprint; 192 source pixels
        // (1.5 tiles) covers the tallest casters under the key light.
        for(auto const& source:shadow_dirty){if(overflow)break;
            auto r=source_region_rect(source[0],source[1],source[2],source[3],view,192);
            r={std::max(r.left,c.left),std::max(r.top,c.top),std::min(r.right,c.right),std::min(r.bottom,c.bottom)};
            if(!r.empty()){dirty.push_back(r);if(dirty.size()>4096)overflow=true;}
        }
        ring=ring.clipped(c);
        if(!ring.empty())dirty.push_back(ring);
        static_decision.repair_changed=unsigned(dirty.size());static_decision.repair_reordered=unsigned(reordered.size());
        if(overflow){static_decision.repair=2;return false;}
        // A changed dependency without a changed contributor cannot be located.
        if(dirty.empty()){static_decision.repair=4;return false;}
        // Merge overlapping rectangles; fall back to a bounding box when many.
        bool merged=true;
        while(merged && dirty.size()>1){
            merged=false;
            for(std::size_t i=0;i<dirty.size() && !merged;++i)for(std::size_t j=i+1;j<dirty.size();++j){
                auto& a=dirty[i];auto const& b=dirty[j];
                if(a.left<=b.right && b.left<=a.right && a.top<=b.bottom && b.top<=a.bottom){
                    a={std::min(a.left,b.left),std::min(a.top,b.top),std::max(a.right,b.right),std::max(a.bottom,b.bottom)};
                    dirty.erase(dirty.begin()+std::ptrdiff_t(j));merged=true;break;
                }
            }
        }
        if(dirty.size()>24){
            // Camera steps rebuild tiles along a whole edge of the resident
            // region. A single bounding box of those bands covers most of the
            // raster; cells keep the repair proportional to the changed area.
            constexpr int cell=128;
            int columns=(c.right-c.left+cell-1)/cell,rows=(c.bottom-c.top+cell-1)/cell;
            std::vector<unsigned char> marked(std::size_t(std::max(0,columns))*std::size_t(std::max(0,rows)));
            for(auto const& r:dirty)
                for(int y=(r.top-c.top)/cell;y<=(r.bottom-1-c.top)/cell && y<rows;++y)
                    for(int x=(r.left-c.left)/cell;x<=(r.right-1-c.left)/cell && x<columns;++x)marked[std::size_t(y)*columns+x]=1;
            dirty.clear();
            for(int y=0;y<rows;++y)for(int x=0;x<columns;){
                if(!marked[std::size_t(y)*columns+x]){++x;continue;}
                int end=x;while(end<columns && marked[std::size_t(y)*columns+end])++end;
                StaticRect run={c.left+x*cell,c.top+y*cell,std::min(c.right,c.left+end*cell),std::min(c.bottom,c.top+(y+1)*cell)};
                // Extend the previous row's identical run downward.
                bool joined=false;
                for(auto& r:dirty)if(r.left==run.left && r.right==run.right && r.bottom==run.top){r.bottom=run.bottom;joined=true;break;}
                if(!joined)dirty.push_back(run);
                x=end;
            }
        }
        long long area=0;for(auto const& r:dirty)area+=r.area();
        static_decision.repair_rects=unsigned(dirty.size());static_decision.repair_percent=unsigned(area*100/std::max(1ll,c.area()));
        if(area*100>c.area()*45){static_decision.repair=5;return false;}
        static_decision.repair=6;
        // Repair in the existing back target. A draw can reject a not-yet-ready
        // mesh/instance range after earlier layers have succeeded; clearing the
        // displayed target first would then expose holes as the stale preview.
        unsigned patch_index=index^1u;
        auto& patch=static_rasters.states[patch_index];
        if(!ensure_linear_target(patch.region,region_width_px,region_height_px,scene_samples,false))return false;
        patch.valid=false;patch.stale=true;patch.refining=false;patch.covered={};patch.unshadowed={};++patch.revision;
        raster_inputs[patch_index].clear();
        patch.camera_x=slot.camera_x;patch.camera_y=slot.camera_y;
        patch.projection=slot.projection;patch.depth_translation=slot.depth_translation;
        // The retained overlay layer is repaired in the same rectangles.
        auto& repaired=overlay_slots[index];auto& patch_overlay=overlay_slots[patch_index];
        bool overlays=retain_overlays && repaired.revision==slot.revision && overlay_target(patch_index);
        patch_overlay.revision=overlays?patch.revision:~0ull;
        for(auto const& r:dirty){
            D3D11_RECT rect={r.left,r.top,r.right,r.bottom};
            if(!static_restore.draw(renderer.context,patch.region,nullptr,nullptr,0,0,{},nullptr,
                    patch.region.width,patch.region.height,false,true,0,&rect))return false;
            work.draw(3);
            if(overlays && patch_overlay.revision==patch.revision){
                if(!static_restore.draw(renderer.context,patch_overlay.layer,nullptr,nullptr,0,0,{},nullptr,
                        patch_overlay.layer.width,patch_overlay.layer.height,false,true,0,&rect))return false;
                work.draw(3);
            }
            if(!write_slot(patch_index,patch,screen,r,1.f,false))return false;
        }
        overlays=overlays && patch_overlay.revision==patch.revision;
        // All rectangles completed. Copy color and depth together, touching
        // only the repaired pixels; no new full-screen scratch owner is needed.
        for(auto const& r:dirty){
            D3D11_RECT rect={r.left,r.top,r.right,r.bottom};
            if(!static_restore.draw(renderer.context,slot.region,patch.region.samples,patch.region.depth_samples,
                    0,0,{},nullptr,patch.region.width,patch.region.height,false,false,0,&rect,1))return false;
            work.draw(3);
            if(overlays){
                if(!static_restore.draw(renderer.context,repaired.layer,patch_overlay.layer.samples,patch_overlay.layer.depth_samples,
                        0,0,{},nullptr,patch_overlay.layer.width,patch_overlay.layer.height,false,false,0,&rect,1))return false;
                work.draw(3);
            }
        }
        ++slot.revision;slot.unshadowed=slot.unshadowed.joined(patch.unshadowed);
        repaired.revision=overlays?slot.revision:~0ull;
        renderer.context->OMSetRenderTargets(0,nullptr,nullptr);
        static_decision.repair=0;
        // Re-register only the repaired regions (a full slot re-registration
        // cost 150-300 ms per camera step on busy maps).
        inputs.forget(stale);
        for(auto const& r:dirty)
            if(!raster_dependencies(inputs,view,{r.left,r.top,r.right,r.bottom},true)){inputs.complete=false;break;}
        ++partial_repairs;partial_repair_pixels+=std::uint64_t(area);
        refine_worked=true;
        return true;
#else
        (void)index;(void)slot;(void)screen;return false;
#endif
    }
    bool reset_slot(unsigned index,float projection,ViewportShaderSettings const& screen){
        auto& slot=static_rasters.states[index];
        if(!ensure_linear_target(slot.region,region_width_px,region_height_px,scene_samples,false))return false;
        float clear[4]={};
        renderer.context->ClearRenderTargetView(slot.region.target,clear);work.clear(slot.region.target);
        renderer.context->ClearDepthStencilView(slot.region.depth,D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);work.clear(slot.region.depth);
        slot.camera_x=camera_x;slot.camera_y=camera_y;slot.projection=projection;slot.raster_scale=1;
        slot.depth_translation=screen.depth_translation;slot.depth_origin=renderer.scene_depth_origin;
        slot.translation={screen.translation[0],screen.translation[1]};
        slot.key=static_key();slot.signature=raster_scope();slot.lighting_revision=lighting_revision;
        slot.environment_hour=visual_hour;slot.environment_season=previous_season;slot.shadow_builds=shadow.builds;
        slot.shadow_serial=shadow.change_serial;
        slot.geometry_epoch=renderer.tile_geometry_epoch;
        slot.covered={};slot.unshadowed={};slot.valid=false;slot.stale=false;slot.refining=true;++slot.revision;
        raster_inputs[index].clear();
        auto& retained=overlay_slots[index];retained.revision=~0ull;
        if(overlay_enabled() && overlay_target(index)){
            renderer.context->ClearRenderTargetView(retained.layer.target,clear);work.clear(retained.layer.target);
            renderer.context->ClearDepthStencilView(retained.layer.depth,D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);work.clear(retained.layer.depth);
            retained.revision=slot.revision;
        }
        return true;
    }
    // Scrolling past the guard band keeps the overlapping pixels: copy them
    // into the lane's other slot at the new anchor instead of redrawing all.
    bool recenter(unsigned lane,ViewportShaderSettings const& screen,
            c3x_renderer::render_core::StaticRegionShift& shift,int w,int h){
        auto& source=static_rasters.front(lane);
        unsigned source_index=static_rasters.front_slot[lane];
        unsigned destination_index=static_rasters.back_index(lane);
        auto& destination=static_rasters.states[destination_index];
        if(sandbox_perf_options().legacy || !source.valid || source.covered.empty()){static_decision.recenter=-1;return false;}
        if(destination.refining){static_decision.recenter=-2;return false;}
        // The new slot must stay on the source's raster lattice. At a ladder
        // zoom k/8 a camera move is rarely a whole number of raster pixels
        // (an odd step at 0.5x is half a pixel), which used to refuse every
        // recenter and fall back to previews plus a full progressive redraw
        // while scrolling. Anchor the new slot at the nearest camera step
        // whose raster shift is whole; the display path absorbs the small
        // remainder exactly as for any retained slot.
        int anchor_x=camera_x,anchor_y=camera_y;
        if(std::abs(shift.snap_x)>1e-3 || std::abs(shift.snap_y)>1e-3){
            double eighths=double(source.projection)*8.;long k=std::lround(eighths);
            if(k<=0 || std::abs(eighths-double(k))>1e-4){static_decision.recenter=-3;return false;}
            int lattice=1;while(lattice<8 && (k*lattice)%8)lattice*=2;
            auto align=[lattice](int camera,int origin){
                auto d=std::int64_t(camera)-origin,half=std::int64_t(lattice/2);
                auto steps=(d>=0?d+half:d-half)/lattice;return int(origin+steps*lattice);};
            anchor_x=align(camera_x,source.camera_x);anchor_y=align(camera_y,source.camera_y);
            shift=c3x_renderer::render_core::StaticRegionShift::between(source.projection,anchor_x,anchor_y,
                source.camera_x,source.camera_y,region_margin_x,region_margin_y);
            if(std::abs(shift.snap_x)>1e-3 || std::abs(shift.snap_y)>1e-3)return false;
        }
        // New-slot pixel r samples old-slot pixel r-shift, so the old coverage
        // lands at covered+shift in the new slot.
        StaticRect moved={std::max(source.covered.left+shift.x,0),std::max(source.covered.top+shift.y,0),
            std::min(source.covered.right+shift.x,int(region_width_px)),std::min(source.covered.bottom+shift.y,int(region_height_px))};
        if(moved.area()*4<(long long)(w)*h){static_decision.recenter=-4;return false;}
        if(!ensure_linear_target(destination.region,region_width_px,region_height_px,scene_samples,false))return false;
        if(!static_restore.draw(renderer.context,destination.region,source.region.samples,source.region.depth_samples,
                shift.x,shift.y,{},nullptr,source.region.width,source.region.height,false,false,0,nullptr,1,0.f))return false;
        renderer.context->OMSetRenderTargets(0,nullptr,nullptr);
        work.draw(3);
        destination.camera_x=anchor_x;destination.camera_y=anchor_y;
        destination.projection=source.projection;destination.raster_scale=1;
        destination.depth_translation=source.depth_translation;destination.depth_origin=source.depth_origin;
        destination.translation=source.translation;destination.key=source.key;destination.signature=source.signature;
        destination.lighting_revision=source.lighting_revision;destination.environment_hour=source.environment_hour;
        destination.environment_season=source.environment_season;destination.shadow_builds=source.shadow_builds;
        destination.shadow_serial=source.shadow_serial;
        destination.geometry_epoch=source.geometry_epoch;
        destination.covered=moved;destination.valid=true;destination.stale=source.stale;destination.refining=false;
        auto const& u=source.unshadowed;
        destination.unshadowed=StaticRect{u.left+shift.x,u.top+shift.y,u.right+shift.x,u.bottom+shift.y}.clipped(moved);
        ++destination.revision;
        raster_inputs[destination_index].clear();
        {
            auto& from=overlay_slots[source_index];auto& to=overlay_slots[destination_index];
            bool kept=overlay_enabled() && from.revision==source.revision && overlay_target(destination_index) &&
                static_restore.draw(renderer.context,to.layer,from.layer.samples,from.layer.depth_samples,
                    shift.x,shift.y,{},nullptr,from.layer.width,from.layer.height,false,false,0,nullptr,1,0.f);
            if(kept)work.draw(3);
            to.revision=kept?destination.revision:~0ull;
        }
        if(!destination.stale){
            ZoomScope scope_zoom(*this,destination.projection);
            if(!raster_dependencies(raster_inputs[destination_index],slot_settings(destination,screen),
                    {moved.left,moved.top,moved.right,moved.bottom},true))
                raster_inputs[destination_index].complete=false;
        }
        static_rasters.promote(lane);
        ++recenter_copies;
        shift=c3x_renderer::render_core::StaticRegionShift::between(destination.projection,camera_x,camera_y,
            destination.camera_x,destination.camera_y,region_margin_x,region_margin_y);
        return true;
    }
    // First image after a jump, at the current zoom and camera. Keep native
    // pixel resolution by default: routine scene edits must never soften the
    // entire viewport while shadows/full-quality lighting finish preparing.
    // Covers the view plus a 128 px band, so it survives small camera moves.
    bool render_bootstrap(unsigned lane,ViewportShaderSettings const& screen,int w,int h,StaticState const* ring=nullptr){
        auto& image=bootstrap[lane];
        float scale=sandbox_perf_options().bootstrap_scale;
        unsigned image_width=std::max(1u,unsigned(std::ceil(float(region_width_px)*scale)));
        unsigned image_height=std::max(1u,unsigned(std::ceil(float(region_height_px)*scale)));
        if(!ensure_linear_target(image.region,image_width,image_height,1,false))return false;
        float clear[4]={};
        renderer.context->ClearRenderTargetView(image.region.target,clear);work.clear(image.region.target);
        renderer.context->ClearDepthStencilView(image.region.depth,D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);work.clear(image.region.depth);
        image.camera_x=camera_x;image.camera_y=camera_y;
        // An outward destination covers every intermediate view. Preparing it
        // once avoids repeated full-view draws and a resampled depth plane at
        // the final zoom while its native-resolution refinement is pending.
        image.projection=std::min(projection_zoom,zoom_destination());image.raster_scale=scale;
        image.depth_translation=screen.depth_translation;image.key=static_key();
        image.covered={};image.valid=false;image.stale=false;image.refining=false;++image.revision;
        constexpr int band=128;
        double unbounded=-1.;
        StaticRect needed={std::max(0,region_margin_x-band),std::max(0,region_margin_y-band),
            std::min(int(region_width_px),region_margin_x+w+band),std::min(int(region_height_px),region_margin_y+h+band)};
        bootstrap_ring[lane]=nullptr;
        if(ring && ring->valid && !ring->covered.empty()){
            // Zooming out, the displayed raster still covers the inner view and
            // the preview takes it wherever it covers, so a full-view draw
            // (~270 ms at 0.5x on busy maps) is needed only for the exposed
            // ring. Both rasters are world-anchored at this camera, so the hole
            // holds for every frame of the zoom. Insets keep bilinear and
            // depth taps on drawn texels.
            float zoom=projection_zoom;
            float cx=float(renderer.content_view_width/2)+4,cy=float(renderer.content_view_height/2)+4;
            float rf=ring->projection/zoom,rb=image.projection/zoom;
            float fx=cx+float(region_margin_x)-cx*rf+float(ring->camera_x-camera_x)*ring->projection;
            float fy=cy+float(region_margin_y)-cy*rf+float(ring->camera_y-camera_y)*ring->projection;
            float ox=cx+float(region_margin_x)-cx*rb,oy=cy+float(region_margin_y)-cy*rb;
            auto const& c=ring->covered;constexpr float front_inset=4,ring_inset=3;
            StaticRect hole={int(std::ceil((float(c.left)+front_inset-fx)/rf*rb+ox+ring_inset)),
                int(std::ceil((float(c.top)+front_inset-fy)/rf*rb+oy+ring_inset)),
                int(std::floor((float(c.right)-front_inset-fx)/rf*rb+ox-ring_inset)),
                int(std::floor((float(c.bottom)-front_inset-fy)/rf*rb+oy-ring_inset))};
            hole={std::max(hole.left,needed.left),std::max(hole.top,needed.top),
                std::min(hole.right,needed.right),std::min(hole.bottom,needed.bottom)};
            if(!hole.empty()){image.covered=hole;bootstrap_ring[lane]=ring;bootstrap_ring_revision[lane]=ring->revision;}
        }
        auto timed=std::chrono::steady_clock::now();auto hole_area=image.covered.area();
        if(!extend_coverage(0,image,screen,needed,unbounded,0,scale,false))return false;
        static_decision.boot_area=image.covered.area()-hole_area;
        static_decision.boot_draw_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-timed).count();
        timed=std::chrono::steady_clock::now();
        bootstrap_stamp[lane]=bootstrap_identity();
        static_decision.boot_deps_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-timed).count();
        image.valid=true;++bootstrap_draws;
        return true;
    }
    // Affine map from static_cache pixels to a slot's texels and its coverage
    // of the displayed screen: 2 complete, 1 partial, 0 none.
    int resample_source(StaticState const& slot,int w,int h,ViewportShaderSettings const& screen,
            c3x_renderer::render_core::LinearResample::Source& source)const{
        float zoom=projection_zoom,q=slot.raster_scale,ratio=slot.projection/zoom;
        float bx=float(renderer.content_view_width/2)+4,by=float(renderer.content_view_height/2)+4;
        float ox=bx+float(region_margin_x)-bx*ratio+float(slot.camera_x-camera_x)*slot.projection;
        float oy=by+float(region_margin_y)-by*ratio+float(slot.camera_y-camera_y)*slot.projection;
        float left=ox,top=oy,right=float(w)*ratio+ox,bottom=float(h)*ratio+oy;
        auto const& c=slot.covered;
        source.color=slot.region.samples;source.depth=slot.region.depth_samples;
        source.map[0]=source.map[1]=ratio*q;source.map[2]=ox*q;source.map[3]=oy*q;
        source.covered[0]=float(c.left)*q+1;source.covered[1]=float(c.top)*q+1;
        source.covered[2]=float(c.right)*q-1;source.covered[3]=float(c.bottom)*q-1;
        source.size[0]=float(slot.region.width);source.size[1]=float(slot.region.height);
        source.depth_shift=-(screen.depth_translation-slot.depth_translation)/16384.f;
        if(!slot.valid || c.empty() || !source.color || !source.depth)return 0;
        if(left>=float(c.left)+1 && top>=float(c.top)+1 && right<=float(c.right)-1 && bottom<=float(c.bottom)-1)return 2;
        return left<float(c.right) && right>float(c.left) && top<float(c.bottom) && bottom>float(c.top)?1:0;
    }
    bool static_pixels_current(StaticState& slot,RasterInputs& inputs,ViewportShaderSettings const& screen){
        if(!slot.valid || slot.covered.empty())return false;
        ZoomScope scope_zoom(*this,slot.projection);
        auto const& c=slot.covered;
        if(!raster_dependencies(inputs,slot_settings(slot,screen),{c.left,c.top,c.right,c.bottom},false)){
            slot.valid=false;slot.stale=true;++slot.revision;
            return false;
        }
        return true;
    }
    bool compose_static(ViewportShaderSettings& settings,int w,int h){
        using Shift=c3x_renderer::render_core::StaticRegionShift;
        overlay_frame.valid=false;static_preview=false;preview_overlays=false;static_decision={};
        static_decision.camera_x=camera_x;static_decision.camera_y=camera_y;
        struct StaticClock {double& out;std::chrono::steady_clock::time_point begin=std::chrono::steady_clock::now();
            explicit StaticClock(double& o):out(o){}
            ~StaticClock(){out=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();}
        } static_clock{last_static_ms};
        auto const& options=sandbox_perf_options();
        auto* context=renderer.context;
        float zoom=projection_zoom;
        unsigned lane=StaticRasters::lane_of(zoom);
        static_rasters.select_lane(lane);
        auto key=static_key();
        float hint=zoom_destination();
        bool previewable=!options.legacy && scene_samples==1 && static_resample.ensure(renderer.device);
        bool settled=lane_still[lane]>=3;
        // Lane 1 refines toward the zoom destination while the zoom animates;
        // 0 means "do not refine this lane" (it is animating back to 1x).
        float goal=lane==0?1.f:(!previewable || settled)?zoom:(hint!=1.f?hint:0.f);
        double budget=previewable?refinement_budget(lane):-1.;
        StaticRect limits={0,0,int(region_width_px),int(region_height_px)};
        auto clip_region=[&](StaticRect r){return StaticRect{std::max(r.left,limits.left),std::max(r.top,limits.top),
            std::min(r.right,limits.right),std::min(r.bottom,limits.bottom)};};
        auto visible_rect=[&](StaticState const& slot,Shift& slot_shift){
            slot_shift=Shift::between(slot.projection,camera_x,camera_y,slot.camera_x,slot.camera_y,region_margin_x,region_margin_y);
            return clip_region(slot_shift.needed<StaticRect>(w,h,region_margin_x,region_margin_y));
        };
        unsigned front_index=static_rasters.front_slot[lane];
        {
            auto& displayed=static_rasters.states[front_index];
            static_decision.entry=(displayed.valid?1u:0u)|(displayed.stale?2u:0u)|(displayed.key!=key?4u:0u)|
                (raster_inputs[front_index].complete?32u:0u)|(layout_reset?64u:0u);layout_reset=false;
            static_decision.input_bytes=raster_inputs[front_index].bytes();
            if(displayed.valid && !displayed.stale && displayed.key!=key)
                static_rasters.invalidate(front_index,c3x_renderer::render_core::raster_environment);
            if(displayed.valid && !displayed.stale && displayed.shadow_serial<shadow.change_floor)
                static_rasters.invalidate(front_index,c3x_renderer::render_core::raster_environment);
            if(displayed.valid && !displayed.covered.empty()){
                // Exact content proofs (journal fast path when nothing changed).
                // The borrowed completed view (a camera job is running) is the
                // snapshot this slot was drawn from. Topology the job has already
                // applied does not change it; judging it by that topology
                // repaired the same changed region on every frame after a move.
                bool proven=renderer.borrowed_scene_frame;
                if(!proven){
                    ZoomScope scope_zoom(*this,displayed.projection);
                    proven=raster_dependencies(raster_inputs[front_index],slot_settings(displayed,settings),
                        {displayed.covered.left,displayed.covered.top,displayed.covered.right,displayed.covered.bottom},false);
                }
                shadow_dirty.clear();
                StaticRect field{},ring{};
                if(!displayed.stale){
                    auto const& changes=shadow.shadow_changes;auto serial=displayed.shadow_serial;
                    auto first=std::upper_bound(changes.begin(),changes.end(),serial,
                        [](std::uint64_t value,auto const& change){return value<change.serial;});
                    // Footprints are chunk coordinates. Draw records and raster
                    // keys add the resident basis (and wrapped copies add one
                    // map width), so convert them the same way before repair.
                    int bx=int(resident_basis_x),by=int(resident_basis_y);
                    for(;first!=changes.end();++first)for(int copy=-1;copy<=1;++copy){
                        if(copy&&!wrap_pixels)continue;auto const& r=first->source;int wx=bx+copy*wrap_pixels;
                        shadow_dirty.push_back({r[0]+wx,r[1]+by,r[2]+wx,r[3]+by});}
                    // Pixels drawn while the receiver field was narrower (the
                    // hidden canonical lane during a zoom-in, or a lane refined
                    // toward a closer destination) sampled no shadow page.
                    // Repair the part the current field covers.
                    if(!displayed.unshadowed.empty() && shadow.atlas_complete && !(lane==0 && canonical_hidden())){
                        ZoomScope scope_zoom(*this,displayed.projection);
                        auto const& f=shadow_field;
                        field=f.left==INT_MIN/4?displayed.covered:
                            source_region_rect(f.left,f.top,f.right,f.bottom,slot_settings(displayed,settings),0);
                        ring=displayed.unshadowed.clipped(field).clipped(displayed.covered);
                    }
                }
                if(!displayed.unshadowed.empty())static_decision.entry|=ring.empty()?128u:256u;
                bool shadow_only=proven && shadow_dirty.empty();
                if(shadow_only && ring.empty()){static_decision.entry|=8u;displayed.shadow_serial=shadow.change_serial;}
                else if(shadow_only && ring.area()*100>displayed.covered.area()*45)
                    static_rasters.invalidate(front_index,c3x_renderer::render_core::raster_shadow); // refine behind it
                else if(repair_front(front_index,displayed,settings,ring)){static_decision.entry|=16u;displayed.shadow_serial=shadow.change_serial;
                    // Bounds that still reach into the field cannot shrink; stop
                    // tracking them rather than repairing on every frame.
                    if(!ring.empty()){auto rest=displayed.unshadowed.outside(field);
                        displayed.unshadowed=rest.clipped(field).empty()?rest:StaticRect{};}}
                else if(proven && static_decision.repair==4)displayed.shadow_serial=shadow.change_serial; // footprints outside the slot
                else static_rasters.invalidate(front_index,shadow_only?c3x_renderer::render_core::raster_shadow:
                    c3x_renderer::render_core::raster_scene);
                shadow_dirty.clear();
            }
        }
        Shift shift{};bool shifted=false;
        {
            auto& displayed=static_rasters.states[front_index];
            static_decision.lane=lane;static_decision.slot_x=displayed.camera_x;static_decision.slot_y=displayed.camera_y;
            if(displayed.valid && displayed.projection==zoom){
                visible_rect(displayed,shift);static_decision.reusable=shift.reusable?1:0;
                if(!shift.reusable && recenter(lane,settings,shift,w,h)){front_index=static_rasters.front_slot[lane];static_decision.recenter=1;}
                shifted=shift.reusable;
            }
        }
        // Builds the back slot under `spend`; promotes it once the visible
        // area at `projection` is complete. Returns -1 on draw failure.
        auto refine=[&](float projection,double& spend)->int{
            unsigned back_index=static_rasters.back_index(lane);
            auto& building=static_rasters.states[back_index];
            Shift back_shift{};
            bool restart=!building.refining || building.stale || building.key!=key || building.projection!=projection;
            StaticRect needed{};
            if(!restart){needed=visible_rect(building,back_shift);restart=!back_shift.reusable;}
            if(!restart && !building.covered.empty()){
                // A reveal or city/terrain edit can arrive between budgeted
                // bands without changing the environment key. Validate pixels
                // already drawn before appending today's contributors; otherwise
                // promotion briefly exposes a mixture of two world versions.
                ZoomScope scope_zoom(*this,building.projection);
                auto const& c=building.covered;
                restart=!raster_dependencies(raster_inputs[back_index],slot_settings(building,settings),
                    {c.left,c.top,c.right,c.bottom},false);
            }
            if(restart){
                if(building.refining && !building.covered.empty())++refine_restarts;
                if(!reset_slot(back_index,projection,settings))return -1;
                needed=visible_rect(building,back_shift);
            }
            if(!extend_coverage(back_index,building,settings,needed,spend,0,1.f,true))return -1;
            refine_worked=true;++refine_slices;
            if(!building.covered.contains(needed))return 0;
            building.valid=true;building.refining=false;building.stale=false;refine_restarts=0;
            ++building.metrics.full_draws;++cache_full_draws;++refine_promotions;
            static_rasters.promote(lane);front_index=static_rasters.front_slot[lane];
            auto& promoted=static_rasters.states[front_index];
            shifted=false;
            if(promoted.projection==zoom){visible_rect(promoted,shift);shifted=shift.reusable;}
            return 1;
        };
        {
            auto const& displayed=static_rasters.states[front_index];
            bool fresh=displayed.fresh(key);
            bool needs=goal>0 && (!(fresh && displayed.projection==goal) || (goal==zoom && !shifted));
            if(!needs){
                // The front became adequate again (zoom reversed, scrolled back):
                // abandon the half-built back so nothing keeps requesting frames.
                auto& abandoned=static_rasters.back(lane);
                if(abandoned.refining){abandoned.refining=false;abandoned.stale=true;}
            }
            else if(budget!=0){static_decision.refine=1;if(refine(goal,budget)<0)return false;}
        }
        // Strips the displayed raster needs now. Small (scroll) gaps are drawn
        // immediately; a large gap is filled progressively behind a preview.
        if(shifted){
            auto& displayed=static_rasters.states[front_index];
            auto needed=clip_region(shift.needed<StaticRect>(w,h,region_margin_x,region_margin_y));
            StaticRect overlap=clip_region({std::max(needed.left,displayed.covered.left),std::max(needed.top,displayed.covered.top),
                std::min(needed.right,displayed.covered.right),std::min(needed.bottom,displayed.covered.bottom)});
            long long missing=needed.area()-overlap.area();static_decision.missing=missing;
            if(missing>0){
                bool progressive=previewable && missing*100>35LL*w*h;
                double spend=progressive?std::max(budget,0.):-1.;
                // A strip costs roughly the records it crosses, whatever its
                // height, so a scroll that exposes a few rows per frame paid
                // nearly a full strip every frame. Draw a band ahead so most
                // scroll frames need none; background guard fill continues.
                int ahead=previewable?96:128;
                if(spend!=0 && !extend_coverage(front_index,displayed,settings,needed,spend,progressive?0:ahead,1.f,!displayed.stale))return false;
                if(progressive){budget=spend;refine_worked=true;}
                if(!displayed.covered.contains(needed))shifted=false;
            }
            // Background guard-band fill: grow a fresh raster toward its full
            // region on otherwise quiet frames, so later scroll steps (and
            // zoom-out previews) find their pixels already drawn.
            if(shifted && previewable && budget>0 && displayed.fresh(key) && !static_rasters.lane_refining(lane) &&
                    !displayed.covered.contains(limits)){
                double spend=budget*.5;
                if(!extend_coverage(front_index,displayed,settings,limits,spend,0,1.f,true))return false;
                refine_worked=true;
            }
        }
        // Without a preview path, finish the visible raster synchronously.
        if(!shifted && (!previewable || options.bootstrap_scale<=0)){
            auto const& displayed=static_rasters.states[front_index];
            c3x_renderer::render_core::LinearResample::Source probe{};
            bool covered=previewable && displayed.valid && resample_source(displayed,w,h,settings,probe)==2;
            if(!covered){static_decision.sync=1;double unbounded=-1.;if(refine(zoom,unbounded)<0)return false;}
        }
        auto bits=[](float value){std::uint32_t result=0;std::memcpy(&result,&value,sizeof(result));return std::uint64_t(result);};
        if(shifted){
            auto& displayed=static_rasters.states[front_index];
            std::array<std::uint64_t,12> next={1,front_index,displayed.revision,std::uint64_t(std::uint32_t(shift.x)),
                std::uint64_t(std::uint32_t(shift.y)),bits(settings.depth_translation),bits(displayed.depth_translation),
                std::uint64_t(displayed.region.width),bits(zoom),0,0,0};
            if(next!=restore_key){
                if(!static_restore.draw(context,static_cache,displayed.region.samples,
                        displayed.region.depth_samples,shift.x-region_margin_x,
                        shift.y-region_margin_y,{},nullptr,displayed.region.width,
                        displayed.region.height,false,false,0,nullptr,1,
                        -(settings.depth_translation-displayed.depth_translation)/16384.f))return false;
                work.draw(3);if(work.enabled)work.row().target_pixels+=std::uint64_t(static_cache.width)*static_cache.height;
                context->OMSetRenderTargets(0,nullptr,nullptr);
                if(scene_samples==1){
                    if(static_cache.resolved!=static_cache.color){context->CopyResource(static_cache.resolved,static_cache.color);work.copy(static_cache.resolved,true);}
                }
                else context->ResolveSubresource(static_cache.resolved,0,static_cache.color,0,
                    DXGI_FORMAT_R16G16B16A16_FLOAT);
                restore_key=next;++cache_scrolls;++displayed.metrics.restores;
            }else ++displayed.metrics.reuses;
            static_decision.shifted=1;
            overlay_frame={true,front_index,shift.x-region_margin_x,shift.y-region_margin_y,
                -(settings.depth_translation-displayed.depth_translation)/16384.f};
            // Dynamic layers move with the whole-pixel static raster.
            settings.translation[0]+=float(shift.snap_x/zoom);
            settings.translation[1]+=float(shift.snap_y/zoom);
            return true;
        }
        // Preview: resample the best retained rasters for this projection.
        using Source=c3x_renderer::render_core::LinearResample::Source;
        Source front_source{},home_source{},boot_source{};
        auto& displayed=static_rasters.states[front_index];
        auto& home=static_rasters.front(0);
        auto& image=bootstrap[lane];
        int front_cover=resample_source(displayed,w,h,settings,front_source);
        if(lane==1)static_decision.home_entry=(home.valid?1u:0u)|(home.stale?2u:0u)|(home.key!=key?4u:0u)|
            (raster_inputs[static_rasters.front_slot[0]].complete?32u:0u);
        int home_cover=lane==1 && static_pixels_current(home,raster_inputs[static_rasters.front_slot[0]],settings)?
            resample_source(home,w,h,settings,home_source):0;
        if(lane==1 && home.valid)static_decision.home_entry|=8u;
        int boot_cover=image.valid && bootstrap_stamp[lane]==bootstrap_identity()?
            resample_source(image,w,h,settings,boot_source):0;
        // Once zoom reaches its destination, a scene edit or camera move must
        // not fall back to an upsampled raster: magnifying terrain softens
        // every unchanged forest until refinement catches up. A current raster
        // at up to twice the resolution stays sharp when minified, so it is
        // kept (as map tiles are) while the destination refines behind it.
        if(zoom==hint){
            auto sharp=[&](StaticState const& slot){return slot.projection>=zoom && slot.projection<=2.f*zoom;};
            if(!sharp(displayed))front_cover=0;
            if(!sharp(home))home_cover=0;
            if(image.projection!=zoom)boot_cover=0;
        }
        // A ring-only bootstrap has no pixels inside the raster it extends,
        // which must remain the primary preview source.
        StaticState const* ring=front_cover==1?&displayed:front_cover==0 && home_cover==1?&home:nullptr;
        if(bootstrap_ring[lane] && !(bootstrap_ring[lane]==ring && bootstrap_ring_revision[lane]==ring->revision))boot_cover=0;
        Source const* primary=nullptr;Source const* secondary=nullptr;
        StaticState const* primary_slot=nullptr;StaticState const* secondary_slot=nullptr;
        if(front_cover==2){primary=&front_source;primary_slot=&displayed;}
        else if(front_cover==1 && home_cover==2){primary=&front_source;primary_slot=&displayed;secondary=&home_source;secondary_slot=&home;}
        else if(home_cover==2){primary=&home_source;primary_slot=&home;}
        else{
            if(boot_cover<2){
                if(!render_bootstrap(lane,settings,w,h,ring))return false;
                boot_cover=resample_source(image,w,h,settings,boot_source);
                if(boot_cover==0)return false;
            }
            if(front_cover==1){primary=&front_source;primary_slot=&displayed;secondary=&boot_source;secondary_slot=&image;}
            else if(ring==&home){primary=&home_source;primary_slot=&home;secondary=&boot_source;secondary_slot=&image;}
            else{primary=&boot_source;primary_slot=&image;}
        }
        std::array<std::uint64_t,12> next={2,std::uint64_t(reinterpret_cast<std::uintptr_t>(primary_slot)),primary_slot->revision,
            std::uint64_t(reinterpret_cast<std::uintptr_t>(secondary_slot)),secondary_slot?secondary_slot->revision:0,
            bits(zoom),std::uint64_t(std::uint32_t(camera_x)),std::uint64_t(std::uint32_t(camera_y)),
            bits(settings.depth_translation),0,0,0};
        // Each source slot's retained near-water layer is composited with the
        // same mapping. Baking them into bootstraps instead made a 0.5x
        // bootstrap 5x slower; any source without a layer keeps live overlays.
        auto overlay_source=[&](StaticState const* slot,Source const& base,Source& out){
            out=base;if(!slot || !overlay_enabled())return false;
            for(unsigned i=0;i<overlay_slots.size();++i)if(slot==&static_rasters.states[i]){
                auto const& retained=overlay_slots[i];
                if(retained.revision!=slot->revision || !retained.layer.samples || !retained.layer.depth_samples)return false;
                out.color=retained.layer.samples;out.depth=retained.layer.depth_samples;return true;}
            return false;
        };
        Source primary_overlay{},secondary_overlay{};
        bool overlays=overlay_source(primary_slot,*primary,primary_overlay) &&
            (!secondary || overlay_source(secondary_slot,*secondary,secondary_overlay));
        next[9]=std::uint64_t(overlays);
        if(next!=restore_key){
            if(!static_resample.draw(context,static_cache,*primary,secondary))return false;
            work.draw(3);if(work.enabled)work.row().target_pixels+=std::uint64_t(static_cache.width)*static_cache.height;
            if(overlays){
                if(!static_resample.draw(context,static_cache,primary_overlay,secondary?&secondary_overlay:nullptr,secondary?3:1))return false;
                work.draw(3);
            }
            restore_key=next;restore_overlays=overlays;
        }
        preview_overlays=restore_overlays;
        ++preview_frames;static_preview=true;static_decision.preview=1;
        static_decision.front_cover=front_cover;static_decision.home_cover=home_cover;static_decision.boot_cover=boot_cover;
        return true;
    }
    // Region of interest: the 1x region around a quantized camera. It owns
    // body placements and city lights (superset of every lane's region) and,
    // at the destination zoom, the shadow receivers. It changes only when the
    // camera crosses a quantum, so scrolling does not refit shadows or lights.
    bool update_roi(ViewportShaderSettings const& settings,int w,int h){
        auto floor_to=[](int value,int quantum){return value>=0?value/quantum*quantum:-((-value+quantum-1)/quantum)*quantum;};
        int qx=floor_to(camera_x,roi_quantum),qy=floor_to(camera_y,roi_quantum);
        static constexpr float ladder[]={.5f,.625f,.75f,.875f,1.f,1.25f,1.5f,1.75f,2.f,2.5f,3.f};
        // Keyed to the zoom destination, not the animating zoom, so the field
        // refits at most once per wheel input instead of at every ladder step
        // the animation crosses (edges briefly lack shadows during zoom-in).
        // Hidden 1x transactions share it, so lanes cannot alternate the field.
        float wanted=zoom_destination(),shadow_zoom=1.f;
        if(projection_zoom==1.f && !canonical_hidden())wanted=1.f;
        for(float step:ladder)if(step<=wanted+1e-4f)shadow_zoom=step;
        std::uint32_t zoom_bits=0;std::memcpy(&zoom_bits,&shadow_zoom,sizeof(zoom_bits));
        std::uint64_t visibility_sequence=0;
#ifdef C3X_RENDERER64_FRESH
        visibility_sequence=renderer.topology_cache.visibility_sequence();
#endif
        std::array<std::int64_t,11> next_key={std::int64_t(view_revision()),std::int64_t(wrap_pixels),qx,qy,w,h,
            std::int64_t(renderer.water_scene_active),std::int64_t(visibility_sequence),std::int64_t(zoom_bits),
            std::int64_t(renderer.content_revision),projection_zoom<1.f || shadow_zoom<1.f};
        auto roi=settings;
        roi.translation[0]-=float(camera_x-qx);roi.translation[1]-=float(camera_y-qy);
        int pad=roi_quantum;
        D3D11_RECT body_clip={-2*region_margin_x-pad,-2*region_margin_y-pad,
            w+2*region_margin_x+pad,h+2*region_margin_y+pad};
        // Body/light membership must cover the complete widest supported view,
        // including both endpoints of an in-flight zoom reversal.
        if(projection_zoom<1.f || shadow_zoom<1.f)body_clip=c3x_renderer::SceneProjection(renderer.content_view_width,
            renderer.content_view_height,c3x_renderer::SceneProjection::minimum)
            .source_rect(body_clip,4.f);
        float half_x=(float(w)*.5f+2.f*region_margin_x)/shadow_zoom+float(pad);
        float half_y=(float(h)*.5f+2.f*region_margin_y)/shadow_zoom+float(pad);
        D3D11_RECT shadow_clip={LONG(std::floor(float(w)*.5f-half_x)),LONG(std::floor(float(h)*.5f-half_y)),
            LONG(std::ceil(float(w)*.5f+half_x)),LONG(std::ceil(float(h)*.5f+half_y))};
        bool escaped=false;
        if(next_key==roi_key && body_requirements_valid && roi_receiver_check!=static_receiver_revision){
            // A newly captured receiver outside the retained region forces a rebuild.
            roi_receiver_check=static_receiver_revision;
            for(unsigned layer=0;!escaped && layer<geometry_layer_count;++layer)
                for(auto const& record:all_visible[layer])
                    if(!renderer.chunk_intersects_region(GeometryDrawReference(record),roi,body_clip,false)){escaped=true;break;}
        }
        if(next_key==roi_key && body_requirements_valid && !escaped){++body_requirement_reuses;return true;}
        auto started=std::chrono::steady_clock::now();
        roi_records={};roi_shadow_records={};
        body_requirements.clear();body_requirements_valid=false;
        bool complete=true;
        auto require=[&](unsigned layer,GeometryDrawReference const& reference){
            if(complete && reference.content().instances && !reference.content().instances->empty())
                complete=body_requirements.add(renderer.shared_instances,layer,reference,renderer.shared_instance_draw_key(layer,reference));
        };
#ifndef C3X_RENDERER64_FRESH
        // Legacy consumers submit the original native ranges.
        GeometryDrawView original=renderer.geometry_vertex_buffers;
        for(unsigned layer=0;layer<geometry_layer_count;++layer)
            for(auto const& reference:original[layer])require(layer,reference);
#endif
        for(unsigned layer=0;layer<geometry_layer_count;++layer)
            for(auto const& record:all_visible[layer])require(layer,GeometryDrawReference(record));
        contributors(roi,body_clip,false,[&](unsigned layer,auto const& record){
            GeometryDrawReference reference(record);
            if(!renderer.chunk_intersects_region(reference,roi,body_clip,false))return;
            roi_records[layer].push_back(record);require(layer,reference);
            if(renderer.chunk_intersects_region(reference,roi,shadow_clip,false))roi_shadow_records[layer].push_back(record);
        });
        if(!complete)return false;
        bool any_receiver=false;
        for(unsigned layer=0;layer<geometry_layer_count;++layer)
            if(layer!=geometry_shadow && !roi_shadow_records[layer].empty())any_receiver=true;
        if(!any_receiver)roi_shadow_records=all_visible;
        shadow_field=any_receiver && !sandbox_perf_options().shadow_tight?StaticRect{int(shadow_clip.left)-int(roi.translation[0]),int(shadow_clip.top)-int(roi.translation[1]),
            int(shadow_clip.right)-int(roi.translation[0]),int(shadow_clip.bottom)-int(roi.translation[1])}:
            StaticRect{INT_MIN/4,INT_MIN/4,INT_MAX/4,INT_MAX/4};
        roi_key=next_key;++roi_revision;body_requirements_valid=true;roi_receiver_check=static_receiver_revision;
        ++body_requirement_builds;body_requirement_visits=body_requirements.visits;
        body_requirement_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        return true;
    }
    bool reconstruct() {
        SandboxPassWorkload::Scope pass(work,SandboxPassWorkload::reconstruction);
        auto* context=renderer.context;
        context->OMSetRenderTargets(0,nullptr,nullptr);
        if(scene_samples==1){
            if(glow.linear.resolved!=glow.linear.color){context->CopyResource(glow.linear.resolved,glow.linear.color);work.copy(glow.linear.resolved,true);}
        }
        else context->ResolveSubresource(glow.linear.resolved,0,glow.linear.color,0,
            DXGI_FORMAT_R16G16B16A16_FLOAT);
        return bloom.draw(static_cache.view,glow.linear.view);
    }
    bool draw(c3x_renderer_frame_v1 const& frame,int next_camera_x,int next_camera_y,
            int unit_x,int unit_y,int incarnation,int viewer,bool unit_visible,
            float next_zoom) {
        work.begin();shadow.work=&work;sandbox_direct_units.work=&work;bloom.work=&work;
        pulled_pages.begin_frame();
        if(pulled_device!=renderer.device_generation){pulled_pages.clear();pulled_device=renderer.device_generation;}
        body_requirement_builds=body_requirement_reuses=body_requirement_visits=0;body_requirement_ms=0;
        body_requirements.coverage_probes=0;
        prepare_subspans={};dynamic_subspans={};dynamic_calls={};prepare_unit_plan_reused=prepare_unit_reselected=0;
        char scroll_diagnostics[8]={};
        bool trace_scroll=c3x_renderer::render_core::cached_environment("C3X_SANDBOX_SCROLL_CALLS",
            scroll_diagnostics,sizeof(scroll_diagnostics)) && scroll_diagnostics[0]=='1';
        auto scroll_before=cache_scrolls,full_before=cache_full_draws;

        auto fail=[this](char const* stage){
            discard_static();
            reflection_valid=false;reflected_terrain_material_valid=false;
            renderer.trace.write("fresh-draw-failed",stage,true);
            char diagnostic[160];std::snprintf(diagnostic,sizeof(diagnostic),
                "[C3X renderer] stage=fresh-draw-failed pass=%s\n",stage);
            OutputDebugStringA(diagnostic);
            std::printf("SANDBOX_DRAW_ERROR stage=%s\n",stage);
            std::fflush(stdout);
            return false;
        };
#ifdef C3X_RENDERER64_FRESH
        instance_plans.begin();instance_plan_builds=instance_plan_reuses=0;
#endif
        LARGE_INTEGER frequency{},ticks[7]={};
        QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&ticks[0]);
        LARGE_INTEGER prepare_previous=ticks[0];
        auto mark_prepare=[&](PrepareSpan span){LARGE_INTEGER now{};QueryPerformanceCounter(&now);
            prepare_subspans[span]=double(now.QuadPart-prepare_previous.QuadPart)*1000/frequency.QuadPart;
            prepare_previous=now;};
        int width=renderer.content_view_width,height=renderer.content_view_height;
        if (!renderer.device || !renderer.context || width<1 || height<1)return fail("viewport");
        if(!std::isfinite(next_zoom))return fail("projection");
        float zoom=std::clamp(next_zoom,c3x_renderer::SceneProjection::minimum,c3x_renderer::SceneProjection::maximum);
        // The mirror's key includes the projection; an animating zoom reuses
        // it for a couple of frames instead of invalidating it outright.
        if(sandbox_perf_options().legacy && projection_zoom!=zoom){
            reflection_valid=false;reflected_terrain_material_valid=false;
        }
        {
            unsigned zoom_lane=StaticRasters::lane_of(zoom);
            if(lane_projection[zoom_lane]==zoom)++lane_still[zoom_lane];else lane_still[zoom_lane]=0;
            lane_projection[zoom_lane]=zoom;lane_drawn[zoom_lane]=++draw_serial;
        }
        projection_zoom=zoom;display_zoom=1.f;
        static_rasters.select_lane(c3x_renderer::render_core::StaticRasterStates<
            c3x_renderer::render_core::LinearTarget>::lane_of(projection_zoom));
#ifdef C3X_RENDERER64_FRESH
        if(renderer.profiling){
            gpu_phases.poll(renderer.context,[&](GpuPhases::Sample const& sample){
                double scale=sample.valid?1000.0/sample.frequency:0;
                char detail[640];sprintf_s(detail,"sample_sequence=%llu valid=%u frequency=%llu gpu_span_ms=%.6f prepare_ms=%.6f reflection_ms=%.6f static_ms=%.6f water_ms=%.6f units_ms=%.6f reconstruct_ms=%.6f ring_skipped=%u",
                    sample.sequence,unsigned(sample.valid),sample.frequency,sample.total*scale,
                    sample.ticks[0]*scale,sample.ticks[1]*scale,sample.ticks[2]*scale,sample.ticks[3]*scale,sample.ticks[4]*scale,sample.ticks[5]*scale,sample.skipped);
                renderer.trace.write("fresh-gpu-phases",detail,true);
            });
            gpu_phases.begin(renderer.device,renderer.context,renderer.trace.sequence.load(),6);
            gpu_phases.pass_begin(renderer.context,GpuPhases::background);
        }
        GpuPhases::Scope gpu_scope{gpu_phases,renderer.context};
#endif
        bool cold_setup=!visual.installed;LARGE_INTEGER setup[4]={};
        QueryPerformanceCounter(&setup[0]);
        if (!visual.install()) return fail("visual_setup");QueryPerformanceCounter(&setup[1]);
        renderer.context->PSSetSamplers(15,1,&visual.shadow_comparison);
        if (!shadow.ensure()) return fail("shadow_setup");QueryPerformanceCounter(&setup[2]);
        unsigned w=unsigned(width)+8,h=unsigned(height)+8;
        if (!ensure_targets(w,h)) return fail("targets");QueryPerformanceCounter(&setup[3]);
        if(cold_setup){std::printf("SANDBOX_FRESH_SETUP visual_ms=%.6f shadow_ms=%.6f targets_ms=%.6f\n",
            double(setup[1].QuadPart-setup[0].QuadPart)*1000/frequency.QuadPart,
            double(setup[2].QuadPart-setup[1].QuadPart)*1000/frequency.QuadPart,
            double(setup[3].QuadPart-setup[2].QuadPart)*1000/frequency.QuadPart);std::fflush(stdout);}
        update_environment(frame);
        char reflection_diagnostic[8]{};
        if(c3x_renderer::render_core::cached_environment("C3X_SANDBOX_SKIP_REFLECTION",
                reflection_diagnostic,sizeof(reflection_diagnostic)) &&
            std::strcmp(reflection_diagnostic,"1")==0)
            renderer.reflection.enabled=false;
        if (!active) {
            production_reflection=renderer.scene_reflection_view;
            production_reflection_width=renderer.scene_reflection_width;
            production_reflection_height=renderer.scene_reflection_height;
            active=true;
        }
#ifdef C3X_RENDERER64_FRESH
        // The production state owns its reflection view. Borrow the sandbox
        // view only during this scene draw, including every early exit.
        struct ReflectionBinding {
            ID3D11ShaderResourceView*& view;
            unsigned& width;
            unsigned& height;
            ID3D11ShaderResourceView* old_view;
            unsigned old_width,old_height;
            ~ReflectionBinding(){view=old_view;width=old_width;height=old_height;}
        } reflection_binding{renderer.scene_reflection_view,
            renderer.scene_reflection_width,renderer.scene_reflection_height,
            renderer.scene_reflection_view,renderer.scene_reflection_width,
            renderer.scene_reflection_height};
#endif
        char skip_scene[8]{};
        if (resident_builds && c3x_renderer::render_core::cached_environment("C3X_SANDBOX_SKIP_SCENE",
                skip_scene,sizeof(skip_scene)) && std::strcmp(skip_scene,"1")==0) {
            std::fill(phases,phases+5,0.0);
            return true;
        }
        renderer.scene_reflection_view=reflection.view;
        renderer.scene_reflection_width=unsigned(reflection.width/reflection_scale)/2;
        renderer.scene_reflection_height=unsigned(reflection.height/reflection_scale)/2;
#ifndef C3X_RENDERER64_FRESH
        renderer.geometry_viewport_settings.translation[0]+=float(next_camera_x-camera_x);
        renderer.geometry_viewport_settings.translation[1]+=float(next_camera_y-camera_y);
        renderer.geometry_viewport_settings.depth_translation+=float(next_camera_y-camera_y);
#endif
        camera_x=next_camera_x;camera_y=next_camera_y;
        if(camera_x!=moved_camera_x || camera_y!=moved_camera_y){
            moved_camera_x=camera_x;moved_camera_y=camera_y;QueryPerformanceCounter(&camera_moved_at);
        }
        reflection_shift[0]=reflection_shift[1]=0;
        renderer.water_material=c3x_renderer::render_core::water_material_frame(frame);
        int water_camera_x=frame.world_wrap_x && frame.world_width_tiles>0?
            camera_x%(frame.world_width_tiles*frame.tile_width/2):camera_x;
        renderer.water_material.camera[0]-=float(water_camera_x)/frame.tile_width+
            float(camera_y)/frame.tile_height;
        renderer.water_material.camera[1]-=float(water_camera_x)/frame.tile_width-
            float(camera_y)/frame.tile_height;
        renderer.water_time_seconds=renderer.water_material.time;
        renderer.wave_time_seconds=renderer.water_material.time;
        renderer.cities.time=renderer.water_material.time; // attached city effects
        if(!renderer.compose_resource_animations(frame,true))return fail("resource_poses");
        mark_prepare(prepare_setup_resources);
        auto settings=renderer.geometry_viewport_settings;
#ifdef C3X_RENDERER64_FRESH
        // Static occurrences use a stable world pixel basis. View translation
        // and depth origin are independent, including a rebuilt membership.
        settings.translation[0]=float(camera_x);settings.translation[1]=float(camera_y);
        settings.depth_translation=-float(renderer.scene_depth_origin);
#endif
        settings.translation[0]+=4;settings.translation[1]+=4;
        settings.inverse_size[0]=1.f/w;settings.inverse_size[1]=1.f/h;
        auto reflected=renderer.geometry_viewport_settings;
#ifdef C3X_RENDERER64_FRESH
        reflected.translation[0]=float(camera_x);reflected.translation[1]=float(camera_y);
        reflected.depth_translation=-float(renderer.scene_depth_origin);
#endif
        reflected.translation[0]+=8;reflected.translation[1]+=8;
        reflected.inverse_size[0]=1.f/float(w+8);
        reflected.inverse_size[1]=1.f/float(h+8);
        int next_wrap_pixels=frame.world_wrap_x?
            frame.world_width_tiles*frame.tile_width/2:0;
        if (!capture(settings,reflected,int(w),int(h),next_wrap_pixels))return fail("visibility_capture");
        mark_prepare(prepare_capture);
        // One stable region of interest feeds shadow receivers, city lights and
        // body placements (see update_roi). Camera steps inside it change none
        // of them, so retained static pixels and shadow pages stay valid.
        mark_prepare(prepare_raster_proof);
        if(!update_roi(settings,int(w),int(h)))return fail("body_requirements");
        auto const& body_inputs=body_requirements;
        mark_prepare(prepare_body_requirements);
        auto retire_completed_instance_plans=[&]{
#ifdef C3X_RENDERER64_FRESH
            instance_plans={};instance_plans.begin();instance_plan_bytes=0;
#endif
        };
        bool tight_shadows=sandbox_perf_options().shadow_tight;
        auto lights_begin=std::chrono::steady_clock::now();
        if(!update_city_lights())return fail("city_lights");
        if(renderer.trace.level>=2){double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-lights_begin).count();
            if(ms>1){char detail[64];sprintf_s(detail,"ms=%.2f lights=%zu",ms,selected_lighting.size());renderer.trace.write("fresh-city-lights",detail);}}
        if(!shadow.render(tight_shadows?all_visible:roi_shadow_records,raster_scope(),
                tight_shadows?static_receiver_revision:roi_revision,view_revision(),body_inputs,retire_completed_instance_plans))
            return fail("shadow");
#ifdef C3X_RENDERER64_FRESH
        if(shared_front!=shadow.shared_front){instance_plans={};instance_plans.begin();instance_plan_bytes=0;shared_front=shadow.shared_front;}
#else
        shared_front=shadow.shared_front;
#endif
        if(!renderer.shared_instances.valid(shared_front))return fail("shared_submission");
        mark_prepare(prepare_city_shadow);
        D3D11_RECT full={0,0,LONG(w),LONG(h)};
        char unit_control[8]{};
        bool units=c3x_renderer::render_core::cached_environment("C3X_SANDBOX_UNITS",unit_control,
            sizeof(unit_control)) && std::strcmp(unit_control,"1")==0;
        // One BIQ water plane currently consumes this mirror target. Each
        // additional authored water level can own another target and pass.
        std::vector<D3D11_RECT> water_receivers;
        auto mirror=reflected_water_bounds(settings,int(w),int(h),&water_receivers);
#ifdef C3X_RENDERER64_FRESH
        // Standalone callers can supply a fresh captured occurrence vector
        // directly; production calls this same preselection before asset work.
        bool unit_input_changed=!same_unit_inputs(renderer.fresh_unit_poses,unit_contribution_required);
        if(unit_input_changed || unit_contribution_scope!=view_revision() || unit_plan_x!=camera_x || unit_plan_y!=camera_y || unit_plan_zoom!=projection_zoom || unit_plan_hour!=visual_hour || unit_plan_season!=frame.season ||
                unit_plan_bounds!=renderer.unit_bodies.contribution_sequence || unit_plan_reflection!=renderer.reflection.enabled){
            ++prepare_unit_reselected;
            auto candidates=unit_input_changed?renderer.fresh_unit_poses:unit_contribution_candidates;
            std::vector<SandboxDirectUnits::ScenePose> required;
            if(!select_unit_contributors(frame,candidates,required,projection_zoom))return fail("unit_preselection");
        }else ++prepare_unit_plan_reused;
#endif
        // This span includes consumer receiver setup and any selection missed
        // by the caller's preselection; source pose preparation is separate.
        mark_prepare(prepare_unit_selection);
#ifdef C3X_RENDERER64_FRESH
        if(!sandbox_direct_units.prepare_real(frame,unit_contribution_candidates,visual_hour,unit_contribution_plan))return fail("shared_unit_preparation");
#endif
        QueryPerformanceCounter(&ticks[1]);c3x_renderer::render_core::gpu_timeline().mark(renderer.context,"scene_prepare");
        prepare_subspans[prepare_unit_pose]=double(ticks[1].QuadPart-prepare_previous.QuadPart)*1000/frequency.QuadPart;
#ifdef C3X_RENDERER64_FRESH
        gpu_phases.pass_end(renderer.context);gpu_phases.pass_begin(renderer.context,GpuPhases::import);
#endif
        work.pass=SandboxPassWorkload::reflection_scene;
        if (renderer.reflection.enabled && mirror.left<mirror.right && mirror.top<mirror.bottom) {
            auto key=reflection_writer(reflected,mirror);
            bool redraw=!reflection_valid || reflection_key!=key;
            // While the zoom animates, the blurred mirror may lag up to two
            // frames; scrolling and settled zoom still redraw it immediately.
            unsigned mirror_lane=StaticRasters::lane_of(projection_zoom);
            bool hidden=mirror_lane==0 && canonical_hidden();
            if(redraw && reflection_valid && !sandbox_perf_options().legacy &&
                    ((lane_still[mirror_lane]==0 && reflection_skips<2) || hidden)){
                redraw=false;++reflection_skips;
            }
            else if(redraw)reflection_skips=0;
            if(redraw && reflection_valid && !sandbox_perf_options().legacy && mirror_camera_only(key,reflection_key)){
                LARGE_INTEGER now{};QueryPerformanceCounter(&now);
                bool moving=frequency.QuadPart>0 && now.QuadPart-camera_moved_at.QuadPart<frequency.QuadPart*3/20;
                double shift_x=double(camera_x-reflection_camera_x)*projection_zoom;
                double shift_y=double(camera_y-reflection_camera_y)*projection_zoom;
                if(moving && std::abs(shift_x)<=160 && std::abs(shift_y)<=96){redraw=false;++reflection_shift_reuses;}
            }
            if(!redraw){
                ++reflection_reuses;if(work.enabled)++work.row().reuses;
                reflection_shift[0]=float(double(camera_x-reflection_camera_x)*projection_zoom);
                reflection_shift[1]=float(double(camera_y-reflection_camera_y)*projection_zoom);
            }
            else {
            if(work.enabled)++work.row().rebuilds;
            reflection_valid=false;
            ID3D11ShaderResourceView* none=nullptr;
            renderer.context->PSSetShaderResources(121,1,&none);
            if(!prepare_reflected_terrain_material(reflected,mirror))return fail("reflected_terrain");
            float mirror_clear[4]={};
            renderer.context->ClearRenderTargetView(reflection_static.target,mirror_clear);work.clear(reflection_static.target);
            renderer.context->ClearDepthStencilView(reflection_static.depth,
                D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);work.clear(reflection_static.depth);
            relight_reflected_terrain();
            if(!draw_scene(reflection_visible,reflected,mirror,reflection_static.target,
                    reflection_static.depth,true,reflection_scale,false,true))return fail("reflection_scene");
            renderer.context->OMSetRenderTargets(0,nullptr,nullptr);
            reflection_key=key;reflection_valid=true;
            reflection_camera_x=camera_x;reflection_camera_y=camera_y;
            ++reflection_draws;
            }
            bool dynamic_reflection=units
#ifdef C3X_RENDERER64_FRESH
                || sandbox_direct_units.reflection_contributors!=0
#endif
                ;
            // A shifted mirror cannot take unit reflections drawn at this camera;
            // they return when the exact mirror is redrawn after the camera settles.
            bool mirror_shifted=reflection_shift[0]!=0 || reflection_shift[1]!=0;
            if(mirror_shifted)dynamic_reflection=false;
            if(redraw || dynamic_reflection || reflection_dynamic_written){
                renderer.context->OMSetRenderTargets(0,nullptr,nullptr);
                renderer.context->CopyResource(reflection.color,reflection_static.color);work.copy(reflection.color,true);
                renderer.context->CopyResource(reflection.depth_texture,reflection_static.depth_texture);work.copy(reflection.depth_texture,true);
            }
#ifdef C3X_RENDERER64_FRESH
            if(!mirror_shifted && !sandbox_direct_units.draw_real(frame,unit_contribution_candidates,reflection,
                    reflection_scale,visual_hour,true,projection_zoom))return fail("reflected_real_units");
#endif
            if(units && !mirror_shifted && !sandbox_direct_units.draw(frame,unit_x,unit_y,
                    incarnation,viewer,unit_visible,camera_x,camera_y,reflection,
                    reflection_scale,next_zoom,visual_hour,true))return fail("reflected_units");
            reflection_dynamic_written=dynamic_reflection;
        }
        QueryPerformanceCounter(&ticks[2]);c3x_renderer::render_core::gpu_timeline().mark(renderer.context,"reflection");
#ifdef C3X_RENDERER64_FRESH
        gpu_phases.pass_end(renderer.context);gpu_phases.pass_begin(renderer.context,GpuPhases::receivers);
#endif
        work.pass=SandboxPassWorkload::main_scene;
        auto* context=renderer.context;
        // Retained static layer: whole-pixel reuse, preview resampling and
        // budgeted refinement (never a synchronous full redraw while a usable
        // preview exists). May move `settings` by a sub-pixel snap.
        if(!compose_static(settings,int(w),int(h)))return fail("static_layer");
        if(renderer.trace.level>=2){auto const& d=static_decision;char detail[512];
            sprintf_s(detail,"zoom=%.4f lane=%u camera=%d,%d slot=%d,%d reusable=%d recenter=%d shifted=%d refine=%d sync=%d preview=%d covers=%d,%d,%d missing=%lld static_ms=%.2f entry=%u home_entry=%u input_kb=%zu repair=%u changed=%u reordered=%u rects=%u percent=%u boot_draw_ms=%.1f boot_deps_ms=%.1f boot_kpx=%lld borrowed=%u",
                projection_zoom,d.lane,d.camera_x,d.camera_y,d.slot_x,d.slot_y,d.reusable,d.recenter,d.shifted,d.refine,d.sync,d.preview,
                d.front_cover,d.home_cover,d.boot_cover,d.missing,last_static_ms,d.entry,d.home_entry,d.input_bytes/1024,
                d.repair,d.repair_changed,d.repair_reordered,d.repair_rects,d.repair_percent,d.boot_draw_ms,d.boot_deps_ms,d.boot_area/1000,
                unsigned(renderer.borrowed_scene_frame));
            renderer.trace.write("static-compose",detail,false);}
        QueryPerformanceCounter(&ticks[3]);c3x_renderer::render_core::gpu_timeline().mark(renderer.context,"static");
#ifdef C3X_RENDERER64_FRESH
        gpu_phases.pass_end(renderer.context);gpu_phases.pass_begin(renderer.context,GpuPhases::shadow);
#endif
        if(trace_scroll){
            std::printf("SCROLL_CALL camera=%d,%d zoom=%.9f native_width=%d epoch=%llu translation=%.6f,%.6f depth=%.6f depth_origin=%lld region_camera=%d,%d reuse=%u full=%u\n",
                camera_x,camera_y,projection_zoom,frame.tile_width,
                static_cast<unsigned long long>(renderer.tile_geometry_epoch),
                settings.translation[0]-4,settings.translation[1]-4,settings.depth_translation,
                static_cast<long long>(renderer.scene_depth_origin),current_raster().camera_x,current_raster().camera_y,
                cache_scrolls-scroll_before,cache_full_draws-full_before);
            std::fflush(stdout);
        }
        LARGE_INTEGER dynamic_previous=ticks[3];auto dynamic_previous_calls=work.calls;
        auto mark_dynamic=[&](DynamicSpan span){
            if(!renderer.trace.level)return;
            LARGE_INTEGER now{};QueryPerformanceCounter(&now);
            dynamic_subspans[span]=double(now.QuadPart-dynamic_previous.QuadPart)*1000/frequency.QuadPart;
            dynamic_calls[span]={work.calls.draws-dynamic_previous_calls.draws,work.calls.copies-dynamic_previous_calls.copies,
                work.calls.uploads-dynamic_previous_calls.uploads};
            dynamic_previous=now;dynamic_previous_calls=work.calls;
        };
        context->OMSetRenderTargets(0,nullptr,nullptr);
        float dynamic_clear[4]={};
        context->ClearRenderTargetView(glow.linear.target,dynamic_clear);work.clear(glow.linear.target);
        context->CopyResource(glow.linear.depth_texture,static_cache.depth_texture);work.copy(glow.linear.depth_texture,true);
        ++depth_copies;
        bool const aquatic_visible=renderer.water_scene_active &&
            !renderer.sandbox_aquatic_resource_poses[geometry_feature].empty();
        float aquatic_bounds[8]={float(w),float(h),0,0,float(width/2)+4,float(height/2)+4,0,0};
        if(aquatic_visible)for(auto const& record:
                renderer.sandbox_aquatic_resource_poses[geometry_feature]){
            GeometryDrawReference chunk(record);
            auto const& box=chunk.bounds();
            float dx=settings.translation[0]+float(chunk.translation_x())+resident_basis_x;
            float dy=settings.translation[1]+float(chunk.translation_y())+resident_basis_y;
            aquatic_bounds[0]=std::max(0.f,std::min(aquatic_bounds[0],box.left+dx-8));
            aquatic_bounds[1]=std::max(0.f,std::min(aquatic_bounds[1],box.top+dy-8));
            aquatic_bounds[2]=std::min(float(w),std::max(aquatic_bounds[2],box.right+dx+8));
            aquatic_bounds[3]=std::min(float(h),std::max(aquatic_bounds[3],box.bottom+dy+8));
        }
        context->UpdateSubresource(aquatic_bounds_buffer,0,nullptr,aquatic_bounds,0,0);work.upload_buffer(aquatic_bounds_buffer);
        mark_dynamic(dynamic_depth_setup);
        if(aquatic_visible) {
            if(!aquatic_scene.ensure(renderer.device,w,h,DXGI_FORMAT_R8G8B8A8_UNORM))
                return false;
            ID3D11ShaderResourceView* none=nullptr;
            context->PSSetShaderResources(123,1,&none);
            context->ClearRenderTargetView(aquatic_scene.target,dynamic_clear);work.clear(aquatic_scene.target);
            if(!draw_resource_poses(renderer.sandbox_aquatic_resource_poses,settings,full,
                    aquatic_scene.target,nullptr,true))return fail("aquatic_resources");
            context->OMSetRenderTargets(0,nullptr,nullptr);
        }
        mark_dynamic(dynamic_aquatic);
        work.pass=SandboxPassWorkload::water;
        char water_diagnostic[8]{};
        bool skip_water=c3x_renderer::render_core::cached_environment("C3X_SANDBOX_SKIP_WATER_PASS",
            water_diagnostic,sizeof(water_diagnostic)) &&
            std::strcmp(water_diagnostic,"1")==0;
        if (renderer.water_scene_active && !skip_water) {
            // Cache lighting only once the view settles. During motion the
            // direct shader avoids paying an extra preparation pass per frame.
            // Every quantity affecting lighting/coverage is in this identity;
            // animated normals, marine objects and reflections stay live.
            auto bits=[](float value){std::uint32_t v=0;std::memcpy(&v,&value,4);return std::uint64_t(v);};
            std::array<std::uint64_t,13> key={view_revision(),visibility_revision,renderer.device_generation,
                renderer.content_revision,shadow.sampling_identity,shadow.page_contents.rebuilt,
                lighting_revision,bits(visual_hour),std::uint64_t(previous_season),bits(projection_zoom),
                bits(settings.translation[0]),bits(settings.translation[1]),(std::uint64_t(w)<<32)|h};
            char direct_water[8]={};
            bool reference=c3x_renderer::render_core::cached_environment("C3X_RENDERER_WATER_LIGHTING_REFERENCE",direct_water,sizeof(direct_water))&&direct_water[0]=='1';
            bool eligible=!reference && scene_samples==1 && scene_scale==1 && shadow.atlas_complete &&
                lane_still[StaticRasters::lane_of(projection_zoom)]>=3 &&
                !(projection_zoom==1.f && canonical_hidden());
            bool cached=eligible && water_lighting_valid && water_lighting_key==key;
            if(eligible && !cached){
                ID3D11ShaderResourceView* none=nullptr;context->PSSetShaderResources(122,1,&none);
                water_lighting_valid=false;
                if(water_lighting.ensure(renderer.device,w,h,DXGI_FORMAT_R16G16B16A16_FLOAT)){
                    context->ClearRenderTargetView(water_lighting.target,dynamic_clear);work.clear(water_lighting.target);
                    capture_water_lighting=true;
                    bool drawn=draw_layer(water_visible,geometry_water,settings,full,water_lighting.target,nullptr,false,float(scene_scale));
                    capture_water_lighting=false;context->OMSetRenderTargets(0,nullptr,nullptr);
                    if(!drawn)return fail("water_lighting");
                    water_lighting_key=key;water_lighting_valid=cached=true;++water_lighting_builds;
                }
            }else if(cached)++water_lighting_reuses;
            aquatic_bounds[6]=cached?1.f:0.f;
            context->UpdateSubresource(aquatic_bounds_buffer,0,nullptr,aquatic_bounds,0,0);work.upload_buffer(aquatic_bounds_buffer);
            // The frame's static pixels came from a slot whose retained overlay
            // layer matches it: draw only the animated water live, then the
            // layer, then anything after the overlays in scene order.
            bool composited=false;
            if(overlay_frame.valid && overlay_enabled()){
                auto const& slot=static_rasters.states[overlay_frame.slot];
                auto const& retained=overlay_slots[overlay_frame.slot];
                bool covered=retained.revision==slot.revision && retained.layer.samples;
                if(!covered)++overlay_miss_revision;
                for(unsigned layer=0;covered && layer<geometry_layer_count;++layer)
                    if(!water_visible[layer].empty() && layer!=geometry_water && layer!=geometry_river &&
                       layer!=geometry_wave && !overlay_layer(layer)){covered=false;++overlay_miss_layer;overlay_miss_last_layer=layer;}
                if(covered){
                    if(!draw_layer(water_visible,geometry_water,settings,full,glow.linear.target,glow.linear.depth,false,float(scene_scale)) ||
                       !draw_layer(water_visible,geometry_river,settings,full,glow.linear.target,glow.linear.depth,false,float(scene_scale)))
                        return fail("water_scene");
                    auto const& c=slot.covered;
                    if(!overlay_composite.draw(context,glow.linear,retained.layer,overlay_frame.move_x,overlay_frame.move_y,
                            {c.left,c.top,c.right,c.bottom},overlay_frame.depth_shift))return fail("water_overlay");
                    work.draw(3);
                    char wave_control[8]{};
                    bool skip_scene_wave=c3x_renderer::render_core::cached_environment("C3X_SANDBOX_SKIP_WAVE",
                        wave_control,sizeof(wave_control)) && std::strcmp(wave_control,"1")==0;
                    if(!skip_scene_wave && !draw_layer(water_visible,geometry_wave,settings,full,glow.linear.target,
                            glow.linear.depth,false,float(scene_scale)))return fail("water_scene");
                    composited=true;++overlay_hits;
                }
            }
            if(!composited && static_preview && preview_overlays){
                // The preview already holds the near-water overlays; only the
                // animated water is live (over resampled depth, so biased).
                if(preview_raster_device!=renderer.device_generation){
                    if(preview_water_raster){preview_water_raster->Release();preview_water_raster=nullptr;}
                    D3D11_RASTERIZER_DESC desc={};renderer.rasterizer_state->GetDesc(&desc);
                    desc.DepthBias=-INT(4.0*16777216.0/16384.0);
                    if(SUCCEEDED(renderer.device->CreateRasterizerState(&desc,&preview_water_raster)))
                        preview_raster_device=renderer.device_generation;
                }
                auto* raster=renderer.rasterizer_state;
                if(preview_water_raster)renderer.rasterizer_state=preview_water_raster;
                bool drawn=draw_layer(water_visible,geometry_water,settings,full,glow.linear.target,glow.linear.depth,false,float(scene_scale)) &&
                    draw_layer(water_visible,geometry_river,settings,full,glow.linear.target,glow.linear.depth,false,float(scene_scale));
                renderer.rasterizer_state=raster;
                if(!drawn)return fail("water_scene");
                char wave_control[8]{};
                bool skip_scene_wave=c3x_renderer::render_core::cached_environment("C3X_SANDBOX_SKIP_WAVE",
                    wave_control,sizeof(wave_control)) && std::strcmp(wave_control,"1")==0;
                if(!skip_scene_wave && !draw_layer(water_visible,geometry_wave,settings,full,glow.linear.target,
                        glow.linear.depth,false,float(scene_scale)))return fail("water_scene");
                composited=true;++overlay_preview_hits;
            }
            if(!composited){
                ++overlay_misses;if(!overlay_frame.valid)++overlay_miss_unshifted;
                if(static_preview && preview_raster_device!=renderer.device_generation){
                    if(preview_water_raster){preview_water_raster->Release();preview_water_raster=nullptr;}
                    D3D11_RASTERIZER_DESC desc={};renderer.rasterizer_state->GetDesc(&desc);
                    // Four pixel-depth units (depth = 0.5 - pixels/16384) in D24 steps.
                    desc.DepthBias=-INT(4.0*16777216.0/16384.0);
                    if(SUCCEEDED(renderer.device->CreateRasterizerState(&desc,&preview_water_raster)))
                        preview_raster_device=renderer.device_generation;
                }
                auto* raster=renderer.rasterizer_state;
                if(static_preview && preview_water_raster)renderer.rasterizer_state=preview_water_raster;
                bool drawn=draw_scene(water_visible,settings,full,glow.linear.target,
                        glow.linear.depth,false,float(scene_scale));
                renderer.rasterizer_state=raster;
                if(!drawn)return fail("water_scene");
            }
        }
        mark_dynamic(dynamic_water_scene);
        if(!draw_resource_poses(renderer.sandbox_resource_poses,settings,full,
                glow.linear.target,glow.linear.depth))
            return fail("resource_scene");
        mark_dynamic(dynamic_resources);
        char wave_diagnostic[8]{};
        bool skip_wave=c3x_renderer::render_core::cached_environment("C3X_SANDBOX_SKIP_WAVE",
            wave_diagnostic,sizeof(wave_diagnostic)) &&
            std::strcmp(wave_diagnostic,"1")==0;
        if (!skip_water && !skip_wave && !renderer.wave_chunks.empty()) {
            GeometryDrawView::Records waves{};
            for(auto const& chunk:renderer.wave_chunks){
                GeometryDrawRecord record(chunk);
#ifdef C3X_RENDERER64_FRESH
                record.translation_x+=int(resident_basis_x);record.translation_y+=int(resident_basis_y);
#endif
                waves[geometry_wave].push_back(record);
            }
            if(!draw_layer(waves,geometry_wave,settings,full,glow.linear.target,
                    glow.linear.depth,false,float(scene_scale)))return fail("coastal_waves");
        }
        mark_dynamic(dynamic_waves);
        // Flames, smoke and night lights above city bodies, below units. A
        // static preview (resampled from another zoom while the new static
        // layer builds) skips them for its one or two frames: drawn over the
        // resampled color and depth they flashed as bright bursts in game.
        if(!static_preview && !effect_visible[geometry_city].empty() && !draw_layer(effect_visible,geometry_city,settings,full,
                glow.linear.target,glow.linear.depth,false,float(scene_scale)))return fail("city_effects");
        // World overlays finish before the foreground unit layer.
        auto border_clip=source_bounds(settings,full,false);
        for(auto const* visible_scene:{&static_visible,&water_visible})
          for(unsigned layer:{unsigned(geometry_underlay),unsigned(geometry_natural_terrain),
                              unsigned(geometry_natural_mountain),unsigned(geometry_water)})
            if(!territory_borders.draw(renderer.device,context,(*visible_scene)[layer],settings,glow.linear,
                    renderer.content_view_width,renderer.content_view_height,projection_zoom,float(scene_scale),
                    [&](auto const& r){return !border_static(r) &&
                        renderer.chunk_intersects_region(GeometryDrawReference(r),settings,border_clip,false);}))
                return fail("territory_borders");
        QueryPerformanceCounter(&ticks[4]);c3x_renderer::render_core::gpu_timeline().mark(renderer.context,"water_dynamic");
#ifdef C3X_RENDERER64_FRESH
        gpu_phases.pass_end(renderer.context);gpu_phases.pass_begin(renderer.context,GpuPhases::body);
#endif
#ifdef C3X_RENDERER64_FRESH
        if(!sandbox_direct_units.draw_real(frame,unit_contribution_candidates,glow.linear,
                float(scene_scale),visual_hour,false,projection_zoom))return fail("real_units");
        // Combat effects over and under the bodies they touch (depth only).
        // A failed effect draw never costs the frame.
        sandbox_combat_effects.update(frame,sandbox_direct_units);
        if(!sandbox_combat_effects.draw(frame,sandbox_direct_units,glow.linear,float(scene_scale),visual_hour,projection_zoom))
            renderer.trace.write("combat-effects","draw failed",true);
#endif
        if(units && !sandbox_direct_units.draw(frame,unit_x,unit_y,
            incarnation,viewer,unit_visible,camera_x,camera_y,glow.linear,
            float(scene_scale),next_zoom,visual_hour))
            return fail("direct_units");
        QueryPerformanceCounter(&ticks[5]);c3x_renderer::render_core::gpu_timeline().mark(renderer.context,"units");
#ifdef C3X_RENDERER64_FRESH
        gpu_phases.pass_end(renderer.context);gpu_phases.pass_begin(renderer.context,GpuPhases::finish);
#endif
        if(!reconstruct())return fail("reconstruct");
        renderer.context->OMSetRenderTargets(0,nullptr,nullptr);
        QueryPerformanceCounter(&ticks[6]);c3x_renderer::render_core::gpu_timeline().mark(renderer.context,"reconstruct");
#ifdef C3X_RENDERER64_FRESH
        gpu_phases.pass_end(renderer.context);gpu_phases.end(renderer.context);
#endif
        for (int i=0;i<6;++i) phases[i]=1000.0*double(ticks[i+1].QuadPart-ticks[i].QuadPart)/
            double(frequency.QuadPart);
        return true;
    }
};

SandboxFreshPipeline sandbox_fresh;

extern "C" __declspec(dllexport) void c3x_sandbox_pass_counts(SandboxPassCounts* output) {
    if(output)*output=sandbox_fresh.work;
    std::fflush(stdout);
}

extern "C" __declspec(dllexport) void c3x_sandbox_fresh_metrics(double* phases,
        unsigned* visible,unsigned* shadows,unsigned* resident_builds,
        std::size_t* gpu_bytes,float* shadow_box) {
    if (phases) std::copy(sandbox_fresh.phases,sandbox_fresh.phases+6,phases);
    if (visible) *visible=sandbox_fresh.visible;
    if (shadows) *shadows=sandbox_fresh.shadow.builds;
    if (resident_builds) *resident_builds=sandbox_fresh.resident_builds;
    if (gpu_bytes) *gpu_bytes=
#ifdef C3X_RENDERER64_FRESH
        sandbox_fresh.instance_plan_bytes+sandbox_direct_units.gpu_preparation_bytes()+
#endif
        sandbox_fresh.water_parameters.gpu_bytes()+
        sandbox_fresh.static_cache.bytes()+
        sandbox_fresh.static_rasters.bytes()+
        sandbox_fresh.bootstrap[0].region.bytes()+sandbox_fresh.bootstrap[1].region.bytes()+
        sandbox_fresh.material_albedo.bytes()+
        std::size_t(sandbox_fresh.material_normal.width)*sandbox_fresh.material_normal.height*4+
        std::size_t(sandbox_fresh.material_world.width)*sandbox_fresh.material_world.height*8+
        sandbox_fresh.terrain_albedo.bytes()+
        std::size_t(sandbox_fresh.terrain_normal.width)*sandbox_fresh.terrain_normal.height*4+
        std::size_t(sandbox_fresh.terrain_world.width)*sandbox_fresh.terrain_world.height*8+
        std::size_t(sandbox_fresh.terrain_properties.width)*sandbox_fresh.terrain_properties.height*8+
        sandbox_fresh.reflected_terrain_albedo.bytes()+
        std::size_t(sandbox_fresh.reflected_terrain_normal.width)*
            sandbox_fresh.reflected_terrain_normal.height*4+
        std::size_t(sandbox_fresh.reflected_terrain_world.width)*
            sandbox_fresh.reflected_terrain_world.height*8+
        std::size_t(sandbox_fresh.reflected_terrain_properties.width)*
            sandbox_fresh.reflected_terrain_properties.height*8+
        std::size_t(sandbox_fresh.aquatic_scene.width)*
            sandbox_fresh.aquatic_scene.height*4+
        std::size_t(sandbox_fresh.water_lighting.width)*sandbox_fresh.water_lighting.height*8+
        sandbox_fresh.glow.linear.bytes()+
        sandbox_fresh.reflection.bytes()+sandbox_fresh.reflection_static.bytes()+
        sandbox_fresh.bloom.bytes()+sandbox_fresh.shadow.bytes();
    if (shadow_box) std::copy(sandbox_fresh.shadow.box,sandbox_fresh.shadow.box+4,shadow_box);
}

// Diagnostic snapshot only; ownership stays in the fixed pipeline slots.
extern "C" __declspec(dllexport) int c3x_sandbox_static_state_metrics(unsigned slot,
        c3x_renderer::render_core::StaticRasterMetrics* output) {
    if(!output || slot>=sandbox_fresh.static_rasters.states.size())return 0;
    auto const& state=sandbox_fresh.static_rasters.states[slot];
    *output=state.metrics;output->version=1;output->slot=slot;
    output->valid=unsigned(state.valid);output->projection=state.projection;
    output->region_revision=state.revision;output->region_bytes=state.region.bytes();
    output->total_region_bytes=sandbox_fresh.static_rasters.bytes();
    output->shared_viewport_bytes=sandbox_fresh.static_cache.bytes();
    output->sample_count=state.region.sample_count;
    std::size_t gpu_bytes=0;
    c3x_sandbox_fresh_metrics(nullptr,nullptr,nullptr,nullptr,&gpu_bytes,nullptr);
    output->total_gpu_bytes=gpu_bytes;return 1;
}

extern "C" __declspec(dllexport) void c3x_sandbox_cache_metrics(unsigned* depth_copies,
        unsigned* scrolls,unsigned* full_draws,unsigned* reflection_reuses,
        unsigned* reflection_draws,unsigned* pose_builds) {
    *depth_copies=sandbox_fresh.depth_copies;
    *scrolls=sandbox_fresh.cache_scrolls;
    *full_draws=sandbox_fresh.cache_full_draws;
    *reflection_reuses=sandbox_fresh.reflection_reuses;
    *reflection_draws=sandbox_fresh.reflection_draws;
    *pose_builds=sandbox_direct_units.pose_builds;
}

extern "C" __declspec(dllexport) int c3x_sandbox_prewarm_units(int hour,int season) {
#ifdef C3X_RENDERER64_FRESH
    // Synchronize the device and catalogue before adopting prewarmed meshes.
    c3x_renderer64_begin_unit_assets();
#endif
    return sandbox_direct_units.prewarm(hour,season)?0:1;
}

extern "C" __declspec(dllexport) void c3x_sandbox_combat_event(int serial,
        c3x_renderer_i64 presentation_ticks) {
    sandbox_direct_units.combat_event(serial,presentation_ticks);
}

// Untimed witnesses can force an independent raster without rebuilding world
// generations or rewinding animation. Ordinary production draws never call it.
extern "C" __declspec(dllexport) void c3x_sandbox_scroll_cache_invalidate() {
    sandbox_fresh.static_rasters.discard_all(c3x_renderer::render_core::raster_explicit_reset);
    for(auto& image:sandbox_fresh.bootstrap){image.valid=false;image.covered={};}
    sandbox_fresh.restore_key={};
    sandbox_fresh.reflection_valid=false;
    sandbox_fresh.reflected_terrain_material_valid=false;
}
extern "C" __declspec(dllexport) void c3x_sandbox_scroll_cache_metrics(unsigned* depth_copies,
        unsigned* scrolls,unsigned* full_draws,unsigned* reflection_reuses,
        unsigned* reflection_draws,unsigned* pose_builds) {
    c3x_sandbox_cache_metrics(depth_copies,scrolls,full_draws,reflection_reuses,reflection_draws,pose_builds);
}
extern "C" __declspec(dllexport) void c3x_sandbox_scroll_reasons(unsigned* values) {
    if(values)for(unsigned i=0;i<11;++i){values[i]=0;
        for(auto const& state:sandbox_fresh.static_rasters.states)values[i]+=state.metrics.reasons[i];}
}

extern "C" __declspec(dllexport) int c3x_sandbox_draw_fresh(
        c3x_renderer_frame_v1 const* frame,char const*,int camera_x,int camera_y,
        int unit_x,int unit_y,int incarnation,int viewer,int unit_visible,
        float zoom) {
    if (!frame) return 1;
    if (!sandbox_fresh.draw(*frame,camera_x,camera_y,unit_x,unit_y,
            incarnation,viewer,unit_visible!=0,zoom)) {
        std::printf("SANDBOX_FRESH_DRAW_ERROR visible=%u culled=%u shadow_builds=%u\n",
            sandbox_fresh.visible,sandbox_fresh.culled,sandbox_fresh.shadow.builds);
        std::fflush(stdout);
        return 2;
    }
    static bool reported=false;
    if (!reported) {
        std::printf("SANDBOX_FRESH_SCENE visible=%u culled=%u reflection=%u shadow_builds=%u shadow_draws=%u unit_draws=%u pose_builds=%u viewport=%dx%d\n",
            sandbox_fresh.visible,sandbox_fresh.culled,sandbox_fresh.reflection_count,
            sandbox_fresh.shadow.builds,sandbox_fresh.shadow.draws,
            sandbox_direct_units.draws,sandbox_direct_units.pose_builds,
            renderer.content_view_width,renderer.content_view_height);
        reported=true;
    }
    return 0;
}
