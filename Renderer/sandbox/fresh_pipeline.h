#pragma once
#include <fstream>
#include "pass_workload.h"
#include "scroll_region.h"
#include "static_raster_state.h"
#include "../native/render_core/raster_contributors.h"
#include "../native/render_core/shadow_sampling_grid.h"
#include "../native/render_core/body_placement_requirements.h"
#include "../native/render_core/prepared_draw_parameters.h"
#include "../native/render_core/submission_census.h"
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
    ID3D11PixelShader* water_surface = nullptr;
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
    ~SandboxVisualShaders() {
        if(water_surface) water_surface->Release();
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
 int layer=local_page.y*int(range.z)+local_page.x;
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
 [unroll]for(int y=-1;y<=1;y++)[unroll]for(int x=-1;x<=1;x++) {
  int2 sample=center+int2(x,y);
  float blocker=sandbox_shadow_load(field,sample);
  float receiver=z+dot(gradient,float2(sample)+.5-uv);
  sum+=step(blocker,receiver+(water?.00060:texel*.35));
 }
 return sum/9;
}
)");
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
    bool install() {
        if (installed) return true;
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
#endif
        installed = true;
        return true;
    }
};

// One light-space field for all receivers in the visible scene. The retained
// renderer supplies caster meshes, cutout textures and the light basis; the
// sandbox owns target selection, residency and the field's lifetime.
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
        struct Part {Shadow::Bounds bounds;unsigned first=0,count=0;};
        Shadow::Caster source;
        std::vector<Part> parts;

    };
    using Submission=c3x_renderer::render_core::SharedInstanceSubmission;
    Submission::Lease shared_front;
    Submission::CpuLease shared_metadata;
    std::uint64_t submission_generation=0;
    std::vector<InstanceGroup> instance_groups;
    std::vector<Shadow::Caster> casters;
    std::vector<Shadow::Caster> caster_inputs;
    using Membership=c3x_renderer::render_core::SceneMembership<CachedVertexChunk,geometry_layer_count>;
    Membership::Lease caster_lease;
    using AtlasInputs=c3x_renderer::render_core::RasterContributors<CachedGeometryProof,20>;
    AtlasInputs atlas_inputs;
    std::uint64_t prepared_signature=~std::uint64_t(0);
    std::array<float,4> prepared_box{};
    std::array<float,12> prepared_light{};
    std::vector<ID3D11Buffer*> patch_buffers;
    unsigned resolution=Grid::page_texels;
    bool ready=false;
    std::size_t production_field_bytes=0;
    std::size_t bytes()const{return (texture?Grid::texture_bytes:0)+(constants?80u:0)+production_field_bytes;}
    std::size_t metadata_bytes()const{return sizeof(sampling_grid)+sizeof(targets);}
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
    template<class T> static void drop(T*& pointer) {if (pointer) pointer->Release(); pointer=nullptr;}
    ~SandboxSceneShadow() {
        if(renderer.source_shadow.borrowed_view==view)renderer.source_shadow.borrowed_view=nullptr;
        renderer.fresh_shadow_working_bytes=0;
        drop(production_view);production_field_bytes=0;
        shared_front.reset();
        for(auto* buffer:patch_buffers)drop(buffer);
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
        casters.clear();instance_groups.clear();
        for(auto*& buffer:patch_buffers)drop(buffer);
        patch_buffers.clear();
        renderer.collect_shadow_casters(renderer.geometry_vertex_buffers,casters);
        // collect_shadow_casters already includes cliffs. Their old extra
        // submission duplicated identical maximum-height samples.
        std::unordered_set<AtlasInputs::Key,AtlasInputs::Hash> unique;
        casters.erase(std::remove_if(casters.begin(),casters.end(),[&](auto const& caster){
            return !unique.insert(caster_key(caster)).second;
        }),casters.end());
        caster_inputs=casters;
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
            Submission::Generation const* checked=nullptr,bool covered=false) {
        // Close one body/reflection/shadow placement union before any pass.
        // A serial identifies a new exact prepared caster selection; its input
        // equality is already checked by membership/box/light preparation.
        Submission::Key identity={caster_signature,renderer.device_generation,renderer.content_revision,
            renderer.geometry_vertex_buffers.revision(),++submission_generation,0,0,1};
        auto reused=renderer.shared_instances.find_covering([&](Submission::Generation const& candidate){
            if(candidate.identity[1]!=renderer.device_generation || candidate.identity[2]!=renderer.content_revision)return false;
            if(!(checked==&candidate?covered:body_placements_covered(candidate,inputs)))return false;
            for(auto const& caster:casters)if(caster.instances && !caster.instances->empty()){
                auto range=candidate.find(shared_caster_placement_key(caster));if(range.count!=caster.instances->size())return false;
            }return true;
        });
        // Completed pass selections pin the old placement generation.
        // Retire them before charging a replacement, while preserving
        // the active union and any independently held consumer leases.
        if(!reused)retire_completed_plans();
        if(!renderer.shared_instances.reserve_index_scratch())return false;
        shared_metadata.reset();std::size_t parts=0;
        for(auto const& caster:casters)if(caster.instances && !caster.instances->empty())++parts;
        auto metadata=renderer.shared_instances.retain_metadata(parts*(sizeof(InstanceGroup)*2+sizeof(InstanceGroup::Part)*2+sizeof(std::array<std::uintptr_t,9>)+96u));
        if(!metadata)return false;
        // Packed placements own no draw sources. The current resident/caster
        // publications below each pass already own the actually bound meshes.
        auto builder=reused?Submission::Builder{}:renderer.shared_instances.begin_retained(identity,{},true);
        if(!reused && !builder)return false;
        if(!reused && !append_body_placements(builder,inputs))return false;
        // Only the required current membership owns source meshes. Reusable
        // placement ranges retain weak, epoch-protected identities instead.
        std::map<std::uint64_t,c3x_renderer::render_core::ContentHandle> source_handles;
        Submission::CpuLease source_handle_metadata;
        if(!reused){
            using SourceHandle=decltype(source_handles)::value_type;
            source_handle_metadata=renderer.shared_instances.retain_metadata(sizeof(source_handles)+
                caster_lease->content.size()*(sizeof(SourceHandle)+64u));
            if(!source_handle_metadata)return false;
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
        if(!reused){bool retained=true;
            inputs.visit([&](unsigned,Submission::Key const& key,GeometryDrawRecord const& record,unsigned){
                auto draw=GeometryDrawReference(record);auto const& mesh=draw.content();if(!retained || !draw.occurrence)return;
                retained=retain_source(key,draw.occurrence->owner,
                    mesh.instances.get(),mesh.instance_material);
            });if(!retained)return false;
        }
        std::map<std::array<std::uintptr_t,9>,std::size_t> lookup;
        std::size_t total=0;
        for(auto const& caster:casters){
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
                        if(builder->complete)return false;
                    }
                }
            }
            if(!reused && !range && !renderer.shared_instances.append(builder,placement_key,caster.instances,caster.instances->data(),
                unsigned(caster.instances->size()),caster.instances->front().projection,
                caster.offset[0],caster.offset[1],caster.offset[2],caster.instance_material,range))return false;
            if(!reused && caster.content_generation){
                auto owner=source_handles.find(caster.content_generation);
                if(owner!=source_handles.end() && !retain_source(placement_key,owner->second,caster.instances,caster.instance_material))return false;
            }
            InstanceGroup::Part part;part.first=range.first;part.count=range.count;
            for(int axis=0;axis<3;++axis){part.bounds.low[axis]=caster.bounds.low[axis]+caster.offset[axis];
                part.bounds.high[axis]=caster.bounds.high[axis]+caster.offset[axis];}
            group.parts.push_back(part);total+=range.count;
        }
        if(!reused && !renderer.shared_instances.carry_forward(builder,[&](Submission::RetainedSource const& proof){
            auto source=proof.source.lock();
            auto cached=renderer.resident_content.resolve({std::size_t(proof.owner[0]),proof.owner[1]});
            return source && cached && cached->mesh && cached->mesh->proof && cached->mesh.get()==source.get() &&
                renderer.raster_content_valid(*cached->mesh->proof);
        }))return false;
        auto uploads=renderer.shared_instances.uploads;auto bytes=renderer.shared_instances.uploaded_bytes;
        shared_front=reused?reused:renderer.shared_instances.upload(builder,renderer.device,renderer.context);
        if(!shared_front)return false;
        shared_metadata=std::move(metadata);
        renderer.frame_content_uploads+=renderer.shared_instances.uploads-uploads;
        renderer.frame_upload_bytes+=renderer.shared_instances.uploaded_bytes-bytes;
        work->upload(renderer.shared_instances.uploaded_bytes-bytes);
        std::printf("SANDBOX_SHADOW_INSTANCES casters=%zu groups=%zu instances=%zu bytes=%zu\n",
            casters.size(),instance_groups.size(),total,total*sizeof(unsigned));std::fflush(stdout);return true;
    }
    void batch_terrain_casters() {
        auto& captured=renderer.sandbox_shadow_meshes;
        if(captured.empty())return;
        auto selected=[](unsigned layer){return layer==geometry_land ||
            layer==geometry_natural_terrain || layer==geometry_natural_mountain;};
        std::map<std::pair<unsigned,std::array<float,6>>,unsigned> prepared_bounds;
        for(auto const& entry:captured){
            auto const& mesh=entry.second;
            prepared_bounds[{std::get<2>(entry.first),{mesh.world_low[0],mesh.world_low[1],mesh.world_low[2],
                mesh.world_high[0],mesh.world_high[1],mesh.world_high[2]}}]++;
        }
        unsigned original=0;
        bool matched=true;
        for(auto const& caster:casters)if(selected(caster.layer) &&
            caster.offset[0]==0 && caster.offset[1]==0){
            ++original;
            std::array<float,6> bounds={caster.bounds.low[0],caster.bounds.low[1],caster.bounds.low[2],
                caster.bounds.high[0],caster.bounds.high[1],caster.bounds.high[2]};
            auto found=prepared_bounds.find({caster.layer,bounds});
            if(found==prepared_bounds.end() || !found->second)matched=false;
            else --found->second;
        }
        for(auto const& entry:prepared_bounds)if(entry.second)matched=false;
        if(original!=captured.size() || !matched){
            std::printf("SANDBOX_TERRAIN_BATCH fallback captured=%zu resident=%u\n",captured.size(),original);
            captured.clear();return;
        }
        struct Region {
            std::vector<std::uint8_t> vertices;
            std::vector<std::uint32_t> indices;
            Shadow::Bounds bounds;
            unsigned stride=0,layer=0;
            bool filled=false;
        };
        std::map<std::tuple<int,int,unsigned>,Region> regions;
        for(auto const& entry:captured){
            auto const& mesh=entry.second;
            if(!mesh.vertex_stride || (mesh.index_stride!=2 && mesh.index_stride!=4) ||
                mesh.vertices.size()%mesh.vertex_stride ||
                mesh.indices.size()!=std::size_t(mesh.index_count)*mesh.index_stride){
                std::printf("SANDBOX_TERRAIN_BATCH fallback invalid mesh\n");captured.clear();return;
            }
            auto& region=regions[{std::get<0>(entry.first)/8,std::get<1>(entry.first)/8,
                std::get<2>(entry.first)}];
            if(region.stride && region.stride!=mesh.vertex_stride){
                std::printf("SANDBOX_TERRAIN_BATCH fallback mixed vertex strides\n");captured.clear();return;
            }
            region.stride=mesh.vertex_stride;
            region.layer=std::get<2>(entry.first);
            auto base=region.vertices.size()/region.stride;
            region.vertices.insert(region.vertices.end(),mesh.vertices.begin(),mesh.vertices.end());
            for(unsigned i=0;i<mesh.index_count;++i){
                std::uint32_t index=0;
                std::memcpy(&index,mesh.indices.data()+std::size_t(i)*mesh.index_stride,mesh.index_stride);
                region.indices.push_back(std::uint32_t(base+index));
            }
            for(int axis=0;axis<3;++axis){
                if(!region.filled){region.bounds.low[axis]=mesh.world_low[axis];
                    region.bounds.high[axis]=mesh.world_high[axis];}
                else {region.bounds.low[axis]=std::min(region.bounds.low[axis],mesh.world_low[axis]);
                    region.bounds.high[axis]=std::max(region.bounds.high[axis],mesh.world_high[axis]);}
            }
            region.filled=true;
        }
        auto dims=renderer.world_coast.world().dimensions();
        std::vector<ID3D11Buffer*> created;
        std::vector<Shadow::Caster> replacements;
        bool okay=true;
        for(auto const& entry:regions){
            auto const& region=entry.second;
            if(region.vertices.empty() || region.indices.empty() ||
                region.vertices.size()>std::numeric_limits<UINT>::max() ||
                region.indices.size()>std::numeric_limits<UINT>::max()/4u){okay=false;break;}
            D3D11_BUFFER_DESC desc={};desc.Usage=D3D11_USAGE_IMMUTABLE;
            desc.BindFlags=D3D11_BIND_VERTEX_BUFFER;desc.ByteWidth=UINT(region.vertices.size());
            D3D11_SUBRESOURCE_DATA initial={region.vertices.data(),0,0};
            ID3D11Buffer *vertices=nullptr,*indices=nullptr;
            if(FAILED(renderer.device->CreateBuffer(&desc,&initial,&vertices))){okay=false;break;}
            desc.BindFlags=D3D11_BIND_INDEX_BUFFER;desc.ByteWidth=UINT(region.indices.size()*4u);
            initial.pSysMem=region.indices.data();
            if(FAILED(renderer.device->CreateBuffer(&desc,&initial,&indices))){vertices->Release();okay=false;break;}
            created.push_back(vertices);created.push_back(indices);
            for(int wy=dims.wrap_y?-1:0;wy<=(dims.wrap_y?1:0);++wy)
                for(int wx=dims.wrap_x?-1:0;wx<=(dims.wrap_x?1:0);++wx){
                    Shadow::Caster caster;
                    caster.vertices=vertices;caster.indices=indices;caster.count=UINT(region.indices.size());
                    caster.stride=region.stride;caster.layer=region.layer;
                    caster.index_format=DXGI_FORMAT_R32_UINT;caster.bounds=region.bounds;
                    caster.offset[0]=float(wx*dims.width+wy*dims.height)*.5f;
                    caster.offset[1]=float(wx*dims.width-wy*dims.height)*.5f;
                    replacements.push_back(caster);
                }
        }
        captured.clear();
        if(!okay){
            for(auto* buffer:created)buffer->Release();
            std::printf("SANDBOX_TERRAIN_BATCH fallback GPU allocation\n");return;
        }
        casters.erase(std::remove_if(casters.begin(),casters.end(),[&](Shadow::Caster const& caster){
            return selected(caster.layer);
        }),casters.end());
        casters.insert(casters.end(),replacements.begin(),replacements.end());
        patch_buffers=std::move(created);
        std::printf("SANDBOX_TERRAIN_BATCH meshes=%u regions=%zu draws=%zu\n",
            original,regions.size(),replacements.size());std::fflush(stdout);
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
        auto exact=[&](){
        if(!append && !atlas_inputs.valid([&](auto const& proof){return renderer.raster_content_valid(proof);},
            [&](auto tile){auto record=renderer.topology_cache.retained(tile);return record?record->visibility_revision:0;}))return false;
        bool valid=true;if(!append)atlas_inputs.begin_membership();
        for(auto const& caster:caster_inputs){
            ++atlas_inputs.validation_counts.membership;
            auto p=Shadow::project(caster.bounds,caster.offset,renderer.shadow_basis);
            if(p[2]<box[0] || p[3]<box[1] || p[0]>box[0]+box[2] || p[1]>box[1]+box[3])continue;
            auto key=caster_key(caster);
            if(!append){if(!atlas_inputs.visit_membership(key)){valid=false;++renderer.raster_proof_rejections[5];}continue;}
            auto mesh=std::static_pointer_cast<CachedMeshGeneration>(caster_lease->content.get({0,caster.content_generation}));
            auto tile=mesh?mesh->proof->tile:0;
            auto observed=renderer.topology_cache.retained(tile);
            bool new_proof=false;
            valid=atlas_inputs.add(key,mesh?mesh->proof:nullptr,tile,observed?observed->visibility_revision:0,&new_proof) && valid;
            if(new_proof)valid=renderer.watch_raster_dependencies(*mesh->proof,atlas_inputs)&&valid;
        }
        if(append)atlas_inputs.finish_dependencies();
        else if(!atlas_inputs.exact_membership()){valid=false;++renderer.raster_proof_rejections[5];}
        return valid;
        };
        if(append){auto begin=std::chrono::steady_clock::now();bool valid=exact();
            atlas_inputs.validation_counts.append_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();return valid;}
        AtlasInputs::ValidationKey key={caster_signature,renderer.topology_cache.scope_sequence(),renderer.content_revision,
            renderer.device_generation,unsigned(renderer.geometry_canonical_world)};
        auto bits=[](float value){std::uint32_t result;std::memcpy(&result,&value,sizeof(result));return std::uint64_t(result);};
        for(unsigned i=0;i<4;++i)key[5+i]=bits(box[i]);
        for(unsigned i=0;i<renderer.shadow_basis.size();++i)key[9+i]=bits(renderer.shadow_basis[i]);
        return atlas_inputs.validate(renderer.raster_dependency_revisions,key,exact);
#else
        return true;
#endif
    }
    std::array<std::uint64_t,9> receiver_identity(std::uint64_t revision,std::uint64_t scene)const{
        auto dims=renderer.world_coast.world().dimensions();
        return {revision,scene,renderer.content_revision,renderer.device_generation,unsigned(renderer.geometry_canonical_world),
            std::uint64_t(dims.width),std::uint64_t(dims.height),unsigned(dims.wrap_x),unsigned(dims.wrap_y)};
    }
    template<class BodyInputs,class RetireCompletedPlans> bool render(GeometryDrawView::Records const& receivers,std::uint64_t scene,std::uint64_t revision,std::uint64_t membership,BodyInputs const& inputs,RetireCompletedPlans const& retire_completed_plans) {
        SandboxPassWorkload::Scope pass(*work,SandboxPassWorkload::shadow);
        if (!ensure() || !refresh_casters(membership)) return false;
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
        if (!any) return false;
        if(!query_grid.configure(needed,renderer.shadow_basis))return false;
        receiver_grid=query_grid;receiver_wrap=wrap_query;receiver_key=next_receiver_key;receiver_light=renderer.shadow_basis;
        receiver_grid_valid=true;++receiver_builds;
        }
        receiver_revision=revision;
        bool body_covered=renderer.shared_instances.valid(shared_front) && inputs.covers(shared_front);
        if (renderer.shared_instances.valid(shared_front) && prepared_signature==membership &&
            signature==scene && wrap_basis==wrap_query &&
            light_basis==renderer.shadow_basis &&
            sampling_grid.covers(query_grid) && atlas_dependencies(false) && body_covered){if(work->enabled)++work->row().reuses;return true;}
        if(work->enabled)++work->row().rebuilds;
        sampling_grid=query_grid;
        auto coverage=sampling_grid.coverage();
        std::copy(coverage.begin(),coverage.end(),box);
        atlas_inputs.clear();if(!atlas_dependencies(true))atlas_inputs.complete=false;
        std::array<float,4> next_box={box[0],box[1],box[2],box[3]};
        if(!renderer.shared_instances.valid(shared_front) || prepared_signature!=membership || prepared_box!=next_box || prepared_light!=renderer.shadow_basis ||
                !body_covered){
            instance_groups.clear();
            ID3D11Buffer* empty[]={nullptr,nullptr};UINT cleared_strides[2]={};
            renderer.context->IASetVertexBuffers(0,2,empty,cleared_strides,cleared_strides);
            renderer.context->IASetIndexBuffer(nullptr,DXGI_FORMAT_UNKNOWN,0);
            for(auto*& buffer:patch_buffers)drop(buffer);patch_buffers.clear();
            casters.clear();
            for(auto const& caster:caster_inputs){auto p=Shadow::project(caster.bounds,caster.offset,renderer.shadow_basis);
                if(p[2]<box[0] || p[3]<box[1] || p[0]>box[0]+box[2] || p[1]>box[1]+box[3])continue;
                casters.push_back(caster);
            }
            batch_terrain_casters();
            if(!prepare_instances(inputs,retire_completed_plans,shared_front.get(),body_covered))return false;
            prepared_signature=membership;prepared_box=next_box;prepared_light=renderer.shadow_basis;
        }
        auto* context=renderer.context;
        std::array<ID3D11ShaderResourceView*,128> empty{};
        context->PSSetShaderResources(0,128,empty.data());
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetConstantBuffers(0,1,&constants);
        context->RSSetState(raster);
        D3D11_VIEWPORT viewport={0,0,float(resolution),float(resolution),0,1};context->RSSetViewports(1,&viewport);
        context->OMSetDepthStencilState(nullptr,0);
        context->OMSetBlendState(maximum,nullptr,0xffffffffu);
        float clear[4]={-1e6f,-1e6f,-1e6f,-1e6f};
        draws=0;
        for(unsigned page_slot=0;page_slot<sampling_grid.pages();++page_slot){
        auto* target=targets[page_slot];
        context->ClearRenderTargetView(target,clear);work->clear(target);
        context->OMSetRenderTargets(1,&target,nullptr);
        auto page_box=sampling_grid.page_box(page_slot);
        auto page= sampling_grid.page(page_slot);
        for (auto const& caster:casters) {
            if(caster.instances)continue;
            if(work->enabled)++work->row(caster.layer).tested_records;
            auto bounds=Shadow::project(caster.bounds,caster.offset,renderer.shadow_basis);
            if(bounds[2]<page_box[0] || bounds[0]>page_box[0]+page_box[2] ||
               bounds[3]<page_box[1] || bounds[1]>page_box[1]+page_box[3])continue;
            if(work->enabled)++work->row(caster.layer).accepted_records;
            float settings[20]={};
            std::copy(renderer.shadow_basis.begin(),renderer.shadow_basis.end(),settings);
            for (int i=0;i<3;++i) {
                settings[i]*=6/page_box[2];settings[4+i]*=6/page_box[3];
                settings[16+i]=caster.offset[i];
            }
            settings[12]=float(page[0]);settings[13]=float(page[1]);
            context->UpdateSubresource(constants,0,nullptr,settings,0,0);work->upload_buffer(constants);
            context->PSSetShader(bind_cutout(caster.binding==0xffffffffu?
                caster.layer:caster.binding)?cutout:opaque,nullptr,0);
            context->VSSetShader(vertex,nullptr,0);
            context->IASetInputLayout(caster.stride==88?city_layout:
                caster.stride==92?natural_layout:caster.stride==48?feature_layout:layout);
            UINT stride=caster.stride,offset=caster.vertex_offset;
            context->IASetVertexBuffers(0,1,&caster.vertices,&stride,&offset);
            context->IASetIndexBuffer(caster.indices,caster.index_format,caster.index_offset);
            context->DrawIndexed(caster.count,0,0);work->draw(caster.count,1,caster.layer);
            ++draws;
        }
        std::vector<unsigned> selected;
        for(auto& group:instance_groups){
            selected.clear();
            for(auto const& part:group.parts){
                if(work->enabled){++work->row(group.source.layer).tested_records;work->row(group.source.layer).tested_instances+=part.count;}
                auto bounds=Shadow::project(part.bounds,zero,renderer.shadow_basis);
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
            context->VSSetShaderResources(15,1,&shared_front->view);
            context->IASetIndexBuffer(caster.indices,caster.index_format,caster.index_offset);
            for(std::size_t begin=0;begin<selected.size();begin+=Submission::record_limit){
                auto count=std::min(selected.size()-begin,std::size_t(Submission::record_limit));
                if(!renderer.shared_instances.select_indices(renderer.device,context,shared_front,selected.data()+begin,unsigned(count)))return false;
                ID3D11Buffer* streams[]={caster.vertices,renderer.shared_instances.selection_buffer};
                UINT strides[]={32,4},offsets[]={caster.vertex_offset,renderer.shared_instances.selection_offset};
                context->IASetVertexBuffers(0,2,streams,strides,offsets);
                context->DrawIndexedInstanced(caster.count,UINT(count),0,0,0);work->draw(caster.count,count,caster.layer);
                work->upload(count*sizeof(unsigned),caster.layer);
                ++draws;
            }
        }
        }
        context->OMSetRenderTargets(0,nullptr,nullptr);
        std::array<std::array<float,4>,64> table{};
        table[0]={float(sampling_grid.low[0]),float(sampling_grid.low[1]),
            float(sampling_grid.count[0]),float(sampling_grid.count[1])};
        table[2]={sampling_grid.inverse_pitch(0),sampling_grid.inverse_pitch(1),
            sampling_grid.page_span(0),sampling_grid.page_span(1)};
        table[1]=wrap_query;
        wrap_basis=wrap_query;
        context->UpdateSubresource(renderer.source_shadow.table,0,nullptr,table.data(),0,0);work->upload_buffer(renderer.source_shadow.table);
        signature=scene;
        light_basis=renderer.shadow_basis;
        ++builds;
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
    ID3D11Buffer* aquatic_bounds_buffer=nullptr;
    ID3D11BlendState* terrain_material_blend=nullptr;
    ID3D11DepthStencilState* aquatic_depth=nullptr;
    c3x_renderer::render_core::LinearRestore static_restore;
    c3x_renderer::render_core::DrawParameterStream parameters;
    using PreparedParameters=c3x_renderer::render_core::PreparedDrawParameters<ViewportShaderSettings,geometry_layer_count>;
    PreparedParameters water_parameters;
    c3x_renderer::render_core::SubmissionCensus<std::array<std::uint64_t,12>> submission_census;
    SandboxMirrorTarget reflection,reflection_static;
    GeometryDrawView::Records resident, static_visible, water_visible,
        reflection_visible, all_visible;
    using Membership=c3x_renderer::render_core::SceneMembership<CachedVertexChunk,geometry_layer_count>;
    Membership::Lease resident_lease;
    using RasterInputs=c3x_renderer::render_core::RasterContributors<CachedGeometryProof>;
    std::array<RasterInputs,2> raster_inputs;
    float resident_basis_x=0,resident_basis_y=0;
    std::vector<RasterInputs::Key> reflection_inputs;
    std::uint64_t static_receiver_revision=0;
    std::uint64_t reflection_revision=0;
    std::uint64_t resident_signature=0;
    std::uint64_t resident_order_revision=0;
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
    bool reflected_terrain_material_valid=false;
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
    };
    StaticValidationCounts static_validation_counts()const{
        StaticValidationCounts result;
        for(auto const& inputs:raster_inputs){auto const& counts=inputs.validation_counts;
            result.raster.full+=counts.full;result.raster.content+=counts.content;result.raster.visibility+=counts.visibility;
            result.raster.membership+=counts.membership;result.raster.regions+=counts.regions;result.raster.reused+=counts.reused;result.raster.changes+=counts.changes;
            result.raster.proof_registrations+=counts.proof_registrations;result.raster.dependency_watch_calls+=counts.dependency_watch_calls;
            result.raster.source_expansions+=counts.source_expansions;result.raster.source_reuses+=counts.source_reuses;result.raster.append_ms+=counts.append_ms;
        }
        result.atlas=shadow.atlas_inputs.validation_counts;result.receiver_visits=shadow.receiver_visits;
        result.receiver_builds=shadow.receiver_builds;result.receiver_reuses=shadow.receiver_reuses;
        result.placement_probes=body_requirements.coverage_total_probes;result.placement_reuses=body_requirements.coverage_total_reuses;
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
    bool capture(ViewportShaderSettings const& settings,
            ViewportShaderSettings const& reflected,int width,int height,int next_wrap_pixels) {
#ifdef C3X_RENDERER64_FRESH
        // A set proof can certify additions/removals, but alpha pixels also
        // depend on primitive order. Retire both pixel owners once after an
        // actual boundary reorder; unchanged camera frames only compare epochs.
        auto order=renderer.geometry_vertex_buffers.order_revision();
        if(resident_order_revision!=order){
            static_rasters.invalidate_all(c3x_renderer::render_core::raster_scene);
            resident_order_revision=order;
        }
#endif
        if (resident_signature!=view_revision() ||
                wrap_pixels!=next_wrap_pixels) {
#ifndef C3X_RENDERER64_FRESH
            static_rasters.invalidate_all(c3x_renderer::render_core::raster_scene);
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
                view_key==visibility_view_key){if(work.enabled)++work.counts[SandboxPassWorkload::selection][SandboxPassWorkload::screen].reuses;return visible>0;}
        if(work.enabled)++work.counts[SandboxPassWorkload::selection][SandboxPassWorkload::screen].rebuilds;
        static_visible={};water_visible={};reflection_visible={};
        auto prior_receivers=std::move(all_visible);all_visible={};
        visible=culled=reflection_count=0;
        D3D11_RECT rect=source_bounds(settings,{0,0,width,height},false);
        contributors(settings,rect,false,[&](unsigned layer,auto const& record) {
                if(work.enabled)++work.counts[SandboxPassWorkload::selection][layer].tested_records;
                if (!renderer.chunk_intersects_region(GeometryDrawReference(record),settings,rect,false)) {
                    ++culled;return;
                }
                auto& output=renderer.water_scene_active && record.water_dependent?
                    water_visible:static_visible;
                if(work.enabled)++work.counts[SandboxPassWorkload::selection][layer].accepted_records;
                output[layer].push_back(record);
                all_visible[layer].push_back(record);++visible;
            });
        auto mirror=source_bounds(reflected,reflected_water_bounds(settings,width,height),true);
        contributors(reflected,mirror,true,[&](unsigned layer,auto const& record) {
            if (layer==geometry_underlay || layer==geometry_bed ||
                layer==geometry_water || layer==geometry_river ||
                layer==geometry_route || layer==geometry_shadow || layer==geometry_wave)
                return;
                if (record.content().animation_texture ||
                    !renderer.chunk_intersects_region(GeometryDrawReference(record),
                        reflected,mirror,true)) return;
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
        if(!same_receivers)++static_receiver_revision;
        visibility_scene_key=scene_key;visibility_view_key=view_key;
        visibility_valid=true;++visibility_revision;
        return visible>0;
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
        bool valid=true;if(!append)inputs.begin_membership();auto clip=source_bounds(settings,rect,false);
        contributors(settings,clip,false,[&](unsigned layer,auto const& record){
            ++inputs.validation_counts.membership;
            if(renderer.water_scene_active && record.water_dependent)return;
            if(!renderer.chunk_intersects_region(GeometryDrawReference(record),settings,clip,false))return;
            auto key=contributor_key(layer,record);
            if(!append){if(!inputs.visit_membership(key)){valid=false;++renderer.raster_proof_rejections[5];}return;}
            auto mesh=std::static_pointer_cast<CachedMeshGeneration>(resident_lease->content.get(record.owner));
            auto tile=renderer.topology_cache.key(record.tile_x,record.tile_y);
            auto observed=renderer.topology_cache.retained(tile);
            bool new_proof=false;
            valid=inputs.add(key,mesh?mesh->proof:nullptr,tile,observed?observed->visibility_revision:0,&new_proof) && valid;
            if(new_proof)valid=renderer.watch_raster_dependencies(*mesh->proof,inputs)&&valid;
            valid=inputs.watch(RasterInputs::Revisions::Domain::visibility,tile)&&valid;
        });
        if(append)inputs.finish_dependencies();
        else if(!inputs.exact_membership()){valid=false;++renderer.raster_proof_rejections[5];}
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
        bool region_target=target==static_region().target ||
            target==material_albedo.target || target==terrain_albedo.target;
        D3D11_VIEWPORT viewport={0,0,float(mirrored?reflection.width:
                region_target?static_region().width:glow.linear.width),
            float(mirrored?reflection.height:
                region_target?static_region().height:glow.linear.height),0,1};
        c3x_renderer::SceneProjection(renderer.content_view_width,renderer.content_view_height,projection_zoom)
            .viewport(viewport,mirrored?8.f:4.f,region_target?float(region_margin_x):0.f,
                region_target?float(region_margin_y):0.f,scale);
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
                left.cursor==right.cursor && left.predict==right.predict && left.animated==right.animated && left.travelling==right.travelling;
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
        char cycle[8]{};bool day_night=GetEnvironmentVariableA("C3X_SANDBOX_DAY_NIGHT",cycle,sizeof(cycle)) && cycle[0]=='1';
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
        char emission_control[8]={};
        bool original_emission=GetEnvironmentVariableA("C3X_SANDBOX_CITY_SUBMISSION_REFERENCE",emission_control,sizeof(emission_control)) && emission_control[0]=='1';
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
                    settings.depth_translation=renderer.city_profile?viewport.depth_translation+float(chunk.translation_y()):settings.translation[1];generated_values[i]=settings;
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
            for(unsigned i=0;i<selected.size();++i){auto const& chunk=selected[i];if(!chunk.content().rigid_source)continue;
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
                        context->UpdateSubresource(renderer.viewport_settings_buffer,0,nullptr,&viewport,0,0);work.upload_buffer(renderer.viewport_settings_buffer);
                        previous=viewport;previous_valid=true;
                    }context->VSSetConstantBuffers(1,1,&renderer.viewport_settings_buffer);
                }else if(prepared)prepared->bind(parameters.context,1,i);
                else if(streamed)parameters.bind(1,parameter_index[i]);
                else if(!previous_valid || std::memcmp(&previous,&settings,sizeof(settings))){
                    context->UpdateSubresource(renderer.viewport_settings_buffer,0,nullptr,&settings,0,0);work.upload_buffer(renderer.viewport_settings_buffer);
                    previous=settings;previous_valid=true;
                }
                if(packets[i]){
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
                if(mesh.city_material!=0xffffffffu){
                    ID3D11SamplerState* samplers[]={renderer.natural_wrap,renderer.natural_clamp};
                    context->PSSetSamplers(0,2,samplers);
                    renderer.cities.bind(context,mesh.city_material,mesh.city_environment,
                        mesh.city_atlas,mirrored,false,stride==88);
                    context->DrawIndexed(mesh.index_count,0,0);work.draw(mesh.index_count,1,unsigned(layer));++renderer.frame_draw_calls;
                    if(!renderer.cities.library.materials[mesh.city_material].ground &&
                       (original_emission || renderer.cities.emits(mesh.city_material))){
                        if(original_emission)renderer.cities.bind(context,mesh.city_material,mesh.city_environment,
                            mesh.city_atlas,mirrored,true,stride==88);
                        else renderer.cities.bind_emission(context,mirrored);
                        context->DrawIndexed(mesh.index_count,0,0);work.draw(mesh.index_count,1,unsigned(layer));++renderer.frame_draw_calls;
                    }
                    context->OMSetBlendState(renderer.blend_state,nullptr,0xffffffffu);
                    context->OMSetDepthStencilState(renderer.depth_state,0);
                    ++i;continue;
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
                    ++phase_constant_counts.water_records;
                    if(!last_water_valid || !same_water(sample,last_water)){
                        context->UpdateSubresource(renderer.water_frame,0,nullptr,&sample,0,0);work.upload_buffer(renderer.water_frame);
                        last_water=sample;last_water_valid=true;++phase_constant_counts.water_updates;
                    }else ++phase_constant_counts.water_hits;
                    if(!water_bound){context->PSSetConstantBuffers(10,1,&renderer.water_frame);water_bound=true;}
                }
                if(layer==geometry_wave){
                    std::array<float,4> sample={mesh.visual_time<0?renderer.wave_time_seconds:mesh.visual_time,0,0,0};
                    bool same=last_wave_valid;
                    for(unsigned j=0;same && j<sample.size();++j)same=same_float(sample[j],last_wave[j]);
                    ++phase_constant_counts.wave_records;
                    if(!same){
                        context->UpdateSubresource(renderer.wave_frame,0,nullptr,sample.data(),0,0);work.upload_buffer(renderer.wave_frame);
                        last_wave=sample;last_wave_valid=true;++phase_constant_counts.wave_updates;
                    }else ++phase_constant_counts.wave_hits;
                }
                context->DrawIndexed(mesh.index_count,0,0);work.draw(mesh.index_count,1,unsigned(layer));++renderer.frame_draw_calls;
                if(mesh.resource_instance){
                    context->IASetInputLayout(layer==geometry_shadow?
                        renderer.input_layout:renderer.feature_input_layout);
                    context->VSSetShader(layer==geometry_shadow?
                        renderer.vertex_shader:renderer.feature_vertex_shader,nullptr,0);
                }
                if(mesh.animation_texture)context->PSSetShaderResources(116,1,
                    renderer.resource_texture_views.data());
                ++i;
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
        context->OMSetRenderTargets(1,&target,depth);
        context->OMSetDepthStencilState(renderer.depth_state,0);
        context->OMSetBlendState(renderer.blend_state,nullptr,0xffffffffu);
        context->RSSetState(renderer.rasterizer_state);
        bool region_target=target==static_region().target ||
            target==material_albedo.target || target==terrain_albedo.target;
        D3D11_VIEWPORT viewport={0,0,float(mirrored?reflection.width:
                region_target?static_region().width:glow.linear.width),
            float(mirrored?reflection.height:
                region_target?static_region().height:glow.linear.height),0,1};
        c3x_renderer::SceneProjection(renderer.content_view_width,renderer.content_view_height,projection_zoom)
            .viewport(viewport,mirrored?8.f:4.f,region_target?float(region_margin_x):0.f,
                region_target?float(region_margin_y):0.f,scale);
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
            unsigned(reflection.height/reflection_scale),4,4);
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
        context->UpdateSubresource(renderer.viewport_settings_buffer,0,nullptr,&settings,0,0);work.upload_buffer(renderer.viewport_settings_buffer);
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
            if(!cached.valid || !renderer.shared_instances.valid(plan.selection) || plan.selection->content!=shared_front){
                std::vector<unsigned> prepared;prepared.reserve(count);
                for(auto const* record:selected){auto draw=GeometryDrawReference(*record);auto range=shared_front->find(renderer.shared_instance_draw_key(unsigned(layer),draw));
                    if(!range || range.count!=record->content().instances->size())return renderer.reject_shared_instance_range(unsigned(layer),draw,shared_front,range);
                    for(unsigned i=0;i<range.count;++i)prepared.push_back(range.first+i);}
                auto candidate=renderer.shared_instances.prepare_selection(renderer.device,shared_front,prepared.data(),unsigned(prepared.size()),
                    key.capacity()*sizeof(key[0])+sizeof(typename decltype(instance_plans)::Entry));
                if(candidate){instance_plan_bytes=instance_plan_bytes-plan.bytes+candidate->bytes();plan.selection=std::move(candidate);
                    plan.bytes=plan.selection->bytes();plan.count=unsigned(count);cached.valid=true;++instance_plan_builds;work.upload(count*sizeof(unsigned),unsigned(layer));}
                else cached.valid=false;
            }else ++instance_plan_reuses;
            if(cached.valid){
                context->IASetInputLayout(renderer.natural.resident_instance_layout);
                context->VSSetShader(visual.vegetation_instances[mirrored?1:0],nullptr,0);context->VSSetShaderResources(15,1,&shared_front->view);
                ID3D11Buffer* streams[]={mesh.buffer,plan.selection->buffer};UINT strides[]={32,4},offsets[]={0,0};
                context->IASetVertexBuffers(0,2,streams,strides,offsets);context->IASetIndexBuffer(mesh.indices,mesh.index_format,0);
                char skip_depth[8]{};
                if(!(GetEnvironmentVariableA("C3X_SANDBOX_SKIP_VEGETATION_DEPTH",skip_depth,sizeof(skip_depth)) && skip_depth[0]=='1')){
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
            if(!(GetEnvironmentVariableA("C3X_SANDBOX_SKIP_VEGETATION_DEPTH",skip_depth,sizeof(skip_depth)) && skip_depth[0]=='1')){
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
        if(layer>=geometry_natural_forest0 && GetEnvironmentVariableA("C3X_SANDBOX_SKIP_VEGETATION",skip_vegetation,sizeof(skip_vegetation)) && skip_vegetation[0]=='1')return true;
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
                if(!GetEnvironmentVariableA("C3X_SANDBOX_PROFILE_COMPLETION",option,sizeof(option)) || option[0]!='1')return;
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
            if(!(GetEnvironmentVariableA("C3X_SANDBOX_WATER_FULL_SHADER",
                    full_water,sizeof(full_water)) && std::strcmp(full_water,"1")==0))
                context->PSSetShader(visual.water_surface,nullptr,0);
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
        }else for(auto layer:{geometry_natural_terrain,geometry_natural_mountain,
                              geometry_natural_decal})
            if(!draw(layer))return false;
        for(unsigned layer=geometry_natural_forest0;layer<geometry_layer_count;++layer)
            if(!draw(static_cast<GeometryLayer>(layer)))return false;
        if(!mirrored) {
            if(!draw(geometry_bed))return false;
            if(!draw(geometry_water))return false;
            if(!draw(geometry_river) || !draw(geometry_shadow) ||
                    !draw(geometry_route))return false;
        }
        for(unsigned i=0;i<renderer.cliff_bundle.assets.size();++i)
            if(!draw(static_cast<GeometryLayer>(geometry_cliff0+i)))return false;
        if(!draw_features(records,settings,rect,target,depth,mirrored,scale)){
            char detail[128];sprintf_s(detail,"layer=%u mirrored=%u records=%zu",unsigned(geometry_feature),unsigned(mirrored),records[geometry_feature].size());
            renderer.trace.write("fresh-layer-draw-failed",detail,true);return false;
        }
        for(auto layer:{geometry_site,geometry_mine,geometry_farm,geometry_city,geometry_wall})
            if(!draw(layer))return false;
        char wave_diagnostic[8]{};
        bool skip_wave=GetEnvironmentVariableA("C3X_SANDBOX_SKIP_WAVE",
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
        bool day_night=GetEnvironmentVariableA("C3X_SANDBOX_DAY_NIGHT",cycle,
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
        std::vector<c3x_renderer::city_fidelity::Lighting const*> selected;
        for(auto const& record:all_visible[geometry_city])
            if(record.content().city_lighting){
                auto* light=record.content().city_lighting.get();
                if(std::find(selected.begin(),selected.end(),light)==selected.end())
                    selected.push_back(light);
            }
        // Selected immutable generations own these light payloads. Changing the
        // field can relight already covered pixels even if shadows still fit.
        if(selected!=selected_lighting){
            static_rasters.invalidate_all(c3x_renderer::render_core::raster_lights);
            ++lighting_revision;reflection_valid=false;reflected_terrain_material_valid=false;
        }
        selected_lighting=selected;
        return renderer.cities.lights(renderer.context,selected);
    }
    bool ensure_linear_target(c3x_renderer::render_core::LinearTarget& target,
            unsigned width,unsigned height,unsigned samples,bool resolved) {
        char reference[8]={};
        bool copy_reference=GetEnvironmentVariableA("C3X_SANDBOX_RESOLVE_COPY_REFERENCE",reference,sizeof(reference)) && reference[0]=='1';
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
        if(GetEnvironmentVariableA("C3X_SANDBOX_MSAA_2X",option,sizeof(option)) && option[0]=='1')samples=2;
        if(GetEnvironmentVariableA("C3X_RENDERER_SCENE_SAMPLES",option,sizeof(option)))
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
        reflection_scale=GetEnvironmentVariableA("C3X_SANDBOX_REFLECTION_FULL",value,sizeof(value)) &&
            std::strcmp(value,"1")==0?1.f:.375f;
        unsigned reflection_width=unsigned((width+8)*reflection_scale);
        unsigned reflection_height=unsigned((height+8)*reflection_scale);
        unsigned region_width=width+2*region_margin_x;
        unsigned region_height=height+2*region_margin_y;
        static_rasters.set_layout(width,height,scene_samples);
        if(static_region().width!=region_width || static_region().height!=region_height ||
                static_region().sample_count!=scene_samples)
            static_rasters.invalidate(static_rasters.selected,c3x_renderer::render_core::raster_layout);
        if(static_cache.width!=width || static_cache.height!=height || static_cache.sample_count!=scene_samples)
            static_rasters.begin_restore();
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
        if(!ensure_linear_target(static_cache,width*scene_scale,
                height*scene_scale,scene_samples,true) ||
           !ensure_linear_target(static_region(),region_width*scene_scale,
                region_height*scene_scale,scene_samples,false) ||
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
            std::uint64_t(renderer.scene_depth_origin),lighting_revision,shadow.builds,
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
        context->OMSetBlendState(renderer.blend_state,nullptr,0xffffffffu);
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
        body_requirement_builds=body_requirement_reuses=body_requirement_visits=0;body_requirement_ms=0;
        body_requirements.coverage_probes=0;
        prepare_subspans={};dynamic_subspans={};dynamic_calls={};prepare_unit_plan_reused=prepare_unit_reselected=0;
        char scroll_diagnostics[8]={};
        bool trace_scroll=GetEnvironmentVariableA("C3X_SANDBOX_SCROLL_CALLS",
            scroll_diagnostics,sizeof(scroll_diagnostics)) && scroll_diagnostics[0]=='1';
        auto scroll_before=cache_scrolls,full_before=cache_full_draws;

        auto fail=[this](char const* stage){
            static_rasters.invalidate_all(c3x_renderer::render_core::raster_error);
            reflection_valid=false;reflected_terrain_material_valid=false;
            renderer.trace.write("fresh-draw-failed",stage,true);
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
        if(projection_zoom!=zoom){
            reflection_valid=false;reflected_terrain_material_valid=false;
        }
        projection_zoom=zoom;display_zoom=1.f;
        auto& raster=static_rasters.select(projection_zoom);
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
        if (!shadow.ensure()) return fail("shadow_setup");QueryPerformanceCounter(&setup[2]);
        unsigned w=unsigned(width)+8,h=unsigned(height)+8;
        if (!ensure_targets(w,h)) return fail("targets");QueryPerformanceCounter(&setup[3]);
        if(cold_setup){std::printf("SANDBOX_FRESH_SETUP visual_ms=%.6f shadow_ms=%.6f targets_ms=%.6f\n",
            double(setup[1].QuadPart-setup[0].QuadPart)*1000/frequency.QuadPart,
            double(setup[2].QuadPart-setup[1].QuadPart)*1000/frequency.QuadPart,
            double(setup[3].QuadPart-setup[2].QuadPart)*1000/frequency.QuadPart);std::fflush(stdout);}
        update_environment(frame);
        char reflection_diagnostic[8]{};
        if(GetEnvironmentVariableA("C3X_SANDBOX_SKIP_REFLECTION",
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
        if (resident_builds && GetEnvironmentVariableA("C3X_SANDBOX_SKIP_SCENE",
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
        renderer.water_material=c3x_renderer::render_core::water_material_frame(frame);
        int water_camera_x=frame.world_wrap_x && frame.world_width_tiles>0?
            camera_x%(frame.world_width_tiles*frame.tile_width/2):camera_x;
        renderer.water_material.camera[0]-=float(water_camera_x)/frame.tile_width+
            float(camera_y)/frame.tile_height;
        renderer.water_material.camera[1]-=float(water_camera_x)/frame.tile_width-
            float(camera_y)/frame.tile_height;
        renderer.water_time_seconds=renderer.water_material.time;
        renderer.wave_time_seconds=renderer.water_material.time;
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
        auto region_shift=c3x_renderer::render_core::StaticRegionShift::between(
            projection_zoom,camera_x,camera_y,current_raster().camera_x,current_raster().camera_y,
            region_margin_x,region_margin_y);
        bool recenter_region=!current_raster().valid ||
            current_raster().signature!=raster_scope() ||
            current_raster().lighting_revision!=lighting_revision ||
            current_raster().environment_hour!=visual_hour ||
            current_raster().environment_season!=previous_season ||
            current_raster().depth_origin!=renderer.scene_depth_origin ||
            settings.translation[0]!=current_raster().translation[0]+float(std::int64_t(camera_x)-current_raster().camera_x) ||
            settings.translation[1]!=current_raster().translation[1]+float(std::int64_t(camera_y)-current_raster().camera_y) ||
            !region_shift.reusable;
#ifdef C3X_RENDERER64_FRESH
        if(!recenter_region){
            auto retained=settings;
            retained.translation[0]+=float(region_margin_x-(camera_x-current_raster().camera_x));
            retained.translation[1]+=float(region_margin_y-(camera_y-current_raster().camera_y));
            retained.inverse_size[0]=1.f/static_region().width;retained.inverse_size[1]=1.f/static_region().height;
            auto const& covered=current_raster().covered;
            if(!raster_dependencies(raster_inputs[static_rasters.selected],retained,
                    {covered.left,covered.top,covered.right,covered.bottom},false)){
                recenter_region=true;++raster.metrics.reasons[c3x_renderer::render_core::raster_scene];
            }
        }
#endif
        if(recenter_region){
            if(cache_full_draws){
                // Existing phase counter includes fractional and derivative-quad lattice misses.
                if(current_raster().valid && (region_shift.reason==c3x_renderer::render_core::StaticRegionShift::fractional_phase ||
                    region_shift.reason==c3x_renderer::render_core::StaticRegionShift::derivative_phase))++raster.metrics.reasons[c3x_renderer::render_core::raster_fractional_phase];
                if(current_raster().valid && region_shift.reason==c3x_renderer::render_core::StaticRegionShift::guard_bounds)++raster.metrics.reasons[c3x_renderer::render_core::raster_guard_bounds];
                if(current_raster().depth_origin!=renderer.scene_depth_origin)++raster.metrics.reasons[c3x_renderer::render_core::raster_depth_origin];
                if(settings.translation[0]!=current_raster().translation[0]+float(std::int64_t(camera_x)-current_raster().camera_x) ||
                        settings.translation[1]!=current_raster().translation[1]+float(std::int64_t(camera_y)-current_raster().camera_y))++raster.metrics.reasons[c3x_renderer::render_core::raster_anchors];
            }
            current_raster().valid=false;
            static_rasters.begin_restore();
            current_raster().camera_x=camera_x;current_raster().camera_y=camera_y;
        }
        auto region_settings=settings;
        region_settings.translation[0]+=float(region_margin_x-(camera_x-current_raster().camera_x));
        region_settings.translation[1]+=float(region_margin_y-(camera_y-current_raster().camera_y));
        // Captured world depth can stay constant while screen anchors move.
        // Refill in the retained basis, then translate to the current basis.
        region_settings.depth_translation=current_raster().valid?
            current_raster().depth_translation:settings.depth_translation;
        region_settings.inverse_size[0]=1.f/static_region().width;
        region_settings.inverse_size[1]=1.f/static_region().height;
        D3D11_RECT region_rect={region_margin_x,region_margin_y,
            LONG(region_margin_x+w),LONG(region_margin_y+h)};
        mark_prepare(prepare_raster_proof);
        std::array<std::uint64_t,4> body_scene={view_revision(),visibility_revision,
            (std::uint64_t(static_region().width)<<32)|static_region().height,std::uint64_t(renderer.water_scene_active)};
        std::array<float,7> body_view={region_settings.translation[0],region_settings.translation[1],
            region_settings.inverse_size[0],region_settings.inverse_size[1],projection_zoom,resident_basis_x,resident_basis_y};
        if(!body_requirements_valid || body_scene!=body_requirement_scene || body_view!=body_requirement_view){
            auto started=std::chrono::steady_clock::now();body_requirements.clear();body_requirements_valid=false;
            bool complete=true;
            auto visit=[&](unsigned layer,GeometryDrawReference const& draw){
                if(complete && draw.content().instances && !draw.content().instances->empty())
                    complete=body_requirements.add(renderer.shared_instances,layer,draw,renderer.shared_instance_draw_key(layer,draw));
            };
#ifndef C3X_RENDERER64_FRESH
            // Legacy consumers submit the original native ranges. Renderer64
            // production uses the selected occurrences and guarded strips;
            // its explicit CPU oracle prepares its own original-range union.
            GeometryDrawView original=renderer.geometry_vertex_buffers;
            for(unsigned layer=0;layer<geometry_layer_count;++layer)
                for(auto const& draw:original[layer])visit(layer,draw);
#endif
            for(unsigned layer=0;layer<geometry_layer_count;++layer)
                for(auto const& record:all_visible[layer])visit(layer,GeometryDrawReference(record));
            auto clip=source_bounds(region_settings,{0,0,LONG(static_region().width),LONG(static_region().height)},false);
            contributors(region_settings,clip,false,[&](unsigned layer,auto const& record){
                if(renderer.water_scene_active && record.water_dependent)return;
                auto draw=GeometryDrawReference(record);
                if(renderer.chunk_intersects_region(draw,region_settings,clip,false))visit(layer,draw);
            });
            if(!complete)return fail("body_requirements");
            body_requirement_scene=body_scene;body_requirement_view=body_view;body_requirements_valid=true;
            ++body_requirement_builds;body_requirement_visits=body_requirements.visits;
            body_requirement_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        }else ++body_requirement_reuses;
        auto const& body_inputs=body_requirements;
        mark_prepare(prepare_body_requirements);
        auto retire_completed_instance_plans=[&]{
#ifdef C3X_RENDERER64_FRESH
            instance_plans={};instance_plans.begin();instance_plan_bytes=0;
#endif
        };
        if(!update_city_lights() || !shadow.render(all_visible,raster_scope(),static_receiver_revision,view_revision(),body_inputs,retire_completed_instance_plans))return fail("lights_or_shadow");
#ifdef C3X_RENDERER64_FRESH
        if(shared_front!=shadow.shared_front){instance_plans={};instance_plans.begin();instance_plan_bytes=0;shared_front=shadow.shared_front;}
#else
        shared_front=shadow.shared_front;
#endif
        if(!renderer.shared_instances.valid(shared_front))return fail("shared_submission");
        mark_prepare(prepare_city_shadow);
        D3D11_RECT full={0,0,LONG(w),LONG(h)};
        char unit_control[8]{};
        bool units=GetEnvironmentVariableA("C3X_SANDBOX_UNITS",unit_control,
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
        QueryPerformanceCounter(&ticks[1]);
        prepare_subspans[prepare_unit_pose]=double(ticks[1].QuadPart-prepare_previous.QuadPart)*1000/frequency.QuadPart;
#ifdef C3X_RENDERER64_FRESH
        gpu_phases.pass_end(renderer.context);gpu_phases.pass_begin(renderer.context,GpuPhases::import);
#endif
        work.pass=SandboxPassWorkload::reflection_scene;
        if (renderer.reflection.enabled && mirror.left<mirror.right && mirror.top<mirror.bottom) {
            auto key=reflection_writer(reflected,mirror);
            bool redraw=!reflection_valid || reflection_key!=key;
            if(!redraw){++reflection_reuses;if(work.enabled)++work.row().reuses;}
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
            ++reflection_draws;
            }
            bool dynamic_reflection=units
#ifdef C3X_RENDERER64_FRESH
                || sandbox_direct_units.reflection_contributors!=0
#endif
                ;
            if(redraw || dynamic_reflection || reflection_dynamic_written){
                renderer.context->OMSetRenderTargets(0,nullptr,nullptr);
                renderer.context->CopyResource(reflection.color,reflection_static.color);work.copy(reflection.color,true);
                renderer.context->CopyResource(reflection.depth_texture,reflection_static.depth_texture);work.copy(reflection.depth_texture,true);
            }
#ifdef C3X_RENDERER64_FRESH
            if(!sandbox_direct_units.draw_real(frame,unit_contribution_candidates,reflection,
                    reflection_scale,visual_hour,true,projection_zoom))return fail("reflected_real_units");
#endif
            if(units && !sandbox_direct_units.draw(frame,unit_x,unit_y,
                    incarnation,viewer,unit_visible,camera_x,camera_y,reflection,
                    reflection_scale,next_zoom,visual_hour,true))return fail("reflected_units");
            reflection_dynamic_written=dynamic_reflection;
        }
        QueryPerformanceCounter(&ticks[2]);
#ifdef C3X_RENDERER64_FRESH
        gpu_phases.pass_end(renderer.context);gpu_phases.pass_begin(renderer.context,GpuPhases::receivers);
#endif
        work.pass=SandboxPassWorkload::main_scene;
        auto* context=renderer.context;
        bool const cache_ready=scene_scale==1 && current_raster().valid &&
            current_raster().signature==raster_scope() &&
            current_raster().shadow_builds==shadow.builds;
        if(raster.metrics.full_draws && raster.shadow_builds!=shadow.builds)
            ++raster.metrics.reasons[c3x_renderer::render_core::raster_shadow];
        if(cache_ready)++raster.metrics.reuses;
        if(work.enabled){if(cache_ready)++work.row().reuses;else ++work.row().rebuilds;}
        if(!cache_ready){
            // A light or scene change replaces the whole resident image at
            // the current camera; its center is the first covered rectangle.
            current_raster().camera_x=camera_x;current_raster().camera_y=camera_y;
            region_settings.translation[0]=settings.translation[0]+region_margin_x;
            region_settings.translation[1]=settings.translation[1]+region_margin_y;
            region_settings.depth_translation=settings.depth_translation;
            current_raster().depth_translation=settings.depth_translation;
            current_raster().depth_origin=renderer.scene_depth_origin;
            current_raster().translation={settings.translation[0],settings.translation[1]};
            static_rasters.begin_write();
            float clear[4]={};
            context->ClearRenderTargetView(static_region().target,clear);work.clear(static_region().target);
            context->ClearDepthStencilView(static_region().depth,
                D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);work.clear(static_region().depth);
            if(!draw_scene(static_visible,region_settings,region_rect,
                    static_region().target,static_region().depth,false,float(scene_scale)))return fail("static_scene");
            current_raster().covered={region_rect.left,region_rect.top,region_rect.right,region_rect.bottom};
            ++cache_full_draws;++raster.metrics.full_draws;
            current_raster().signature=raster_scope();
#ifdef C3X_RENDERER64_FRESH
            auto& dependencies=raster_inputs[static_rasters.selected];dependencies.clear();
            if(!raster_dependencies(dependencies,region_settings,region_rect,true))dependencies.complete=false;
#endif
            current_raster().shadow_builds=shadow.builds;
            raster.geometry_epoch=renderer.tile_geometry_epoch;
            raster.lighting_revision=lighting_revision;
            raster.environment_hour=visual_hour;raster.environment_season=previous_season;
            current_raster().valid=true;
        }
        if(cache_ready){
            auto needed=region_shift.needed<D3D11_RECT>(int(w),int(h),
                region_margin_x,region_margin_y);
            auto fill_strip=[&](D3D11_RECT strip){
                if(strip.left>=strip.right || strip.top>=strip.bottom)return true;
                ++raster.metrics.reasons[c3x_renderer::render_core::raster_strip_fills];
                ++raster.metrics.strip_fills;
                static_rasters.begin_write();
                GeometryDrawView::Records selected{};
                auto clip=source_bounds(region_settings,strip,false);
                contributors(region_settings,clip,false,[&](unsigned layer,auto const& record){
                        if(renderer.water_scene_active && record.water_dependent)return;
                        if(renderer.chunk_intersects_region(GeometryDrawReference(record),
                                region_settings,clip,false))selected[layer].push_back(record);
                    });
                if(!draw_scene(selected,region_settings,strip,static_region().target,
                    static_region().depth,false,float(scene_scale)))return false;
#ifdef C3X_RENDERER64_FRESH
                if(!raster_dependencies(raster_inputs[static_rasters.selected],region_settings,strip,true))
                    raster_inputs[static_rasters.selected].complete=false;
#endif
                raster.valid=true;return true;
            };
            constexpr LONG ahead=128;
            if(needed.left<current_raster().covered.left){
                LONG left=std::max<LONG>(0,needed.left-ahead);
                if(!fill_strip({left,current_raster().covered.top,current_raster().covered.left,
                        current_raster().covered.bottom}))return fail("static_strip");
                current_raster().covered.left=left;
            }
            if(needed.right>current_raster().covered.right){
                LONG right=std::min<LONG>(LONG(static_region().width),needed.right+ahead);
                if(!fill_strip({current_raster().covered.right,current_raster().covered.top,right,
                        current_raster().covered.bottom}))return fail("static_strip");
                current_raster().covered.right=right;
            }
            if(needed.top<current_raster().covered.top){
                LONG top=std::max<LONG>(0,needed.top-ahead);
                if(!fill_strip({current_raster().covered.left,top,current_raster().covered.right,
                        current_raster().covered.top}))return fail("static_strip");
                current_raster().covered.top=top;
            }
            if(needed.bottom>current_raster().covered.bottom){
                LONG bottom=std::min<LONG>(LONG(static_region().height),needed.bottom+ahead);
                if(!fill_strip({current_raster().covered.left,current_raster().covered.bottom,
                        current_raster().covered.right,bottom}))return fail("static_strip");
                current_raster().covered.bottom=bottom;
            }
        }
        if(static_rasters.needs_restore(cache_ready,camera_x,camera_y,settings.depth_translation)){
            static_rasters.begin_restore();
            region_shift=c3x_renderer::render_core::StaticRegionShift::between(
                projection_zoom,camera_x,camera_y,current_raster().camera_x,current_raster().camera_y,
                region_margin_x,region_margin_y);
            if(!static_restore.draw(context,static_cache,static_region().samples,
                    static_region().depth_samples,region_shift.x-region_margin_x,
                    region_shift.y-region_margin_y,{},nullptr,static_region().width,
                    static_region().height,false,false,0,nullptr,1,
                    -(settings.depth_translation-current_raster().depth_translation)/16384.f))return fail("static_restore");
            work.draw(3);if(work.enabled)work.row().target_pixels+=std::uint64_t(static_cache.width)*static_cache.height;
            context->OMSetRenderTargets(0,nullptr,nullptr);
            if(scene_samples==1){
                if(static_cache.resolved!=static_cache.color){context->CopyResource(static_cache.resolved,static_cache.color);work.copy(static_cache.resolved,true);}
            }
            else context->ResolveSubresource(static_cache.resolved,0,static_cache.color,0,
                DXGI_FORMAT_R16G16B16A16_FLOAT);
            if(cache_ready)++cache_scrolls;
            static_rasters.restored(camera_x,camera_y,settings.depth_translation);
        }
        QueryPerformanceCounter(&ticks[3]);
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
        bool skip_water=GetEnvironmentVariableA("C3X_SANDBOX_SKIP_WATER_PASS",
            water_diagnostic,sizeof(water_diagnostic)) &&
            std::strcmp(water_diagnostic,"1")==0;
        if (renderer.water_scene_active && !skip_water) {
            if(!draw_scene(water_visible,settings,full,glow.linear.target,
                    glow.linear.depth,false,float(scene_scale)))return fail("water_scene");
        }
        mark_dynamic(dynamic_water_scene);
        if(!draw_resource_poses(renderer.sandbox_resource_poses,settings,full,
                glow.linear.target,glow.linear.depth))
            return fail("resource_scene");
        mark_dynamic(dynamic_resources);
        char wave_diagnostic[8]{};
        bool skip_wave=GetEnvironmentVariableA("C3X_SANDBOX_SKIP_WAVE",
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
        QueryPerformanceCounter(&ticks[4]);
#ifdef C3X_RENDERER64_FRESH
        gpu_phases.pass_end(renderer.context);gpu_phases.pass_begin(renderer.context,GpuPhases::body);
#endif
#ifdef C3X_RENDERER64_FRESH
        if(!sandbox_direct_units.draw_real(frame,unit_contribution_candidates,glow.linear,
                float(scene_scale),visual_hour,false,projection_zoom))return fail("real_units");
#endif
        if(units && !sandbox_direct_units.draw(frame,unit_x,unit_y,
            incarnation,viewer,unit_visible,camera_x,camera_y,glow.linear,
            float(scene_scale),next_zoom,visual_hour))
            return fail("direct_units");
        QueryPerformanceCounter(&ticks[5]);
#ifdef C3X_RENDERER64_FRESH
        gpu_phases.pass_end(renderer.context);gpu_phases.pass_begin(renderer.context,GpuPhases::finish);
#endif
        for(auto const* visible_scene:{&static_visible,&water_visible})
          for(unsigned layer:{unsigned(geometry_underlay),unsigned(geometry_natural_terrain),
                              unsigned(geometry_natural_mountain),unsigned(geometry_water)})
            if(!territory_borders.draw(renderer.device,context,(*visible_scene)[layer],settings,glow.linear,
                    renderer.content_view_width,renderer.content_view_height,projection_zoom,float(scene_scale),
                    [&](auto const& r){return renderer.chunk_intersects_region(GeometryDrawReference(r),settings,full,false);}))
                return fail("territory_borders");
        if(!reconstruct())return fail("reconstruct");
        renderer.context->OMSetRenderTargets(0,nullptr,nullptr);
        QueryPerformanceCounter(&ticks[6]);
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
    sandbox_fresh.static_rasters.invalidate_all(c3x_renderer::render_core::raster_explicit_reset);
    sandbox_fresh.reflection_valid=false;
    sandbox_fresh.reflected_terrain_material_valid=false;
}
extern "C" __declspec(dllexport) void c3x_sandbox_scroll_cache_metrics(unsigned* depth_copies,
        unsigned* scrolls,unsigned* full_draws,unsigned* reflection_reuses,
        unsigned* reflection_draws,unsigned* pose_builds) {
    c3x_sandbox_cache_metrics(depth_copies,scrolls,full_draws,reflection_reuses,reflection_draws,pose_builds);
}
extern "C" __declspec(dllexport) void c3x_sandbox_scroll_reasons(unsigned* values) {
    if(values)for(unsigned i=0;i<11;++i)values[i]=sandbox_fresh.static_rasters.states[0].metrics.reasons[i]+
        sandbox_fresh.static_rasters.states[1].metrics.reasons[i];
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
