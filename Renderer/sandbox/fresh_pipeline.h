#pragma once
#include <fstream>
#include <limits>
#include <sstream>

// Keep production's material equations and preparation. Change only receiver
// addressing from regional pages to the camera-framed field.
struct SandboxVisualShaders {
    bool installed = false;
    ID3D11PixelShader* water_surface = nullptr;
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
    ~SandboxVisualShaders() {
        if(water_surface) water_surface->Release();
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
    }
    static std::string source(std::string const& relative) {
        char const* profile=renderer.city_profile?"city_fidelity":
            renderer.environment_profile?"environment_refresh":"source_fidelity";
        std::ifstream file(renderer.fidelity_root + "/Renderer/native/" + profile + "/" + relative,
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
float c3x_paged_visibility(Texture2DArray field,float4 world,float3 normal,bool water,
 float4 ShadowU,float4 ShadowV,float4 ShadowL,float4 ShadowFlags) {
 if(world.w<=.5 || ShadowFlags.x<=.5)return 1;
 float4 box=pickup_pages[0]; // minimum light-plane u/v and full u/v span
 // The 4096px shared frame spans roughly 140 world units in this BIQ view;
 // the earlier 0.006 world-unit offset was sub-texel and caused self-shadow bands.
 const float texel=.035;
 float3 offset=world.xyz+normal*(water?6./1024.:texel*1.5);
 float2 uv=(float2(dot(offset,ShadowU.xyz),dot(offset,ShadowV.xyz))-box.xy)/box.zw*4096.;
 float z=dot(offset,ShadowL.xyz);
 float2 plane=(float2(dot(world.xyz,ShadowU.xyz),dot(world.xyz,ShadowV.xyz))-box.xy)/box.zw*4096.;
 float plane_z=dot(world.xyz,ShadowL.xyz);
 float2 ux=ddx(plane),uy=ddy(plane);float zx=ddx(plane_z),zy=ddy(plane_z);
 float determinant=ux.x*uy.y-ux.y*uy.x;
 float2 gradient=0;
 if(abs(determinant)>1e-12)gradient=float2(zx*uy.y-zy*ux.y,zy*ux.x-zx*uy.x)/determinant;
 int2 center=int2(floor(uv));float sum=0;
 [unroll]for(int y=-1;y<=1;y++)[unroll]for(int x=-1;x<=1;x++) {
  int2 sample=center+int2(x,y);
  float blocker=all(sample>=0)&&all(sample<4096)?field.Load(int4(sample,0,0)).r:-1e6;
  float receiver=z+dot(gradient,float2(sample)+.5-uv);
  sum+=step(blocker,receiver+(water?.00060:texel*.35));
 }
 float soft=sum/9;
 return soft;
}
)");
        return true;
    }
    static bool tune_native_terrain(std::string& shader) {
        // At native resolution the prepared sand relief covers fewer samples.
        // Keep its authored textures and placement, while restoring the
        // material normal response visible in the production-sized target.
        auto replace=[&](char const* old_text,char const* new_text) {
            auto at=shader.find(old_text);
            if(at==std::string::npos)return false;
            shader.replace(at,std::strlen(old_text),new_text);return true;
        };
        if(!replace("packed = lerp(0.5.xx, packed, 0.58);",
                    "packed = lerp(0.5.xx, packed, 1.80);"))return false;
        if(!replace("geometric = detail_normal(geometric, input.world, height_detail);",
                    "geometric = detail_normal_strength(geometric, input.world, height_detail, Detail.y * (0.85 + 3.00 * desert_weight));"))return false;
        return true;
    }
    static bool compile(char const* file, char const* entry, ID3D11PixelShader** output) {
        bool water_variant = std::strcmp(file,"water_surface.hlsl")==0;
        std::string shader = source(water_variant?"hydrology.hlsl":file);
        bool albedo_variant=std::strcmp(entry,"PSSandboxMaterialAlbedo")==0;
        bool normal_variant=std::strcmp(entry,"PSSandboxMaterialNormalWorld")==0;
        bool terrain_variant=std::strcmp(entry,"PSSandboxTerrainMaterial")==0 ||
            std::strcmp(entry,"PSSandboxReflectionTerrainMaterial")==0;
        if(albedo_variant)shader.insert(0,"#define Q8_DEBUG_ALBEDO 1\n");
        if(normal_variant)shader.insert(0,"#define Q8_DEBUG_NORMAL 1\n");
        if(terrain_variant)shader.insert(0,"#define SANDBOX_TERRAIN_MATERIAL 1\n");
        if(water_variant){
            std::ifstream variant(renderer.fidelity_root+
                "/Renderer/sandbox/water_surface.hlsl",std::ios::binary);
            if(!variant)return false;
            shader.append(std::istreambuf_iterator<char>(variant),
                std::istreambuf_iterator<char>());
        }
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
        if (shader.empty() || !patch(shader) ||
            (std::strcmp(file,"terrain.hlsl")==0 && !tune_native_terrain(shader))) return false;
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
            if (FAILED(hr)) std::printf("SANDBOX_SHADER_ERROR file=%s entry=%s %s\n", file, entry,
                static_cast<char const*>(errors->GetBufferPointer()));
            errors->Release();
        }
        if (SUCCEEDED(hr)) hr = renderer.device->CreatePixelShader(
            code->GetBufferPointer(), code->GetBufferSize(), nullptr, output);
        if (SUCCEEDED(hr)) {
            std::ofstream saved(cache,std::ios::binary);
            saved.write(static_cast<char const*>(code->GetBufferPointer()),
                std::streamsize(code->GetBufferSize()));
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
        if(!compile("water_surface.hlsl","PSWaterSurface",&water_surface))return false;
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
        char const* names[]={"hydrology.hlsl","feature.hlsl","terrain.hlsl",
            "mountain.hlsl","objects.hlsl"};
        auto& reflection=renderer.scene_region_size==128?
            renderer.reflection:renderer.region_reflection;
        for (unsigned i=0;i<5;++i) {
            ID3D11PixelShader* replacement=nullptr;
            if (!compile(names[i],"PSReflection",&replacement)) return false;
            if (!reflection.ps[i]) return false;
            reflection.ps[i]->Release();
            reflection.ps[i]=replacement;
        }
        installed = true;
        return true;
    }
};

// One light-space field for all receivers in the visible scene. The retained
// renderer supplies caster meshes, cutout textures and the light basis; the
// sandbox owns target selection, residency and the field's lifetime.
struct SandboxSceneShadow {
    using Shadow = c3x_renderer::render_core::SourceShadow;
    ID3D11Texture2D* texture = nullptr;
    ID3D11RenderTargetView* target = nullptr;
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
        std::vector<Instance> prepared,selected;
    };
    c3x_renderer::render_core::InstanceStream instance_stream;
    std::vector<InstanceGroup> instance_groups;
    std::vector<Shadow::Caster> casters;
    std::vector<ID3D11Buffer*> patch_buffers;
    unsigned resolution=4096;
    ID3D11ShaderResourceView* production_view = nullptr;
    float box[4] = {};
    std::uint64_t signature = 0;
    std::array<float,12> light_basis{};
    unsigned builds = 0, draws = 0;
    template<class T> static void drop(T*& pointer) {if (pointer) pointer->Release(); pointer=nullptr;}
    ~SandboxSceneShadow() {
        if (production_view) renderer.source_shadow.view = production_view;
        instance_stream.clear();
        for(auto* buffer:patch_buffers)drop(buffer);
        drop(texture); drop(target); drop(view); drop(vertex); drop(instance_vertex); drop(rigid_vertex);
        drop(opaque); drop(cutout); drop(layout); drop(feature_layout); drop(natural_layout);
        drop(city_layout); drop(instance_layout); drop(constants); drop(maximum); drop(raster);
    }
    bool ensure() {
        if (view) return true;
        auto root = renderer.fidelity_root + "/Renderer/native/";
        auto shader = [&](std::string const& path, char const* entry, char const* profile,
                ID3DBlob** result) {
            std::wstring wide(path.begin(),path.end()); ID3DBlob* errors=nullptr;
            HRESULT hr=c3x_renderer::render_core::compile_cached(wide.c_str(),entry,profile,result,&errors);
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
        if (SUCCEEDED(hr) && shader(root+"source_fidelity/instance_caster.hlsl","VSInstance","vs_5_0",&code)) {
            hr=renderer.device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&instance_vertex);
            if (SUCCEEDED(hr)) hr=c3x_renderer::render_core::create_instance_layout(renderer.device,code,&instance_layout);
        } else hr=E_FAIL;
        drop(code);
        if (SUCCEEDED(hr) && shader(root+"city_fidelity/rigid_caster.hlsl","VSSharedCaster","vs_5_0",&code))
            hr=renderer.device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&rigid_vertex);
        else hr=E_FAIL;
        drop(code);
        D3D11_TEXTURE2D_DESC d={};d.Width=d.Height=resolution;d.ArraySize=d.MipLevels=d.SampleDesc.Count=1;
        d.Format=DXGI_FORMAT_R32_FLOAT;
        d.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
        if (SUCCEEDED(hr)) hr=renderer.device->CreateTexture2D(&d,nullptr,&texture);
        D3D11_RENDER_TARGET_VIEW_DESC rt={};rt.Format=d.Format;
        rt.ViewDimension=D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
        rt.Texture2DArray.ArraySize=1;
        if (SUCCEEDED(hr)) hr=renderer.device->CreateRenderTargetView(texture,&rt,&target);
        D3D11_SHADER_RESOURCE_VIEW_DESC sr={};sr.Format=d.Format;
        sr.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
        sr.Texture2DArray.ArraySize=1;sr.Texture2DArray.MipLevels=1;
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
        if (FAILED(hr)) return false;
        production_view=renderer.source_shadow.view;
        renderer.source_shadow.view=view;
        renderer.collect_shadow_casters(renderer.geometry_vertex_buffers,casters);
        batch_terrain_casters();
        return prepare_instances();
    }
    bool prepare_instances() {
        using Instance=c3x_renderer::fidelity::MeshInstance;
        std::map<std::array<std::uintptr_t,9>,std::size_t> lookup;
        std::size_t total=0;
        for(auto const& caster:casters){
            if(!caster.instances || caster.instances->empty())continue;
            std::array<std::uintptr_t,9> key={
                reinterpret_cast<std::uintptr_t>(caster.vertices),
                reinterpret_cast<std::uintptr_t>(caster.indices),
                caster.vertex_offset,caster.index_offset,caster.count,
                std::uintptr_t(caster.index_format),
                caster.binding==0xffffffffu?caster.layer:caster.binding,
                std::uintptr_t(caster.rigid),caster.stride};
            auto found=lookup.find(key);
            if(found==lookup.end()){
                found=lookup.emplace(key,instance_groups.size()).first;
                instance_groups.emplace_back();
                instance_groups.back().source=caster;
            }
            auto& group=instance_groups[found->second];
            if(group.prepared.size()+caster.instances->size()>std::numeric_limits<UINT>::max())return false;
            InstanceGroup::Part part;part.first=UINT(group.prepared.size());
            part.count=UINT(caster.instances->size());
            for(int axis=0;axis<3;++axis){
                part.bounds.low[axis]=caster.bounds.low[axis]+caster.offset[axis];
                part.bounds.high[axis]=caster.bounds.high[axis]+caster.offset[axis];
            }
            group.parts.push_back(part);
            for(auto instance:*caster.instances){
                std::copy(caster.offset,caster.offset+3,instance.view);
                instance.view[3]=caster.instance_material;
                group.prepared.push_back(instance);
            }
            total+=caster.instances->size();
        }
        for(auto& group:instance_groups)group.selected.reserve(group.prepared.size());
        std::printf("SANDBOX_SHADOW_INSTANCES casters=%zu groups=%zu instances=%zu bytes=%zu\n",
            casters.size(),instance_groups.size(),total,total*sizeof(Instance));
        std::fflush(stdout);
        return true;
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
        if (layer==geometry_wall) views[29]=views[30]=views[31]=views[32]=renderer.wall_texture_view;
        if (layer==geometry_site) std::copy(renderer.site_views.begin(),renderer.site_views.end(),views.begin()+21);
        if (layer==geometry_mine) std::copy(renderer.mine_base_views.begin(),renderer.mine_base_views.end(),views.begin()+21);
        if (layer==geometry_farm) std::copy(renderer.farm_base_views.begin(),renderer.farm_base_views.end(),views.begin()+21);
        if (layer>=geometry_cliff0 && layer<geometry_natural_terrain)
            views[0]=renderer.cliff_views[renderer.cliff_bundle.assets[layer-geometry_cliff0].texture_index];
        context->PSSetShaderResources(0,33,views.data());return true;
    }
    bool render(GeometryDrawView::Records const& receivers) {
        if (!ensure()) return false;
        float needed[4]={std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),
            -std::numeric_limits<float>::max(),-std::numeric_limits<float>::max()};
        bool any=false;
        float zero[3]={};
        for (unsigned layer=0;layer<geometry_layer_count;++layer)
            for (auto const& record:receivers[layer]) {
                if (layer==geometry_shadow) continue;
                auto p=Shadow::project(record.content().world_bounds,zero,renderer.shadow_basis);
                needed[0]=std::min(needed[0],p[0]);needed[1]=std::min(needed[1],p[1]);
                needed[2]=std::max(needed[2],p[2]);needed[3]=std::max(needed[3],p[3]);
                any=true;
            }
        if (!any) return false;
        if (signature==renderer.cached_signature.complete &&
            light_basis==renderer.shadow_basis &&
            needed[0]>box[0]+1 && needed[1]>box[1]+1 &&
            needed[2]<box[0]+box[2]-1 && needed[3]<box[1]+box[3]-1) return true;
        box[0]=std::floor((needed[0]-4)/2)*2;box[1]=std::floor((needed[1]-4)/2)*2;
        box[2]=std::ceil((needed[2]+4-box[0])/2)*2;
        box[3]=std::ceil((needed[3]+4-box[1])/2)*2;
        if (box[2]<=0 || box[3]<=0) return false;
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
        context->ClearRenderTargetView(target,clear);
        context->OMSetRenderTargets(1,&target,nullptr);
        draws=0;
        for (auto const& caster:casters) {
            if(caster.instances)continue;
            auto bounds=Shadow::project(caster.bounds,caster.offset,renderer.shadow_basis);
            if(bounds[2]<box[0] || bounds[0]>box[0]+box[2] ||
               bounds[3]<box[1] || bounds[1]>box[1]+box[3])continue;
            float settings[20]={};
            std::copy(renderer.shadow_basis.begin(),renderer.shadow_basis.end(),settings);
            for (int i=0;i<3;++i) {
                settings[i]*=6/box[2];settings[4+i]*=6/box[3];
                settings[16+i]=caster.offset[i];
            }
            settings[12]=box[0]/box[2];settings[13]=box[1]/box[3];
            context->UpdateSubresource(constants,0,nullptr,settings,0,0);
            context->PSSetShader(bind_cutout(caster.binding==0xffffffffu?
                caster.layer:caster.binding)?cutout:opaque,nullptr,0);
            context->VSSetShader(vertex,nullptr,0);
            context->IASetInputLayout(caster.stride==88?city_layout:
                caster.stride==92?natural_layout:caster.stride==48?feature_layout:layout);
            UINT stride=caster.stride,offset=caster.vertex_offset;
            context->IASetVertexBuffers(0,1,&caster.vertices,&stride,&offset);
            context->IASetIndexBuffer(caster.indices,caster.index_format,caster.index_offset);
            context->DrawIndexed(caster.count,0,0);
            ++draws;
        }
        for(auto& group:instance_groups){
            group.selected.clear();
            for(auto const& part:group.parts){
                auto bounds=Shadow::project(part.bounds,zero,renderer.shadow_basis);
                if(bounds[2]<box[0] || bounds[0]>box[0]+box[2] ||
                   bounds[3]<box[1] || bounds[1]>box[1]+box[3])continue;
                group.selected.insert(group.selected.end(),
                    group.prepared.begin()+part.first,
                    group.prepared.begin()+part.first+part.count);
            }
            if(group.selected.empty())continue;
            auto const& caster=group.source;
            float settings[20]={};
            std::copy(renderer.shadow_basis.begin(),renderer.shadow_basis.end(),settings);
            for(int axis=0;axis<3;++axis){
                settings[axis]*=6/box[2];settings[4+axis]*=6/box[3];
            }
            settings[12]=box[0]/box[2];settings[13]=box[1]/box[3];
            context->UpdateSubresource(constants,0,nullptr,settings,0,0);
            context->PSSetShader(bind_cutout(caster.binding==0xffffffffu?
                caster.layer:caster.binding)?cutout:opaque,nullptr,0);
            context->VSSetShader(caster.rigid?rigid_vertex:instance_vertex,nullptr,0);
            context->IASetInputLayout(instance_layout);
            context->IASetIndexBuffer(caster.indices,caster.index_format,caster.index_offset);
            for(std::size_t begin=0;begin<group.selected.size();
                    begin+=c3x_renderer::render_core::InstanceStream::limit){
                auto count=std::min(group.selected.size()-begin,
                    std::size_t(c3x_renderer::render_core::InstanceStream::limit));
                if(!instance_stream.upload(nullptr,context,group.selected.data()+begin,count))return false;
                ID3D11Buffer* streams[]={caster.vertices,instance_stream.buffer};
                UINT strides[]={32,64},offsets[]={caster.vertex_offset,instance_stream.offset};
                context->IASetVertexBuffers(0,2,streams,strides,offsets);
                context->DrawIndexedInstanced(caster.count,UINT(count),0,0,0);
                ++draws;
            }
        }
        context->OMSetRenderTargets(0,nullptr,nullptr);
        std::array<std::array<float,4>,64> table{};
        std::copy(box,box+4,table[0].begin());
        context->UpdateSubresource(renderer.source_shadow.table,0,nullptr,table.data(),0,0);
        signature=renderer.cached_signature.complete;
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

// Frame ownership follows 0 A.D.'s submit/prepare/render split. Immutable
// occurrences survive camera changes; each pass gets only visible references.
struct SandboxFreshPipeline {
    SandboxVisualShaders visual;
    SandboxSceneShadow shadow;
    c3x_renderer::city_fidelity::Glow glow;
    SandboxBloom bloom;
    c3x_renderer::render_core::LinearTarget static_cache;
    c3x_renderer::render_core::LinearTarget material_albedo;
    SandboxMaterialChannel material_normal,material_world;
    c3x_renderer::render_core::LinearTarget terrain_albedo;
    SandboxMaterialChannel terrain_normal,terrain_world,terrain_properties;
    c3x_renderer::render_core::LinearTarget reflected_terrain_albedo;
    SandboxMaterialChannel reflected_terrain_normal,reflected_terrain_world,
        reflected_terrain_properties;
    SandboxMaterialChannel aquatic_scene;
    ID3D11BlendState* terrain_material_blend=nullptr;
    ID3D11DepthStencilState* aquatic_depth=nullptr;
    c3x_renderer::render_core::LinearRestore static_restore;
    c3x_renderer::render_core::DrawParameterStream parameters;
    SandboxMirrorTarget reflection,reflection_static;
    GeometryDrawView::Records resident, static_visible, water_visible,
        reflection_visible, all_visible;
    std::uint64_t resident_signature=0;
    unsigned resident_builds=0;
    int wrap_pixels=0;
    bool visibility_valid=false;
    std::array<std::uint64_t,5> visibility_scene_key{};
    std::array<float,5> visibility_view_key{};
    ID3D11ShaderResourceView* production_reflection=nullptr;
    unsigned production_reflection_width=0,production_reflection_height=0;
    int camera_x=0,camera_y=0;
    unsigned visible=0,culled=0;
    unsigned reflection_count=0;
    unsigned scene_scale=1,scene_samples=2;
    // A close crop in the output pass keeps the static cache resident.
    float display_zoom=1.f;
    float reflection_scale=1;
    unsigned depth_copies=0,cache_scrolls=0,cache_full_draws=0;
    unsigned reflection_reuses=0,reflection_draws=0;
    float visual_hour=12.f,previous_hour=-1.f;
    int previous_season=INT_MIN;
    std::uint64_t static_signature=0;
    std::uint64_t reflection_signature=0;
    unsigned static_shadow_builds=0;
    unsigned reflection_shadow_builds=0;
    int static_camera_x=0,static_camera_y=0;
    int reflection_camera_x=0,reflection_camera_y=0;
    bool static_valid=false;
    bool material_valid=false;
    std::uint64_t material_signature=0;
    int material_camera_x=0,material_camera_y=0;
    bool terrain_material_valid=false;
    std::uint64_t terrain_material_signature=0;
    int terrain_material_camera_x=0,terrain_material_camera_y=0;
    bool reflected_terrain_material_valid=false;
    std::uint64_t reflected_terrain_material_signature=0;
    int reflected_terrain_camera_x=0,reflected_terrain_camera_y=0;
    bool reflection_valid=false;
    double phases[6]={};
    bool active=false;
    ~SandboxFreshPipeline() {
        if(terrain_material_blend)terrain_material_blend->Release();
        if(aquatic_depth)aquatic_depth->Release();
        if (production_reflection) {
            renderer.scene_reflection_view=production_reflection;
            renderer.scene_reflection_width=production_reflection_width;
            renderer.scene_reflection_height=production_reflection_height;
        }
    }
    bool capture(ViewportShaderSettings const& settings,
            ViewportShaderSettings const& reflected,int width,int height,int next_wrap_pixels) {
        if (resident_signature!=renderer.cached_signature.complete ||
                wrap_pixels!=next_wrap_pixels) {
            visibility_valid=false;
            resident={};wrap_pixels=next_wrap_pixels;
            for (unsigned layer=0;layer<geometry_layer_count;++layer)
                for (auto const& record:renderer.geometry_vertex_buffers[layer]) {
                    resident[layer].push_back(record);
                    if (wrap_pixels) for (int direction:{-1,1}) {
                        auto wrapped=record;
                        wrapped.translation_x+=direction*wrap_pixels;
                        resident[layer].push_back(wrapped);
                    }
                }
            resident_signature=renderer.cached_signature.complete;
            ++resident_builds;
            shadow.signature=0;
        }
        std::array<std::uint64_t,5> scene_key={renderer.cached_signature.complete,
            std::uint64_t(next_wrap_pixels),std::uint64_t(width),
            std::uint64_t(height),std::uint64_t(renderer.water_scene_active)};
        std::array<float,5> view_key={settings.translation[0],settings.translation[1],
            reflected.translation[0],reflected.translation[1],
            renderer.reflection.height_pixels};
        if(visibility_valid && scene_key==visibility_scene_key &&
                view_key==visibility_view_key)return visible>0;
        static_visible={};water_visible={};reflection_visible={};all_visible={};
        visible=culled=reflection_count=0;
        D3D11_RECT rect={0,0,width,height};
        for(unsigned layer=0;layer<geometry_layer_count;++layer)
            for(auto const& record:resident[layer]) {
                if (!renderer.chunk_intersects_region(GeometryDrawReference(record),settings,rect,false)) {
                    ++culled;continue;
                }
                auto& output=renderer.water_scene_active && record.water_dependent?
                    water_visible:static_visible;
                output[layer].push_back(record);
                all_visible[layer].push_back(record);++visible;
            }
        D3D11_RECT mirror={0,0,width+8,height+8};
        for(unsigned layer=0;layer<geometry_layer_count;++layer) {
            if (layer==geometry_underlay || layer==geometry_bed ||
                layer==geometry_water || layer==geometry_river ||
                layer==geometry_route || layer==geometry_shadow || layer==geometry_wave)
                continue;
            for (auto const& record:resident[layer]) {
                if (record.content().animation_texture ||
                    !renderer.chunk_intersects_region(GeometryDrawReference(record),
                        reflected,mirror,true)) continue;
                reflection_visible[layer].push_back(record);
                all_visible[layer].push_back(record);
                ++reflection_count;
            }
        }
        visibility_scene_key=scene_key;visibility_view_key=view_key;
        visibility_valid=true;
        return visible>0;
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
        D3D11_VIEWPORT viewport={0,0,float(mirrored?reflection.width:glow.linear.width),
            float(mirrored?reflection.height:glow.linear.height),0,1};
        context->RSSetViewports(1,&viewport);
        D3D11_RECT scissor={LONG(rect.left*scale),LONG(rect.top*scale),
            LONG(rect.right*scale),LONG(rect.bottom*scale)};
        context->RSSetScissorRects(1,&scissor);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->IASetInputLayout(renderer.feature_input_layout);
        auto& active_reflection=renderer.scene_region_size==128?
            renderer.reflection:renderer.region_reflection;
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
        context->PSSetShaderResources(17,1,&renderer.source_shadow.view);
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
        if(!submerged){
            bind_common(settings,rect,target,depth,false,float(scene_scale));
            auto* context=renderer.context;
            context->PSSetShader(renderer.resource_shadow_shader,nullptr,0);
            context->PSSetShaderResources(17,1,&renderer.source_shadow.view);
            context->PSSetShaderResources(25,1,&renderer.source_shadow.view);
            context->PSSetShaderResources(116,8,renderer.resource_texture_views.data());
            ID3D11SamplerState* shadow_samplers[]={renderer.natural_wrap,renderer.natural_clamp};
            context->PSSetSamplers(0,2,shadow_samplers);
            context->OMSetDepthStencilState(renderer.natural.decal_depth,0);
            if(!issue_records(poses,geometry_shadow,settings,rect,false))return false;
        }
        if(!draw_features(poses,settings,rect,target,depth,false,float(scene_scale),
                submerged))return false;
        return true;
    }
    // Like CPatchRData::RenderBases, visible records are submitted from resident
    // buffers after the pass has bound its material variant. The producer owns
    // meshes and materials; this pipeline owns per-camera selection and draws.
    bool issue_records(GeometryDrawView::Records const& records,GeometryLayer layer,
            ViewportShaderSettings const& viewport,D3D11_RECT rect,bool mirrored) {
        auto* context=renderer.context;
        bool streamed=parameters.available(renderer.device,context);
        ViewportShaderSettings previous{};
        bool previous_valid=false;
        std::array<ViewportShaderSettings,c3x_renderer::render_core::DrawParameterStream::limit> values{};
        std::vector<GeometryDrawReference> selected;
        selected.reserve(values.size());
        auto flush=[&]() {
            if(selected.empty())return true;
            if(streamed && !parameters.upload(values.data(),unsigned(selected.size())))return false;
            std::vector<c3x_renderer::fidelity::MeshInstance> rigid;
            std::array<unsigned,c3x_renderer::render_core::DrawParameterStream::limit> rigid_offset{};
            for(unsigned i=0;i<selected.size();++i){
                auto const& chunk=selected[i];
                if(!chunk.content().rigid_source)continue;
                if(!chunk.content().instances || chunk.content().instances->size()!=1)return false;
                auto instance=chunk.content().instances->front();
                std::copy(std::begin(chunk.natural_projection()),
                    std::end(chunk.natural_projection()),instance.projection);
                instance.view[0]=values[i].translation[0];
                instance.view[1]=values[i].translation[1];
                instance.view[2]=values[i].depth_translation;
                instance.view[3]=chunk.content().instance_material;
                rigid_offset[i]=unsigned(rigid.size());rigid.push_back(instance);
            }
            if(!rigid.empty() && !renderer.rigid_sources.stream.upload(renderer.device,context,rigid))
                return false;
            auto& mirror=renderer.scene_region_size==128?
                renderer.reflection:renderer.region_reflection;
            for(unsigned i=0;i<selected.size();){
                auto const& chunk=selected[i];
                auto const& mesh=chunk.content();
                auto const& settings=values[i];
                if(streamed)parameters.bind(1,i);
                else if(!previous_valid || std::memcmp(&previous,&settings,sizeof(settings))){
                    context->UpdateSubresource(renderer.viewport_settings_buffer,0,nullptr,&settings,0,0);
                    previous=settings;previous_valid=true;
                }
                UINT stride=mesh.vertex_stride,offset=mesh.vertex_offset;
                context->IASetVertexBuffers(0,1,&mesh.buffer,&stride,&offset);
                context->IASetIndexBuffer(mesh.indices,mesh.index_format,mesh.index_offset);
                if(mesh.rigid_source){
                    unsigned end=i+1;
                    for(;end<selected.size();++end){
                        auto const& next=selected[end].content();
                        if(!next.rigid_source || next.buffer!=mesh.buffer ||
                            next.indices!=mesh.indices || next.vertex_offset!=mesh.vertex_offset ||
                            next.index_offset!=mesh.index_offset || next.index_count!=mesh.index_count ||
                            next.index_format!=mesh.index_format ||
                            next.projection_kind!=mesh.projection_kind)break;
                    }
                    context->IASetInputLayout(renderer.rigid_sources.layout);
                    context->VSSetShader(renderer.rigid_sources.vertex[mirrored?1:0],nullptr,0);
                    if(layer==geometry_city){
                        context->PSSetShader(mirrored?mirror.ps[1]:renderer.feature_pixel_shader,nullptr,0);
                        context->PSSetShaderResources(116,4,renderer.city_emissive_views.data());
                        context->PSSetShaderResources(124,4,renderer.city_base_views.data());
                        ID3D11SamplerState* samplers[]={renderer.terrain_sampler,renderer.decal_sampler};
                        context->PSSetSamplers(0,2,samplers);
                    }
                    ID3D11Buffer* streams[]={mesh.buffer,renderer.rigid_sources.stream.buffer};
                    UINT strides[]={32,64};
                    UINT offsets[]={mesh.vertex_offset,
                        renderer.rigid_sources.stream.offset+rigid_offset[i]*64};
                    context->IASetVertexBuffers(0,2,streams,strides,offsets);
                    context->DrawIndexedInstanced(mesh.index_count,end-i,0,0,0);
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
                    context->DrawIndexed(mesh.index_count,0,0);++renderer.frame_draw_calls;
                    if(!renderer.cities.library.materials[mesh.city_material].ground){
                        renderer.cities.bind(context,mesh.city_material,mesh.city_environment,
                            mesh.city_atlas,mirrored,true,stride==88);
                        context->DrawIndexed(mesh.index_count,0,0);++renderer.frame_draw_calls;
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
                    auto sample=renderer.water_material;
                    if(!renderer.water_scene_active || !chunk.water_visible() || mesh.visual_time>=0){
                        sample.time=0;sample.drift[0]=sample.drift[1]=sample.drift[2]=0;
                    }
                    context->UpdateSubresource(renderer.water_frame,0,nullptr,&sample,0,0);
                    context->PSSetConstantBuffers(10,1,&renderer.water_frame);
                }
                if(layer==geometry_wave){
                    float sample[]={mesh.visual_time<0?renderer.wave_time_seconds:mesh.visual_time,0,0,0};
                    context->UpdateSubresource(renderer.wave_frame,0,nullptr,sample,0,0);
                }
                context->DrawIndexed(mesh.index_count,0,0);++renderer.frame_draw_calls;
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
        for(auto const& record:records[layer]){
            GeometryDrawReference chunk(record);
            if(mirrored && chunk.content().animation_texture)continue;
            if(!renderer.chunk_intersects_region(chunk,viewport,rect,mirrored))continue;
            ViewportShaderSettings settings=viewport;
            std::copy(std::begin(chunk.natural_projection()),
                std::end(chunk.natural_projection()),settings.natural_projection);
            settings.padding=float(chunk.content().projection_kind);
            if(renderer.pickup_profile)settings.reserved[1]=layer==geometry_underlay?.5f:
                layer==geometry_bed?4.f:layer==geometry_water?5.f:0.f;
            settings.translation[0]+=float(chunk.translation_x());
            settings.translation[1]+=float(chunk.translation_y());
            settings.depth_translation=renderer.city_profile?
                viewport.depth_translation+float(chunk.translation_y()):settings.translation[1];
            values[selected.size()]=settings;selected.push_back(chunk);
            if(selected.size()==values.size() && !flush())return false;
        }
        if(!flush())return false;
        if(streamed)context->VSSetConstantBuffers(1,1,&renderer.viewport_settings_buffer);
        return true;
    }
    void bind_common(ViewportShaderSettings const& settings,D3D11_RECT rect,
            ID3D11RenderTargetView* target,ID3D11DepthStencilView* depth,
            bool mirrored,float scale) {
        auto* context=renderer.context;
        context->OMSetRenderTargets(1,&target,depth);
        context->OMSetDepthStencilState(renderer.depth_state,0);
        context->OMSetBlendState(renderer.blend_state,nullptr,0xffffffffu);
        context->RSSetState(renderer.rasterizer_state);
        D3D11_VIEWPORT viewport={0,0,float(mirrored?reflection.width:glow.linear.width),
            float(mirrored?reflection.height:glow.linear.height),0,1};
        context->RSSetViewports(1,&viewport);
        D3D11_RECT scissor={LONG(rect.left*scale),LONG(rect.top*scale),
            LONG(rect.right*scale),LONG(rect.bottom*scale)};
        context->RSSetScissorRects(1,&scissor);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->IASetInputLayout(renderer.input_layout);
        auto& mirror=renderer.scene_region_size==128?renderer.reflection:renderer.region_reflection;
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
        context->PSSetShaderResources(0,UINT(materials.size()),materials.data());
        context->PSSetShaderResources(25,1,&renderer.source_shadow.view);
        ID3D11ShaderResourceView* reflection_view=mirrored?nullptr:reflection.view;
        context->PSSetShaderResources(121,1,&reflection_view);
        mirror.bind(context,unsigned(reflection.width/reflection_scale),
            unsigned(reflection.height/reflection_scale),4,4);
        (void)settings;
    }
    bool draw_vegetation_instances(GeometryDrawView::Records const& records,
            GeometryLayer layer,ViewportShaderSettings const& settings,bool mirrored) {
        using Stream=c3x_renderer::render_core::InstanceStream;
        auto& group=records[layer];
        if(group.empty())return true;
        auto const& mesh=group.front().content();
        if(!mesh.instances)return false;
        auto* context=renderer.context;
        context->UpdateSubresource(renderer.viewport_settings_buffer,0,nullptr,&settings,0,0);
        renderer.natural.bind_instances(context,unsigned(layer-geometry_natural_forest0));
        if(mirrored)context->VSSetShader(renderer.reflection.instance_vs,nullptr,0);
        std::vector<Stream::Instance> batch;batch.reserve(Stream::limit);
        auto flush=[&]() {
            if(batch.empty())return true;
            if(!renderer.natural.instance_stream.upload(renderer.device,context,batch))return false;
            ID3D11Buffer* streams[]={mesh.buffer,renderer.natural.instance_stream.buffer};
            UINT strides[]={32,64},offsets[]={0,renderer.natural.instance_stream.offset};
            context->IASetVertexBuffers(0,2,streams,strides,offsets);
            context->IASetIndexBuffer(mesh.indices,mesh.index_format,0);
            context->DrawIndexedInstanced(mesh.index_count,UINT(batch.size()),0,0,0);
            batch.clear();return true;
        };
        for(auto const& record:group){
            auto const& content=record.content();
            if(!content.instances || content.buffer!=mesh.buffer)return false;
            for(auto instance:*content.instances){
                if(batch.size()==Stream::limit && !flush())return false;
                std::copy(record.natural_projection,record.natural_projection+4,instance.projection);
                instance.view[0]=settings.translation[0]+float(record.translation_x);
                instance.view[1]=settings.translation[1]+float(record.translation_y);
                instance.view[2]=settings.depth_translation+float(record.translation_y);
                batch.push_back(instance);
            }
        }
        return flush();
    }
    bool draw_layer(GeometryDrawView::Records const& records,GeometryLayer layer,
            ViewportShaderSettings const& settings,D3D11_RECT rect,
            ID3D11RenderTargetView* target,ID3D11DepthStencilView* depth,
            bool mirrored,float scale,bool capture_terrain_material=false) {
        if (records[layer].empty()) return true;
        bind_common(settings,rect,target,depth,mirrored,scale);
        auto* context=renderer.context;
        auto& mirror=renderer.scene_region_size==128?renderer.reflection:renderer.region_reflection;
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
                context->PSSetShaderResources(124,1,&renderer.wall_texture_view);
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
            context->PSSetShaderResources(17,1,&renderer.source_shadow.view);
        if(layer>=geometry_natural_forest0 && records[layer].front().content().instances)
            return draw_vegetation_instances(records,layer,settings,mirrored);
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
        }
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
        auto draw=[&](GeometryLayer layer) {
            return draw_layer(records,layer,settings,rect,target,depth,mirrored,scale);
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
        if(!draw_features(records,settings,rect,target,depth,mirrored,scale))return false;
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
            glow.linear.height==height*scene_scale && glow.linear.samples &&
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
        previous_hour=hour;previous_season=frame.season;
        auto environment=c3x_renderer::evaluate_environment(hour,frame.season);
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
        renderer.context->UpdateSubresource(renderer.terrain_settings_buffer,0,nullptr,&settings,0,0);
        renderer.display_exposure=environment.exposure;
        renderer.shadow_basis=c3x_renderer::fidelity::light_frame(environment);
        float shadow_values[20]={};
        std::copy(renderer.shadow_basis.begin(),renderer.shadow_basis.end(),shadow_values);
        shadow_values[16]=renderer.fidelity_profile && renderer.fidelity_shadow_control?0.f:1.f;
        shadow_values[17]=1.f;
        renderer.context->UpdateSubresource(renderer.shadow_settings_buffer,0,nullptr,shadow_values,0,0);
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
        return renderer.cities.lights(renderer.context,selected);
    }
    bool ensure_linear_target(c3x_renderer::render_core::LinearTarget& target,
            unsigned width,unsigned height,unsigned samples,bool resolved) {
        return target.ensure(renderer.device,width,height,true,resolved,samples);
    }
    bool ensure_targets(unsigned width,unsigned height) {
        char value[8]{};
        scene_scale=1;scene_samples=1;
        char msaa_option[8]={};
        if(GetEnvironmentVariableA("C3X_SANDBOX_MSAA_2X",msaa_option,sizeof(msaa_option)) &&
            msaa_option[0]=='1')scene_samples=2;
        reflection_scale=GetEnvironmentVariableA("C3X_SANDBOX_REFLECTION_FULL",value,sizeof(value)) &&
            std::strcmp(value,"1")==0?1.f:.375f;
        unsigned reflection_width=unsigned((width+8)*reflection_scale);
        unsigned reflection_height=unsigned((height+8)*reflection_scale);
        if(static_cache.width!=width || static_cache.height!=height)static_valid=false;
        if(material_albedo.width!=width || material_albedo.height!=height)
            material_valid=false;
        if(terrain_albedo.width!=width || terrain_albedo.height!=height)
            terrain_material_valid=false;
        if(reflection.width!=reflection_width || reflection.height!=reflection_height){
            reflection_valid=false;
            reflected_terrain_material_valid=false;
        }
        if(!ensure_glow(width,height))return false;
        if(!ensure_linear_target(static_cache,width*scene_scale,
                height*scene_scale,scene_samples,true) ||
           !static_restore.ensure(renderer.device,scene_samples))
            return false;
        if(scene_samples==1 &&
           (!material_albedo.ensure(renderer.device,width,height,true,false,1) ||
            !material_normal.ensure(renderer.device,width,height,DXGI_FORMAT_R8G8B8A8_UNORM) ||
            !material_world.ensure(renderer.device,width,height,DXGI_FORMAT_R16G16B16A16_FLOAT) ||
            !terrain_albedo.ensure(renderer.device,width,height,true,false,1) ||
            !terrain_normal.ensure(renderer.device,width,height,DXGI_FORMAT_R8G8B8A8_UNORM) ||
            !terrain_world.ensure(renderer.device,width,height,DXGI_FORMAT_R16G16B16A16_FLOAT) ||
            !terrain_properties.ensure(renderer.device,width,height,DXGI_FORMAT_R16G16B16A16_FLOAT) ||
            !reflected_terrain_albedo.ensure(renderer.device,reflection_width,
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
    bool prepare_material_cache(ViewportShaderSettings const& settings,D3D11_RECT rect) {
        if(material_valid && material_signature==renderer.cached_signature.complete &&
           material_camera_x==camera_x && material_camera_y==camera_y)return true;
        auto* context=renderer.context;
        float clear[4]={};
        context->ClearRenderTargetView(material_albedo.target,clear);
        context->ClearRenderTargetView(material_normal.target,clear);
        context->ClearRenderTargetView(material_world.target,clear);
        context->ClearDepthStencilView(material_albedo.depth,
            D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);
        auto const& records=static_visible[geometry_underlay];
        if(!records.empty()){
            bind_common(settings,rect,material_albedo.target,material_albedo.depth,
                false,float(scene_scale));
            context->PSSetShader(visual.material_albedo,nullptr,0);
            if(!issue_records(static_visible,geometry_underlay,settings,rect,false))return false;
            bind_common(settings,rect,material_normal.target,material_albedo.depth,
                false,float(scene_scale));
            ID3D11RenderTargetView* outputs[]={material_normal.target,material_world.target};
            context->OMSetRenderTargets(2,outputs,material_albedo.depth);
            context->PSSetShader(visual.material_normal_world,nullptr,0);
            if(!issue_records(static_visible,geometry_underlay,settings,rect,false))return false;
        }
        context->OMSetRenderTargets(0,nullptr,nullptr);
        material_signature=renderer.cached_signature.complete;
        material_camera_x=camera_x;material_camera_y=camera_y;
        material_valid=true;
        return true;
    }
    bool prepare_terrain_material(ViewportShaderSettings const& settings,D3D11_RECT rect) {
        if(terrain_material_valid &&
           terrain_material_signature==renderer.cached_signature.complete &&
           terrain_material_camera_x==camera_x && terrain_material_camera_y==camera_y)
            return true;
        auto* context=renderer.context;
        float clear[4]={};
        context->ClearRenderTargetView(terrain_albedo.target,clear);
        context->ClearRenderTargetView(terrain_normal.target,clear);
        context->ClearRenderTargetView(terrain_world.target,clear);
        context->ClearRenderTargetView(terrain_properties.target,clear);
        context->OMSetRenderTargets(0,nullptr,nullptr);
        context->CopyResource(terrain_albedo.depth_texture,material_albedo.depth_texture);
        for(auto layer:{geometry_natural_terrain,geometry_natural_mountain,
                        geometry_natural_decal})
            if(!draw_layer(static_visible,layer,settings,rect,terrain_albedo.target,
                    terrain_albedo.depth,false,float(scene_scale),true))return false;
        context->OMSetRenderTargets(0,nullptr,nullptr);
        terrain_material_signature=renderer.cached_signature.complete;
        terrain_material_camera_x=camera_x;terrain_material_camera_y=camera_y;
        terrain_material_valid=true;
        return true;
    }
    bool prepare_reflected_terrain_material(ViewportShaderSettings const& settings,
            D3D11_RECT rect) {
        if(reflected_terrain_material_valid &&
           reflected_terrain_material_signature==renderer.cached_signature.complete &&
           reflected_terrain_camera_x==camera_x &&
           reflected_terrain_camera_y==camera_y)return true;
        auto* context=renderer.context;
        float clear[4]={};
        context->ClearRenderTargetView(reflected_terrain_albedo.target,clear);
        context->ClearRenderTargetView(reflected_terrain_normal.target,clear);
        context->ClearRenderTargetView(reflected_terrain_world.target,clear);
        context->ClearRenderTargetView(reflected_terrain_properties.target,clear);
        context->ClearDepthStencilView(reflected_terrain_albedo.depth,
            D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);
        for(auto layer:{geometry_natural_terrain,geometry_natural_mountain,
                        geometry_natural_decal})
            if(!draw_layer(reflection_visible,layer,settings,rect,
                    reflected_terrain_albedo.target,reflected_terrain_albedo.depth,
                    true,reflection_scale,true))return false;
        context->OMSetRenderTargets(0,nullptr,nullptr);
        reflected_terrain_material_signature=renderer.cached_signature.complete;
        reflected_terrain_camera_x=camera_x;reflected_terrain_camera_y=camera_y;
        reflected_terrain_material_valid=true;
        return true;
    }
    void relight_reflected_terrain() {
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
        context->PSSetConstantBuffers(0,1,&renderer.natural.frames[0]);
        context->PSSetConstantBuffers(2,1,&renderer.shadow_settings_buffer);
        context->PSSetConstantBuffers(4,1,&renderer.source_shadow.table);
        context->PSSetShaderResources(17,1,&renderer.source_shadow.view);
        ID3D11ShaderResourceView* inputs[]={reflected_terrain_albedo.samples,
            reflected_terrain_normal.view,reflected_terrain_world.view,
            reflected_terrain_properties.view,reflected_terrain_albedo.depth_samples};
        context->PSSetShaderResources(116,5,inputs);
        context->Draw(3,0);
        ID3D11ShaderResourceView* empty[5]={};
        context->PSSetShaderResources(116,5,empty);
        context->OMSetRenderTargets(0,nullptr,nullptr);
    }
    void relight_terrain(D3D11_RECT rect) {
        auto* context=renderer.context;
        context->OMSetRenderTargets(1,&static_cache.target,static_cache.depth);
        context->OMSetBlendState(renderer.blend_state,nullptr,0xffffffffu);
        context->OMSetDepthStencilState(renderer.depth_state,0);
        context->RSSetState(renderer.rasterizer_state);
        D3D11_VIEWPORT viewport={0,0,float(static_cache.width),float(static_cache.height),0,1};
        context->RSSetViewports(1,&viewport);
        context->RSSetScissorRects(1,&rect);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(static_restore.vertex,nullptr,0);
        context->PSSetShader(visual.terrain_relight,nullptr,0);
        context->PSSetConstantBuffers(0,1,&renderer.natural.frames[0]);
        context->PSSetConstantBuffers(2,1,&renderer.shadow_settings_buffer);
        context->PSSetConstantBuffers(4,1,&renderer.source_shadow.table);
        context->PSSetShaderResources(17,1,&renderer.source_shadow.view);
        ID3D11ShaderResourceView* inputs[]={terrain_albedo.samples,
            terrain_normal.view,terrain_world.view,terrain_properties.view,
            terrain_albedo.depth_samples};
        context->PSSetShaderResources(116,5,inputs);
        context->Draw(3,0);
        ID3D11ShaderResourceView* empty[5]={};
        context->PSSetShaderResources(116,5,empty);
        context->OMSetRenderTargets(0,nullptr,nullptr);
    }
    void relight_underlay(D3D11_RECT rect) {
        auto* context=renderer.context;
        context->OMSetRenderTargets(1,&static_cache.target,nullptr);
        context->OMSetBlendState(nullptr,nullptr,0xffffffffu);
        context->OMSetDepthStencilState(nullptr,0);
        context->RSSetState(renderer.rasterizer_state);
        D3D11_VIEWPORT viewport={0,0,float(static_cache.width),float(static_cache.height),0,1};
        context->RSSetViewports(1,&viewport);
        context->RSSetScissorRects(1,&rect);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(static_restore.vertex,nullptr,0);
        context->PSSetShader(visual.material_relight,nullptr,0);
        context->PSSetConstantBuffers(0,1,&renderer.terrain_settings_buffer);
        context->PSSetConstantBuffers(2,1,&renderer.shadow_settings_buffer);
        context->PSSetConstantBuffers(3,1,&renderer.world_settings_buffer);
        context->PSSetConstantBuffers(4,1,&renderer.source_shadow.table);
        ID3D11ShaderResourceView* inputs[]={material_albedo.samples,
            material_normal.view,material_world.view};
        context->PSSetShaderResources(116,3,inputs);
        context->PSSetShaderResources(25,1,&renderer.source_shadow.view);
        context->Draw(3,0);
        ID3D11ShaderResourceView* empty[3]={};
        context->PSSetShaderResources(116,3,empty);
        context->OMSetRenderTargets(0,nullptr,nullptr);
        context->CopyResource(static_cache.depth_texture,material_albedo.depth_texture);
    }
    bool reconstruct() {
        auto* context=renderer.context;
        context->OMSetRenderTargets(0,nullptr,nullptr);
        if(scene_samples==1)context->CopyResource(glow.linear.resolved,glow.linear.color);
        else context->ResolveSubresource(glow.linear.resolved,0,glow.linear.color,0,
            DXGI_FORMAT_R16G16B16A16_FLOAT);
        return bloom.draw(static_cache.view,glow.linear.view);
    }
    bool draw(c3x_renderer_frame_v1 const& frame,int next_camera_x,int next_camera_y,
            int unit_x,int unit_y,int incarnation,int viewer,bool unit_visible,
            float next_zoom) {
        LARGE_INTEGER frequency{},ticks[7]={};
        QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&ticks[0]);
        int width=renderer.content_view_width,height=renderer.content_view_height;
        if (!renderer.device || !renderer.context || width<1 || height<1) return false;
        if (!visual.install() || !shadow.ensure()) return false;
        unsigned w=unsigned(width)+8,h=unsigned(height)+8;
        if (!ensure_targets(w,h)) return false;
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
        char skip_scene[8]{};
        if (resident_builds && GetEnvironmentVariableA("C3X_SANDBOX_SKIP_SCENE",
                skip_scene,sizeof(skip_scene)) && std::strcmp(skip_scene,"1")==0) {
            std::fill(phases,phases+5,0.0);
            return true;
        }
        renderer.scene_reflection_view=reflection.view;
        renderer.scene_reflection_width=unsigned(reflection.width/reflection_scale)/2;
        renderer.scene_reflection_height=unsigned(reflection.height/reflection_scale)/2;
        renderer.geometry_viewport_settings.translation[0]+=float(next_camera_x-camera_x);
        renderer.geometry_viewport_settings.translation[1]+=float(next_camera_y-camera_y);
        renderer.geometry_viewport_settings.depth_translation+=float(next_camera_y-camera_y);
        camera_x=next_camera_x;camera_y=next_camera_y;
        display_zoom=std::clamp(next_zoom,1.f,1.35f);
        renderer.water_material=c3x_renderer::render_core::water_material_frame(frame);
        int water_camera_x=frame.world_wrap_x && frame.world_width_tiles>0?
            camera_x%(frame.world_width_tiles*frame.tile_width/2):camera_x;
        renderer.water_material.camera[0]-=float(water_camera_x)/frame.tile_width+
            float(camera_y)/frame.tile_height;
        renderer.water_material.camera[1]-=float(water_camera_x)/frame.tile_width-
            float(camera_y)/frame.tile_height;
        renderer.water_time_seconds=renderer.water_material.time;
        renderer.wave_time_seconds=renderer.water_material.time;
        if(!renderer.compose_resource_animations(frame,true))return false;
        auto settings=renderer.geometry_viewport_settings;
        settings.translation[0]+=4;settings.translation[1]+=4;
        settings.inverse_size[0]=1.f/w;settings.inverse_size[1]=1.f/h;
        auto reflected=renderer.geometry_viewport_settings;
        reflected.translation[0]+=8;reflected.translation[1]+=8;
        reflected.inverse_size[0]=1.f/float(w+8);
        reflected.inverse_size[1]=1.f/float(h+8);
        int next_wrap_pixels=frame.world_wrap_x?
            frame.world_width_tiles*frame.tile_width/2:0;
        if (!capture(settings,reflected,int(w),int(h),next_wrap_pixels))return false;
        if(!update_city_lights() || !shadow.render(all_visible))return false;
        QueryPerformanceCounter(&ticks[1]);
        D3D11_RECT full={0,0,LONG(w),LONG(h)};
        char unit_control[8]{};
        bool units=GetEnvironmentVariableA("C3X_SANDBOX_UNITS",unit_control,
            sizeof(unit_control)) && std::strcmp(unit_control,"1")==0;
        // One BIQ water plane currently consumes this mirror target. Each
        // additional authored water level can own another target and pass.
        if (renderer.reflection.enabled &&
            (!water_visible[geometry_water].empty() || !water_visible[geometry_river].empty())) {
            bool redraw=!reflection_valid ||
                reflection_signature!=renderer.cached_signature.complete ||
                reflection_shadow_builds!=shadow.builds ||
                reflection_camera_x!=camera_x || reflection_camera_y!=camera_y;
            if(!redraw)++reflection_reuses;
            else {
            ID3D11ShaderResourceView* none=nullptr;
            renderer.context->PSSetShaderResources(121,1,&none);
            D3D11_RECT mirror={0,0,LONG(w+8),LONG(h+8)};
            if(!prepare_reflected_terrain_material(reflected,mirror))return false;
            float mirror_clear[4]={};
            renderer.context->ClearRenderTargetView(reflection_static.target,mirror_clear);
            renderer.context->ClearDepthStencilView(reflection_static.depth,
                D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);
            relight_reflected_terrain();
            if(!draw_scene(reflection_visible,reflected,mirror,reflection_static.target,
                    reflection_static.depth,true,reflection_scale,false,true))return false;
            renderer.context->OMSetRenderTargets(0,nullptr,nullptr);
            reflection_signature=renderer.cached_signature.complete;
            reflection_shadow_builds=shadow.builds;
            reflection_camera_x=camera_x;reflection_camera_y=camera_y;
            reflection_valid=true;
            ++reflection_draws;
            }
            if(redraw || units){
                renderer.context->OMSetRenderTargets(0,nullptr,nullptr);
                renderer.context->CopyResource(reflection.color,reflection_static.color);
                renderer.context->CopyResource(reflection.depth_texture,reflection_static.depth_texture);
            }
            if(units && !sandbox_direct_units.draw(frame,unit_x,unit_y,
                    incarnation,viewer,unit_visible,camera_x,camera_y,reflection,
                    reflection_scale,next_zoom,visual_hour,true))return false;
        }
        QueryPerformanceCounter(&ticks[2]);
        auto* context=renderer.context;
        bool const cache_ready=scene_scale==1 && static_valid &&
            static_signature==renderer.cached_signature.complete &&
            static_shadow_builds==shadow.builds;
        int shift_x=camera_x-static_camera_x,shift_y=camera_y-static_camera_y;
        bool const scroll_cache=cache_ready && (shift_x || shift_y) &&
            std::abs(shift_x)<int(w/4) && std::abs(shift_y)<int(h/4);
        if(cache_ready && !shift_x && !shift_y){
            // Static color and depth stay resident. Only the dynamic layer
            // is cleared and submitted for this camera.
        }else if(scroll_cache){
            if(!static_restore.draw(context,glow.linear,static_cache.samples,
                    static_cache.depth_samples,shift_x,shift_y,{},nullptr,w,h,
                    false,false,0,nullptr,1))return false;
            std::vector<D3D11_RECT> dirty;
            if(shift_x>0)dirty.push_back({0,0,shift_x,LONG(h)});
            if(shift_x<0)dirty.push_back({LONG(w)+shift_x,0,LONG(w),LONG(h)});
            LONG left=shift_x>0?shift_x:0;
            LONG right=shift_x<0?LONG(w)+shift_x:LONG(w);
            if(shift_y>0)dirty.push_back({left,0,right,shift_y});
            if(shift_y<0)dirty.push_back({left,LONG(h)+shift_y,right,LONG(h)});
            for(auto const& strip:dirty)
                if(!draw_scene(static_visible,settings,strip,glow.linear.target,
                        glow.linear.depth,false,float(scene_scale)))return false;
            context->OMSetRenderTargets(0,nullptr,nullptr);
            glow.linear.swap(static_cache);
            ++cache_scrolls;
        }else{
            float clear[4]={};
            context->ClearRenderTargetView(static_cache.target,clear);
            context->ClearDepthStencilView(static_cache.depth,
                D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);
            if(scene_samples==1){
                if(!prepare_material_cache(settings,full))return false;
                if(!prepare_terrain_material(settings,full))return false;
                relight_underlay(full);
            }
            if(!draw_scene(static_visible,settings,full,static_cache.target,
                    static_cache.depth,false,float(scene_scale),scene_samples==1,
                    scene_samples==1))return false;
            ++cache_full_draws;
        }
        if(scene_scale==1 && (!cache_ready || shift_x || shift_y)){
            context->OMSetRenderTargets(0,nullptr,nullptr);
            if(scene_samples==1)context->CopyResource(static_cache.resolved,static_cache.color);
            else context->ResolveSubresource(static_cache.resolved,0,static_cache.color,0,
                DXGI_FORMAT_R16G16B16A16_FLOAT);
            static_signature=renderer.cached_signature.complete;
            static_shadow_builds=shadow.builds;
            static_camera_x=camera_x;static_camera_y=camera_y;
            static_valid=true;
        }
        QueryPerformanceCounter(&ticks[3]);
        context->OMSetRenderTargets(0,nullptr,nullptr);
        float dynamic_clear[4]={};
        context->ClearRenderTargetView(glow.linear.target,dynamic_clear);
        context->CopyResource(glow.linear.depth_texture,static_cache.depth_texture);
        ++depth_copies;
        bool const aquatic_visible=renderer.water_scene_active &&
            !renderer.sandbox_aquatic_resource_poses[geometry_feature].empty();
        if(aquatic_visible) {
            if(!aquatic_scene.ensure(renderer.device,w,h,DXGI_FORMAT_R8G8B8A8_UNORM))
                return false;
            ID3D11ShaderResourceView* none=nullptr;
            context->PSSetShaderResources(123,1,&none);
            context->ClearRenderTargetView(aquatic_scene.target,dynamic_clear);
            if(!draw_resource_poses(renderer.sandbox_aquatic_resource_poses,settings,full,
                    aquatic_scene.target,nullptr,true))return false;
            context->OMSetRenderTargets(0,nullptr,nullptr);
        }
        char water_diagnostic[8]{};
        bool skip_water=GetEnvironmentVariableA("C3X_SANDBOX_SKIP_WATER_PASS",
            water_diagnostic,sizeof(water_diagnostic)) &&
            std::strcmp(water_diagnostic,"1")==0;
        if (renderer.water_scene_active && !skip_water) {
            if(!draw_scene(water_visible,settings,full,glow.linear.target,
                    glow.linear.depth,false,float(scene_scale)))return false;
        }
        if(!draw_resource_poses(renderer.sandbox_resource_poses,settings,full,
                glow.linear.target,glow.linear.depth))
            return false;
        char wave_diagnostic[8]{};
        bool skip_wave=GetEnvironmentVariableA("C3X_SANDBOX_SKIP_WAVE",
            wave_diagnostic,sizeof(wave_diagnostic)) &&
            std::strcmp(wave_diagnostic,"1")==0;
        if (!skip_water && !skip_wave && !renderer.wave_chunks.empty()) {
            GeometryDrawView::Records waves{};
            for(auto const& chunk:renderer.wave_chunks)
                waves[geometry_wave].emplace_back(chunk);
            if(!draw_layer(waves,geometry_wave,settings,full,glow.linear.target,
                    glow.linear.depth,false,float(scene_scale)))return false;
        }
        QueryPerformanceCounter(&ticks[4]);
        if(units && !sandbox_direct_units.draw(frame,unit_x,unit_y,
            incarnation,viewer,unit_visible,camera_x,camera_y,glow.linear,
            float(scene_scale),next_zoom,visual_hour))
            return false;
        QueryPerformanceCounter(&ticks[5]);
        if(!reconstruct())return false;
        renderer.context->OMSetRenderTargets(0,nullptr,nullptr);
        QueryPerformanceCounter(&ticks[6]);
        for (int i=0;i<6;++i) phases[i]=1000.0*double(ticks[i+1].QuadPart-ticks[i].QuadPart)/
            double(frequency.QuadPart);
        return true;
    }
};

SandboxFreshPipeline sandbox_fresh;

extern "C" __declspec(dllexport) void c3x_sandbox_fresh_metrics(double* phases,
        unsigned* visible,unsigned* shadows,unsigned* resident_builds,
        std::size_t* gpu_bytes,float* shadow_box) {
    if (phases) std::copy(sandbox_fresh.phases,sandbox_fresh.phases+6,phases);
    if (visible) *visible=sandbox_fresh.visible;
    if (shadows) *shadows=sandbox_fresh.shadow.builds;
    if (resident_builds) *resident_builds=sandbox_fresh.resident_builds;
    if (gpu_bytes) *gpu_bytes=sandbox_fresh.static_cache.bytes()+
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
        sandbox_fresh.bloom.bytes()+std::size_t(sandbox_fresh.shadow.resolution)*
            sandbox_fresh.shadow.resolution*4;
    if (shadow_box) std::copy(sandbox_fresh.shadow.box,sandbox_fresh.shadow.box+4,shadow_box);
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
    return sandbox_direct_units.prewarm(hour,season)?0:1;
}

extern "C" __declspec(dllexport) void c3x_sandbox_combat_event(int serial,
        c3x_renderer_i64 presentation_ticks) {
    sandbox_direct_units.combat_event(serial,presentation_ticks);
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
