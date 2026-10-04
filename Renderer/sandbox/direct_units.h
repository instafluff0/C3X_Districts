#pragma once
#include "../native/render_core/frame_sample_cache.h"
#include "pass_workload.h"
#include "../native/render_core/unit_pose_transition.h"
#include "../native/scene_projection.h"
#include "../native/render_core/skin_shadow_bounds.h"
#include "../native/render_core/unit_contribution_plan.h"

// 0 A.D.'s GPUSkinnedModelRenderer keeps mesh inputs resident and updates only
// the animation palette for visible models. Here each authored frame is already
// an immutable palette: the vertex shader selects one exact source frame.
struct SandboxDirectUnits {
    SandboxPassWorkload* work=nullptr;
    struct Vertex {
        float position[3],normal[3],uv[2],tangent[3],bitangent[3];
        std::uint32_t joints[4];float weights[4];
    };
    static_assert(sizeof(Vertex)==88);
    struct Mesh {
        std::size_t bytes=0;std::uint64_t used=0;
        c3x_renderer::render_core::SkinShadowBounds shadow_bounds;
        Microsoft::WRL::ComPtr<ID3D11Buffer> vertices,indices,palettes;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> palette_view;
    };
    std::vector<Mesh> meshes;
    std::size_t mesh_bytes=0,mesh_peak_bytes=0;std::uint64_t mesh_serial=0,mesh_evictions=0;
    c3x_renderer::render_core::UnitPoseTransitions transitions;
    Microsoft::WRL::ComPtr<ID3D11Buffer> transition_palette;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> transition_view;
    ID3D11VertexShader* vertex=nullptr;
    ID3D11PixelShader* pixel=nullptr,*shadow_pixel=nullptr,*height_pixel=nullptr;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> self_shadow;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> self_shadow_target;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> self_shadow_view;
    Microsoft::WRL::ComPtr<ID3D11BlendState> shadow_maximum;
    std::vector<c3x_renderer::UnitShadow::Point> shadow_points;
    c3x_renderer::UnitShadow shadow_fit{512,false};
    ID3D11InputLayout* layout=nullptr;
    ID3D11DepthStencilState* visible_depth=nullptr;
    ID3D11Buffer *material=nullptr,*beauty=nullptr,*placement=nullptr;
    ID3D11Texture2D* unshadowed=nullptr;
    ID3D11ShaderResourceView* unshadowed_view=nullptr;
    ID3D11SamplerState* samplers[4]={};
    int previous_x=INT_MIN,previous_y=INT_MIN;
    int previous_viewer=INT_MIN,previous_incarnation=INT_MIN;
    int move_from_x=0,move_from_y=0;
    int moving_subject=0;
    c3x_renderer_i64 move_started=-1,combat_started=-1;
    int combat_serial=0;
    unsigned draws=0,pose_builds=0,mesh_builds=0;
#ifdef C3X_RENDERER64_FRESH
    using ScenePose=c3x_renderer::render_core::UnitInstances::ScenePose;
    struct MaterialSample {
        std::array<float,32> values{};
        Microsoft::WRL::ComPtr<ID3D11Buffer> constants;
    };
    struct PartSample {
        unsigned frame=0;
        float const* blended=nullptr;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> palette;
        std::array<float,32> material{};
        Microsoft::WRL::ComPtr<ID3D11Buffer> material_buffer;
        std::array<ID3D11ShaderResourceView*,4> textures{};
    };
    struct ShadowSlot {
        Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
        Microsoft::WRL::ComPtr<ID3D11RenderTargetView> target;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> view;
        c3x_renderer::UnitShadow fit{512,false};
        c3x_renderer::render_core::UnitReflectionBounds reflection_bounds;
    };
    struct PreparedUnit {
        ScenePose instance;
        c3x_renderer_unit_v1 draw{};
        c3x_renderer::UnitAnimationPose pose{};
        float low=0,ground_pixels=0,ground_depth=0,angle=0;
        std::vector<PartSample> parts;
        bool main=false,shadow=false,reflected=false;
        unsigned shadow_slot=UINT_MAX;
        c3x_renderer::UnitShadow fit{512,false};
        c3x_renderer::render_core::UnitReflectionBounds reflection_bounds;
    };
    struct PaletteSlot {
        Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> view;
    };
    std::vector<PreparedUnit> prepared_units;
    c3x_renderer::render_core::FrameSampleCache<std::vector<std::uint64_t>,ShadowSlot,64> shared_shadows;
    ShadowSlot working_shadow;
    c3x_renderer::render_core::FrameSampleCache<std::vector<std::uint64_t>,MaterialSample,256> material_samples;
    unsigned material_builds=0,material_reuses=0,shadow_contributors=0;
    unsigned reflection_bounds_builds=0,reflection_bounds_reuses=0,reflection_bounds_rejected=0;
    unsigned material_buffer_builds=0,material_buffer_reuses=0,material_buffer_uploads=0,material_upload_fallbacks=0;
    std::vector<PaletteSlot> prepared_palettes;
    static constexpr unsigned palette_slot_limit=1024;
    unsigned required_samples=0,part_samples=0,shadow_samples=0,shadow_reuses=0,
        shadow_overflow=0,main_contributors=0,reflection_contributors=0,palette_uploads=0;
    float prepared_light[2]={};
    decltype(c3x_renderer::evaluate_environment(12,0)) prepared_environment{};
    std::array<float,20> prepared_beauty{};
#endif
    // Texture aliases are borrowed slots; count each reachable allocation once.
    std::size_t gpu_preparation_bytes()const{
#ifdef C3X_RENDERER64_FRESH
        std::unordered_set<ID3D11Texture2D*> textures;
        auto add=[&](auto const& texture){if(texture)textures.insert(texture.Get());};
        add(self_shadow);add(working_shadow.texture);
        for(unsigned i=0;i<shared_shadows.size();++i)add(shared_shadows[i].value.texture);
        std::size_t bytes=mesh_bytes+textures.size()*512u*512u*4u;
        for(auto const& palette:prepared_palettes)if(palette.buffer)bytes+=16384u;
        if(transition_palette)bytes+=16384u;
        for(unsigned i=0;i<material_samples.size();++i)if(material_samples[i].value.constants)bytes+=128u;
        return bytes;
#else
        return 0;
#endif
    }
    float low_ground(float column,float row){
        if(renderer.natural.low_relief.fields[0].pixels.empty())return 0;
        c3x_renderer::render_core::ExactPointCache<c3x_renderer::render_core::ShoreSample> samples;
        auto ignore=[](auto,auto){};
        int c=int(std::floor(column)),r=int(std::floor(row));
        c3x_renderer::fidelity::SurfaceQueries query(renderer.world_coast,samples,c+r,c-r,ignore,ignore,true,&renderer.natural);
        return query.low_height(renderer.natural,column,row);
    }
    float unit_low_ground(c3x_renderer_frame_v1 const& frame,float x,float y){
        if(renderer.natural.low_relief.fields[0].pixels.empty()||!frame.tile_count)return 0;
        // Invert the captured isometric basis from one authoritative tile.
        // This also follows sub-tile movement without a per-unit tile search.
        auto const& tile=frame.tiles[0];
        float dx=(x-tile.anchor_x-frame.tile_width*.5f)/(frame.tile_width*.5f);
        float dy=(y-tile.anchor_y-frame.tile_height*.5f)/(frame.tile_height*.5f);
        return low_ground((tile.tile_x+tile.tile_y)*.5f+.5f+(dx+dy)*.5f,
            (tile.tile_x-tile.tile_y)*.5f+.5f+(dx-dy)*.5f);
    }
    template<class T>static void drop(T*& p){if(p)p->Release();p=nullptr;}
    ~SandboxDirectUnits(){drop(vertex);drop(pixel);drop(shadow_pixel);drop(height_pixel);drop(layout);drop(visible_depth);drop(material);drop(beauty);
        drop(placement);drop(unshadowed_view);drop(unshadowed);
        for(auto& sampler:samplers)drop(sampler);}
    c3x_renderer::UnitBodyRenderer::Unit const* unit_for(int subject){
        char const* keys[]={"PRTO_Warrior","PRTO_Archer","PRTO_Worker",
            "PRTO_Spearman","PRTO_Scout"};
        auto& units=renderer.unit_bodies.units;
        auto unit=std::find_if(units.begin(),units.end(),[&](auto const& item){
            return std::find(item.keys.begin(),item.keys.end(),keys[subject])!=item.keys.end();});
        return unit==units.end()?nullptr:&*unit;
    }
    c3x_renderer::UnitBodyRenderer::Action const* action_named(
            c3x_renderer::UnitBodyRenderer::Unit const& unit,char const* name){
        auto action=std::find_if(unit.actions.begin(),unit.actions.end(),
            [&](auto const& item){return item.name==name;});
        return action==unit.actions.end()?nullptr:&*action;
    }
    c3x_renderer::UnitBodyRenderer::Action const* action_for(
            c3x_renderer::UnitBodyRenderer::Unit const& unit,int number){
        return action_named(unit,c3x_renderer::native_unit_action(number));
    }
    bool initialize(){
        if(pixel)return true;
        char const* source=R"(
StructuredBuffer<float4> Palettes:register(t0);
cbuffer ScenePlacement:register(b2){
 float2 origin;float2 extent;float scale;float depth_base;float2 padding;
 float4 skin_frame;float4 skin_shape;
 float4 pass_control;float4 shadow_rect;float4 shadow_light;
};
struct Input{float3 position:POSITION;float3 normal:NORMAL;float2 uv:TEXCOORD0;
 float3 tangent:TANGENT;float3 bitangent:BINORMAL;uint4 joints:BLENDINDICES;
 float4 weights:BLENDWEIGHT;};
struct Output{float4 p:SV_Position;float3 n:NORMAL;float2 uv:TEXCOORD0;
 float3 shadow:TEXCOORD1;float3 tangent:TANGENT;float3 bitangent:BINORMAL;};
Output VS(Input i){
 float3 position=0,normal=0,tangent=0,bitangent=0;
 [unroll]for(uint influence=0;influence<4;++influence){
  float weight=i.weights[influence];if(weight<=0)continue;
  uint address=((uint)skin_frame.x*(uint)skin_frame.y+i.joints[influence])*4;
  float3 c0=Palettes[address].xyz,c1=Palettes[address+1].xyz;
  float3 c2=Palettes[address+2].xyz,c3=Palettes[address+3].xyz;
  position+=weight*(i.position.x*c0+i.position.y*c1+i.position.z*c2+c3);
  float3 n0=cross(c1,c2),n1=cross(c2,c0),n2=cross(c0,c1);
  float determinant=dot(c0,n0);
  if(abs(determinant)>1e-12)
   normal+=weight*(i.normal.x*n0+i.normal.y*n1+i.normal.z*n2)/determinant;
  tangent+=weight*(i.tangent.x*c0+i.tangent.y*c1+i.tangent.z*c2);
  bitangent+=weight*(i.bitangent.x*c0+i.bitangent.y*c1+i.bitangent.z*c2);
 }
 float c=skin_frame.z,s=skin_frame.w,unit_scale=skin_shape.x;
 float x=(position.x*c-position.y*s)*unit_scale;
 float y=(position.x*s+position.y*c)*unit_scale;
 float z=(position.z+skin_shape.y)*unit_scale;
 if(pass_control.x>.5 && pass_control.x<1.5){
  x+=z*pass_control.y;y+=z*pass_control.z;z=0;
 }
 float2 local=float2((x-y)*64,(x+y)*32-z*(150.0*128/224));
 float2 pixel=(origin+local)*scale;
 if(pass_control.x>1.5 && pass_control.x<2.5)pixel.y+=2*z*(150.0*128/224)*scale;
 Output o;o.p=float4(pixel/extent*float2(2,-2)+float2(-1,1),
  clamp(.5-(depth_base+(x+y)*5+z*.1)/16384,.001,.999),1);
 float3 n=float3(normal.x*c-normal.y*s,normal.x*s+normal.y*c,normal.z);
 float3 t=float3(tangent.x*c-tangent.y*s,tangent.x*s+tangent.y*c,tangent.z);
 float3 b=float3(bitangent.x*c-bitangent.y*s,bitangent.x*s+bitangent.y*c,bitangent.z);
 o.n=normalize(float3(n.x,-n.y,n.z/(150.0/(112*.82))));
 o.tangent=normalize(float3(t.x,-t.y,t.z/(150.0/(112*.82))));
 o.bitangent=normalize(float3(b.x,-b.y,b.z/(150.0/(112*.82))));
 float2 shadow_uv=(float2(x,y)-shadow_light.xy*z-shadow_rect.xy)*shadow_rect.zw;
 o.uv=i.uv;o.shadow=float3(z,shadow_uv);
 if(pass_control.x>2.5)o.p=float4(shadow_uv*float2(2,-2)+float2(-1,1),0,1);
 return o;
}
Texture2D<float4> shadow_base:register(t0);
SamplerState shadow_sampler:register(s0);
float PSHeight(Output i):SV_Target {
 clip(i.shadow.x-.002);
 if(pass_control.w>.5)clip(shadow_base.Sample(shadow_sampler,i.uv).a-.5);
 return i.shadow.x;
}
float4 PSShadow(Output i):SV_Target {
 if(pass_control.w>.5)clip(shadow_base.Sample(shadow_sampler,i.uv).a-.5);
 return float4(0,0,0,.28);
})";
        ID3DBlob *vs=nullptr,*ps=nullptr,*shadow_ps=nullptr,*height_ps=nullptr,*errors=nullptr;
        HRESULT hr=D3DCompile(source,std::strlen(source),"sandbox_gpu_skin",nullptr,nullptr,
            "VS","vs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&vs,&errors);
        if(errors){if(FAILED(hr))std::printf("SANDBOX_UNIT_SHADER %s\n",
            static_cast<char const*>(errors->GetBufferPointer()));drop(errors);}
        if(SUCCEEDED(hr))hr=D3DCompile(c3x_renderer::unit_material_shader(),
            std::strlen(c3x_renderer::unit_material_shader()),"production_unit_material",
            nullptr,nullptr,"PS","ps_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&ps,&errors);
        if(errors){if(FAILED(hr))std::printf("SANDBOX_UNIT_MATERIAL %s\n",
            static_cast<char const*>(errors->GetBufferPointer()));drop(errors);}
        if(SUCCEEDED(hr))hr=D3DCompile(source,std::strlen(source),"sandbox_unit_shadow",
            nullptr,nullptr,"PSShadow","ps_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,
            &shadow_ps,&errors);
        if(errors){if(FAILED(hr))std::printf("SANDBOX_UNIT_SHADOW %s\n",
            static_cast<char const*>(errors->GetBufferPointer()));drop(errors);}
        if(SUCCEEDED(hr))hr=D3DCompile(source,std::strlen(source),"resident_unit_self_shadow",
            nullptr,nullptr,"PSHeight","ps_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&height_ps,&errors);
        if(errors){if(FAILED(hr))std::printf("UNIT_SELF_SHADOW %s\n",
            static_cast<char const*>(errors->GetBufferPointer()));drop(errors);}
        if(SUCCEEDED(hr))hr=renderer.device->CreateVertexShader(vs->GetBufferPointer(),
            vs->GetBufferSize(),nullptr,&vertex);
        if(SUCCEEDED(hr))hr=renderer.device->CreatePixelShader(ps->GetBufferPointer(),
            ps->GetBufferSize(),nullptr,&pixel);
        if(SUCCEEDED(hr))hr=renderer.device->CreatePixelShader(shadow_ps->GetBufferPointer(),
            shadow_ps->GetBufferSize(),nullptr,&shadow_pixel);
        if(SUCCEEDED(hr))hr=renderer.device->CreatePixelShader(height_ps->GetBufferPointer(),
            height_ps->GetBufferSize(),nullptr,&height_pixel);
        D3D11_INPUT_ELEMENT_DESC elements[]={
            {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,24,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"TANGENT",0,DXGI_FORMAT_R32G32B32_FLOAT,0,32,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"BINORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,44,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"BLENDINDICES",0,DXGI_FORMAT_R32G32B32A32_UINT,0,56,D3D11_INPUT_PER_VERTEX_DATA,0},
            {"BLENDWEIGHT",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,72,D3D11_INPUT_PER_VERTEX_DATA,0}};
        if(SUCCEEDED(hr))hr=renderer.device->CreateInputLayout(elements,7,
            vs->GetBufferPointer(),vs->GetBufferSize(),&layout);
        drop(vs);drop(ps);drop(shadow_ps);drop(height_ps);
        D3D11_BUFFER_DESC b={};b.ByteWidth=128;b.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        if(SUCCEEDED(hr))hr=renderer.device->CreateBuffer(&b,nullptr,&material);
        b.ByteWidth=80;if(SUCCEEDED(hr))hr=renderer.device->CreateBuffer(&b,nullptr,&beauty);
        b.ByteWidth=112;if(SUCCEEDED(hr))hr=renderer.device->CreateBuffer(&b,nullptr,&placement);
        float empty_height=-1.f;
        D3D11_TEXTURE2D_DESC t={};t.Width=t.Height=t.MipLevels=t.ArraySize=t.SampleDesc.Count=1;
        t.Format=DXGI_FORMAT_R32_FLOAT;t.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA blank={&empty_height,4,0};
        if(SUCCEEDED(hr))hr=renderer.device->CreateTexture2D(&t,&blank,&unshadowed);
        if(SUCCEEDED(hr))hr=renderer.device->CreateShaderResourceView(unshadowed,nullptr,&unshadowed_view);
        for(unsigned mode=0;mode<4 && SUCCEEDED(hr);++mode){
            D3D11_SAMPLER_DESC s={};s.Filter=D3D11_FILTER_ANISOTROPIC;s.MaxAnisotropy=16;
            s.AddressU=(mode&1)?D3D11_TEXTURE_ADDRESS_CLAMP:D3D11_TEXTURE_ADDRESS_WRAP;
            s.AddressV=(mode&2)?D3D11_TEXTURE_ADDRESS_CLAMP:D3D11_TEXTURE_ADDRESS_WRAP;
            s.AddressW=D3D11_TEXTURE_ADDRESS_WRAP;s.MaxLOD=D3D11_FLOAT32_MAX;
            hr=renderer.device->CreateSamplerState(&s,&samplers[mode]);
        }
        return SUCCEEDED(hr);
    }
    bool reserve_mesh_bytes(std::size_t bytes){
        auto const& bodies=renderer.unit_bodies;auto limit=bodies.source_gpu_limit;
        if(bytes>limit)return false;
        while(mesh_bytes>limit-bytes){
            unsigned victim=UINT_MAX;std::uint64_t oldest=UINT64_MAX;
            for(unsigned i=0;i<meshes.size();++i)if(meshes[i].bytes &&
                (i>=bodies.meshes.size()||!bodies.meshes[i].source_pinned) &&
                (i>=renderer.frame_mesh_leases.size()||!renderer.frame_mesh_leases[i]) && meshes[i].used<oldest){victim=i;oldest=meshes[i].used;}
            if(victim==UINT_MAX)return false;
            mesh_bytes-=meshes[victim].bytes;meshes[victim]=Mesh{};++mesh_evictions;
        }
        return true;
    }
    bool prepare_mesh(unsigned index){
        auto& bodies=renderer.unit_bodies;
        if(index>=bodies.meshes.size())return false;
        if(meshes.size()<bodies.meshes.size())meshes.resize(bodies.meshes.size());
        auto& gpu=meshes[index];if(gpu.vertices){
            if(!reserve_mesh_bytes(0))return false;
            if(gpu.vertices){gpu.used=++mesh_serial;return true;}}
        auto const& mesh=bodies.meshes[index].animation;
        if(!mesh||mesh->vertices.empty()||mesh->indices.empty()||mesh->palettes.empty())return false;
        auto bytes=mesh->vertices.size()*sizeof(Vertex)+(mesh->indices.size()+mesh->palettes.size())*4;
        if(!reserve_mesh_bytes(bytes))return false;
        Mesh next;
        std::vector<Vertex> input(mesh->vertices.size());
        for(std::size_t n=0;n<input.size();++n){
            auto const& source=mesh->vertices[n];auto& output=input[n];
            std::copy(source.source.position,source.source.position+3,output.position);
            std::copy(source.source.normal,source.source.normal+3,output.normal);
            std::copy(source.source.uv,source.source.uv+2,output.uv);
            std::copy(source.tangent.begin(),source.tangent.end(),output.tangent);
            std::copy(source.bitangent.begin(),source.bitangent.end(),output.bitangent);
            std::copy(source.joints.begin(),source.joints.end(),output.joints);
            std::copy(source.weights.begin(),source.weights.end(),output.weights);
        }
        D3D11_BUFFER_DESC b={};D3D11_SUBRESOURCE_DATA data={};
        b.ByteWidth=UINT(input.size()*sizeof(Vertex));b.Usage=D3D11_USAGE_IMMUTABLE;
        b.BindFlags=D3D11_BIND_VERTEX_BUFFER;data.pSysMem=input.data();
        if(FAILED(renderer.device->CreateBuffer(&b,&data,&next.vertices)))return false;
        b.ByteWidth=UINT(mesh->indices.size()*4);b.BindFlags=D3D11_BIND_INDEX_BUFFER;
        data.pSysMem=mesh->indices.data();
        if(FAILED(renderer.device->CreateBuffer(&b,&data,&next.indices)))return false;
        b.ByteWidth=UINT(mesh->palettes.size()*4);b.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        b.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;b.StructureByteStride=16;
        data.pSysMem=mesh->palettes.data();
        if(FAILED(renderer.device->CreateBuffer(&b,&data,&next.palettes)))return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC view={};view.Format=DXGI_FORMAT_UNKNOWN;
        view.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;
        view.Buffer.NumElements=UINT(mesh->palettes.size()/4);
        if(FAILED(renderer.device->CreateShaderResourceView(next.palettes.Get(),&view,
            &next.palette_view)))return false;
        next.shadow_bounds.prepare(*mesh);next.bytes=bytes;next.used=++mesh_serial;
        gpu=std::move(next);mesh_bytes+=bytes;mesh_peak_bytes=std::max(mesh_peak_bytes,mesh_bytes);
        ++mesh_builds;return true;
    }
#ifdef C3X_RENDERER64_FRESH
    int prepare_frame_meshes(){
        if(!initialize())return C3X_RENDERER_RESULT_ERROR;
        if(!reserve_mesh_bytes(0))return C3X_RENDERER_RESULT_ERROR;
        auto began=std::chrono::steady_clock::now();unsigned adopted=0;bool ready=true;
        for(unsigned i=0;i<renderer.frame_mesh_leases.size();++i)if(renderer.frame_mesh_leases[i]){
            if(i<meshes.size()&&meshes[i].vertices)continue;
            if(adopted>=2 || (adopted&&std::chrono::steady_clock::now()-began>std::chrono::milliseconds(3))){ready=false;continue;}
            if(!prepare_mesh(i))return C3X_RENDERER_RESULT_ERROR;
            ++adopted;
        }
        if(adopted){char detail[256];std::snprintf(detail,sizeof(detail),
            "adopted=%u resident_bytes=%zu peak_bytes=%zu evictions=%llu ms=%.3f complete=%u",
            adopted,mesh_bytes,mesh_peak_bytes,mesh_evictions,
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count(),unsigned(ready));
            renderer.trace.write("frame-mesh-turn",detail,true);}
        return ready?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_PENDING;
    }
#endif
    bool prewarm(int,int){
        if(!initialize())return false;
        char moving[16]{};
        if(GetEnvironmentVariableA("C3X_SANDBOX_MOVING_UNIT",moving,sizeof(moving))){
            if(std::strcmp(moving,"Spearman")==0)moving_subject=3;
            else if(std::strcmp(moving,"Scout")==0)moving_subject=4;
        }
        char const* needed[5][4]={{"idle","move",nullptr,nullptr},
            {"idle","attack","defend",nullptr},{"idle","road",nullptr,nullptr},
            {"idle","attack","defend",moving_subject==3?"move":nullptr},
            {"idle","move",nullptr,nullptr}};
        for(int subject=0;subject<(moving_subject==4?5:4);++subject){
            auto* unit=unit_for(subject);if(!unit)return false;
            for(auto const* name:needed[subject]){
                if(!name)continue;
                auto* action=action_named(*unit,name);if(!action)return false;
                if(!renderer.prepare_unit_action(*action))return false;
                for(auto const& part:action->parts)if(!prepare_mesh(part.mesh))return false;
            }
        }
        return true;
    }
    void combat_event(int serial,c3x_renderer_i64 ticks){
        if(serial>combat_serial){combat_serial=serial;combat_started=ticks;}
    }
    template<class Target>bool draw(c3x_renderer_frame_v1 const& frame,int world_x,int world_y,
            int incarnation,int viewer,bool visible,int camera_x,int camera_y,
            Target& scene,float scene_scale,float zoom,float visual_hour,
            bool reflected=false){
        SandboxPassWorkload::Scope pass(*work,reflected?SandboxPassWorkload::reflected_units:SandboxPassWorkload::units);
        if(!initialize())return false;
        auto& bodies=renderer.unit_bodies;
        if(incarnation!=previous_incarnation || viewer!=previous_viewer || previous_x==INT_MIN){
            previous_x=world_x;previous_y=world_y;move_started=-1;
        }else if(world_x!=previous_x || world_y!=previous_y){
            move_from_x=previous_x;move_from_y=previous_y;
            previous_x=world_x;previous_y=world_y;move_started=frame.presentation_time_ticks;
        }
        previous_incarnation=incarnation;previous_viewer=viewer;
        int positions[4][2]={{world_x,world_y},{25,57},{24,58},{24,56}};
        auto locate=[&](int x,int y){
            c3x_renderer_tile_v1 const* found=nullptr;int best=INT_MAX;
            for(unsigned i=0;i<frame.tile_count;++i){auto const& tile=frame.tiles[i];
                if(tile.tile_y!=y || (tile.tile_x%frame.world_width_tiles+
                    frame.world_width_tiles)%frame.world_width_tiles!=x)continue;
                int distance=std::abs(tile.anchor_x-frame.target_width/2)+
                    std::abs(tile.anchor_y-frame.target_height/2);
                if(distance<best){best=distance;found=&tile;}
            }return found;
        };
        auto* context=renderer.context;
        context->OMSetRenderTargets(1,&scene.target,scene.depth);
        context->OMSetDepthStencilState(renderer.depth_state,0);
        context->OMSetBlendState(nullptr,nullptr,0xffffffffu);
        context->RSSetState(renderer.rasterizer_state);
        D3D11_VIEWPORT viewport={0,0,float(scene.width),float(scene.height),0,1};
        c3x_renderer::SceneProjection(frame.target_width,frame.target_height,zoom)
            .viewport(viewport,reflected?8.f:4.f,0,0,scene_scale);
        D3D11_RECT scissor={0,0,LONG(scene.width),LONG(scene.height)};
        context->RSSetViewports(1,&viewport);context->RSSetScissorRects(1,&scissor);
        context->IASetInputLayout(layout);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex,nullptr,0);context->PSSetShader(pixel,nullptr,0);
        context->VSSetConstantBuffers(2,1,&placement);
        context->PSSetConstantBuffers(2,1,&placement);
        context->PSSetConstantBuffers(0,1,&material);
        context->PSSetConstantBuffers(1,1,&beauty);
        context->PSSetShaderResources(1,1,&unshadowed_view);
        context->PSSetSamplers(1,1,&samplers[3]);
        auto environment=c3x_renderer::evaluate_environment(visual_hour,frame.season);
        auto noon=c3x_renderer::evaluate_environment(12,0);
        auto key_light=c3x_renderer::lighting::key_light(environment);
        float beauty_values[20]={};float const ambient_source[]={.34f,.45f,.60f};
        float const chromatic[]={1.f,4.5f/6.2f,3.5f/6.2f};
        auto beauty_light=key_light.direction.data();auto light_color=key_light.color.data();
        for(unsigned axis=0;axis<3;++axis){
            beauty_values[axis]=beauty_light[axis];
            beauty_values[4+axis]=chromatic[axis]*light_color[axis]/
                std::max(.001f,noon.sun_color[axis]);
            beauty_values[8+axis]=ambient_source[axis]*environment.ambient_color[axis]/
                std::max(.001f,noon.ambient_color[axis]);
        }
        beauty_values[3]=2.05f*(environment.sun_intensity+environment.moon_intensity)/
            (noon.sun_intensity+noon.moon_intensity);
        beauty_values[7]=1;beauty_values[11]=.62f;
        beauty_values[12]=.490290f;beauty_values[13]=-.735435f;
        beauty_values[14]=.469979f;beauty_values[16]=1;
        context->UpdateSubresource(beauty,0,nullptr,beauty_values,0,0);work->upload_buffer(beauty);
        double seconds=double(frame.presentation_time_ticks)/
            double(std::max<c3x_renderer_i64>(1,frame.presentation_frequency));
        double combat_seconds=combat_started<0?-1:double(frame.presentation_time_ticks-combat_started)/
            double(std::max<c3x_renderer_i64>(1,frame.presentation_frequency));
        auto clip_duration=[&](int subject,char const* action_name){
            auto* unit=unit_for(subject);
            auto* action=unit?action_named(*unit,action_name):nullptr;
            return action&&!action->parts.empty()&&action->parts[0].mesh<bodies.meshes.size()&&
                bodies.meshes[action->parts[0].mesh].animation?
                double(bodies.meshes[action->parts[0].mesh].animation->duration):0.0;
        };
        double archer_attack=clip_duration(1,"attack"),spearman_attack=clip_duration(3,"attack");
        double first_exchange=archer_attack,second_exchange=archer_attack+spearman_attack;
        double third_exchange=second_exchange+archer_attack;
        double combat_end=third_exchange+spearman_attack;
        for(int index=0;index<4;++index){
            if(index==0&&!visible)continue;
            auto* unit=unit_for(index==0?moving_subject:index);if(!unit)return false;
            auto* anchor=locate(positions[index][0],positions[index][1]);
            if(!anchor)continue;
            float x=float(anchor->anchor_x),y=float(anchor->anchor_y);
            float world_row=float(anchor->tile_y);
            float world_column=float(anchor->tile_x);
            double move_seconds=0,travel=1;
            if(index==0&&move_started>=0)if(auto* from=locate(move_from_x,move_from_y)){
                double dx=double(x-float(from->anchor_x))*zoom;
                double dy=double(y-float(from->anchor_y))*zoom;
                double distance=std::hypot(dx,2.0*dy);
                move_seconds=double(frame.presentation_time_ticks-move_started)/
                    double(std::max<c3x_renderer_i64>(1,frame.presentation_frequency));
                travel=distance>0?std::clamp(move_seconds*225.0/distance,0.0,1.0):1;
                x=float(from->anchor_x)+(x-float(from->anchor_x))*float(travel);
                y=float(from->anchor_y)+(y-float(from->anchor_y))*float(travel);
                world_row=float(from->tile_y)+(world_row-float(from->tile_y))*float(travel);
                world_column=float(from->tile_x)+(world_column-float(from->tile_x))*float(travel);
            }
            if(index==0&&move_started>=0&&travel>=1)move_started=-1;
            float low=low_ground((world_column+world_row)*.5f+.5f,(world_column-world_row)*.5f+.5f);
            float ground_pixels=low*frame.tile_width/224.f*.82f;
            int body_x=int(std::lround(x))+camera_x+frame.tile_width/2-95;
            int body_y=int(std::lround(y))+camera_y+frame.tile_height/2-95;
            if(frame.world_wrap_x && frame.world_width_tiles>0){
                int span=frame.world_width_tiles*frame.tile_width/2;
                while(body_x>frame.target_width+191)body_x-=span;
                while(body_x< -191)body_x+=span;
            }
            if(body_x>frame.target_width || body_x+191<0 ||
                body_y>frame.target_height || body_y+191<0)continue;
            int action_number=index==0&&move_started>=0?2:index==2?13:1;
            double action_seconds=action_number==2?move_seconds:seconds+index*.137;
            char const* combat_action=nullptr;
            if(combat_seconds>=0&&combat_seconds<combat_end&&(index==1||index==3)){
                int exchange=combat_seconds<first_exchange?0:
                    combat_seconds<second_exchange?1:combat_seconds<third_exchange?2:3;
                double start=exchange==0?0:exchange==1?first_exchange:
                    exchange==2?second_exchange:third_exchange;
                bool archer_attacks=(exchange&1)==0;
                combat_action=(archer_attacks==(index==1))?"attack":"defend";
                action_seconds=combat_seconds-start;
            }
            auto* action=combat_action?action_named(*unit,combat_action):
                action_for(*unit,action_number);
            if(!action)return false;
            float ground_depth=world_row*frame.tile_height*.5f+
                renderer.geometry_viewport_settings.depth_translation+
                frame.tile_height*.5f+4.f;
            for(auto const& part:action->parts){
                if(part.mesh>=bodies.meshes.size() || part.texture>=bodies.textures.size() ||
                    !bodies.textures[part.texture].view)return false;
                auto const& source=bodies.meshes[part.mesh].animation;
                if(!source||part.mesh>=meshes.size()||!meshes[part.mesh].vertices)return false;
                auto& gpu=meshes[part.mesh];
                double duration=std::max(.001,double(source->duration));
                double local=action->loop?std::fmod(std::max(0.0,action_seconds),duration):
                    std::clamp(action_seconds,0.0,duration);
                unsigned frame_number=std::min(source->frames-1,
                    unsigned(std::floor(local/duration*double(source->frames-1)+1e-7)));
                float angle=(unit->yaw_offset+float((index==1?5:index==3?1:3)%8)*45)*.01745329252f;
                float placement_values[28]={float(body_x+(reflected?8:4))+95.5f,
                    float(body_y+(reflected?8:4))+95.5f+(reflected?ground_pixels:-ground_pixels),
                    float(scene.width),float(scene.height),float(scene_scale),ground_depth+low*.0016f*frame.target_height,0,0,
                    float(frame_number),float(source->bones),std::cos(angle),std::sin(angle),
                    unit->scale,unit->offset_z,0,0,
                    reflected?2.f:0.f,0,0,part.cutout};
                float values[32]={part.tint[0],part.tint[1],part.tint[2],part.mask};
                for(unsigned axis=0;axis<3;++axis){
                    float color=float((0x205bddu>>(16-axis*8))&255)/255;
                    values[4+axis]=color<=.04045f?color/12.92f:
                        std::pow((color+.055f)/1.055f,2.4f);
                    values[8+axis]=environment.sun_direction[axis];
                    values[12+axis]=environment.sun_color[axis];
                    values[16+axis]=environment.moon_direction[axis];
                    values[20+axis]=environment.moon_color[axis];
                    values[24+axis]=environment.ambient_color[axis];
                }
                values[7]=part.strength;values[11]=environment.sun_intensity;
                values[19]=environment.moon_intensity;values[27]=part.cutout;
                ID3D11ShaderResourceView* extra[4]={};
                for(unsigned channel=0;channel<4;++channel)
                    if(part.material_textures[channel]!=UINT32_MAX){
                        unsigned texture=part.material_textures[channel];
                        if(texture>=bodies.textures.size()||!bodies.textures[texture].view)return false;
                        extra[channel]=bodies.textures[texture].view;values[28+channel]=1;
                    }
                values[23]=part.material_model;
                context->UpdateSubresource(material,0,nullptr,values,0,0);work->upload_buffer(material);
                ID3D11Buffer* static_vertices=gpu.vertices.Get();
                UINT stride=sizeof(Vertex),offset=0;
                context->IASetVertexBuffers(0,1,&static_vertices,&stride,&offset);
                context->IASetIndexBuffer(gpu.indices.Get(),DXGI_FORMAT_R32_UINT,0);
                ID3D11ShaderResourceView* palette=gpu.palette_view.Get();
                context->VSSetShaderResources(0,1,&palette);
                context->PSSetShaderResources(0,1,&bodies.textures[part.texture].view);
                context->PSSetShaderResources(2,4,extra);
                context->PSSetSamplers(0,1,&samplers[part.address]);
                if(!reflected && key_light.intensity>.001f){
                    placement_values[16]=1;
                    placement_values[17]=-key_light.direction[0]/key_light.direction[2]*
                        c3x_renderer::lighting::object_height_to_world;
                    // Source-local Y runs opposite world V; match the shared
                    // resource and terrain shadow projection.
                    placement_values[18]=key_light.direction[1]/key_light.direction[2]*
                        c3x_renderer::lighting::object_height_to_world;
                    context->UpdateSubresource(placement,0,nullptr,placement_values,0,0);work->upload_buffer(placement);
                    context->OMSetDepthStencilState(renderer.natural.decal_depth,0);
                    context->OMSetBlendState(renderer.blend_state,nullptr,0xffffffffu);
                    context->PSSetShader(shadow_pixel,nullptr,0);
                    context->DrawIndexed(UINT(source->indices.size()),0,0);work->draw(source->indices.size());
                    placement_values[16]=0;
                    context->OMSetDepthStencilState(renderer.depth_state,0);
                    context->OMSetBlendState(nullptr,nullptr,0xffffffffu);
                    context->PSSetShader(pixel,nullptr,0);
                }
                context->UpdateSubresource(placement,0,nullptr,placement_values,0,0);work->upload_buffer(placement);
                context->DrawIndexed(UINT(source->indices.size()),0,0);work->draw(source->indices.size());
                ++draws;
            }
        }
        ID3D11ShaderResourceView* empty[6]={};context->PSSetShaderResources(0,6,empty);
        context->VSSetShaderResources(0,1,empty);
        context->OMSetRenderTargets(0,nullptr,nullptr);
        return true;
    }

#ifdef C3X_RENDERER64_FRESH
    template<class Unit,class Action,class Instance>bool draw_self_shadow(Unit const& unit,
            Action const& action,Instance const& instance,c3x_renderer::UnitAnimationPose const& pose,
            float angle,c3x_renderer_frame_v1 const& frame,float light_x,float light_y
#ifdef C3X_RENDERER64_FRESH
            ,PreparedUnit const* prepared=nullptr
#endif
            ){
        SandboxPassWorkload::Scope pass(*work,SandboxPassWorkload::unit_shadow);
        auto* context=renderer.context;auto& bodies=renderer.unit_bodies;
        if(!prepared){self_shadow=working_shadow.texture;self_shadow_target=working_shadow.target;self_shadow_view=working_shadow.view;}
        if(!self_shadow){
            D3D11_TEXTURE2D_DESC t={};t.Width=t.Height=shadow_fit.extent;
            t.MipLevels=t.ArraySize=t.SampleDesc.Count=1;t.Format=DXGI_FORMAT_R32_FLOAT;
            t.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
            if(FAILED(renderer.device->CreateTexture2D(&t,nullptr,&self_shadow)) ||
               FAILED(renderer.device->CreateRenderTargetView(self_shadow.Get(),nullptr,&self_shadow_target)) ||
               FAILED(renderer.device->CreateShaderResourceView(self_shadow.Get(),nullptr,&self_shadow_view)))return false;
        }
        if(!shadow_maximum){
            D3D11_BLEND_DESC b={};auto& r=b.RenderTarget[0];r.BlendEnable=TRUE;
            r.SrcBlend=r.DestBlend=r.SrcBlendAlpha=r.DestBlendAlpha=D3D11_BLEND_ONE;
            r.BlendOp=r.BlendOpAlpha=D3D11_BLEND_OP_MAX;r.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_RED;
            if(FAILED(renderer.device->CreateBlendState(&b,&shadow_maximum)))return false;
        }
        unsigned part_index=0;
#ifdef C3X_RENDERER64_FRESH
        if(prepared)shadow_fit=prepared->fit;
        else
#endif
        {
        shadow_points.clear();
        for(auto const& part:action.parts){
            if(part.mesh>=bodies.meshes.size()||part.texture>=bodies.textures.size()||
               !bodies.textures[part.texture].view||!prepare_mesh(part.mesh))return false;
            auto const& source=*bodies.meshes[part.mesh].animation;
            unsigned n=std::min(source.frames-1,unsigned(std::floor(pose.phase*double(source.frames-1)+1e-7)));
            float const* blended;
#ifdef C3X_RENDERER64_FRESH
            if(prepared){auto const& sample=prepared->parts[part_index++];n=sample.frame;blended=sample.blended;}
            else
#endif
                blended=transitions.sample(instance.draw.unit_id,instance.pose_identity,instance.draw.action,
                    frame.presentation_time_ticks,frame.presentation_frequency,source,n);
            auto* palette=blended?blended:source.palettes.data()+std::size_t(n)*source.bones*16;
            meshes[part.mesh].shadow_bounds.append(palette,angle,unit.scale,unit.offset_z,shadow_points);
        }
        if(!shadow_fit.fit(shadow_points,light_x,light_y,true))return false;
        }
        ID3D11ShaderResourceView* empty=nullptr;context->PSSetShaderResources(1,1,&empty);
        auto target=self_shadow_target.Get();float clear[4]={-1,-1,-1,-1};
        context->ClearRenderTargetView(target,clear);work->clear(target);context->OMSetRenderTargets(1,&target,nullptr);
        context->OMSetBlendState(shadow_maximum.Get(),nullptr,~0u);
        D3D11_VIEWPORT viewport={0,0,float(shadow_fit.extent),float(shadow_fit.extent),0,1};
        D3D11_RECT scissor={0,0,shadow_fit.extent,shadow_fit.extent};
        context->RSSetViewports(1,&viewport);context->RSSetScissorRects(1,&scissor);
        context->PSSetShader(height_pixel,nullptr,0);
        part_index=0;
        for(auto const& part:action.parts){
            auto const& source=*bodies.meshes[part.mesh].animation;auto& gpu=meshes[part.mesh];
            unsigned n=std::min(source.frames-1,unsigned(std::floor(pose.phase*double(source.frames-1)+1e-7)));
            float const* blended;
#ifdef C3X_RENDERER64_FRESH
            if(prepared){auto const& sample=prepared->parts[part_index++];n=sample.frame;blended=sample.blended;}
            else
#endif
                blended=transitions.sample(instance.draw.unit_id,instance.pose_identity,instance.draw.action,
                    frame.presentation_time_ticks,frame.presentation_frequency,source,n);
#ifdef C3X_RENDERER64_FRESH
            if(prepared){auto* palette=prepared->parts[part_index-1].palette.Get();
                if(!palette && blended){if(!bind_palette(gpu,blended))return false;++palette_uploads;}
                else context->VSSetShaderResources(0,1,&palette);}
            else
#endif
                if(!bind_palette(gpu,blended))return false;
            float values[28]={0,0,1,1,1,0,0,0,float(blended?0:n),float(source.bones),
                std::cos(angle),std::sin(angle),unit.scale,unit.offset_z,0,0,3,0,0,part.cutout,
                shadow_fit.left,shadow_fit.top,1/shadow_fit.width,1/shadow_fit.height,shadow_fit.dx,shadow_fit.dy};
            context->UpdateSubresource(placement,0,nullptr,values,0,0);work->upload_buffer(placement);
            ID3D11Buffer* vertices=gpu.vertices.Get();UINT stride=sizeof(Vertex),offset=0;
            context->IASetVertexBuffers(0,1,&vertices,&stride,&offset);
            context->IASetIndexBuffer(gpu.indices.Get(),DXGI_FORMAT_R32_UINT,0);
            context->PSSetShaderResources(0,1,&bodies.textures[part.texture].view);
            context->PSSetSamplers(0,1,&samplers[part.address]);
            context->DrawIndexed(UINT(source.indices.size()),0,0);work->draw(source.indices.size());
        }
        context->OMSetRenderTargets(0,nullptr,nullptr);
        if(!prepared){working_shadow.texture=self_shadow;working_shadow.target=self_shadow_target;working_shadow.view=self_shadow_view;}
        return true;
    }
    bool bind_palette(Mesh const& mesh,float const* blended){
        if(blended&&!transition_palette){
            D3D11_BUFFER_DESC b={};b.ByteWidth=4096*4;b.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            b.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;b.StructureByteStride=16;
            if(FAILED(renderer.device->CreateBuffer(&b,nullptr,&transition_palette)))return false;
            D3D11_SHADER_RESOURCE_VIEW_DESC v={};v.Format=DXGI_FORMAT_UNKNOWN;
            v.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;v.Buffer.NumElements=1024;
            if(FAILED(renderer.device->CreateShaderResourceView(transition_palette.Get(),&v,&transition_view)))return false;
        }
        if(blended){renderer.context->UpdateSubresource(transition_palette.Get(),0,nullptr,blended,0,0);work->upload_buffer(transition_palette.Get());}
        ID3D11ShaderResourceView* palette=blended?transition_view.Get():mesh.palette_view.Get();
        renderer.context->VSSetShaderResources(0,1,&palette);return true;
    }
    using ContributionPlan=c3x_renderer::render_core::UnitContributionPlan;
    // Selection runs before payload admission. No body palette, ground query,
    // GPU allocation or material evaluation is required to form the union.
    bool select_real(c3x_renderer_frame_v1 const& frame,std::vector<ScenePose> const& candidates,
            float visual_hour,float zoom,std::vector<D3D11_RECT> const& water_receivers,ContributionPlan& plan){
        using namespace c3x_renderer::render_core;
        UnitContributionView view;view.width=renderer.content_view_width;view.height=renderer.content_view_height;view.zoom=zoom;
        auto environment=c3x_renderer::evaluate_environment(visual_hour,frame.season);
        auto light=c3x_renderer::lighting::key_light(environment);
        view.shadow=light.intensity>.001f;view.reflection=renderer.reflection.enabled;
        view.shadow_x=-light.direction[0]/light.direction[2]*c3x_renderer::lighting::object_height_to_world;
        view.shadow_y=light.direction[1]/light.direction[2]*c3x_renderer::lighting::object_height_to_world;
        for(auto const& r:water_receivers)view.receivers.push_back({double(r.left),double(r.top),double(r.right),double(r.bottom)});
        // low_height is a convex nonnegative two-field sum, multiplied by
        // clamped coast/river ramps. The retained field amplitudes bound every
        // camera/anchor without constructing per-unit terrain query scratch.
        double maximum=0;bool ground_known=true;
        if(frame.tile_count&&!renderer.natural.low_relief.fields[0].pixels.empty())
            for(auto const& field:renderer.natural.low_relief.fields){
                ground_known&=std::isfinite(field.amplitude)&&field.amplitude>=0;
                maximum=std::max(maximum,double(field.amplitude));}
        double ground=maximum*frame.tile_width/224.*.82;
        ground_known&=std::isfinite(ground)&&ground>=0;
        std::vector<UnitContributionCandidate> inputs;inputs.reserve(candidates.size());
        auto& bodies=renderer.unit_bodies;
        for(auto const& instance:candidates){
            if(instance.unit>=bodies.units.size()||instance.action>=bodies.units[instance.unit].actions.size()){plan={};plan.valid=false;return false;}
            auto const& unit=bodies.units[instance.unit];auto const& action=unit.actions[instance.action];auto const& d=instance.draw;
            c3x_renderer::NativeUnitDraw native{};native.expected_sprite=native.sprite=native.expected_canvas=native.canvas=1;
            native.unit_id=d.unit_id;native.action=d.action;native.direction=d.direction;
            native.action_cursor=d.action_cursor;native.frame_count=d.frame_count;native.body_x=d.body_x;native.body_y=d.body_y;
            native.sprite_width=d.sprite_width;native.sprite_height=d.sprite_height;native.reduced=d.reduced!=0;
            native.projection_scale_milli=d.projection_scale_milli;
            c3x_renderer::UnitAnimationPose pose;
            if(!c3x_renderer::prepare_native_unit_pose(native,action.loop,pose)){plan={};plan.valid=false;return false;}
            UnitContributionCandidate input;input.visible=true; // caller's native VISIBLE occurrence contract
            input.anchor_x=pose.anchor_x;input.anchor_y=pose.anchor_y;input.projection_scale=pose.projection_scale;
            input.model_scale=unit.scale;input.offset_z=unit.offset_z;input.bounds=bodies.contribution_bound(instance.unit);
            input.ground_known=ground_known;input.ground_min=0;input.ground_max=ground*1.0001+1e-4;
            inputs.push_back(input);
        }
        plan=ContributionPlan::build(inputs,view);return true;
    }
    bool prepare_real(c3x_renderer_frame_v1 const& frame,std::vector<ScenePose> const& visible,
            float visual_hour,float zoom,std::vector<D3D11_RECT> const& water_receivers){
        ContributionPlan plan;if(!select_real(frame,visible,visual_hour,zoom,water_receivers,plan))return false;
        return prepare_real(frame,visible,visual_hour,plan);
    }
    bool prepare_real(c3x_renderer_frame_v1 const& frame,std::vector<ScenePose> const& visible,
            float visual_hour,ContributionPlan const& plan){
        if(!plan.valid)return false;
        auto required=plan.required(visible);
        prepared_units.clear();transitions.retain(required);shared_shadows.begin();material_samples.begin();
        required_samples=part_samples=shadow_samples=shadow_reuses=shadow_overflow=0;
        main_contributors=reflection_contributors=shadow_contributors=palette_uploads=material_builds=material_reuses=0;
        material_buffer_builds=material_buffer_reuses=material_buffer_uploads=material_upload_fallbacks=0;
        reflection_bounds_builds=reflection_bounds_reuses=reflection_bounds_rejected=0;
        if(plan.entries.empty())return true;
        if(plan.entries.size()>4096 || !initialize())return false;
        auto environment=c3x_renderer::evaluate_environment(visual_hour,frame.season);
        prepared_environment=environment;
        auto light=c3x_renderer::fidelity::light_frame(environment);
        prepared_light[0]=light[0];prepared_light[1]=light[1];

        auto noon=c3x_renderer::evaluate_environment(12,0);
        auto key_light=c3x_renderer::lighting::key_light(environment);
        float beauty_values[20]={};float const ambient_source[]={.34f,.45f,.60f};
        float const chromatic[]={1.f,4.5f/6.2f,3.5f/6.2f};
        auto beauty_light=key_light.direction.data();auto light_color=key_light.color.data();
        for(unsigned axis=0;axis<3;++axis){
            beauty_values[axis]=beauty_light[axis];
            beauty_values[4+axis]=chromatic[axis]*light_color[axis]/
                std::max(.001f,noon.sun_color[axis]);
            beauty_values[8+axis]=ambient_source[axis]*environment.ambient_color[axis]/
                std::max(.001f,noon.ambient_color[axis]);
        }
        beauty_values[3]=2.05f*(environment.sun_intensity+environment.moon_intensity)/
            (noon.sun_intensity+noon.moon_intensity);
        beauty_values[7]=1;beauty_values[11]=.62f;
        beauty_values[12]=.490290f;beauty_values[13]=-.735435f;
        beauty_values[14]=.469979f;beauty_values[16]=float(shadow_fit.extent);
        std::copy(std::begin(beauty_values),std::end(beauty_values),prepared_beauty.begin());
        auto& bodies=renderer.unit_bodies;unsigned palette_slot=0;
        for(auto const& entry:plan.entries){
            if(entry.candidate>=visible.size())return false;
            auto const& instance=visible[entry.candidate];
            if(instance.unit>=bodies.units.size())return false;
            auto const& unit=bodies.units[instance.unit];
            if(instance.action>=unit.actions.size())return false;
            auto const& action=unit.actions[instance.action];
            PreparedUnit sample;sample.instance=instance;sample.draw=instance.draw;
            auto& draw=sample.draw;auto& pose=sample.pose;
            int projection=draw.projection_scale_milli>0?draw.projection_scale_milli:(draw.reduced?500:1000);
            c3x_renderer::NativeUnitDraw source_draw{};
            source_draw.expected_sprite=source_draw.sprite=source_draw.expected_canvas=source_draw.canvas=1;
            source_draw.unit_id=draw.unit_id;source_draw.action=draw.action;source_draw.direction=draw.direction;
            source_draw.action_cursor=draw.action_cursor;source_draw.frame_count=draw.frame_count;
            source_draw.body_x=draw.body_x;source_draw.body_y=draw.body_y;source_draw.sprite_width=draw.sprite_width;
            source_draw.sprite_height=draw.sprite_height;source_draw.reduced=draw.reduced!=0;
            source_draw.projection_scale_milli=projection;
            if(!c3x_renderer::prepare_native_unit_pose(source_draw,action.loop,pose) ||
                !c3x_renderer::expand_unit_canvas(draw.body_x,draw.body_y,draw.sprite_width,draw.sprite_height,projection,unit.minimum_canvas))return false;
            // Native occurrences carry their own exact wrap/anchor. A canvas
            // heuristic must neither relocate nor discard a pass contributor.
            sample.main=bool(entry.mask&c3x_renderer::render_core::unit_main_body);
            sample.shadow=bool(entry.mask&c3x_renderer::render_core::unit_ground_shadow);
            sample.reflected=bool(entry.mask&c3x_renderer::render_core::unit_reflection);
            main_contributors+=sample.main;shadow_contributors+=sample.shadow;
            reflection_contributors+=sample.reflected;++required_samples;
            sample.low=unit_low_ground(frame,float(pose.anchor_x),float(pose.anchor_y));
            sample.ground_pixels=sample.low*frame.tile_width/224.f*.82f;
            sample.ground_depth=float(instance.tile_y)*frame.tile_height*.5f+renderer.geometry_viewport_settings.depth_translation+frame.tile_height*.5f+4.f;
            // Payloads and mesh buffers are admitted before this frame starts.
            sample.angle=transitions.facing(draw.unit_id,instance.pose_identity,frame.presentation_time_ticks,
                frame.presentation_frequency,c3x_renderer::native_unit_yaw(unit.yaw_offset,draw.direction));
            auto bits=[](float value){std::uint32_t result;std::memcpy(&result,&value,sizeof(result));return std::uint64_t(result);};
            std::vector<std::uint64_t> shadow_key={bodies.catalogue_generation,bits(unit.scale),bits(unit.offset_z),
                bits(sample.angle),bits(light[0]),bits(light[1]),action.parts.size()};
            for(auto const& part:action.parts){
                if(part.mesh>=bodies.meshes.size() || part.texture>=bodies.textures.size() ||
                    !bodies.textures[part.texture].view || !bodies.meshes[part.mesh].animation || part.mesh>=meshes.size() || !meshes[part.mesh].vertices)return false;
                auto const& source=*bodies.meshes[part.mesh].animation;
                PartSample prepared;prepared.frame=std::min(source.frames-1,unsigned(std::floor(pose.phase*double(source.frames-1)+1e-7)));
                prepared.blended=transitions.sample(draw.unit_id,instance.pose_identity,draw.action,frame.presentation_time_ticks,
                    frame.presentation_frequency,source,prepared.frame,std::uint64_t(part.mesh)+1);
                prepared.palette=meshes[part.mesh].palette_view;
                if(prepared.blended){
                    if(palette_slot==palette_slot_limit){prepared.palette.Reset();}
                    else {
                    if(palette_slot==prepared_palettes.size())prepared_palettes.emplace_back();
                    auto& slot=prepared_palettes[palette_slot++];
                    if(!slot.buffer){
                        D3D11_BUFFER_DESC desc={};desc.ByteWidth=4096*4;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
                        desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;desc.StructureByteStride=16;
                        if(FAILED(renderer.device->CreateBuffer(&desc,nullptr,&slot.buffer)))return false;
                        D3D11_SHADER_RESOURCE_VIEW_DESC view={};view.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;view.Buffer.NumElements=1024;
                        if(FAILED(renderer.device->CreateShaderResourceView(slot.buffer.Get(),&view,&slot.view)))return false;
                    }
                    renderer.context->UpdateSubresource(slot.buffer.Get(),0,nullptr,prepared.blended,0,0);work->upload_buffer(slot.buffer.Get());
                    prepared.palette=slot.view;++palette_uploads;
                    }
                }
                if(sample.main||sample.reflected){
                    shadow_key.insert(shadow_key.end(),{part.mesh,part.texture,part.address,bits(part.cutout),prepared.frame,source.bones,std::uint64_t(prepared.blended!=nullptr)});
                    if(prepared.blended)for(unsigned i=0;i<source.bones*16;++i)shadow_key.push_back(bits(prepared.blended[i]));
                }
                std::vector<std::uint64_t> material_key={bodies.catalogue_generation,part.mesh,part.texture,part.address,
                    draw.display_color_rgb,bits(part.tint[0]),bits(part.tint[1]),bits(part.tint[2]),bits(part.mask),bits(part.strength),
                    bits(part.material_model),bits(part.cutout),bits(environment.sun_intensity),bits(environment.moon_intensity)};
                for(unsigned axis=0;axis<3;++axis)material_key.insert(material_key.end(),{bits(environment.sun_direction[axis]),
                    bits(environment.sun_color[axis]),bits(environment.moon_direction[axis]),bits(environment.moon_color[axis]),bits(environment.ambient_color[axis])});
                for(auto texture:part.material_textures)material_key.push_back(texture);
                auto material_slot=material_samples.select(material_key);
                auto& values=prepared.material;
                bool material_ready=material_slot!=UINT_MAX&&material_samples[material_slot].valid;
                if(material_ready){values=material_samples[material_slot].value.values;
                    prepared.material_buffer=material_samples[material_slot].value.constants;
                    material_buffer_reuses+=unsigned(bool(prepared.material_buffer));++material_reuses;}
                else {
                    float player_color[3]={};
                    for(unsigned axis=0;axis<3;++axis){float color=float((draw.display_color_rgb>>(16-axis*8))&255)/255;
                        player_color[axis]=color<=.04045f?color/12.92f:std::pow((color+.055f)/1.055f,2.4f);}
                    std::copy(std::begin(part.tint),std::end(part.tint),values.begin());values[3]=part.mask;
                    for(unsigned axis=0;axis<3;++axis){
                        values[4+axis]=player_color[axis];values[8+axis]=environment.sun_direction[axis];
                        values[12+axis]=environment.sun_color[axis];values[16+axis]=environment.moon_direction[axis];
                        values[20+axis]=environment.moon_color[axis];values[24+axis]=environment.ambient_color[axis];
                    }
                    values[7]=part.strength;values[11]=environment.sun_intensity;values[19]=environment.moon_intensity;
                    values[23]=part.material_model;values[27]=part.cutout;
                    for(unsigned channel=0;channel<4;++channel)values[28+channel]=part.material_textures[channel]!=UINT32_MAX;
                    ++material_builds;
                }
                for(unsigned channel=0;channel<4;++channel)if(part.material_textures[channel]!=UINT32_MAX){
                    auto texture=part.material_textures[channel];
                    if(texture>=bodies.textures.size() || !bodies.textures[texture].view)return false;
                    prepared.textures[channel]=bodies.textures[texture].view;values[28+channel]=1;
                }
                if(material_slot!=UINT_MAX&&!material_ready){
                    auto& cached=material_samples[material_slot];cached.value.values=values;
                    // The existing sample cache pins selected slots for this
                    // entire frame. A slot is updated only when its exact
                    // material changes; both passes borrow the prepared value.
                    if(!cached.value.constants){
                        D3D11_BUFFER_DESC desc={};desc.ByteWidth=sizeof(values);
                        desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
                        if(SUCCEEDED(renderer.device->CreateBuffer(&desc,nullptr,&cached.value.constants)))
                            ++material_buffer_builds;
                    }
                    if(cached.value.constants){
                        renderer.context->UpdateSubresource(cached.value.constants.Get(),0,nullptr,values.data(),0,0);
                        prepared.material_buffer=cached.value.constants;++material_buffer_uploads;
                        work->upload_buffer(cached.value.constants.Get());
                    }
                    // Allocation failure and the bounded sample overflow keep
                    // the existing per-draw upload path without retrying this
                    // failed sample on every animation frame.
                    cached.valid=true;
                }
                sample.parts.push_back(std::move(prepared));++part_samples;
            }
            if(sample.main||sample.reflected){
                // One exact pose-local result can serve many occurrences and both
                // consumers. Native identity/timing still selects every sample.
                if(shadow_key.size()*sizeof(shadow_key[0])<=64u*1024u)sample.shadow_slot=shared_shadows.select(shadow_key);
                if(sample.shadow_slot==UINT_MAX)++shadow_overflow;
                bool cached=sample.shadow_slot!=UINT_MAX && shared_shadows[sample.shadow_slot].prepared;
                if(cached){auto const& slot=shared_shadows[sample.shadow_slot].value;
                    sample.fit=slot.fit;sample.reflection_bounds=slot.reflection_bounds;++reflection_bounds_reuses;}
                else {
                    shadow_points.clear();unsigned part_index=0;
                    for(auto const& part:action.parts){auto const& source=*bodies.meshes[part.mesh].animation;
                        auto const& p=sample.parts[part_index++];
                        auto* palette=p.blended?p.blended:source.palettes.data()+std::size_t(p.frame)*source.bones*16;
                        auto const& bounds=meshes[part.mesh].shadow_bounds;auto first=shadow_points.size();
                        bounds.append(palette,sample.angle,unit.scale,unit.offset_z,shadow_points);
                        sample.reflection_bounds.append(shadow_points,first,bounds.known,bounds.weight_low,bounds.weight_high,
                            double(unit.offset_z)*unit.scale);
                    }
                    ++reflection_bounds_builds;
                    if(!sample.fit.fit(shadow_points,light[0],light[1],true))return false;
                    if(sample.shadow_slot!=UINT_MAX){auto& cached_shadow=shared_shadows[sample.shadow_slot];auto& slot=cached_shadow.value;
                        slot.fit=sample.fit;slot.reflection_bounds=sample.reflection_bounds;cached_shadow.prepared=true;}
                }
            }
            if(sample.reflected&&!sample.reflection_bounds.overlaps(plan.view,pose.anchor_x,pose.anchor_y,
                    pose.projection_scale,sample.ground_pixels)){
                sample.reflected=false;--reflection_contributors;++reflection_bounds_rejected;
            }
            prepared_units.push_back(std::move(sample));
        }
        return true;
    }
    std::size_t material_sample_bytes()const{
        std::size_t bytes=material_samples.size()*sizeof(decltype(material_samples)::Entry);
        for(unsigned i=0;i<material_samples.size();++i)bytes+=material_samples[i].key.capacity()*sizeof(std::uint64_t);
        return bytes;
    }
    bool borrow_shadow(PreparedUnit& sample,c3x_renderer_frame_v1 const& frame){
        auto const& unit=renderer.unit_bodies.units[sample.instance.unit];auto const& action=unit.actions[sample.instance.action];
        if(sample.shadow_slot!=UINT_MAX && shared_shadows[sample.shadow_slot].valid){auto& slot=shared_shadows[sample.shadow_slot].value;
            ++shadow_reuses;self_shadow=slot.texture;self_shadow_target=slot.target;self_shadow_view=slot.view;shadow_fit=slot.fit;return true;}
        if(sample.shadow_slot==UINT_MAX){self_shadow=working_shadow.texture;self_shadow_target=working_shadow.target;self_shadow_view=working_shadow.view;}
        else {
            auto& slot=shared_shadows[sample.shadow_slot].value;self_shadow=slot.texture;self_shadow_target=slot.target;self_shadow_view=slot.view;
        }
        if(!draw_self_shadow(unit,action,sample.instance,sample.pose,sample.angle,frame,prepared_light[0],prepared_light[1],&sample))return false;
        ++shadow_samples;
        if(sample.shadow_slot==UINT_MAX){working_shadow.texture=self_shadow;working_shadow.target=self_shadow_target;working_shadow.view=self_shadow_view;}
        if(sample.shadow_slot!=UINT_MAX){auto& entry=shared_shadows[sample.shadow_slot];auto& slot=entry.value;
            slot.texture=self_shadow;slot.target=self_shadow_target;slot.view=self_shadow_view;slot.fit=shadow_fit;entry.valid=true;}
        return true;
    }
    template<class Target>bool draw_real(c3x_renderer_frame_v1 const& frame,
            std::vector<c3x_renderer::render_core::UnitInstances::ScenePose> const& visible,
            Target& scene,float scene_scale,float /*visual_hour*/,bool reflected,float zoom){
        SandboxPassWorkload::Scope pass(*work,reflected?SandboxPassWorkload::reflected_units:SandboxPassWorkload::units);
        // The map fog pass consumes this exact body coverage after tone mapping.
        // Keep terrain depth for projected shadows; bodies start their own depth pass.
        if(!reflected){renderer.context->ClearDepthStencilView(scene.depth,D3D11_CLEAR_STENCIL,1,0);work->clear(scene.depth);}
        if(visible.empty())return true;
        if(!initialize())return false;
        if(!reflected&&!visible_depth){
            D3D11_DEPTH_STENCIL_DESC d={};renderer.depth_state->GetDesc(&d);
            d.StencilEnable=TRUE;d.StencilReadMask=d.StencilWriteMask=0xff;
            d.FrontFace.StencilFunc=D3D11_COMPARISON_ALWAYS;
            d.FrontFace.StencilFailOp=d.FrontFace.StencilDepthFailOp=D3D11_STENCIL_OP_KEEP;
            d.FrontFace.StencilPassOp=D3D11_STENCIL_OP_REPLACE;d.BackFace=d.FrontFace;
            if(FAILED(renderer.device->CreateDepthStencilState(&d,&visible_depth)))return false;
        }
        auto* body_depth=reflected?renderer.depth_state:visible_depth;
        auto& bodies=renderer.unit_bodies;
        auto* context=renderer.context;
        if(!reflected){
            c3x_renderer::tactical::Input cursors;
            for(auto const& instance:visible)if(instance.cursor){
                auto const& draw=instance.draw;
                int projection=draw.projection_scale_milli>0?draw.projection_scale_milli:(draw.reduced?500:1000);
                // Same native center and sampled travel as the body below.
                int x=draw.body_x+int(std::int64_t(draw.sprite_width)*projection/2000);
                int y=draw.body_y+int(std::int64_t(draw.sprite_height)*projection/2000);
                float low=unit_low_ground(frame,float(x),float(y));
                cursors.ring(float(x+4),float(y+4)-low*frame.tile_width/224.f*.82f,draw.reduced?64.f:128.f,true);
            }
            if(!cursors.primitives.empty())renderer.tactical_gpu.draw_into(renderer.device,context,cursors,
                {0,0,int(scene.width/scene_scale),int(scene.height/scene_scale)},
                double(frame.presentation_time_ticks)/double(std::max(1ll,frame.presentation_frequency)),
                scene.target,scene.width,scene.height,zoom,4.f);
        }
        context->OMSetRenderTargets(1,&scene.target,scene.depth);
        context->OMSetDepthStencilState(body_depth,reflected?0:1);
        context->OMSetBlendState(nullptr,nullptr,0xffffffffu);
        context->RSSetState(renderer.rasterizer_state);
        D3D11_VIEWPORT viewport={0,0,float(scene.width),float(scene.height),0,1};
        c3x_renderer::SceneProjection(frame.target_width,frame.target_height,zoom)
            .viewport(viewport,reflected?8.f:4.f,0,0,scene_scale);
        D3D11_RECT scissor={0,0,LONG(scene.width),LONG(scene.height)};
        context->RSSetViewports(1,&viewport);context->RSSetScissorRects(1,&scissor);
        context->IASetInputLayout(layout);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vertex,nullptr,0);context->PSSetShader(pixel,nullptr,0);
        context->VSSetConstantBuffers(2,1,&placement);
        context->PSSetConstantBuffers(2,1,&placement);
        context->PSSetConstantBuffers(0,1,&material);
        context->PSSetConstantBuffers(1,1,&beauty);
        context->PSSetShaderResources(1,1,&unshadowed_view);
        context->PSSetSamplers(1,1,&samplers[3]);
        auto const& environment=prepared_environment;
        auto key_light=c3x_renderer::lighting::key_light(environment);
        context->UpdateSubresource(beauty,0,nullptr,prepared_beauty.data(),0,0);work->upload_buffer(beauty);
        ID3D11Buffer* bound_material=nullptr;
        // Ground shadows still test against terrain. Visible bodies then share
        // a fresh depth plane, preserving self/other-unit occlusion while every
        // world feature stays behind them. The next frame restores cached world
        // depth before its dynamic passes; reflections keep their world depth.
        for(int layer=reflected?1:0;layer<2;++layer){
          if(!reflected && layer==1){
            context->ClearDepthStencilView(scene.depth,D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);
            work->clear(scene.depth);
          }
          for(auto& prepared:prepared_units){
            if(reflected?!prepared.reflected:layer==0?!prepared.shadow:!prepared.main)continue;
            auto const& instance=prepared.instance;auto const& unit=bodies.units[instance.unit];
            auto const& action=unit.actions[instance.action];auto const& pose=prepared.pose;
            float low=prepared.low,ground_pixels=prepared.ground_pixels,ground_depth=prepared.ground_depth,angle=prepared.angle;
            if(layer==1&&!borrow_shadow(prepared,frame))return false;
            context->OMSetRenderTargets(1,&scene.target,scene.depth);
            context->OMSetDepthStencilState(body_depth,reflected?0:1);
            context->OMSetBlendState(nullptr,nullptr,~0u);
            context->RSSetViewports(1,&viewport);context->RSSetScissorRects(1,&scissor);
            context->PSSetShader(pixel,nullptr,0);
            auto* shadow_view=(prepared.main||prepared.reflected)?self_shadow_view.Get():unshadowed_view;context->PSSetShaderResources(1,1,&shadow_view);
            unsigned part_index=0;
            for(auto const& part:action.parts){
                if(part.mesh>=bodies.meshes.size()||part.texture>=bodies.textures.size()||
                   !bodies.textures[part.texture].view)return false;
                auto const& source=bodies.meshes[part.mesh].animation;
                if(!source||!source->frames)return false;
                auto& gpu=meshes[part.mesh];
                auto const& part_sample=prepared.parts[part_index++];
                unsigned frame_number=part_sample.frame;auto* blended=part_sample.blended;
                float scale=pose.projection_scale*scene_scale;
                float guard=reflected?8.f:4.f;
                // The ground point is Civ III's captured center. Sprite size
                // and expanded dirty canvases must not move the resident mesh.
                float placement_values[28]={float(pose.anchor_x+guard)/pose.projection_scale,
                    (float(pose.anchor_y+guard)+(reflected?ground_pixels:-ground_pixels))/pose.projection_scale,
                    float(scene.width),float(scene.height),scale,ground_depth+low*.0016f*frame.target_height,0,0,
                    float(blended?0:frame_number),float(source->bones),std::cos(angle),std::sin(angle),
                    unit.scale,unit.offset_z,0,0,
                    reflected?2.f:0.f,0,0,part.cutout,
                    shadow_fit.left,shadow_fit.top,1/shadow_fit.width,1/shadow_fit.height,shadow_fit.dx,shadow_fit.dy};
                auto* material_constants=part_sample.material_buffer.Get();
                if(!material_constants){
                    context->UpdateSubresource(material,0,nullptr,part_sample.material.data(),0,0);work->upload_buffer(material);
                    material_constants=material;++material_upload_fallbacks;
                }
                if(bound_material!=material_constants){
                    context->PSSetConstantBuffers(0,1,&material_constants);bound_material=material_constants;
                }
                ID3D11Buffer* static_vertices=gpu.vertices.Get();
                UINT stride=sizeof(Vertex),offset=0;
                context->IASetVertexBuffers(0,1,&static_vertices,&stride,&offset);
                context->IASetIndexBuffer(gpu.indices.Get(),DXGI_FORMAT_R32_UINT,0);
                ID3D11ShaderResourceView* palette=part_sample.palette.Get();
                if(!palette && blended){if(!bind_palette(gpu,blended))return false;++palette_uploads;}
                else context->VSSetShaderResources(0,1,&palette);
                context->PSSetShaderResources(0,1,&bodies.textures[part.texture].view);
                context->PSSetShaderResources(2,4,part_sample.textures.data());
                context->PSSetSamplers(0,1,&samplers[part.address]);
                if(layer==0&&key_light.intensity>.001f){
                    placement_values[16]=1;
                    placement_values[17]=-key_light.direction[0]/key_light.direction[2]*
                        c3x_renderer::lighting::object_height_to_world;
                    placement_values[18]=key_light.direction[1]/key_light.direction[2]*
                        c3x_renderer::lighting::object_height_to_world;
                    context->UpdateSubresource(placement,0,nullptr,placement_values,0,0);work->upload_buffer(placement);
                    context->OMSetDepthStencilState(renderer.natural.decal_depth,0);
                    context->OMSetBlendState(renderer.blend_state,nullptr,0xffffffffu);
                    context->PSSetShader(shadow_pixel,nullptr,0);
                    context->DrawIndexed(UINT(source->indices.size()),0,0);work->draw(source->indices.size());
                    placement_values[16]=0;
                    context->OMSetDepthStencilState(body_depth,reflected?0:1);
                    context->OMSetBlendState(nullptr,nullptr,0xffffffffu);
                    context->PSSetShader(pixel,nullptr,0);
                }
                if(layer==1){
                    context->UpdateSubresource(placement,0,nullptr,placement_values,0,0);work->upload_buffer(placement);
                    context->DrawIndexed(UINT(source->indices.size()),0,0);work->draw(source->indices.size());
                    ++draws;
                }
            }
          }
        }
        if(!reflected)transitions.finish(frame.presentation_time_ticks);
        ID3D11ShaderResourceView* empty[6]={};context->PSSetShaderResources(0,6,empty);
        context->VSSetShaderResources(0,1,empty);
        context->OMSetRenderTargets(0,nullptr,nullptr);
        return true;
    }
#endif
};

SandboxDirectUnits sandbox_direct_units;
