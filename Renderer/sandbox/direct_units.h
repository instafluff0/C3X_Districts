#pragma once

// 0 A.D.'s GPUSkinnedModelRenderer keeps mesh inputs resident and updates only
// the animation palette for visible models. Here each authored frame is already
// an immutable palette: the vertex shader selects one exact source frame.
struct SandboxDirectUnits {
    struct Vertex {
        float position[3],normal[3],uv[2],tangent[3],bitangent[3];
        std::uint32_t joints[4];float weights[4];
    };
    static_assert(sizeof(Vertex)==88);
    struct Mesh {
        Microsoft::WRL::ComPtr<ID3D11Buffer> vertices,indices,palettes;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> palette_view;
    };
    std::vector<Mesh> meshes;
    ID3D11VertexShader* vertex=nullptr;
    ID3D11PixelShader* pixel=nullptr,*shadow_pixel=nullptr;
    ID3D11InputLayout* layout=nullptr;
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
    template<class T>static void drop(T*& p){if(p)p->Release();p=nullptr;}
    ~SandboxDirectUnits(){drop(vertex);drop(pixel);drop(shadow_pixel);drop(layout);drop(material);drop(beauty);
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
 float4 pass_control;
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
 float2 local=float2(95.5+(x-y)*64,95.5+(x+y)*32-z*(150.0*128/224));
 float2 pixel=(origin+local)*scale;
 if(pass_control.x>1.5)pixel.y+=2*z*(150.0*128/224)*scale;
 Output o;o.p=float4(pixel/extent*float2(2,-2)+float2(-1,1),
  clamp(.5-(depth_base+(x+y)*5+z*.1)/16384,.001,.999),1);
 float3 n=float3(normal.x*c-normal.y*s,normal.x*s+normal.y*c,normal.z);
 float3 t=float3(tangent.x*c-tangent.y*s,tangent.x*s+tangent.y*c,tangent.z);
 float3 b=float3(bitangent.x*c-bitangent.y*s,bitangent.x*s+bitangent.y*c,bitangent.z);
 o.n=normalize(float3(n.x,-n.y,n.z/(150.0/(112*.82))));
 o.tangent=normalize(float3(t.x,-t.y,t.z/(150.0/(112*.82))));
 o.bitangent=normalize(float3(b.x,-b.y,b.z/(150.0/(112*.82))));
 o.uv=i.uv;o.shadow=float3(z,.5,.5);return o;
}
Texture2D<float4> shadow_base:register(t0);
SamplerState shadow_sampler:register(s0);
float4 PSShadow(Output i):SV_Target {
 if(pass_control.w>.5)clip(shadow_base.Sample(shadow_sampler,i.uv).a-.5);
 return float4(0,0,0,.28);
})";
        ID3DBlob *vs=nullptr,*ps=nullptr,*shadow_ps=nullptr,*errors=nullptr;
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
        if(SUCCEEDED(hr))hr=renderer.device->CreateVertexShader(vs->GetBufferPointer(),
            vs->GetBufferSize(),nullptr,&vertex);
        if(SUCCEEDED(hr))hr=renderer.device->CreatePixelShader(ps->GetBufferPointer(),
            ps->GetBufferSize(),nullptr,&pixel);
        if(SUCCEEDED(hr))hr=renderer.device->CreatePixelShader(shadow_ps->GetBufferPointer(),
            shadow_ps->GetBufferSize(),nullptr,&shadow_pixel);
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
        drop(vs);drop(ps);drop(shadow_ps);
        D3D11_BUFFER_DESC b={};b.ByteWidth=128;b.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        if(SUCCEEDED(hr))hr=renderer.device->CreateBuffer(&b,nullptr,&material);
        b.ByteWidth=80;if(SUCCEEDED(hr))hr=renderer.device->CreateBuffer(&b,nullptr,&beauty);
        b.ByteWidth=80;if(SUCCEEDED(hr))hr=renderer.device->CreateBuffer(&b,nullptr,&placement);
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
    bool prepare_mesh(unsigned index){
        auto& bodies=renderer.unit_bodies;
        if(index>=bodies.meshes.size())return false;
        if(meshes.size()<bodies.meshes.size())meshes.resize(bodies.meshes.size());
        auto& gpu=meshes[index];if(gpu.vertices)return true;
        auto const& mesh=bodies.meshes[index].animation;
        if(!mesh||mesh->vertices.empty()||mesh->indices.empty()||mesh->palettes.empty())return false;
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
        if(FAILED(renderer.device->CreateBuffer(&b,&data,&gpu.vertices)))return false;
        b.ByteWidth=UINT(mesh->indices.size()*4);b.BindFlags=D3D11_BIND_INDEX_BUFFER;
        data.pSysMem=mesh->indices.data();
        if(FAILED(renderer.device->CreateBuffer(&b,&data,&gpu.indices)))return false;
        b.ByteWidth=UINT(mesh->palettes.size()*4);b.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        b.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;b.StructureByteStride=16;
        data.pSysMem=mesh->palettes.data();
        if(FAILED(renderer.device->CreateBuffer(&b,&data,&gpu.palettes)))return false;
        D3D11_SHADER_RESOURCE_VIEW_DESC view={};view.Format=DXGI_FORMAT_UNKNOWN;
        view.ViewDimension=D3D11_SRV_DIMENSION_BUFFER;
        view.Buffer.NumElements=UINT(mesh->palettes.size()/4);
        if(FAILED(renderer.device->CreateShaderResourceView(gpu.palettes.Get(),&view,
            &gpu.palette_view)))return false;
        ++mesh_builds;return true;
    }
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
        auto light=key_light.direction.data();auto light_color=key_light.color.data();
        for(unsigned axis=0;axis<3;++axis){
            beauty_values[axis]=light[axis];
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
        context->UpdateSubresource(beauty,0,nullptr,beauty_values,0,0);
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
            }
            if(index==0&&move_started>=0&&travel>=1)move_started=-1;
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
                float placement_values[20]={float(body_x+(reflected?8:4)),
                    float(body_y+(reflected?8:4)),
                    float(scene.width),float(scene.height),float(scene_scale),ground_depth,0,0,
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
                context->UpdateSubresource(material,0,nullptr,values,0,0);
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
                    placement_values[18]=-key_light.direction[1]/key_light.direction[2]*
                        c3x_renderer::lighting::object_height_to_world;
                    context->UpdateSubresource(placement,0,nullptr,placement_values,0,0);
                    context->OMSetDepthStencilState(renderer.natural.decal_depth,0);
                    context->OMSetBlendState(renderer.blend_state,nullptr,0xffffffffu);
                    context->PSSetShader(shadow_pixel,nullptr,0);
                    context->DrawIndexed(UINT(source->indices.size()),0,0);
                    placement_values[16]=0;
                    context->OMSetDepthStencilState(renderer.depth_state,0);
                    context->OMSetBlendState(nullptr,nullptr,0xffffffffu);
                    context->PSSetShader(pixel,nullptr,0);
                }
                context->UpdateSubresource(placement,0,nullptr,placement_values,0,0);
                context->DrawIndexed(UINT(source->indices.size()),0,0);
                ++draws;
            }
        }
        ID3D11ShaderResourceView* empty[6]={};context->PSSetShaderResources(0,6,empty);
        context->VSSetShaderResources(0,1,empty);
        context->OMSetRenderTargets(0,nullptr,nullptr);
        return true;
    }
};

SandboxDirectUnits sandbox_direct_units;
