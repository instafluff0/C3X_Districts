// Untimed hardware oracle for the actual Renderer64 unit preparation/draw path.
// This executable alone drops prepared CB leases to select the existing upload
// fallback. Production contains no test switch, readback or additional export.
#ifndef C3X_HELPER_TRIAL
#define C3X_HELPER_TRIAL
#endif
#ifndef C3X_RENDERER64_FRESH
#define C3X_RENDERER64_FRESH
#endif
#include "../sandbox/resident_scene.cpp"
#include <cassert>

namespace {
using Microsoft::WRL::ComPtr;
using c3x_renderer::render_core::LinearTarget;
using Direct=SandboxDirectUnits;

void require_gpu(bool valid,char const* label){
    if(!valid)throw std::runtime_error(label);
}
void checked_gpu(HRESULT result,char const* label){require_gpu(SUCCEEDED(result),label);}

std::vector<unsigned char> read_single_sample(ID3D11Texture2D* source,unsigned stride){
    D3D11_TEXTURE2D_DESC desc={};source->GetDesc(&desc);
    require_gpu(desc.SampleDesc.Count==1,"single-sample diagnostic target");
    desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=desc.MiscFlags=0;
    desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;
    checked_gpu(renderer.device->CreateTexture2D(&desc,nullptr,&staging),"oracle staging allocation");
    renderer.context->CopyResource(staging.Get(),source);
    D3D11_MAPPED_SUBRESOURCE mapped={};
    checked_gpu(renderer.context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped),"oracle staging map");
    std::vector<unsigned char> bytes(std::size_t(desc.Width)*desc.Height*stride);
    for(unsigned y=0;y<desc.Height;++y)
        std::memcpy(bytes.data()+std::size_t(y)*desc.Width*stride,
            static_cast<char const*>(mapped.pData)+std::size_t(y)*mapped.RowPitch,
            std::size_t(desc.Width)*stride);
    renderer.context->Unmap(staging.Get(),0);return bytes;
}

void initialize_fixture(Direct& direct,SandboxPassWorkload& work){
    D3D_FEATURE_LEVEL level;
    checked_gpu(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,
        D3D11_SDK_VERSION,&renderer.device,&level,&renderer.context),"hardware device");
    renderer.content_view_width=renderer.content_view_height=256;
    D3D11_DEPTH_STENCIL_DESC depth={};depth.DepthEnable=TRUE;
    depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;depth.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
    checked_gpu(renderer.device->CreateDepthStencilState(&depth,&renderer.depth_state),"body depth state");
    depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;
    checked_gpu(renderer.device->CreateDepthStencilState(&depth,&renderer.natural.decal_depth),"shadow depth state");
    D3D11_RASTERIZER_DESC raster={};raster.FillMode=D3D11_FILL_SOLID;
    raster.CullMode=D3D11_CULL_NONE;raster.DepthClipEnable=raster.ScissorEnable=TRUE;
    checked_gpu(renderer.device->CreateRasterizerState(&raster,&renderer.rasterizer_state),"raster state");
    D3D11_BLEND_DESC blend={};auto& output=blend.RenderTarget[0];output.BlendEnable=TRUE;
    output.SrcBlend=D3D11_BLEND_SRC_ALPHA;output.DestBlend=D3D11_BLEND_INV_SRC_ALPHA;
    output.SrcBlendAlpha=D3D11_BLEND_ONE;output.DestBlendAlpha=D3D11_BLEND_ZERO;
    output.BlendOp=output.BlendOpAlpha=D3D11_BLEND_OP_ADD;
    output.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;
    checked_gpu(renderer.device->CreateBlendState(&blend,&renderer.blend_state),"shadow blend state");
    auto mesh=std::make_shared<c3x_renderer::AnimationMesh>();mesh->frames=2;mesh->bones=1;mesh->duration=1;
    mesh->vertices.resize(3);mesh->indices={0,1,2};mesh->palettes.resize(32);
    float const positions[3][3]={{-.6f,-.3f,.1f},{.6f,-.3f,.1f},{0,.6f,.7f}};
    for(unsigned i=0;i<3;++i){auto& vertex=mesh->vertices[i];
        std::copy(positions[i],positions[i]+3,vertex.source.position);
        vertex.source.normal[2]=vertex.tangent[0]=vertex.bitangent[1]=1;
        vertex.source.uv[0]=i==1?1.f:0.f;vertex.source.uv[1]=i==2?1.f:0.f;
        vertex.joints={0,0,0,0};vertex.weights={1,0,0,0};}
    for(unsigned frame=0;frame<2;++frame){auto* palette=mesh->palettes.data()+frame*16;
        palette[0]=palette[5]=palette[10]=palette[15]=1;palette[12]=float(frame)*.13f;}
    auto& bodies=renderer.unit_bodies;bodies.meshes.resize(1);bodies.meshes[0].animation=mesh;
    bodies.textures.resize(5);
    // Generic local pixels exercise cutout, optional channels and both material models.
    unsigned const texels[5][4]={{0xff457ca1u,0x00457ca1u,0xffd5a273u,0xff69a88au},
        {0xff8f8f8fu,0xffc0c0c0u,0xffffffffu,0xff5f5f5fu},
        {0xff306080u,0xff507040u,0xff609090u,0xff807050u},
        {0xff100a20u,0xff302010u,0xff204020u,0xff102030u},
        {0xffff8080u,0xffffa070u,0xffff6090u,0xffff8080u}};
    for(unsigned index=0;index<5;++index){
        D3D11_TEXTURE2D_DESC texture={};texture.Width=texture.Height=2;
        texture.MipLevels=texture.ArraySize=texture.SampleDesc.Count=1;
        texture.Format=DXGI_FORMAT_R8G8B8A8_UNORM;texture.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA data={texels[index],8,0};ComPtr<ID3D11Texture2D> image;
        checked_gpu(renderer.device->CreateTexture2D(&texture,&data,&image),"material texture");
        checked_gpu(renderer.device->CreateShaderResourceView(image.Get(),nullptr,&bodies.textures[index].view),"material texture view");
    }
    bodies.units.resize(1);auto& unit=bodies.units[0];unit.scale=.7f;
    c3x_renderer::UnitBodyRenderer::Action action;action.name="idle";action.loop=true;action.frames=2;
    c3x_renderer::UnitBodyRenderer::Part first;first.cutout=1;first.mask=2;first.strength=.7f;
    auto second=first;second.material_model=1;second.address=3;second.tint[0]=.8f;
    for(unsigned channel=0;channel<4;++channel)second.material_textures[channel]=channel+1;
    action.parts={first,second};unit.actions.push_back(action);
    direct.work=&work;require_gpu(direct.initialize()&&direct.prepare_mesh(0),"production mesh/shader initialization");
}

struct Snapshot{std::vector<unsigned char> main_color,main_depth,reflection_color,reflection_depth;};

Snapshot render_passes(Direct& direct,c3x_renderer_frame_v1 const& frame,
        std::vector<Direct::ScenePose> const& poses,float hour,float zoom){
    Snapshot result;LinearTarget main,reflection;
    require_gpu(main.ensure(renderer.device,256,256,false,true,1)&&
        reflection.ensure(renderer.device,256,256,false,true,1),"oracle HDR/depth targets");
    auto draw=[&](LinearTarget& target,bool reflected,std::vector<unsigned char>& color,
            std::vector<unsigned char>& depth){
        float clear[4]={.08f,.12f,.16f,1};renderer.context->ClearRenderTargetView(target.target,clear);
        renderer.context->ClearDepthStencilView(target.depth,D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);
        auto before_color=read_single_sample(target.color,8);
        auto before_depth=read_single_sample(target.depth_texture,4);
        require_gpu(direct.draw_real(frame,poses,target,1,hour,reflected,zoom),"production unit draw");
        color=read_single_sample(target.color,8);depth=read_single_sample(target.depth_texture,4);
        require_gpu(color!=before_color,"unit pass must draw nonempty HDR pixels");
        require_gpu(depth!=before_depth,"unit pass must write nonempty depth/stencil");
    };
    // Reflection precedes the main pass in the production pipeline.
    draw(reflection,true,result.reflection_color,result.reflection_depth);
    draw(main,false,result.main_color,result.main_depth);return result;
}

unsigned compare_case(Direct& direct,c3x_renderer_frame_v1& frame,unsigned count,
        unsigned unique,float hour,float zoom,unsigned revision){
    using namespace c3x_renderer::render_core;
    std::vector<Direct::ScenePose> poses;UnitContributionPlan plan;
    for(unsigned i=0;i<count;++i){Direct::ScenePose pose;pose.unit=pose.action=0;pose.pose_identity=i+1;
        auto& draw=pose.draw;draw.struct_size=sizeof(draw);draw.unit_id=int(i+1);draw.action=1;
        draw.direction=1+int(revision%8);draw.frame_count=2;draw.action_cursor=int(revision%2);
        draw.sprite_width=128;draw.sprite_height=64;draw.projection_scale_milli=1000;
        draw.body_x=48+int(i%3)*25+int(revision%3);draw.body_y=75+int(i%4)*12;
        draw.display_color_rgb=((i%unique)+1)*0x00010101u;
        poses.push_back(pose);plan.entries.push_back({i,unit_main_body|unit_ground_shadow|unit_reflection});}
    require_gpu(direct.prepare_real(frame,poses,hour,plan),"production unit preparation");
    auto uploads=direct.material_buffer_uploads,builds=direct.material_buffer_builds;
    require_gpu(direct.part_samples==count*2,"complete multipart preparation");
    if(unique>256)require_gpu(direct.material_samples.size()==256,"bounded material slot overflow");
    auto cached=render_passes(direct,frame,poses,hour,zoom);
    auto cached_body_draws=direct.draws,cached_fallbacks=direct.material_upload_fallbacks;
    for(auto& unit:direct.prepared_units)for(auto& part:unit.parts)part.material_buffer.Reset();
    auto fallback=render_passes(direct,frame,poses,hour,zoom);
    require_gpu(cached.main_color==fallback.main_color&&cached.main_depth==fallback.main_depth&&
        cached.reflection_color==fallback.reflection_color&&cached.reflection_depth==fallback.reflection_depth,
        "cached/fallback HDR + raw D24/stencil + reflection parity");
    require_gpu(direct.draws-cached_body_draws==count*4,"all main/reflection multipart bodies preserved");
    require_gpu(direct.material_upload_fallbacks-cached_fallbacks==count*4,"oracle forced actual upload fallback");
    std::printf("PASS unit material submission: ticks=%lld hour=%.1f zoom=%.2f units=%u parts=%u slots=%u builds=%u uploads=%u cached_fallbacks=%u hdr_depth_stencil_reflection_exact=1\n",
        frame.presentation_time_ticks,hour,zoom,count,direct.part_samples,direct.material_samples.size(),builds,uploads,cached_fallbacks);
    return 4;
}

unsigned compare_reflection_selection(Direct& direct,c3x_renderer_frame_v1& frame,float zoom,unsigned revision){
    using namespace c3x_renderer::render_core;
    std::vector<Direct::ScenePose> poses;UnitContributionPlan plan;
    for(unsigned i=0;i<12;++i){Direct::ScenePose pose;pose.unit=pose.action=0;pose.pose_identity=100+i;
        auto& draw=pose.draw;draw.struct_size=sizeof(draw);draw.unit_id=100+int(i);draw.action=1;
        draw.direction=1+int(revision%8);draw.frame_count=2;draw.action_cursor=int(revision%2);
        draw.sprite_width=128;draw.sprite_height=64;draw.projection_scale_milli=1000;
        draw.body_x=-80+int(i%4)*110;draw.body_y=65+int(i/4)*35;
        draw.display_color_rgb=0x00010101u*(i+1);
        poses.push_back(pose);plan.entries.push_back({i,unit_main_body|unit_ground_shadow|unit_reflection});}
    require_gpu(direct.prepare_real(frame,poses,12,plan),"conservative reflected unit preparation");
    auto baseline=render_passes(direct,frame,poses,12,zoom);
    auto baseline_draws=direct.draws;
    plan.view.width=plan.view.height=256;plan.view.zoom=zoom;plan.view.reflection=true;
    plan.view.receivers={{64,96,128,220},{160,180,184,240}};
    require_gpu(direct.prepare_real(frame,poses,12,plan),"current-pose reflected unit preparation");
    unsigned removed=direct.reflection_bounds_rejected;
    require_gpu(removed>0&&removed<poses.size(),"both retained and rejected reflected units");
    require_gpu(direct.main_contributors==poses.size()&&direct.shadow_contributors==poses.size(),
        "reflection selection preserves main and shadow contributors");
    auto selected=render_passes(direct,frame,poses,12,zoom);
    require_gpu(baseline.main_color==selected.main_color&&baseline.main_depth==selected.main_depth,
        "main HDR/depth/stencil unchanged by reflected-unit selection");
    unsigned checked=0;
    for(unsigned y=0;y<256;++y)for(unsigned x=0;x<256;++x){
        bool receiver=std::any_of(plan.view.receivers.begin(),plan.view.receivers.end(),[&](auto const& rect){
            return x>=rect.left&&x<rect.right&&y>=rect.top&&y<rect.bottom;});
        if(!receiver)continue;auto offset=std::size_t(y)*256+x;
        require_gpu(!std::memcmp(baseline.reflection_color.data()+offset*8,
            selected.reflection_color.data()+offset*8,8)&&
            !std::memcmp(baseline.reflection_depth.data()+offset*4,
            selected.reflection_depth.data()+offset*4,4),"receiver HDR and raw depth unchanged");
        ++checked;
    }
    require_gpu(direct.draws-baseline_draws==(poses.size()*2-removed)*2,
        "rejected reflected multipart bodies reduce actual draws");
    std::printf("PASS current-pose reflection selection: ticks=%lld zoom=%.2f direction=%u removed=%u receiver_pixels=%u main_hdr_depth_exact=1 receiver_hdr_depth_exact=1\n",
        frame.presentation_time_ticks,zoom,revision%8+1,removed,checked);
    return 4;
}
}

int main(){
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
    try{
        Direct direct;SandboxPassWorkload work;initialize_fixture(direct,work);
        c3x_renderer_frame_v1 frame={};frame.target_width=frame.target_height=256;
        frame.tile_width=128;frame.tile_height=64;frame.presentation_frequency=1000;
        frame.presentation_time_ticks=1000;
        unsigned comparisons=compare_case(direct,frame,36,9,12,1,0);
        frame.presentation_time_ticks=1033;comparisons+=compare_case(direct,frame,36,9,12,1,0);
        require_gpu(direct.material_buffer_builds==0&&direct.material_buffer_uploads==0,"warm material CB reuse");
        frame.presentation_time_ticks=1200;comparisons+=compare_case(direct,frame,36,9,12,1.5f,1);
        ++renderer.unit_bodies.catalogue_generation;
        frame.presentation_time_ticks=1600;comparisons+=compare_case(direct,frame,36,9,19,1,2);
        frame.presentation_time_ticks=2000;comparisons+=compare_case(direct,frame,300,300,6,1,3);
        frame.presentation_time_ticks=2033;comparisons+=compare_case(direct,frame,300,300,6,1,4);
        for(unsigned revision=0;revision<8;++revision){
            frame.presentation_time_ticks=3000+33*revision;
            comparisons+=compare_reflection_selection(direct,frame,revision%2?1.5f:1.f,revision);
        }
        std::printf("PASS actual Renderer64 unit material oracle: exact_surfaces=%u sample_count=1 synthetic_generic_assets=1 production_prepare_draw=1 untimed_readback=1\n",comparisons);
        return 0;
    }catch(std::exception const& error){std::fprintf(stderr,"UNIT_MATERIAL_ORACLE_FAILED %s\n",error.what());return 1;}
}
