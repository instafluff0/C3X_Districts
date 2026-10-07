// Lab unit sheet: chosen units through the live Renderer64 unit path.
// Loads a generic unit pack and calls the production prepare_real/draw_real
// (sandbox/direct_units.h) into a transparent scene-linear layer, the same
// "moving" layer the game composites over the static map. Writes that layer as
// RGBA16F plus the noon display exposure; Python composites and applies the
// production display transfer. Diagnostic only: no game, staging or packs edited.
// With "reflect", the production reflected pass (the mirror the water samples)
// is also written to OUTPUT.reflect.f16, with the noon water response in
// OUTPUT.env.json.
//
//   unit_sheet.exe PACK_ROOT UNITS_FILE OUTPUT.f16 OWNER_RGB_HEX [reflect]
// UNITS_FILE lines: "PRTO_Key direction" in sheet order (6 columns).
#ifndef C3X_HELPER_TRIAL
#define C3X_HELPER_TRIAL
#endif
#ifndef C3X_RENDERER64_FRESH
#define C3X_RENDERER64_FRESH
#endif
#include "../../../sandbox/resident_scene.cpp"
#include <fstream>
#include <sstream>

namespace {
using Microsoft::WRL::ComPtr;
using c3x_renderer::render_core::LinearTarget;
constexpr int columns=6,cell_w=160,cell_h=150;
void require(bool valid,char const* label){if(!valid)throw std::runtime_error(label);}
void checked(HRESULT result,char const* label){require(SUCCEEDED(result),label);}
}

int main(int argc,char** argv){
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
    try{
        require(argc==5||(argc==6&&std::string(argv[5])=="reflect"),"usage: unit_sheet PACK_ROOT UNITS_FILE OUTPUT.f16 OWNER_RGB_HEX [reflect]");
        bool reflect=argc==6;
        std::vector<std::pair<std::string,int>> wanted;
        {std::ifstream file(argv[2]);std::string key;int direction;
         while(file>>key>>direction)wanted.push_back({key,direction});}
        require(!wanted.empty()&&wanted.size()<=36,"unit list");
        unsigned owner=unsigned(std::strtoul(argv[4],nullptr,16));
        int rows=int(wanted.size()+columns-1)/columns,width=columns*cell_w,height=rows*cell_h+20;

        D3D_FEATURE_LEVEL level;
        checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,
            D3D11_SDK_VERSION,&renderer.device,&level,&renderer.context),"hardware device");
        renderer.content_view_width=width;renderer.content_view_height=height;
        // Production state descriptors (c3x_renderer.cpp / source_fidelity/runtime.h).
        D3D11_DEPTH_STENCIL_DESC depth={};depth.DepthEnable=TRUE;
        depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;depth.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
        checked(renderer.device->CreateDepthStencilState(&depth,&renderer.depth_state),"depth");
        depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;
        checked(renderer.device->CreateDepthStencilState(&depth,&renderer.natural.decal_depth),"decal depth");
        D3D11_RASTERIZER_DESC raster={};raster.FillMode=D3D11_FILL_SOLID;
        raster.CullMode=D3D11_CULL_NONE;raster.DepthClipEnable=raster.ScissorEnable=TRUE;
        checked(renderer.device->CreateRasterizerState(&raster,&renderer.rasterizer_state),"raster");
        D3D11_BLEND_DESC blend={};auto& output=blend.RenderTarget[0];output.BlendEnable=TRUE;
        output.SrcBlend=D3D11_BLEND_SRC_ALPHA;output.DestBlend=D3D11_BLEND_INV_SRC_ALPHA;output.BlendOp=D3D11_BLEND_OP_ADD;
        output.SrcBlendAlpha=D3D11_BLEND_ONE;output.DestBlendAlpha=D3D11_BLEND_INV_SRC_ALPHA;output.BlendOpAlpha=D3D11_BLEND_OP_ADD;
        output.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;
        checked(renderer.device->CreateBlendState(&blend,&renderer.blend_state),"blend");
        renderer.reflection.enabled=false;
        require(renderer.load_unit_animations(argv[1]),"unit pack");
        auto& bodies=renderer.unit_bodies;

        c3x_renderer_frame_v1 frame={};frame.struct_size=sizeof(frame);
        frame.target_width=width;frame.target_height=height;frame.tile_width=128;frame.tile_height=64;
        frame.hour=12;frame.presentation_frequency=1000;frame.presentation_time_ticks=1000;
        std::vector<SandboxDirectUnits::ScenePose> poses;c3x_renderer::render_core::UnitContributionPlan plan;
        for(unsigned i=0;i<wanted.size();++i){
            auto unit=std::find_if(bodies.units.begin(),bodies.units.end(),[&](auto const& item){
                return std::find(item.keys.begin(),item.keys.end(),wanted[i].first)!=item.keys.end();});
            if(unit==bodies.units.end()){std::printf("MISSING %s\n",wanted[i].first.c_str());continue;}
            auto idle=std::find_if(unit->actions.begin(),unit->actions.end(),[](auto const& a){return a.name=="idle";});
            require(idle!=unit->actions.end(),"idle action");
            SandboxDirectUnits::ScenePose pose;pose.unit=std::size_t(unit-bodies.units.begin());
            pose.action=std::size_t(idle-unit->actions.begin());pose.pose_identity=i+1;pose.pose_ticks=0;
            int cx=int(i%columns)*cell_w+cell_w/2,cy=int(i/columns)*cell_h+cell_h/2+25;
            auto& d=pose.draw;d.struct_size=sizeof(d);d.unit_id=int(i+1);d.action=1;d.direction=wanted[i].second;
            d.frame_count=16;d.sprite_width=d.sprite_height=191;d.projection_scale_milli=1000;
            d.body_x=cx-95;d.body_y=cy-95;d.hour=12;d.display_color_rgb=owner;
            d.presentation_frequency=1000;d.presentation_time_ticks=0;strcpy_s(d.unit_key,wanted[i].first.c_str());
            pose.tile_y=int(i/columns)*2;
            plan.entries.push_back({unsigned(poses.size()),
                c3x_renderer::render_core::unit_main_body|c3x_renderer::render_core::unit_ground_shadow|
                (reflect?c3x_renderer::render_core::unit_reflection:0u)});
            poses.push_back(pose);
        }
        plan.valid=true;
        SandboxPassWorkload work;sandbox_direct_units.work=&work;
        require(sandbox_direct_units.initialize(),"unit shaders");
        for(unsigned attempt=0;;++attempt){
            int ready=renderer.prepare_frame_unit_assets(poses);
            if(ready==C3X_RENDERER_RESULT_OK)ready=c3x_renderer64_prepare_unit_meshes();
            if(ready==C3X_RENDERER_RESULT_OK)break;
            require(ready==C3X_RENDERER_RESULT_PENDING&&attempt<20000,"unit asset preparation");
            Sleep(1);
        }
        require(sandbox_direct_units.prepare_real(frame,poses,12,plan),"prepare_real");
        LinearTarget target;require(target.ensure(renderer.device,width,height,false,true,1),"linear target");
        float clear[4]={0,0,0,0};renderer.context->ClearRenderTargetView(target.target,clear);
        renderer.context->ClearDepthStencilView(target.depth,D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);
        require(sandbox_direct_units.draw_real(frame,poses,target,1,12,false,1),"draw_real");

        auto environment=c3x_renderer::evaluate_environment(12,0);
        float exposure=environment.exposure;
        auto save=[&](LinearTarget& layer,std::string const& path){
            D3D11_TEXTURE2D_DESC desc={};layer.color->GetDesc(&desc);
            desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=desc.MiscFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            ComPtr<ID3D11Texture2D> staging;checked(renderer.device->CreateTexture2D(&desc,nullptr,&staging),"staging");
            renderer.context->CopyResource(staging.Get(),layer.color);
            D3D11_MAPPED_SUBRESOURCE mapped={};checked(renderer.context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped),"map");
            std::ofstream out(path,std::ios::binary);
            std::uint32_t header[4]={0x36314643u,std::uint32_t(width),std::uint32_t(height),0};
            std::memcpy(&header[3],&exposure,4);
            out.write(reinterpret_cast<char const*>(header),sizeof(header));
            for(int y=0;y<height;++y)out.write(static_cast<char const*>(mapped.pData)+std::size_t(y)*mapped.RowPitch,std::size_t(width)*8);
            renderer.context->Unmap(staging.Get(),0);
            require(bool(out),"write");
        };
        save(target,argv[3]);
        if(reflect){
            std::string base=argv[3];base=base.substr(0,base.size()-4);
            LinearTarget mirror;require(mirror.ensure(renderer.device,width,height,false,true,1),"mirror target");
            renderer.context->ClearRenderTargetView(mirror.target,clear);
            renderer.context->ClearDepthStencilView(mirror.depth,D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);
            require(sandbox_direct_units.draw_real(frame,poses,mirror,1,12,true,1),"reflected draw_real");
            save(mirror,base+".reflect.f16");
            auto const& e=environment;std::ofstream json(base+".env.json");
            json<<"{\"exposure\":"<<e.exposure<<",\"water_fresnel\":"<<e.water_fresnel<<",\"water_specular\":"<<e.water_specular
                <<",\"sun_intensity\":"<<e.sun_intensity<<",\"moon_intensity\":"<<e.moon_intensity
                <<",\"ambient\":["<<e.ambient_color[0]<<","<<e.ambient_color[1]<<","<<e.ambient_color[2]
                <<"],\"sun\":["<<e.sun_color[0]<<","<<e.sun_color[1]<<","<<e.sun_color[2]
                <<"],\"moon\":["<<e.moon_color[0]<<","<<e.moon_color[1]<<","<<e.moon_color[2]
                <<"],\"sun_direction\":["<<e.sun_direction[0]<<","<<e.sun_direction[1]<<","<<e.sun_direction[2]<<"]}\n";
            require(bool(json),"environment");
        }
        std::printf("PASS unit sheet: units=%zu draws=%u shadow_contributors=%u exposure=%.4f\n",
            poses.size(),sandbox_direct_units.draws,sandbox_direct_units.shadow_contributors,exposure);
        return 0;
    }catch(std::exception const& error){std::printf("FAIL unit sheet: %s\n",error.what());return 1;}
}
