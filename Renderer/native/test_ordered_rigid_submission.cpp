// Untimed hardware oracle for production farm/mine ordered packet submission.
// Build with the existing Renderer64 native-test link libraries. Run with the
// repository root as argv[1]. Uses local generic vertices/texels, no game/save.
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
void require_packet(bool ok,char const* text){if(!ok)throw std::runtime_error(text);}
void checked_packet(HRESULT hr,char const* text){require_packet(SUCCEEDED(hr),text);}
std::vector<unsigned char> read_packet(ID3D11Texture2D* texture,unsigned stride){
 D3D11_TEXTURE2D_DESC d{};texture->GetDesc(&d);require_packet(d.SampleDesc.Count==1,"single sample oracle");
 d.Usage=D3D11_USAGE_STAGING;d.BindFlags=d.MiscFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 ComPtr<ID3D11Texture2D> staging;checked_packet(renderer.device->CreateTexture2D(&d,nullptr,&staging),"oracle staging");
 renderer.context->CopyResource(staging.Get(),texture);D3D11_MAPPED_SUBRESOURCE mapped{};
 checked_packet(renderer.context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped),"oracle readback");
 std::vector<unsigned char> result(std::size_t(d.Width)*d.Height*stride);
 for(unsigned y=0;y<d.Height;++y)std::memcpy(result.data()+std::size_t(y)*d.Width*stride,
  static_cast<unsigned char const*>(mapped.pData)+std::size_t(y)*mapped.RowPitch,std::size_t(d.Width)*stride);
 renderer.context->Unmap(staging.Get(),0);return result;
}
void buffer_packet(void const* bytes,unsigned size,ID3D11Buffer** output){D3D11_BUFFER_DESC d{};
 d.ByteWidth=(size+15)&~15u;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
 std::vector<unsigned char> aligned(d.ByteWidth);if(bytes)std::memcpy(aligned.data(),bytes,size);
 D3D11_SUBRESOURCE_DATA data{};data.pSysMem=aligned.data();checked_packet(renderer.device->CreateBuffer(&d,&data,output),"oracle constant buffer");}
void initialize_packet(char const* root){
 checked_packet(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&renderer.device,nullptr,&renderer.context),"hardware device");
 renderer.shader_root=root;renderer.content_view_width=renderer.content_view_height=256;
 renderer.city_profile=renderer.pickup_profile=renderer.environment_profile=true;
 renderer.device_generation=1;renderer.content_revision=1;renderer.reflection.height_pixels=16;
 D3D11_DEPTH_STENCIL_DESC depth{};depth.DepthEnable=TRUE;depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;depth.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
 checked_packet(renderer.device->CreateDepthStencilState(&depth,&renderer.depth_state),"production depth rule");
 D3D11_BLEND_DESC blend{};auto& target=blend.RenderTarget[0];target.BlendEnable=TRUE;
 target.SrcBlend=D3D11_BLEND_SRC_ALPHA;target.DestBlend=D3D11_BLEND_INV_SRC_ALPHA;
 target.SrcBlendAlpha=D3D11_BLEND_ONE;target.DestBlendAlpha=D3D11_BLEND_ZERO;
 target.BlendOp=target.BlendOpAlpha=D3D11_BLEND_OP_ADD;target.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;
 checked_packet(renderer.device->CreateBlendState(&blend,&renderer.blend_state),"production alpha rule");
 D3D11_RASTERIZER_DESC raster{};raster.FillMode=D3D11_FILL_SOLID;raster.CullMode=D3D11_CULL_NONE;raster.ScissorEnable=raster.DepthClipEnable=TRUE;
 checked_packet(renderer.device->CreateRasterizerState(&raster,&renderer.rasterizer_state),"raster state");
 D3D11_SAMPLER_DESC sampler{};sampler.Filter=D3D11_FILTER_MIN_MAG_MIP_LINEAR;
 sampler.AddressU=sampler.AddressV=sampler.AddressW=D3D11_TEXTURE_ADDRESS_WRAP;sampler.MaxLOD=D3D11_FLOAT32_MAX;
 checked_packet(renderer.device->CreateSamplerState(&sampler,&renderer.terrain_sampler),"material sampler");
 renderer.terrain_sampler->AddRef();renderer.decal_sampler=renderer.terrain_sampler;
 for(auto* bundle:{&renderer.mine_bundle,&renderer.farm_bundle}){
  bundle->assets.resize(2);
  for(unsigned a=0;a<2;++a){auto& asset=bundle->assets[a];asset.id=a?"generic_rigid_b":"generic_rigid_a";
   asset.vertices.resize(a?4:3);asset.indices=a?std::vector<unsigned>{0,1,2,2,1,3}:std::vector<unsigned>{0,1,2};
   float positions[4][3]={{-.55f,-.45f,.02f},{.55f,-.45f,.02f},{-.2f,.5f,.3f},{.55f,.45f,.1f}};
   for(unsigned i=0;i<asset.vertices.size();++i){auto& v=asset.vertices[i];
    std::copy(positions[i],positions[i]+3,v.position);v.normal[2]=1;v.uv[0]=i&1?1.f:0.f;v.uv[1]=i>1?1.f:0.f;}
  }
 }
 c3x_renderer::objects::Assets assets{{&renderer.bridge_bundle,&renderer.site_bundle,&renderer.mine_bundle,&renderer.farm_bundle,&renderer.city_bundle,&renderer.wall_bundle}};
 require_packet(renderer.rigid_sources.ensure(renderer.device,root,assets),"actual production resident rigid shaders");
 std::wstring path(renderer.shader_root.begin(),renderer.shader_root.end());path+=L"/Renderer/native/city_fidelity/rigid_feature.hlsl";
 for(unsigned reflection=0;reflection<2;++reflection){ComPtr<ID3DBlob> code,error;
  auto hr=c3x_renderer::render_core::compile_cached(path.c_str(),reflection?"PSReflection":"PSIntegratedFeature","ps_5_0",&code,&error);
  if(error)std::fprintf(stderr,"%s\n",static_cast<char const*>(error->GetBufferPointer()));checked_packet(hr,"actual production feature pixel shader");
  checked_packet(renderer.device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,
   reflection?&renderer.reflection.ps[1]:&renderer.feature_pixel_shader),"feature pixel shader");
 }
 // Actual material branches include cutout and fractional ground feathering.
 unsigned texels[]={0xff397ca1u,0x00457ca1u,0xc0d5a273u,0xff69a88au};
 D3D11_TEXTURE2D_DESC image{};image.Width=image.Height=2;image.MipLevels=image.ArraySize=image.SampleDesc.Count=1;
 image.Format=DXGI_FORMAT_R8G8B8A8_UNORM;image.BindFlags=D3D11_BIND_SHADER_RESOURCE;
 D3D11_SUBRESOURCE_DATA pixels{texels,8,0};ComPtr<ID3D11Texture2D> texture;ComPtr<ID3D11ShaderResourceView> view;
 checked_packet(renderer.device->CreateTexture2D(&image,&pixels,&texture),"generic material pixels");
 checked_packet(renderer.device->CreateShaderResourceView(texture.Get(),nullptr,&view),"generic material view");
 std::array<ID3D11ShaderResourceView*,128> views;views.fill(view.Get());views[17]=views[25]=views[121]=nullptr;
 renderer.context->PSSetShaderResources(0,unsigned(views.size()),views.data());
 ID3D11SamplerState* samplers[]={renderer.terrain_sampler,renderer.decal_sampler};renderer.context->PSSetSamplers(0,2,samplers);
 TerrainShaderSettings light{};light.exposure=light.environment_exposure=light.sun_intensity=1;
 light.sun_color[0]=light.sun_color[1]=light.sun_color[2]=1;light.light_direction[2]=1;
 light.ambient_color[0]=.6f;light.ambient_color[1]=.7f;light.ambient_color[2]=.8f;light.hour=12;
 buffer_packet(&light,sizeof(light),&renderer.terrain_settings_buffer);
 buffer_packet(nullptr,sizeof(ViewportShaderSettings),&renderer.viewport_settings_buffer);
 buffer_packet(nullptr,80,&renderer.shadow_settings_buffer);
 std::array<float,4> reflected={16,8,0,0};ComPtr<ID3D11Buffer> reflection;
 buffer_packet(reflected.data(),sizeof(reflected),reflection.GetAddressOf());
 renderer.context->VSSetConstantBuffers(5,1,reflection.GetAddressOf());renderer.context->PSSetConstantBuffers(5,1,reflection.GetAddressOf());
 renderer.context->PSSetConstantBuffers(0,1,&renderer.terrain_settings_buffer);
 renderer.context->PSSetConstantBuffers(2,1,&renderer.shadow_settings_buffer);
}
struct PacketScene {
 std::vector<CachedVertexChunk> meshes;
 GeometryDrawView::Records records;
 explicit PacketScene(unsigned count=36){meshes.resize(count);
  for(unsigned n=0;n<meshes.size();++n){unsigned layer=unsigned(n%2?geometry_farm:geometry_mine);
   auto family=layer==geometry_farm?c3x_renderer::objects::farm_family:c3x_renderer::objects::mine_family;
   unsigned asset=(n/2)%2;auto const& gpu=renderer.rigid_sources.meshes[family][asset];auto& mesh=meshes[n];
   mesh.buffer=mesh.indices=gpu.buffer;mesh.vertex_stride=32;mesh.index_count=gpu.count;mesh.index_offset=gpu.index_offset;
   mesh.rigid_source=true;mesh.version=n+1;mesh.source_tile_width=128;mesh.projection_kind=2;
   mesh.bounds={-10000,-10000,10000,10000};mesh.world_bounds.low[2]=0;mesh.world_bounds.high[2]=1;
   // Deliberately overlapping and fractional-alpha rows prove order, rather
   // than gaining parity from disjoint opaque triangles.
   mesh.instance_material=n%3==0?21.31f:n%3==1?22.1f:23.0f;
   c3x_renderer::fidelity::MeshInstance instance{};
   instance.place[2]=.2f;instance.place[3]=.2f;instance.place[4]=1;instance.place[6]=.65f;
   mesh.instances=std::make_shared<std::vector<c3x_renderer::fidelity::MeshInstance> const>(1,instance);
   std::copy(std::begin(mesh.natural_projection),std::end(mesh.natural_projection),instance.projection);
   mesh.natural_projection[2]=128;mesh.natural_projection[3]=256;
   mesh.translation_x=int(n%3)*8;mesh.translation_y=105+int(n%4)*6;
   GeometryDrawView::Record record(mesh);record.owner={n+1,17};record.ordinal=n;
   records[layer].push_back(record);
  }
 }
 void front(){using Owner=c3x_renderer::render_core::SharedInstanceSubmission;Owner::Key identity={meshes.size(),1,1};
  auto build=renderer.shared_instances.begin_retained(identity,{},true);require_packet(bool(build),"placement builder");
  for(unsigned layer:{unsigned(geometry_mine),unsigned(geometry_farm)})for(auto const& record:records[layer]){
   GeometryDrawReference draw(record);auto const& mesh=draw.content();Owner::Range range;
   require_packet(renderer.shared_instances.append(build,renderer.shared_instance_draw_key(layer,draw),mesh.instances.get(),mesh.instances->data(),1,
    draw.natural_projection(),float(draw.translation_x()),float(draw.translation_y()),float(draw.translation_y()),mesh.instance_material,range),"exact occurrence placement");
  }sandbox_fresh.shared_front=renderer.shared_instances.upload(build,renderer.device,renderer.context);
  require_packet(renderer.shared_instances.valid(sandbox_fresh.shared_front) &&
   sandbox_fresh.shared_front->records==meshes.size() && sandbox_fresh.shared_front->buffer &&
   sandbox_fresh.shared_front->view,"initialized resident placement front");
 }
};
struct Snapshot {std::vector<unsigned char> color,depth;unsigned draws=0;};
Snapshot draw_packet(GeometryDrawView::Records const& records,ViewportShaderSettings viewport,bool reflected){
 LinearTarget target;require_packet(target.ensure(renderer.device,256,256,false,true,1),"single sample HDR/depth target");
 float clear[4]={.08f,.12f,.16f,1};renderer.context->ClearRenderTargetView(target.target,clear);
 renderer.context->ClearDepthStencilView(target.depth,D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);
 auto color_before=read_packet(target.color,8),depth_before=read_packet(target.depth_texture,4);
 renderer.context->OMSetRenderTargets(1,&target.target,target.depth);renderer.context->OMSetBlendState(renderer.blend_state,nullptr,0xffffffffu);
 renderer.context->OMSetDepthStencilState(renderer.depth_state,0);renderer.context->RSSetState(renderer.rasterizer_state);
 D3D11_VIEWPORT area={0,0,256,256,0,1};D3D11_RECT clip={0,0,256,256};
 renderer.context->RSSetViewports(1,&area);renderer.context->RSSetScissorRects(1,&clip);renderer.context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
 renderer.context->PSSetShader(reflected?renderer.reflection.ps[1]:renderer.feature_pixel_shader,nullptr,0);
 sandbox_fresh.work.enabled=true;sandbox_fresh.work.pass=reflected?SandboxPassWorkload::reflection_scene:SandboxPassWorkload::water;
 auto before=renderer.frame_draw_calls;
 for(auto layer:{geometry_mine,geometry_farm})require_packet(sandbox_fresh.issue_records(records,layer,viewport,clip,reflected),"actual production selected draw path");
 Snapshot result{read_packet(target.color,8),read_packet(target.depth_texture,4),unsigned(renderer.frame_draw_calls-before)};
 require_packet(result.color!=color_before&&result.depth!=depth_before,"nonempty actual-shader HDR and raw depth");return result;
}
void compare_packet(PacketScene& scene,ViewportShaderSettings viewport,bool reflected,unsigned scenario,bool reduction=true){
 auto builds=renderer.ordered_rigid_packets.builds,copies=renderer.ordered_rigid_packets.placement_copies;
 auto prepared=draw_packet(scene.records,viewport,reflected);auto warm=draw_packet(scene.records,viewport,reflected);
 require_packet(prepared.color==warm.color&&prepared.depth==warm.depth,"warm packet HDR/raw depth/stencil exact");
 require_packet(renderer.ordered_rigid_packets.builds==builds || copies<renderer.ordered_rigid_packets.placement_copies,"content-only packet build");
 auto warm_builds=renderer.ordered_rigid_packets.builds,warm_copies=renderer.ordered_rigid_packets.placement_copies;
 draw_packet(scene.records,viewport,reflected);
 require_packet(renderer.ordered_rigid_packets.builds==warm_builds&&renderer.ordered_rigid_packets.placement_copies==warm_copies,"unchanged selection allocates/copies nothing");
 // Test-only ablation: source GPU buffers and placement rows stay identical;
 // withdrawing CPU assembly inputs selects the actual existing fallback.
 renderer.ordered_rigid_packets.clear();auto farm=std::move(renderer.farm_bundle.assets),mine=std::move(renderer.mine_bundle.assets);
 auto baseline=draw_packet(scene.records,viewport,reflected);
 renderer.farm_bundle.assets=std::move(farm);renderer.mine_bundle.assets=std::move(mine);
 require_packet(prepared.color==baseline.color&&prepared.depth==baseline.depth,"packet/fallback actual-shader HDR depth stencil alpha/order exact");
 require_packet(reduction?prepared.draws<baseline.draws:prepared.draws<=baseline.draws,"compatible packet draw bound");
 std::printf("PASS ordered rigid packet: scenario=%u reflected=%u records=%zu packet_draws=%u baseline_draws=%u exact_hdr_depth_stencil=1 primitive_alpha_order=1\n",
  scenario,unsigned(reflected),scene.records[geometry_mine].size()+scene.records[geometry_farm].size(),prepared.draws,baseline.draws);
}
}
int main(int argc,char** argv){
 if(argc!=2){std::fprintf(stderr,"usage: test_ordered_rigid_submission REPOSITORY_ROOT\n");return 2;}
 SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
 try{initialize_packet(argv[1]);PacketScene scene;scene.front();ViewportShaderSettings viewport{};
  viewport.inverse_size[0]=viewport.inverse_size[1]=1.f/256;
  unsigned scenario=0;
  for(bool reflected:{false,true}){compare_packet(scene,viewport,reflected,scenario++);
   // Restore the prepared full-view pages after the test-only ablation, then
   // choose their stored subranges without repacking unchanged content.
   draw_packet(scene.records,viewport,reflected);
   auto complete=scene.records;for(auto layer:{geometry_mine,geometry_farm}){
    auto& rows=scene.records[layer];rows.erase(rows.begin()+4,rows.begin()+8);}
   viewport.translation[0]=16;viewport.translation[1]=8;viewport.depth_translation=8;
   compare_packet(scene,viewport,reflected,scenario++);
   draw_packet(scene.records,viewport,reflected);
   for(auto layer:{geometry_mine,geometry_farm})std::reverse(scene.records[layer].begin(),scene.records[layer].end());
   // Reversing overlapping alpha work deliberately breaks contiguous runs.
   // Exact pixels/order remain required even when a draw cannot be reduced.
   compare_packet(scene,viewport,reflected,scenario++,false);scene.records=std::move(complete);
   viewport.translation[0]=viewport.translation[1]=viewport.depth_translation=0;
  }
  // Native navigation can discover many small immutable packet ranges. Own
  // more than 64 individual pages while the actual joint byte budget has room,
  // then compare their real production HDR/depth and reflection draws.
  PacketScene many(192);many.front();
  for(bool reflected:{false,true}){
   renderer.ordered_rigid_packets.clear();unsigned pages=0;
   for(auto layer:{geometry_mine,geometry_farm})for(auto const& record:many.records[layer]){
    std::vector<GeometryDrawReference> one={record};
    std::array<RendererState::OrderedRigidPackets::Range,c3x_renderer::render_core::DrawParameterStream::limit> ranges{};
    renderer.prepare_ordered_rigid_packets(unsigned(layer),one,sandbox_fresh.shared_front,ranges);
    require_packet(bool(ranges[0])&&renderer.ordered_rigid_packets.page_count()==++pages,"more than 64 small native packet pages");
   }
   require_packet(pages==192&&renderer.ordered_rigid_packets.bytes()<=RendererState::OrderedRigidPackets::budget&&
    renderer.ordered_rigid_packets.peak_bytes()<=RendererState::OrderedRigidPackets::budget,"charged many-page owner bound");
   compare_packet(many,viewport,reflected,scenario++,false);
  }
  require_packet(renderer.ordered_rigid_packets.bytes()<=RendererState::OrderedRigidPackets::budget,"joint owner bound");
  std::puts("PASS actual production ordered packet oracle; local generic source assets, untimed native readbacks");return 0;
 }catch(std::exception const& error){std::fprintf(stderr,"ORDERED_RIGID_ORACLE_FAILED %s\n",error.what());return 1;}
}
