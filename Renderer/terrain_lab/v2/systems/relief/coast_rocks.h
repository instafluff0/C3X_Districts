// Included inside the portable combined scene after its source/height helpers.
// Exact normalized source bodies, uniform world transforms and source UVs.
namespace lab_coast_rocks {
struct Batch {
 ID3D11ShaderResourceView* views[4]={};
 ID3D11Buffer* buffer=nullptr;
 std::vector<FeatureVertex> vertices;
};
struct Runtime {
 FeatureBundle bundle;
 std::vector<Batch> batches;
 ~Runtime(){for(auto&b:batches){release(b.buffer);for(auto&v:b.views)release(v);}}
 bool load(ID3D11Device* device){
  auto path=std::getenv("C3X_LAB_V2_COASTAL_ROCKS");if(!path)return true;
  if(!labv2::hydrology_hooks.coast_segment || !load_feature_bundle(path,bundle))return false;
  if(bundle.assets.size()!=6 || bundle.texture_paths.size()!=24)return false;
  batches.resize(bundle.assets.size());
  DXGI_FORMAT formats[]={DXGI_FORMAT_BC1_UNORM_SRGB,DXGI_FORMAT_BC5_UNORM,DXGI_FORMAT_BC4_UNORM,DXGI_FORMAT_BC1_UNORM};
  for(unsigned i=0;i<batches.size();i++)for(unsigned c=0;c<4;c++){
   unsigned width=0,height=0;
   if(!load_dds(device,bundle.texture_paths.at(i*4+c),formats[c],&batches[i].views[c],width,height))return false;
  }
  return true;
 }
 bool build(ID3D11Device* device,HeightField const* heights,HeightField const* blend){
  if(batches.empty())return true;
  auto setting=std::getenv("C3X_LAB_V2_COASTAL_ROCK_PLACEMENT");
  bool grouped=setting && std::atoi(setting)>=2;
  bool joined=setting && std::atoi(setting)>=3;
  std::vector<std::array<float,2>> used;
  for(unsigned index=0;index<65536;index++){
   float segment[5];if(!labv2::hydrology_hooks.coast_segment(index,segment))break;
   float x=(segment[0]+segment[2])*.5f,y=(segment[1]+segment[3])*.5f;
   if(x<.1f||y<.1f||x>biq_window.columns-.1f||y>biq_window.rows-.1f||segment[4]<(grouped?.62f:.24f))continue;
   float spacing=joined?.18f:(grouped?.32f:.24f);
   bool near=false;for(auto p:used)if((p[0]-x)*(p[0]-x)+(p[1]-y)*(p[1]-y)<spacing*spacing){near=true;break;}
   if(near)continue;
   auto distance=labv2::hydrology_hooks.signed_shore_distance;
   float nx=distance(x-.01f,y)-distance(x+.01f,y);
   float ny=distance(x,y-.01f)-distance(x,y+.01f);
   float n=std::sqrt(nx*nx+ny*ny);if(n<1e-6f)continue;nx/=n;ny/=n;
   float top=biq_world_height(x+nx*.38f,y+ny*.38f,heights,blend)/112.f;
   if(joined){
    top=biq_world_height(x+nx*.26f,y+ny*.26f,heights,blend)/112.f;
    if(top<.05f)continue;
   }
   auto tile=biq_tile_at(x+nx*.15f,y+ny*.15f);if(!tile)continue;
   if(grouped){
    auto inland=biq_tile_at(x+nx*.42f,y+ny*.42f);
    if(!inland || inland->real!=5)continue;
   }
   unsigned seed=feature_hash(unsigned(tile->source_x)*73856093u^unsigned(tile->source_y)*19349663u^
                            unsigned(std::floor((x-std::floor(x))*8))*139u^
                            unsigned(std::floor((y-std::floor(y))*8))*367u);
   unsigned asset_index=seed%4;
   auto const& asset=bundle.assets[asset_index];auto& batch=batches[asset_index];
   float maximum=0;for(auto const&v:asset.vertices)maximum=std::max(maximum,v.position[2]);
   float scale=std::clamp((std::max(.12f,top)+.07f)/maximum,.18f,.42f);
   if(grouped)scale=std::clamp((std::max(.16f,top)+.16f)/maximum,.40f,.68f)*(.9f+.2f*feature_random(seed^421u));
   if(joined)scale=std::clamp((top+.24f)/maximum,.48f,.70f);
   float yaw=std::atan2(ny,nx)+feature_random(seed^9347u)*2.4f;
   float cosine=std::cos(yaw),sine=std::sin(yaw);
   used.push_back({x,y});
   float embed=joined?.02f:(grouped?.14f:.06f);
   x+=nx*embed;y+=ny*embed;float z=grouped?-.12f:-.055f;
   if(joined)z=top-maximum*scale-.015f;
   std::vector<FeatureVertex> transformed;transformed.reserve(asset.vertices.size());
   for(auto const&v:asset.vertices){
    float wx=x+(v.position[0]*cosine-v.position[1]*sine)*scale;
    float wy=y+(v.position[0]*sine+v.position[1]*cosine)*scale;
    float wz=z+v.position[2]*scale;
    float base_y=coast_projection.origin_y+(wx-wy)*coast_projection.half_height;
    float sx=coast_projection.origin_x+(wx+wy-1)*coast_projection.half_width;
    float sy=base_y-wz*112*coast_projection.vertical_scale;
    float depth=std::clamp(.94f-base_y/output_height*.75f-wz*112*.0012f,.01f,.99f);
    FeatureVertex result{ndc_x(sx),ndc_y(sy),depth,v.uv[0],v.uv[1],
      v.normal[0]*cosine-v.normal[1]*sine,-(v.normal[0]*sine+v.normal[1]*cosine),v.normal[2],.48f};
    result.world_x=wx;result.world_y=wy;result.world_z=wz;result.world_valid=1;
    transformed.push_back(result);
   }
   for(auto i:asset.indices)batch.vertices.push_back(transformed[i]);
   auto hash=labv2::content_sha256(asset.vertices.data(),asset.vertices.size()*sizeof(FeatureSourceVertex));
   std::ostringstream mesh;mesh<<"{\"id\":"<<labv2::json_string(asset.id)<<",\"bundle_sha256\":"<<labv2::json_string(bundle.source_sha256)<<",\"vertices_sha256\":"<<labv2::json_string(hash)<<",\"source_uv_preserved\":true}";
   labv2::source_meshes[hash]=mesh.str();
   std::ostringstream instance;instance<<std::setprecision(9)<<"{\"mesh_sha256\":"<<labv2::json_string(hash)<<",\"class\":\"source_coastal_rock\",\"uniform_world_scale\":"<<scale<<",\"yaw\":"<<yaw<<",\"translation\":["<<x<<','<<y<<','<<z<<"],\"source_uv_preserved\":true,\"shore_rockiness\":"<<segment[4]<<"}";
   labv2::source_instances.push_back(instance.str());
  }
  for(auto&b:batches)if(!b.vertices.empty()){
   D3D11_BUFFER_DESC desc={};desc.ByteWidth=UINT(b.vertices.size()*sizeof(FeatureVertex));
   desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_VERTEX_BUFFER;
   D3D11_SUBRESOURCE_DATA initial={};initial.pSysMem=b.vertices.data();
   if(FAILED(device->CreateBuffer(&desc,&initial,&b.buffer)))return false;
  }
  return true;
 }
};
}
