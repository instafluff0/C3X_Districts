"""Retained DDS file bytes are freed after upload and re-read after a reset.

Renderer64 kept every pack texture's DDS file in RAM beside its GPU texture,
only so that reset() could upload it again: 591 MB on the busy save (review
89). Once all textures are uploaded, the bytes of texture members with a
recorded pack file are freed; a texture whose view a reset dropped is read
from that file again for its upload. Bytes without a recorded file (held in
another container) stay, and the opt-in CPU terrain preview keeps the terrain
base colors it samples.
"""
import re
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class TextureFileReleaseTests(unittest.TestCase):
    def test_bytes_are_released_and_read_again(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        methods = source[source.index('    // Every retained DDS texture member.'):source.index('    void geometry_census(){')]
        ensure = source[source.index('    bool ensure_dds_texture('):source.index('    bool ensure_pack_texture(')]
        visitor = methods[:methods.index('    std::size_t texture_file_bytes(){')]
        singles = sorted(set(re.findall(r'&(\w+_dds)\b', visitor)))
        arrays = sorted(set(re.findall(r'files\((\w+_dds)\)', visitor)))
        members = ''.join(f' std::vector<std::uint8_t> {name};\n' for name in singles)
        members += ''.join(f' std::array<std::vector<std::uint8_t>,4> {name};\n' for name in arrays)
        run_cpp(r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>
using HRESULT=long;using UINT=unsigned;
inline bool SUCCEEDED(HRESULT hr){return hr>=0;}
template<std::size_t N,class...A>void sprintf_s(char(&s)[N],char const*f,A...a){std::snprintf(s,N,f,a...);}
enum DXGI_FORMAT {DXGI_FORMAT_R16G16B16A16_FLOAT=10,DXGI_FORMAT_R16G16B16A16_UNORM=11,DXGI_FORMAT_R8G8B8A8_UNORM=28,
 DXGI_FORMAT_R16G16_UNORM=35,DXGI_FORMAT_R32_FLOAT=41,DXGI_FORMAT_R8_UNORM=61,DXGI_FORMAT_BC1_UNORM=71,
 DXGI_FORMAT_BC1_UNORM_SRGB=72,DXGI_FORMAT_BC3_UNORM=77,DXGI_FORMAT_BC3_UNORM_SRGB=78,DXGI_FORMAT_BC4_UNORM=80,
 DXGI_FORMAT_BC4_SNORM=81,DXGI_FORMAT_BC5_UNORM=83};
enum {D3D11_USAGE_IMMUTABLE=1,D3D11_BIND_SHADER_RESOURCE=8};
struct D3D11_SUBRESOURCE_DATA {void const* pSysMem=nullptr;UINT SysMemPitch=0,SysMemSlicePitch=0;};
struct D3D11_TEXTURE2D_DESC {UINT Width,Height,MipLevels,ArraySize;DXGI_FORMAT Format;struct{UINT Count;}SampleDesc;UINT Usage,BindFlags;};
struct ID3D11Texture2D {std::vector<std::uint8_t> top;};
struct ID3D11ShaderResourceView {std::vector<std::uint8_t> top;};
struct Device {
 unsigned uploads=0;
 HRESULT CreateTexture2D(D3D11_TEXTURE2D_DESC const*,D3D11_SUBRESOURCE_DATA const* data,ID3D11Texture2D** out){
  ++uploads;auto p=static_cast<std::uint8_t const*>(data[0].pSysMem);
  *out=new ID3D11Texture2D{std::vector<std::uint8_t>(p,p+data[0].SysMemSlicePitch)};return 0;}
 HRESULT CreateShaderResourceView(ID3D11Texture2D* t,void*,ID3D11ShaderResourceView** out){
  *out=new ID3D11ShaderResourceView{t->top};return 0;}
};
namespace c3x_renderer {namespace render_core {
inline unsigned cached_environment(char const* name,char* buffer,unsigned size){
 auto value=std::getenv(name);if(!value)return 0;std::snprintf(buffer,size,"%s",value);return unsigned(std::strlen(value));}
}}
struct TerrainTexture {
 std::vector<std::uint8_t> dds,material_height_dds,specular_dds,elevated_dds,elevated_height_dds,elevated_specular_dds;
 std::array<std::vector<std::uint8_t>,5> relief_layer_dds,water_surface_dds;
};
struct State {
 std::array<TerrainTexture,3> terrain_textures;TerrainTexture dune_surface;
''' + members + r'''
 std::unordered_map<std::vector<std::uint8_t> const*,std::string> texture_files;
 Device owned;Device* device=&owned;bool material_views_valid=true;
 struct {std::vector<std::string> lines;void write(char const* stage,char const* detail,bool){lines.push_back(std::string(stage)+" "+detail);}} trace;
 static std::uint32_t read_u32(std::vector<std::uint8_t> const& d,std::size_t o){
  return std::uint32_t(d[o])|std::uint32_t(d[o+1])<<8|std::uint32_t(d[o+2])<<16|std::uint32_t(d[o+3])<<24;}
 static bool read_file(char const* path,std::vector<std::uint8_t>& out){
  std::ifstream in(path,std::ios::binary);if(!in)return false;
  out.assign(std::istreambuf_iterator<char>(in),{});out.shrink_to_fit();return !out.empty();}
 template<class T>void release(T*& value){delete value;value=nullptr;}
''' + methods + ensure + r'''};
std::vector<std::uint8_t> dds_file(char const* path,std::uint8_t fill){
 // A 4x4 BC1 texture: 148-byte DX10 header and one 8-byte block.
 std::vector<std::uint8_t> d(156,0);std::memcpy(d.data(),"DDS ",4);std::memcpy(d.data()+84,"DX10",4);
 auto put=[&](std::size_t o,std::uint32_t v){for(int i=0;i<4;++i)d[o+i]=std::uint8_t(v>>(8*i));};
 put(4,124);put(12,4);put(16,4);put(28,1);put(128,DXGI_FORMAT_BC1_UNORM);
 for(std::size_t i=148;i<156;++i)d[i]=std::uint8_t(fill+i);
 std::ofstream(path,std::ios::binary).write(reinterpret_cast<char const*>(d.data()),std::streamsize(d.size()));return d;}
int main(int,char** argv){
 // Files live beside the test executable, in run_cpp's temporary folder.
 std::string dir(argv[0]);dir.resize(dir.find_last_of('/')+1);
 auto pa=dir+"a.dds",pb=dir+"b.dds",pc=dir+"c.dds";
 auto a=dds_file(pa.c_str(),1),b=dds_file(pb.c_str(),2),c=dds_file(pc.c_str(),3);
 auto top=[](std::vector<std::uint8_t> const& d){return std::vector<std::uint8_t>(d.begin()+148,d.end());};
 auto s=new State;
 // Member textures record their pack file; bytes held elsewhere do not.
 State::read_file(pa.c_str(),s->terrain_textures[0].dds);s->texture_file(s->terrain_textures[0].dds,pa.c_str());
 State::read_file(pb.c_str(),s->cliff_dds[3]);s->texture_file(s->cliff_dds[3],pb.c_str());
 State::read_file(pc.c_str(),s->site_dds[0]); // loaded without a recorded file
 std::vector<std::uint8_t> moved;s->texture_file(moved,pc.c_str());
 assert(s->texture_files.size()==2);
 ID3D11ShaderResourceView *va=nullptr,*vb=nullptr,*vc=nullptr,*optional=nullptr;
 assert(s->ensure_dds_texture(s->terrain_textures[0].dds,va,true) && s->ensure_dds_texture(s->cliff_dds[3],vb,true) &&
  s->ensure_dds_texture(s->site_dds[0],vc,true) && s->device->uploads==3 && va->top==top(a) && vb->top==top(b));
 assert(s->texture_file_bytes()==3*156u);
 // After every upload: recorded bytes are freed, the rest stay.
 s->release_texture_files();
 assert(s->terrain_textures[0].dds.capacity()==0 && s->cliff_dds[3].capacity()==0 && s->site_dds[0].size()==156);
 assert(s->texture_file_bytes()==156u && s->trace.lines.back()=="texture-files-released released_kb=0 kept_kb=0");
 // Live views need nothing; a reset drops them and the files are read again.
 assert(s->ensure_dds_texture(s->terrain_textures[0].dds,va,true) && s->device->uploads==3);
 s->release(va);s->release(vb);s->release(vc);
 assert(s->ensure_dds_texture(s->terrain_textures[0].dds,va,true) && s->ensure_dds_texture(s->cliff_dds[3],vb,true) &&
  s->ensure_dds_texture(s->site_dds[0],vc,true) && s->device->uploads==6);
 assert(va->top==top(a) && vb->top==top(b) && vc->top==top(c));
 assert(s->terrain_textures[0].dds.empty() && s->cliff_dds[3].empty()); // the re-read bytes are not retained
 // Absent textures keep their contract: optional succeeds, required fails.
 assert(s->ensure_dds_texture(s->dune_decal_base_dds,optional,false) && !optional);
 assert(!s->ensure_dds_texture(s->dune_decal_base_dds,optional,true));
 // A missing file fails the upload instead of drawing nothing.
 std::remove(pb.c_str());s->release(vb);assert(!s->ensure_dds_texture(s->cliff_dds[3],vb,true));
 // The opt-in CPU terrain preview samples terrain base colors: they stay.
 auto p=new State;setenv("C3X_RENDERER_CAMERA_PREVIEW","1",1);
 State::read_file(pa.c_str(),p->terrain_textures[1].dds);p->texture_file(p->terrain_textures[1].dds,pa.c_str());
 State::read_file(pc.c_str(),p->terrain_textures[1].specular_dds);p->texture_file(p->terrain_textures[1].specular_dds,pc.c_str());
 p->release_texture_files();
 assert(p->terrain_textures[1].dds.size()==156 && p->terrain_textures[1].specular_dds.empty());
 std::puts("PASS texture files: released after upload, read again after a reset");
}
''')


if __name__ == '__main__':
    unittest.main()
