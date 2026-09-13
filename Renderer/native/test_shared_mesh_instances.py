"""Bounded production instance streams preserve payloads and device ownership."""
import unittest
from Renderer.native.native_cpp_test import run_cpp
from Renderer.lab.platform import ROOT


class SharedMeshInstanceTests(unittest.TestCase):
    def test_stream_bound_failure_reuse_and_reset(self):
        source=(ROOT/'Renderer/native/render_core/instance_stream.h').read_text()
        owner='struct InstanceStream {'+source.split('struct InstanceStream {',1)[1].split('inline HRESULT create_instance_layout',1)[0]
        run_cpp(r'''
#include <cassert>
#include <cstring>
#include <vector>
#include "Renderer/lab/shared/natural/instance.h"
using UINT=unsigned;using HRESULT=int;
#define FAILED(x) ((x)<0)
enum{D3D11_USAGE_DYNAMIC,D3D11_BIND_VERTEX_BUFFER,D3D11_CPU_ACCESS_WRITE,D3D11_MAP_WRITE_DISCARD,D3D11_MAP_WRITE_NO_OVERWRITE};
struct D3D11_BUFFER_DESC{UINT ByteWidth=0,Usage=0,BindFlags=0,CPUAccessFlags=0;};
struct D3D11_MAPPED_SUBRESOURCE{void*pData=nullptr;};
struct ID3D11Buffer{std::vector<char> data;int released=0;void Release(){++released;}};
struct ID3D11Device{
 ID3D11Buffer buffer;int refs=1,creates=0;bool fail=false;
 HRESULT CreateBuffer(D3D11_BUFFER_DESC const*d,void*,ID3D11Buffer**out){
  ++creates;if(fail)return -1;buffer.data.resize(d->ByteWidth);*out=&buffer;return 0;}
 void Release(){--refs;}
};
struct ID3D11DeviceContext{
 ID3D11Device*device;bool fail=false;int maps=0,unmaps=0;std::vector<UINT> modes;
 void GetDevice(ID3D11Device**out){*out=device;++device->refs;}
 HRESULT Map(ID3D11Buffer*b,UINT,UINT mode,UINT,D3D11_MAPPED_SUBRESOURCE*out){
  ++maps;modes.push_back(mode);if(fail)return -1;out->pData=b->data.data();return 0;}
 void Unmap(ID3D11Buffer*,UINT){++unmaps;}
};
namespace c3x_renderer {namespace render_core {
'''+owner+r'''
}}
int main(){
 using Stream=c3x_renderer::render_core::InstanceStream;
 static_assert(sizeof(Stream::Instance)==64,"instance vertex layout");
 ID3D11Device device;ID3D11DeviceContext context{&device};
 {
  Stream stream;std::vector<Stream::Instance> data;
  assert(!stream.upload(&device,&context,data) && device.creates==0);
  data.resize(Stream::limit+1);assert(!stream.upload(&device,&context,data) && device.creates==0);
  data.resize(Stream::limit);data.back().place[7]=12.5f;data.front().projection[2]=192;
  device.fail=true;assert(!stream.upload(nullptr,&context,data));assert(device.refs==1 && !stream.buffer);
  device.fail=false;assert(stream.upload(nullptr,&context,data));assert(device.refs==1);
  assert(stream.bytes==1024*1024 && stream.uploads==1 && context.unmaps==1);
  assert(!std::memcmp(device.buffer.data.data(),data.data(),stream.bytes));
  auto creates=device.creates;context.fail=true;
  assert(!stream.upload(&device,&context,data));assert(stream.bytes==1024*1024 && stream.uploads==1 && context.unmaps==1);
  context.fail=false;data.resize(1);data[0].view[0]=-129;
  assert(stream.upload(&device,&context,data) && device.creates==creates);
  assert(stream.bytes==1024*1024+64 && stream.uploads==2);
  assert(!std::memcmp(device.buffer.data.data(),data.data(),64));
  assert(stream.offset==0 && stream.cursor==1 && stream.discards==2);
  data[0].view[0]=-128;assert(stream.upload(&device,&context,data));
  assert(stream.offset==64 && stream.cursor==2 && context.modes.back()==D3D11_MAP_WRITE_NO_OVERWRITE);
  assert(!std::memcmp(device.buffer.data.data()+stream.offset,data.data(),64));
  auto old=data[0];old.view[0]=-129;assert(!std::memcmp(device.buffer.data.data(),&old,64));
  context.fail=true;assert(!stream.upload(&device,&context,data));
  assert(stream.offset==64 && stream.cursor==2);context.fail=false;
  data.resize(Stream::limit-2);assert(stream.upload(&device,&context,data));
  assert(stream.offset==128 && stream.cursor==Stream::limit && stream.discards==2);
  data.resize(1);assert(stream.upload(&device,&context,data));
  assert(stream.offset==0 && stream.cursor==1 && stream.discards==3 && context.modes.back()==D3D11_MAP_WRITE_DISCARD);
  stream.clear();assert(device.buffer.released==1 && !stream.buffer && !stream.bytes && !stream.uploads);
  assert(stream.upload(&device,&context,data));
 }
 assert(device.buffer.released==2 && device.refs==1);
}
''')


if __name__=='__main__':
    unittest.main()
