"""Host execution of actual shared-import producer/consumer contracts.

The command/COM model establishes no actual VA, GPU completion or driver-memory
bound; these are independently qualified by native tests."""
import os
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp
ROOT=Path(__file__).resolve().parents[2]
HEADER=(ROOT/'Renderer/native/gpu_native_presenter.h').read_text()
CPP=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
def block(text,marker):
 a=text.index(marker);b=text.index('{',a)+1;level=1
 while level:
  level+=(text[b]=='{')-(text[b]=='}');b+=1
 return text[a:b]
STUB=r'''
#include <cassert>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <stdexcept>
#include <cstdio>
using HANDLE=void*;using HRESULT=int;using BOOL=int;using DWORD=unsigned;using HMODULE=unsigned;
#define WINAPI
#define FAILED(hr) ((hr)<0)
#define SUCCEEDED(hr) ((hr)>=0)
#define IID_PPV_ARGS(value) value
enum{S_OK=0,E_FAIL=-1,FALSE=0,C3X_RENDERER_RESULT_OK=0,C3X_RENDERER_RESULT_BAD_ARGUMENT=2,
 C3X_RENDERER_RESULT_DEVICE_ERROR=3,C3X_RENDERER_RESULT_ERROR=4,C3X_RENDERER_RESULT_PENDING=5,
 DXGI_FORMAT_B8G8R8A8_UNORM=87,D3D11_BIND_RENDER_TARGET=1,D3D11_BIND_SHADER_RESOURCE=2,
 D3D11_RESOURCE_MISC_SHARED_NTHANDLE=4,D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX=8,
 DXGI_SHARED_RESOURCE_READ=16,DXGI_SHARED_RESOURCE_WRITE=32,DUPLICATE_SAME_ACCESS=1,PROCESS_DUP_HANDLE=2};
struct D3D11_TEXTURE2D_DESC{unsigned Width=2240,Height=1260,Format=87,MipLevels=1,ArraySize=1;
 struct{unsigned Count=1,Quality=0;}SampleDesc;unsigned BindFlags=0,MiscFlags=0;};
struct LARGE_INTEGER{long long QuadPart=0;};void QueryPerformanceCounter(LARGE_INTEGER* value){++value->QuadPart;}
struct Store{unsigned value=0;bool master_created=false;};std::unordered_map<unsigned,Store> stores;
std::unordered_map<HANDLE,unsigned> handles;std::vector<HANDLE> closed;std::uintptr_t next_handle=1000;unsigned next_store=10;
HANDLE H(std::uintptr_t value){return reinterpret_cast<HANDLE>(value);}
bool api_available=true,kernelbase_available=true,dup_fail=false,process_fail=false;
unsigned master_creates=0,duplicates=0;HANDLE GetCurrentProcess(){return H(1);}
void CloseHandle(HANDLE value){assert(value&&handles.count(value));closed.push_back(value);handles.erase(value);}
HANDLE incoming(std::uintptr_t numeric,unsigned object){auto value=H(numeric);assert(!handles.count(value));handles[value]=object;return value;}
BOOL CompareObjectHandles(HANDLE a,HANDLE b){return handles.count(a)&&handles.count(b)&&handles[a]!=0&&handles[a]==handles[b];}
HMODULE GetModuleHandleW(wchar_t const* name){return std::wcscmp(name,L"kernelbase.dll")==0?(kernelbase_available?1:0):2;}
using FARPROC=void(*)();FARPROC GetProcAddress(HMODULE,char const*){return api_available?reinterpret_cast<FARPROC>(&CompareObjectHandles):nullptr;}
BOOL DuplicateHandle(HANDLE,HANDLE source,HANDLE,HANDLE* out,unsigned,BOOL,unsigned){
 if(dup_fail||!handles.count(source))return FALSE;++duplicates;*out=incoming(++next_handle,handles[source]);return 1;
}
HANDLE OpenProcess(unsigned,BOOL,DWORD){return process_fail?0:incoming(++next_handle,0);}
struct ID3D11Device1;using ID3D11Device=ID3D11Device1;struct ID3D11Texture2D;struct IDXGIKeyedMutex;struct IDXGIResource1;
struct Ledger{std::vector<ID3D11Texture2D*>deferred;unsigned imported=0,peak=0,destroyed=0,opened=0,acquires=0,releases=0;
 bool as_fail=false,acquire_fail=false,release_fail=false,create_fail=false,master_fail=false;};
template<class T>struct ComPtr{
 T* pointer=nullptr;ComPtr()=default;ComPtr(ComPtr const& x):pointer(x.pointer){if(pointer)pointer->AddRef();}
 ~ComPtr(){Reset();}ComPtr& operator=(ComPtr const& x){if(this!=&x){auto p=x.pointer;if(p)p->AddRef();Reset();pointer=p;}return *this;}
 ComPtr& operator=(T* p){if(p)p->AddRef();Reset();pointer=p;return *this;}
 T* Get()const{return pointer;}T* operator->()const{return pointer;}explicit operator bool()const{return pointer!=nullptr;}
 T** operator&(){assert(!pointer);return &pointer;}void Reset(){if(pointer){auto p=pointer;pointer=nullptr;p->Release();}}
 template<class U>HRESULT As(U** out){return pointer->As(out);}
};
namespace Microsoft{namespace WRL{template<class T>using ComPtr=::ComPtr<T>;}}
struct ID3D11Texture2D{
 Ledger* ledger;ID3D11Device1* device;D3D11_TEXTURE2D_DESC desc;unsigned refs=1,object;bool imported;
 ID3D11Texture2D(Ledger& l,ID3D11Device1* d,unsigned id,bool alias):ledger(&l),device(d),object(id),imported(alias){
  if(alias){++ledger->imported;ledger->peak=std::max(ledger->peak,ledger->imported);}}
 void AddRef(){assert(refs);++refs;}void Release(){assert(refs);if(!--refs){if(imported)ledger->deferred.push_back(this);else delete this;}}
 void GetDesc(D3D11_TEXTURE2D_DESC* out){*out=desc;}void GetDevice(ID3D11Device1**);
 HRESULT As(IDXGIKeyedMutex**);HRESULT As(IDXGIResource1**);
};
struct IDXGIKeyedMutex{ID3D11Texture2D* source;unsigned refs=1;
 explicit IDXGIKeyedMutex(ID3D11Texture2D* s):source(s){source->AddRef();}
 void AddRef(){++refs;}void Release(){assert(refs);if(!--refs){source->Release();delete this;}}
 HRESULT AcquireSync(unsigned key,unsigned timeout){assert((key==0||key==1)&&timeout==1000);++source->ledger->acquires;return source->ledger->acquire_fail?1:S_OK;}
 HRESULT ReleaseSync(unsigned key){assert(key==0||key==1);++source->ledger->releases;return source->ledger->release_fail?E_FAIL:S_OK;}
};
struct IDXGIResource1{ID3D11Texture2D* source;unsigned refs=1;explicit IDXGIResource1(ID3D11Texture2D* s):source(s){source->AddRef();}
 void AddRef(){++refs;}void Release(){if(!--refs){source->Release();delete this;}}
 HRESULT CreateSharedHandle(void*,unsigned,char const*,HANDLE* out){
  if(source->ledger->master_fail)return E_FAIL;assert(!stores[source->object].master_created);
  stores[source->object].master_created=true;++master_creates;*out=incoming(++next_handle,source->object);return S_OK;}
};
struct ID3D11RenderTargetView{ID3D11Texture2D*source;unsigned refs=1;explicit ID3D11RenderTargetView(ID3D11Texture2D*s):source(s){s->AddRef();}
 void AddRef(){++refs;}void Release(){if(!--refs){source->Release();delete this;}}};
HRESULT ID3D11Texture2D::As(IDXGIKeyedMutex** out){if(ledger->as_fail)return E_FAIL;*out=new IDXGIKeyedMutex(this);return S_OK;}
HRESULT ID3D11Texture2D::As(IDXGIResource1** out){if(ledger->as_fail)return E_FAIL;*out=new IDXGIResource1(this);return S_OK;}
struct ID3D11Device1{Ledger&ledger;D3D11_TEXTURE2D_DESC descriptor;unsigned refs=1;bool open_fail=false,removed=false;
 explicit ID3D11Device1(Ledger&l):ledger(l){}void AddRef(){++refs;}void Release(){assert(refs>1);--refs;}
 HRESULT OpenSharedResource1(HANDLE handle,ID3D11Texture2D** out){
  if(open_fail||!handles.count(handle))return E_FAIL;++ledger.opened;
  *out=new ID3D11Texture2D(ledger,this,handles[handle],true);(*out)->desc=descriptor;return S_OK;}
 HRESULT GetDeviceRemovedReason(){return removed?E_FAIL:S_OK;}
 HRESULT CreateTexture2D(D3D11_TEXTURE2D_DESC const*desc,void*,ID3D11Texture2D**out){
  if(ledger.create_fail)return E_FAIL;*out=new ID3D11Texture2D(ledger,this,++next_store,false);(*out)->desc=*desc;return S_OK;}
 HRESULT CreateRenderTargetView(ID3D11Texture2D*t,void*,ID3D11RenderTargetView**out){*out=new ID3D11RenderTargetView(t);return S_OK;}
};
void ID3D11Texture2D::GetDevice(ID3D11Device1**out){*out=device;device->AddRef();}
struct ID3D11DeviceContext{Ledger&ledger;ID3D11Device1*device;
 struct Copy{ID3D11Texture2D*destination,*source;};std::vector<Copy>queued;unsigned copies=0,flushes=0;
 ID3D11DeviceContext(Ledger&l,ID3D11Device1*d):ledger(l),device(d){}
 void GetDevice(ID3D11Device1**out){*out=device;device->AddRef();}
 void CopyResource(ID3D11Texture2D*d,ID3D11Texture2D*s){s->AddRef();d->AddRef();queued.push_back({d,s});++copies;}
 void Flush(){++flushes;for(auto const&c:queued){stores[c.destination->object].value=stores[c.source->object].value;c.source->Release();c.destination->Release();}
  queued.clear();for(auto*t:ledger.deferred){assert(!t->refs);--ledger.imported;++ledger.destroyed;delete t;}ledger.deferred.clear();}
};
struct Trace{int level=0;double milliseconds(long long){return 0.;}void write(char const*,char const*,bool){}};
struct Session{int current_ticket(){return 1;}bool drawn=true;int visual_result=1;
 struct Rect{int left,top,right,bottom;};
 bool display_to(long long,std::uint64_t,ID3D11RenderTargetView*,ID3D11Texture2D*t,ID3D11Texture2D*,unsigned,unsigned,Rect,long long,long long){++stores[t->object].value;return drawn;}
 int visual_frame(long long,long long,ID3D11RenderTargetView*,ID3D11Texture2D*t,ID3D11Texture2D*){++stores[t->object].value;return visual_result;}
};
struct Harness{
 ComPtr<ID3D11Texture2D> display,back;bool swap=true,initialized=true,thread=true;unsigned width=2240,height=1260,owner=11,presents=0,written=0;
 int present_result=0;bool throw_present=false,last_independent=false;
 Harness(Ledger&l,ID3D11Device1*d){display.pointer=new ID3D11Texture2D(l,d,++next_store,false);back.pointer=new ID3D11Texture2D(l,d,++next_store,false);}
 void gpu_written(){++written;}int present(bool independent){++presents;last_independent=independent;if(throw_present)throw std::runtime_error("present");return present_result;}
 unsigned GetCurrentThreadId(){return thread?owner:owner+1;}
 ACTUAL_CACHE
 ACTUAL_COMPARE
 ACTUAL_ADOPT
};
struct Producer{
 ComPtr<ID3D11Texture2D>trial_display,trial_buffer;ComPtr<ID3D11RenderTargetView>trial_display_view;ComPtr<IDXGIKeyedMutex>trial_display_mutex;
 HANDLE trial_display_handle=0;unsigned trial_width=0,trial_height=0;std::uint64_t trial_handle=0;
 std::atomic<bool> trial_front_pending{false};std::uint64_t trial_presented_front_revision=0;
 struct Request{int width=2240,height=1260,ticket=1,image=1,area[4]={0,0,2240,1260};}gpu_present;
 DWORD trial_consumer_pid=99;long long visual_ticks=1,visual_frequency=1000;bool visual_allowed=true;Session session;
 struct State{ID3D11Device1*device;ID3D11DeviceContext*context;Session*gpu_composition;Trace trace;}renderer_state;
 Producer(ID3D11Device1*d,ID3D11DeviceContext*c):renderer_state{d,c,&session,{}}{}
 ~Producer(){retire_trial_display();}
 ACTUAL_RETIRE
 int produce(){int result=C3X_RENDERER_RESULT_ERROR;auto const&p=gpu_present;auto*session=&this->session;
 ACTUAL_PRODUCE
 return result;}
 int animate(){int result=C3X_RENDERER_RESULT_ERROR;
 ACTUAL_ANIMATE
 return result;}
};
'''
def program(main):
 cache=block(HEADER,'    struct SharedImport')+' shared_import;'
 # block() includes the struct terminator only through its closing brace.
 code=STUB.replace('ACTUAL_CACHE',cache).replace('ACTUAL_COMPARE',block(HEADER,'    auto shared_compare()const'))
 code=code.replace('ACTUAL_ADOPT',block(HEADER,'    int adopt_shared('))
 code=code.replace('ACTUAL_RETIRE',block(CPP,'    void retire_trial_display()'))
 code=code.replace('ACTUAL_PRODUCE',block(CPP,'                    if(session&&session->current_ticket()==p.ticket&&trial_consumer_pid&&p.width>0&&p.height>0&&'))
 code=code.replace('ACTUAL_ANIMATE',block(CPP,'                if(trial_display&&trial_display_view&&trial_display_mutex&&trial_buffer&&'))
 assert '#include <windows.h>' not in code
 return code+main
@unittest.skipIf(os.name=='nt','Host-only; never invoke Windows/VM tools')
class SharedImportReleaseOrderTests(unittest.TestCase):
 def test_current_producer_and_consumer_reuse_32_then_replace_dimensions(self):
  run_cpp(program(r'''
int main(){Ledger l;ID3D11Device1 d(l);ID3D11DeviceContext c(l,&d);Harness target(l,&d);Producer producer(&d,&c);
 for(unsigned n=1;n<=32;++n){assert((n%2?producer.produce():producer.animate())==0);
  auto handle=producer.trial_handle;assert(CompareObjectHandles(H(handle),producer.trial_display_handle));
  assert(target.adopt_shared(&d,&c,handle,2240,1260,n%2==0)==0);assert(!handles.count(H(handle)));
  assert(master_creates==1&&l.opened==1&&l.imported==1);assert(target.shared_import.bytes==std::uint64_t(2240)*1260*4);
  assert(stores[target.display->object].value==n&&stores[target.back->object].value==n);
 }
 assert(target.presents==32&&target.written==32&&c.copies==64);
 auto master=producer.trial_display_handle;target.width=320;target.height=240;
 producer.gpu_present.width=320;producer.gpu_present.height=240;producer.gpu_present.area[2]=320;producer.gpu_present.area[3]=240;
 d.descriptor.Width=320;d.descriptor.Height=240;
 assert(producer.produce()==0&&!handles.count(master)&&master_creates==2);
 assert(target.adopt_shared(&d,&c,producer.trial_handle,320,240)==0&&l.opened==2&&l.imported==1);
 assert(target.shared_import.bytes==320u*240u*4u);
 target.shared_import.reset();producer.retire_trial_display();c.Flush();assert(!l.imported&&handles.empty());
}
'''))
 def test_numeric_handle_reuse_different_object_and_failed_replacement_preserve_front(self):
  run_cpp(program(r'''
int main(){Ledger l;ID3D11Device1 d(l);ID3D11DeviceContext c(l,&d);Harness target(l,&d);
 stores[1].value=77;incoming(31,1);assert(target.adopt_shared(&d,&c,31,2240,1260)==0&&l.opened==1);
 auto value=stores[target.back->object].value;d.open_fail=true;incoming(31,2);
 assert(target.adopt_shared(&d,&c,31,2240,1260)==3&&l.opened==1&&target.presents==1);
 assert(stores[target.back->object].value==value&&!target.shared_import.identity&&!target.shared_import.bytes);
 d.open_fail=false;stores[2].value=88;incoming(31,2);assert(target.adopt_shared(&d,&c,31,2240,1260)==0&&l.opened==2);
 assert(stores[target.back->object].value==88);target.shared_import.reset();c.Flush();assert(handles.empty()&&!l.imported);
}
'''))
 def test_compare_unavailable_retains_original_uncached_path(self):
  run_cpp(program(r'''
int main(){api_available=false;Ledger l;ID3D11Device1 d(l);ID3D11DeviceContext c(l,&d);Harness target(l,&d);
 for(unsigned n=0;n<32;++n){incoming(31,1);stores[1].value=n;assert(target.adopt_shared(&d,&c,31,2240,1260)==0);
  assert(!target.shared_import.bytes&&!target.shared_import.identity&&!l.imported);}
 assert(l.opened==32&&target.presents==32&&duplicates==0&&handles.empty());
}
'''))
 def test_kernel32_resolution_duplicate_failure_and_budget_fallback(self):
  run_cpp(program(r'''
int main(){kernelbase_available=false;Ledger l;ID3D11Device1 d(l);ID3D11DeviceContext c(l,&d);Harness target(l,&d);
 dup_fail=true;incoming(31,1);assert(target.adopt_shared(&d,&c,31,2240,1260)==0&&!target.shared_import.identity&&!l.imported);
 dup_fail=false;incoming(31,1);assert(target.adopt_shared(&d,&c,31,2240,1260)==0&&target.shared_import.identity);
 target.shared_import.reset();c.Flush();
 target.width=d.descriptor.Width=4096;target.height=d.descriptor.Height=2048;
 for(unsigned i=0;i<2;++i){incoming(31,1);assert(target.adopt_shared(&d,&c,31,4096,2048)==0&&!target.shared_import.bytes&&!l.imported);}
 target.width=d.descriptor.Width=2048;target.height=d.descriptor.Height=2048;
 incoming(31,1);assert(target.adopt_shared(&d,&c,31,2048,2048)==0&&target.shared_import.bytes==16u*1024u*1024u);
 target.shared_import.reset();c.Flush();assert(handles.empty());
}
'''))
 def test_guard_key_descriptor_device_and_present_failures(self):
  run_cpp(program(r'''
int main(){for(unsigned failure=0;failure<16;++failure){Ledger l;ID3D11Device1 d(l),other(l);ID3D11DeviceContext c(l,&d);Harness target(l,&d);
 auto* device=&d;auto*context=&c;unsigned w=2240,h=1260;bool independent=false;
 incoming(31,1);switch(failure){case 0:d.open_fail=true;break;case 1:d.descriptor.Width=1;break;case 2:d.descriptor.Height=1;break;
 case 3:d.descriptor.Format=1;break;case 4:l.as_fail=true;break;case 5:l.acquire_fail=true;break;case 6:l.release_fail=true;break;
 case 7:d.removed=true;break;case 8:target.thread=false;break;case 9:target.initialized=false;independent=true;break;
 case 10:w=1;break;case 11:target.swap=false;break;case 12:device=nullptr;break;case 13:context=nullptr;break;
 case 14:device=&other;break;case 15:c.device=&other;break;}
 assert(target.adopt_shared(device,context,31,w,h,independent)!=0&&!handles.count(H(31)));
 assert(!target.presents&&!target.written&&!target.shared_import.identity&&!target.shared_import.bytes);
 assert(c.copies==(failure==6||failure==7?2u:0u));c.Flush();assert(!l.imported&&handles.empty());}
 Ledger l;ID3D11Device1 d(l);ID3D11DeviceContext c(l,&d);Harness target(l,&d);
 assert(target.adopt_shared(&d,&c,0,2240,1260)==2&&closed.size());
 incoming(31,1);target.present_result=4;assert(target.adopt_shared(&d,&c,31,2240,1260)==4&&!handles.count(H(31)));
 assert(!target.shared_import.identity);incoming(31,1);target.throw_present=true;
 try{target.adopt_shared(&d,&c,31,2240,1260);assert(false);}catch(std::runtime_error const&){}
 assert(!handles.count(H(31))&&!target.shared_import.identity&&l.opened==2&&target.presents==2);
 target.shared_import.reset();c.Flush();assert(!l.imported&&handles.empty());
}
'''))
 def test_raii_retires_identity_and_reset_order_precedes_detach(self):
  reset=block(HEADER,'    void reset(){\n        shared_import.reset();')
  assert reset.index('shared_import.reset()')<reset.index('composition_target')
  run_cpp(program(r'''
int main(){Ledger l;ID3D11Device1 d(l);ID3D11DeviceContext c(l,&d);HANDLE identity=0;
 {Harness target(l,&d);incoming(31,1);assert(target.adopt_shared(&d,&c,31,2240,1260)==0);
  identity=target.shared_import.identity;assert(handles.count(identity)&&l.imported==1);
  target.present_result=5;incoming(32,1);assert(target.adopt_shared(&d,&c,32,2240,1260)==5);
  assert(target.shared_import.identity==identity&&l.opened==1);}
 // The actual owner destructor closes its identity; command/runtime retirement
 // is modeled separately and is not an actual Windows memory claim.
 assert(!handles.count(identity)&&handles.empty());c.Flush();assert(!l.imported);
}
'''))
 def test_producer_failure_and_source_device_retirement(self):
  run_cpp(program(r'''
int main(){for(unsigned failure=0;failure<6;++failure){Ledger l;ID3D11Device1 d(l);ID3D11DeviceContext c(l,&d);Producer p(&d,&c);
 switch(failure){case 0:l.create_fail=true;break;case 1:l.as_fail=true;break;case 2:l.master_fail=true;break;
 case 3:l.acquire_fail=true;break;case 4:l.release_fail=true;break;case 5:dup_fail=true;break;}
 assert(p.produce()==(failure==4?2:3)&&!p.trial_display_handle&&!p.trial_display&&!p.trial_display_view&&!p.trial_buffer&&!p.trial_display_mutex);
 assert(handles.empty());dup_fail=false;}
 Ledger l;ID3D11Device1 d(l),other(l);ID3D11DeviceContext c(l,&d);Producer p(&d,&c);
 assert(p.produce()==0);CloseHandle(H(p.trial_handle));auto first=p.trial_display_handle;
 p.renderer_state.device=&other;assert(p.animate()==3&&!handles.count(first)&&!p.trial_display_handle);
 assert(p.produce()==0);CloseHandle(H(p.trial_handle));assert(master_creates>=2);p.retire_trial_display();assert(handles.empty());
}
'''))
if __name__=='__main__':unittest.main()
