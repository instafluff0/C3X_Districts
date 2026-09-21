#define NOMINMAX
#include <windows.h>
#include "input_recording/native_calls.h"
#include <cassert>
#include <iostream>
using namespace c3x_inputs;
unsigned leases=0,releases=0;unsigned short pixels[4]={1,2,3,4};
unsigned short* __fastcall lease(void*,int){++leases;return pixels;}
void __fastcall release(void*,int,int){++releases;}
struct FailedRecorder:NativeValuesProvider {
    unsigned short* words(void* p,void* getter,bool write)override{
        c3x_native_access::NativeScope scope;auto value=c3x_native_access::words(p,getter,write);
        borrowed_words=true;last_words=value;throw std::bad_alloc();
    }
};
int main(){
    NativeValues values;Writer scalar;scalar.u32(2240);values[{NativeValuesProvider::scalar,1,0x38}]=scalar.bytes;
    NativeValueStream writer,reader;Writer first,second;writer.encode(first,values);writer.encode(second,values);assert(second.bytes.size()<first.bytes.size());
    Reader a{first.bytes},b{second.bytes};assert(reader.decode(a)==values);assert(reader.decode(b)==values);a.done();b.done();
    reader.retire(1);assert(reader.values.empty());
    bool rejected=false;try{NativeValueStream empty;Reader missing{second.bytes};empty.decode(missing);}catch(std::exception const&){rejected=true;}assert(rejected);
    auto corrupt=first.bytes;corrupt.back()^=1;rejected=false;try{NativeValueStream empty;Reader bad{corrupt};empty.decode(bad);}catch(std::exception const&){rejected=true;}assert(rejected);
    NativeValuesProvider replay;replay.replay=true;replay.values=values;c3x_native_access::provider()=&replay;
    assert(c3x_native_access::field(reinterpret_cast<void*>(1),0x38)==2240);
    rejected=false;try{c3x_native_access::field(reinterpret_cast<void*>(1),0x3c);}catch(std::exception const&){rejected=true;}assert(rejected&&c3x_native_access::provider()==&replay);
    FailedRecorder failing;c3x_native_access::provider()=&failing;
    auto pointer=c3x_native_access::words(nullptr,reinterpret_cast<void*>(lease));
    assert(pointer==pixels&&leases==1&&c3x_native_access::provider()==nullptr&&failing.failed);
    c3x_native_access::release_words(nullptr,reinterpret_cast<void*>(release));assert(releases==1);
    custom_renderer_native_view original{1,2,3,4,5,6,7,8,9,10,11,12},restored{};Writer view;native_view(view,original);Reader view_in{view.bytes};native_view(view_in,restored);view_in.done();assert(!std::memcmp(&original,&restored,sizeof original));
    std::cout<<"PASS native input dependencies: content references, missing/corrupt input rejection, complete navigation fields, capture failure preserves one lease/release; replay fails before native dereference\n";
}
