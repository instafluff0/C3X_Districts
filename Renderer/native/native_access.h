#pragma once
// Native dependencies of composition. A replay supplies owned values through
// this interface; native object addresses never become replay object layouts.
#include <windows.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
namespace c3x_native_access {
struct Provider {
    virtual ~Provider()=default;
    bool borrowed_words=false,borrowed_sprite=false;void* last_words=nullptr;void* last_sprite=nullptr;
    virtual bool recording_failed()noexcept{return false;}
    virtual int field(void*,unsigned)=0;
    virtual void set_field(void*,unsigned,unsigned)=0;
    virtual void* pointer(void*,unsigned)=0;
    virtual bool sprite(void*)=0;
    virtual int scale(unsigned)=0;
    virtual void* palette()=0;
    virtual bool dib(void*,DIBSECTION&)=0;
    virtual HDC dc(void*)=0;
    virtual HWND window(void*)=0;
    virtual unsigned short* words(void*,void*,bool)=0;
    virtual void release_words(void*,void*)=0;
    virtual unsigned char* sprite_bytes(void*)=0;
    virtual void release_sprite(void*)=0;
    virtual unsigned char const* rows(void*)=0;
    virtual unsigned short const* colors(void const*,bool)=0;
    virtual unsigned rgb(void*,unsigned)=0;
    virtual unsigned short const* lookup(void const*,unsigned)=0;
    virtual int readiness(int value){return value;}
};
inline Provider*& provider(){thread_local Provider* value=nullptr;return value;}
template<class F,class Native>auto access(F get,Native native)->decltype(native()){auto current=provider();if(!current)return native();try{return get(*current);}catch(...){if(!current->recording_failed())throw;provider()=nullptr;return native();}}
inline int field(void* p,unsigned offset){if(provider())return access([&](Provider& v){return v.field(p,offset);},[&]{return *reinterpret_cast<int*>(static_cast<char*>(p)+offset);});return *reinterpret_cast<int*>(static_cast<char*>(p)+offset);}
inline void set_field(void* p,unsigned offset,unsigned value){if(provider()){access([&](Provider& v){v.set_field(p,offset,value);return 0;},[&]{set_field(p,offset,value);return 0;});return;}*reinterpret_cast<unsigned*>(static_cast<char*>(p)+offset)=value;}
inline void* pointer(void* p,unsigned offset){if(provider())return access([&](Provider& v){return v.pointer(p,offset);},[&]{return pointer(p,offset);});return *reinterpret_cast<void**>(static_cast<char*>(p)+offset);}
inline bool sprite(void* p){if(provider())return access([&](Provider& v){return v.sprite(p);},[&]{return sprite(p);});auto m=reinterpret_cast<char*>(GetModuleHandleA("jgl.dll"));return m&&p&&*static_cast<void***>(p)==reinterpret_cast<void**>(m+0x68440);}
inline int scale(unsigned axis){if(provider())return access([&](Provider& v){return v.scale(axis);},[&]{return scale(axis);});auto m=reinterpret_cast<char*>(GetModuleHandleA("jgl.dll"));return m?*reinterpret_cast<int*>(m+(axis==0?0x6c0fc:axis==1?0x6c100:0x6c104)):0;}
inline void* palette(){if(provider())return access([&](Provider& v){return v.palette();},[&]{return palette();});auto m=reinterpret_cast<char*>(GetModuleHandleA("jgl.dll"));auto owner=m?*reinterpret_cast<void**>(m+0x70f48):nullptr;return owner?*reinterpret_cast<void**>(static_cast<char*>(owner)+4):nullptr;}
inline bool dib(void* p,DIBSECTION& value){if(provider())return access([&](Provider& v){return v.dib(p,value);},[&]{return dib(p,value);});auto bitmap=*reinterpret_cast<HBITMAP*>(static_cast<char*>(p)+0x4b4);return GetObject(bitmap,sizeof value,&value)==sizeof value;}
inline HDC dc(void* p){if(provider())return access([&](Provider& v){return v.dc(p);},[&]{return dc(p);});return *reinterpret_cast<HDC*>(static_cast<char*>(p)+0x4bc);}
inline HWND window(void* graph){if(provider())return access([&](Provider& v){return v.window(graph);},[&]{return window(graph);});return WindowFromDC(*reinterpret_cast<HDC*>(static_cast<char*>(graph)+0x138));}
inline unsigned short* words(void* p,void* getter,bool write=false){if(provider()){auto current=provider();current->borrowed_words=false;return access([&](Provider& v){return v.words(p,getter,write);},[&]{return current->borrowed_words?static_cast<unsigned short*>(current->last_words):words(p,getter,write);});}return getter?reinterpret_cast<unsigned short*(__thiscall*)(void*)>(getter)(p):*reinterpret_cast<unsigned short**>(static_cast<char*>(p)+0x4c0);}
inline void release_words(void* p,void* release){if(provider()){access([&](Provider& v){v.release_words(p,release);return 0;},[&]{release_words(p,release);return 0;});return;}if(release)reinterpret_cast<void(__thiscall*)(void*,int)>(release)(p,1);}
inline unsigned char* sprite_bytes(void* p){if(provider()){auto current=provider();current->borrowed_sprite=false;return access([&](Provider& v){return v.sprite_bytes(p);},[&]{return current->borrowed_sprite?static_cast<unsigned char*>(current->last_sprite):sprite_bytes(p);});}auto table=*static_cast<void***>(p);return reinterpret_cast<unsigned char*(__thiscall*)(void*)>(table[8])(p);}
inline void release_sprite(void* p){if(provider()){access([&](Provider& v){v.release_sprite(p);return 0;},[&]{release_sprite(p);return 0;});return;}auto table=*static_cast<void***>(p);reinterpret_cast<void(__thiscall*)(void*,int)>(table[9])(p,1);}
inline unsigned char const* rows(void* p){if(provider())return access([&](Provider& v){return v.rows(p);},[&]{return rows(p);});return *reinterpret_cast<unsigned char**>(static_cast<char*>(p)+0x1c);}
inline unsigned short const* colors(void const* p,bool green6){if(provider())return access([&](Provider& v){return v.colors(p,green6);},[&]{return colors(p,green6);});auto table=*static_cast<void* const* const*>(p);return reinterpret_cast<unsigned short const*(__thiscall*)(void const*)>(table[green6?7:6])(p);}
inline unsigned rgb(void* p,unsigned index){if(provider())return access([&](Provider& v){return v.rgb(p,index);},[&]{return rgb(p,index);});auto table=*static_cast<void***>(p);auto bytes=reinterpret_cast<unsigned char*(__thiscall*)(void*,unsigned)>(table[8])(p,index);if(!bytes)throw std::runtime_error("native palette RGB missing");return (unsigned(bytes[0])<<16)|(unsigned(bytes[1])<<8)|bytes[2];}
inline unsigned short const* lookup(void const* p,unsigned blocks){if(provider())return access([&](Provider& v){return v.lookup(p,blocks);},[&]{return lookup(p,blocks);});return static_cast<unsigned short const*>(p);}
inline RECT clip(void* p){return {field(p,0x44),field(p,0x48),field(p,0x4c),field(p,0x50)};}
// Call the original dependency while a recording provider is installed.
struct NativeScope {Provider* prior=provider();NativeScope(){provider()=nullptr;}~NativeScope(){provider()=prior;}};
}
