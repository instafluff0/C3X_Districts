#pragma once
#include <windows.h>
#include <intrin.h>
#include <array>
#include <cstdint>
#include <cstdio>

namespace c3x_native_diagnostic {
// Optimized JGL and injected x86 code do not supply an unwindable frame chain.
// These are bounded stack candidates, not an asserted backtrace. Accept only
// executable game/JGL addresses immediately following a CALL instruction and
// emit module-relative offsets; never emit arbitrary stack contents.
inline bool follows_call(unsigned char const* end) {
    if(end[-5]==0xe8)return true;
    for(unsigned length=2;length<=7;++length){
        auto p=end-length;
        if(p[0]!=0xff||(p[1]&0x38)!=0x10)continue;
        unsigned mod=p[1]>>6,rm=p[1]&7,bytes=2;
        if(mod!=3&&rm==4){if(length<3)continue;++bytes;if(mod==0&&(p[2]&7)==5)bytes+=4;}
        if(mod==1)++bytes;
        if(mod==2||(mod==0&&rm==5))bytes+=4;
        if(bytes==length)return true;
    }
    return false;
}
__declspec(noinline) inline void callsite_candidates(char* output,std::size_t capacity) {
    if(!capacity)return;
    output[0]=0;
#if defined(_M_IX86)
    auto stack=static_cast<unsigned char const*>(_AddressOfReturnAddress());
    MEMORY_BASIC_INFORMATION region={};
    if(!VirtualQuery(stack,&region,sizeof(region))||region.State!=MEM_COMMIT||(region.Protect&PAGE_GUARD))return;
    auto available=static_cast<std::size_t>(static_cast<unsigned char const*>(region.BaseAddress)+region.RegionSize-stack);
    std::array<std::uintptr_t,4096> words={};SIZE_T read=0;
    auto bytes=available<sizeof(words)?available:sizeof(words);
    if(!ReadProcessMemory(GetCurrentProcess(),stack,words.data(),bytes,&read))return;
    auto game=GetModuleHandleA(nullptr),jgl=GetModuleHandleA("jgl.dll");
    std::size_t used=0;unsigned count=0;
    // List image bits/DC vtable calls first so old runtime return addresses in
    // unused stack slots cannot crowd out the immediate native access.
    for(unsigned priority=0;priority<2;++priority)
    for(std::size_t n=0;n<read/sizeof(words[0])&&count<24&&used+48<capacity;++n){
        auto address=words[n];MEMORY_BASIC_INFORMATION code={};
        if(address<8||!VirtualQuery(reinterpret_cast<void const*>(address-7),&code,sizeof(code))||
           code.State!=MEM_COMMIT||(code.Protect&PAGE_GUARD)||
           !(code.Protect&(PAGE_EXECUTE|PAGE_EXECUTE_READ|PAGE_EXECUTE_READWRITE|PAGE_EXECUTE_WRITECOPY))||
           (code.AllocationBase!=game&&code.AllocationBase!=jgl))continue;
        unsigned char instruction[7];SIZE_T got=0;
        if(!ReadProcessMemory(GetCurrentProcess(),reinterpret_cast<void const*>(address-7),instruction,sizeof(instruction),&got)||
           got!=sizeof(instruction)||!follows_call(instruction+7))continue;
        bool lease=instruction[4]==0xff&&(instruction[5]&0xf8)==0x50&&
            (instruction[6]==0x0c||instruction[6]==0x10||instruction[6]==0x14||instruction[6]==0x1c||instruction[6]==0x20||instruction[6]==0x28);
        if(lease!=(priority==0))continue;
        int written=std::snprintf(output+used,capacity-used,"%s%s+%lx@%zu",count?",":"",
            code.AllocationBase==game?"game":"jgl",static_cast<unsigned long>(address-reinterpret_cast<std::uintptr_t>(code.AllocationBase)),n*sizeof(words[0]));
        if(written<0)break;
        used+=static_cast<std::size_t>(written);++count;
    }
#endif
}
}
