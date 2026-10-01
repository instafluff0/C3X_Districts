#pragma once
// Private, generated-source diagnostic. Production does not include this file.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d11.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <cstdint>

struct SandboxCausalState {
    unsigned mode=0,depth=0,serial=0,*pass=nullptr;
    bool ledger=false,active=false;
    unsigned scopes=0,restores=0,bad_state=0;
    std::uint64_t attempts=0,suppressed=0;
    std::map<std::string,std::uint64_t> calls;
    void api(char const* name) {
        if(!active || !ledger)return;
        auto key=std::to_string(pass?*pass:0)+"/"+std::to_string(depth?1:0)+"/"+name;
        ++calls[key];
    }
    bool issue() {
        if(ledger){++attempts;if(mode==2)++suppressed;}
        return mode!=2;
    }
};
inline SandboxCausalState sandbox_causal;

struct SandboxCausalFrame {
    long long clock;
    SandboxCausalFrame(unsigned* pass,long long ticks):clock(ticks) {
        char option[8]={};
        GetEnvironmentVariableA("C3X_SANDBOX_CAUSAL_MODE",option,sizeof(option));
        sandbox_causal.mode=unsigned(std::atoi(option));
        if(sandbox_causal.mode>2)std::abort();
        sandbox_causal.ledger=GetEnvironmentVariableA("C3X_SANDBOX_CAUSAL_LEDGER",option,sizeof(option)) && option[0]=='1';
        sandbox_causal.pass=pass;sandbox_causal.active=true;sandbox_causal.depth=0;
        sandbox_causal.scopes=sandbox_causal.restores=sandbox_causal.bad_state=0;
        sandbox_causal.attempts=sandbox_causal.suppressed=0;
        sandbox_causal.calls.clear();++sandbox_causal.serial;
    }
    ~SandboxCausalFrame() {
        auto& s=sandbox_causal;
        if(s.ledger){
            std::printf("CAUSAL_FRAME serial=%u clock=%lld mode=%u scopes=%u restores=%u bad_state=%u attempts=%llu suppressed=%llu depth=%u\n",
                s.serial,clock,s.mode,s.scopes,s.restores,s.bad_state,
                static_cast<unsigned long long>(s.attempts),static_cast<unsigned long long>(s.suppressed),s.depth);
            for(auto const& row:s.calls)std::printf("CAUSAL_API serial=%u key=%s calls=%llu\n",s.serial,row.first.c_str(),static_cast<unsigned long long>(row.second));
        }
        s.active=false;s.pass=nullptr;
    }
};

// This scope runs after ordinary viewport/scissor binding. Its rectangle is
// never supplied to selection or source_bounds. C uses the same extra RS calls
// as B and suppresses only the enclosed geometry draws.
struct SandboxCausalRaster {
    ID3D11DeviceContext* context;
    UINT count=16;
    D3D11_RECT previous[16]={};
    explicit SandboxCausalRaster(ID3D11DeviceContext* ctx):context(ctx) {
        auto& s=sandbox_causal;++s.depth;
        if(!s.mode)return;
        context->RSGetScissorRects(&count,previous);
        if(count>16)std::abort();
        if(s.ledger){
            ++s.scopes;ID3D11RasterizerState* state=nullptr;context->RSGetState(&state);
            D3D11_RASTERIZER_DESC desc={};if(state){state->GetDesc(&desc);state->Release();}
            if(!state || !desc.ScissorEnable){++s.bad_state;std::abort();}
            s.api("diagnostic_RSGetScissorRects");s.api("diagnostic_RSGetState");
        }
        D3D11_RECT empty={0,0,0,0};context->RSSetScissorRects(1,&empty);
        s.api("diagnostic_RSSetEmpty");
    }
    ~SandboxCausalRaster() {
        auto& s=sandbox_causal;
        if(s.mode){
            context->RSSetScissorRects(count,previous);s.api("diagnostic_RSRestore");
            if(s.ledger){
                D3D11_RECT actual[16]={};UINT n=16;context->RSGetScissorRects(&n,actual);
                if(n!=count || std::memcmp(actual,previous,count*sizeof(D3D11_RECT))){++s.bad_state;std::abort();}
                ++s.restores;
            }
        }
        --s.depth;
    }
};
