#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>

namespace c3x_renderer { namespace profile_v2 {
// The pickup shaders are flattened offline: their complete dependency closure
// is in this file. One bounded atomic cache file per entry avoids shader
// compilation on every game launch or device reset. Corruption/drift recompiles.
inline HRESULT compile_cached(wchar_t const* path,char const* entry,char const* target,
                              ID3DBlob** output,ID3DBlob** errors) {
    auto hash=[](void const* input,std::size_t count,std::uint64_t value=1469598103934665603ull){
        auto bytes=static_cast<unsigned char const*>(input);
        for(std::size_t i=0;i<count;++i)value=(value^bytes[i])*1099511628211ull;
        return value;
    };
    FILE* file=nullptr;std::vector<char> source;
    if(_wfopen_s(&file,path,L"rb")==0 && file){
        std::fseek(file,0,SEEK_END);long size=std::ftell(file);std::rewind(file);
        if(size>0 && size<4*1024*1024){source.resize(size);if(std::fread(source.data(),size,1,file)!=1)source.clear();}
        std::fclose(file);
    }
    if(source.empty())return E_FAIL;
    auto signature=hash(source.data(),source.size());signature=hash(entry,std::strlen(entry),signature);
    signature=hash(target,std::strlen(target),signature);
    unsigned version=D3D_COMPILER_VERSION;signature=hash(&version,sizeof(version),signature);
    std::wstring cache=path;cache+=L".";for(auto p=entry;*p;++p)cache+=wchar_t(*p);cache+=L".cso";
    struct Header {char magic[8];std::uint64_t source,bytes,size;};
    Header header{};
    file=nullptr;
    if(_wfopen_s(&file,cache.c_str(),L"rb")==0 && file){
        if(std::fread(&header,sizeof(header),1,file)==1 && std::memcmp(header.magic,"C3XCSO1",8)==0 &&
           header.source==signature && header.size>0 && header.size<=4*1024*1024 &&
           SUCCEEDED(D3DCreateBlob(static_cast<SIZE_T>(header.size),output))){
            bool valid=std::fread((*output)->GetBufferPointer(),static_cast<std::size_t>(header.size),1,file)==1 &&
                hash((*output)->GetBufferPointer(),static_cast<std::size_t>(header.size))==header.bytes;
            std::fclose(file);file=nullptr;
            if(valid)return S_OK;
            (*output)->Release();*output=nullptr;
        }
        if(file)std::fclose(file);
    }
    HRESULT result=D3DCompile(source.data(),source.size(),nullptr,nullptr,nullptr,
        entry,target,D3DCOMPILE_OPTIMIZATION_LEVEL3,0,output,errors);
    if(FAILED(result))return result;
    std::memcpy(header.magic,"C3XCSO1",8);header.source=signature;header.size=(*output)->GetBufferSize();
    header.bytes=hash((*output)->GetBufferPointer(),static_cast<std::size_t>(header.size));
    std::wstring temp=cache+L"."+std::to_wstring(GetCurrentProcessId())+L".tmp";
    file=nullptr;
    if(_wfopen_s(&file,temp.c_str(),L"wb")==0 && file){
        bool saved=std::fwrite(&header,sizeof(header),1,file)==1 &&
            std::fwrite((*output)->GetBufferPointer(),static_cast<std::size_t>(header.size),1,file)==1;
        saved=std::fclose(file)==0 && saved;
        if(saved)MoveFileExW(temp.c_str(),cache.c_str(),MOVEFILE_REPLACE_EXISTING);
        DeleteFileW(temp.c_str());
    }
    return result;
}
} }
