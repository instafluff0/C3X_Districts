#pragma once
#include <windows.h>
#include <bcrypt.h>
#include <array>
#include <filesystem>
#include <fstream>
#include "codec.h"
#pragma comment(lib,"bcrypt.lib")
namespace c3x_inputs {
struct Sha256 {
    BCRYPT_ALG_HANDLE algorithm=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;
    Sha256(){require(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"asset SHA256 provider failed");
        if(BCryptCreateHash(algorithm,&hash,nullptr,0,nullptr,0,0)<0){BCryptCloseAlgorithmProvider(algorithm,0);algorithm=nullptr;throw std::runtime_error("asset SHA256 creation failed");}}
    ~Sha256(){if(hash)BCryptDestroyHash(hash);if(algorithm)BCryptCloseAlgorithmProvider(algorithm,0);}
    Sha256(Sha256 const&)=delete;Sha256& operator=(Sha256 const&)=delete;
    void add(void const* bytes,std::size_t size){auto data=static_cast<unsigned char const*>(bytes);while(size){auto count=ULONG(std::min(size,std::size_t(1u<<20)));
        require(BCryptHashData(hash,const_cast<unsigned char*>(data),count,0)>=0,"asset SHA256 update failed");data+=count;size-=count;}}
    std::array<unsigned char,32> finish(){std::array<unsigned char,32> result{};require(BCryptFinishHash(hash,result.data(),ULONG(result.size()),0)>=0,"asset SHA256 completion failed");return result;}
};
struct Asset {bool exists=false;std::uint64_t size=0;std::array<unsigned char,32> hash{};
    bool operator==(Asset const& other)const{return exists==other.exists&&size==other.size&&hash==other.hash;}};
inline std::string asset_path(wchar_t const* path){
    wchar_t full[32768];auto n=GetFullPathNameW(path,32768,full,nullptr);require(n&&n<32768,"asset path cannot resolve");
    // pushd creates temporary drive letters for UNC checkouts. Resolve an
    // existing ancestor to its stable filesystem name, including for missing
    // optional assets. A later replay must not depend on that command session.
    std::filesystem::path ancestor(full),suffix;
    std::wstring stable;
    for(;;){
        auto handle=CreateFileW(ancestor.c_str(),0,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
            nullptr,OPEN_EXISTING,FILE_FLAG_BACKUP_SEMANTICS,nullptr);
        if(handle!=INVALID_HANDLE_VALUE){
            wchar_t resolved[32768];auto count=GetFinalPathNameByHandleW(handle,resolved,32768,FILE_NAME_NORMALIZED|VOLUME_NAME_DOS);CloseHandle(handle);
            require(count&&count<32768,"asset filesystem name unavailable");stable.assign(resolved,count);
            if(stable.rfind(L"\\\\?\\UNC\\",0)==0)stable=L"\\\\"+stable.substr(8);
            else if(stable.rfind(L"\\\\?\\",0)==0)stable.erase(0,4);
            if(!suffix.empty())stable=(std::filesystem::path(stable)/suffix).wstring();break;
        }
        auto parent=ancestor.parent_path();require(parent!=ancestor&&!parent.empty(),"asset filesystem root unavailable");
        suffix=suffix.empty()?ancestor.filename():ancestor.filename()/suffix;ancestor=parent;
    }
    auto size=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,stable.data(),int(stable.size()),nullptr,0,nullptr,nullptr);require(size>0&&size<32768,"asset path encoding failed");
    std::string result(std::size_t(size),'\0');require(WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,stable.data(),int(stable.size()),result.data(),size,nullptr,nullptr)==size,"asset path conversion failed");return result;
}
inline std::string asset_path(char const* path){
    auto size=MultiByteToWideChar(CP_ACP,MB_ERR_INVALID_CHARS,path,-1,nullptr,0);require(size>0&&size<32768,"asset ANSI path invalid");
    std::wstring wide(std::size_t(size),L'\0');require(MultiByteToWideChar(CP_ACP,MB_ERR_INVALID_CHARS,path,-1,wide.data(),size)==size,"asset ANSI path conversion failed");return asset_path(wide.c_str());
}
inline Asset asset_file(std::string const& path){
    Asset result;auto file=std::filesystem::u8path(path);if(!std::filesystem::exists(file))return result;
    require(std::filesystem::is_regular_file(file),"asset path is not a file");result.exists=true;result.size=std::filesystem::file_size(file);
    require(result.size<=512u*1024u*1024u,"asset fingerprint size limit");std::ifstream stream(file,std::ios::binary);require(bool(stream),"asset fingerprint open failed");
    Sha256 hash;std::array<char,65536> buffer{};std::uint64_t count=0;
    while(stream){stream.read(buffer.data(),std::streamsize(buffer.size()));auto n=std::size_t(stream.gcount());if(n){hash.add(buffer.data(),n);count+=n;}}
    require(stream.eof()&&count==result.size,"asset fingerprint read failed");result.hash=hash.finish();return result;
}
inline void asset_fields(Writer& out,std::string const& path,Asset const& value){
    out.string(path.c_str(),32768);out.u32(value.exists?1:0);out.u64(value.size);for(auto byte:value.hash)out.u32(byte);
}
inline Asset asset_fields(Reader& in,std::string& path){path=in.string(32768);Asset value;auto exists=in.u32();require(exists<=1,"invalid asset presence");value.exists=exists!=0;value.size=in.u64();
    for(auto& byte:value.hash){auto n=in.u32();require(n<=255,"invalid asset digest");byte=static_cast<unsigned char>(n);}return value;
}
}
