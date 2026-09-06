#pragma once
#include "../contracts/packet_v1.h"
#include <map>
#include <sstream>
#include <iomanip>
namespace labv2 {
inline std::string json_string(std::string value) {
 std::string out="\"";
 for(unsigned char c:value) {if(c=='"'||c=='\\'){out+='\\';out+=c;}else if(c<32)throw std::runtime_error("control character in source identity");else out+=c;}
 return out+'"';
}
inline std::string portable_source(std::string path) {
 for(auto& c:path)if(c=='\\')c='/';
 auto root=path.find("Renderer/packs/");
 if(root!=std::string::npos)return path.substr(root);
 return path.substr(path.find_last_of('/')+1);
}
inline std::map<std::string,std::string> source_meshes,source_textures;
inline std::vector<std::string> source_instances;
inline void write_source_metadata(const std::string& path,const Packet& packet) {
 std::ostringstream out;out<<"{\"schema\":\"c3x.lab_v2.source_metadata.v1\",\"provenance\":\"normalized source meshes; legacy projection calibration explicitly retained\",\"tangent_stream\":\"absent in C3XVEG1; never synthesized as source evidence\",\"meshes\":[";
 bool first=true;for(auto& p:source_meshes){if(!first)out<<',';first=false;out<<p.second;}
 out<<"],\"textures\":[";first=true;for(auto& p:source_textures){if(!first)out<<',';first=false;out<<p.second;}
 out<<"],\"instances\":[";first=true;for(auto& p:source_instances){if(!first)out<<',';first=false;out<<p;}
 out<<"],\"draw_texture_bindings\":[";first=true;
 for(auto& draw:packet.draws){if(!first)out<<',';first=false;out<<"{\"feature\":"<<draw.feature<<",\"slots\":[";for(unsigned i=0;i<128;i++){if(i)out<<',';out<<draw.textures[i];}out<<"]}";}
 out<<"],\"binding_scope\":\"source packet before owner postprocessing; shader declarations name channel semantics\"}\n";auto bytes=out.str();FILE*f=open_path(path,"wb");if(!f)throw std::runtime_error("source metadata write failed");bool ok=fwrite(bytes.data(),1,bytes.size(),f)==bytes.size();fclose(f);if(!ok)throw std::runtime_error("source metadata write truncated");
}
}
