// Append generic aligned data after an existing shared frame-buffer prefix.
#include "../contracts/packet_v1.h"

int main(int argc,char**argv) {
 try {
  if(argc!=5)throw std::runtime_error("usage: append_frame_data input data output prefix_bytes");
  auto p=labv2::read_packet(argv[1]);unsigned prefix=std::stoul(argv[4]);
  FILE* f=labv2::open_path(argv[2],"rb");if(!f)throw std::runtime_error("missing frame payload");
  fseek(f,0,SEEK_END);long size=ftell(f);rewind(f);
  if(size<=0 || size%16 || !prefix || prefix%16 || size+prefix>65536){fclose(f);throw std::runtime_error("invalid frame payload size");}
  std::vector<uint8_t> data(size);bool ok=fread(data.data(),1,data.size(),f)==data.size();fclose(f);
  if(!ok)throw std::runtime_error("truncated frame payload");
  std::vector<uint32_t> mapped(p.buffers.size(),UINT32_MAX);unsigned added=0;
  for(auto& d:p.draws){
   if(d.frame_buffer>=mapped.size() || p.buffers[d.frame_buffer].size()!=prefix)
    throw std::runtime_error("unexpected shared frame prefix");
   auto& id=mapped[d.frame_buffer];
   if(id==UINT32_MAX){
    auto extended=p.buffers[d.frame_buffer];extended.insert(extended.end(),data.begin(),data.end());
    id=uint32_t(p.buffers.size());p.buffers.push_back(std::move(extended));added++;
   }
   d.frame_buffer=id;
  }
  if(!labv2::write_packet(argv[3],p))throw std::runtime_error("frame packet write failed");
  std::printf("{\"added_frame_buffers\":%u,\"prefix_bytes\":%u,\"payload_bytes\":%ld}\n",added,prefix,size);
  return 0;
 }catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
