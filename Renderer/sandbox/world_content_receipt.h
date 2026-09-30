#pragma once

// Explicit standalone diagnostic, called after adoption and outside timings.
// One row per resident owner reconciles canonical identity with occurrence keys.
template<class State,class Layers>
int sandbox_world_content_receipt(State& state,char const* path,Layers layers,
        std::size_t retired,std::size_t peak,std::size_t leases){
    FILE* file=nullptr;if(fopen_s(&file,path,"wb") || !file)return 1;
    std::fprintf(file,"key,generation,version,signature,content,x,y,shared,world_ground,bytes,last_used");
    for(unsigned i=0;i<20;++i)std::fprintf(file,",context%u",i);
    for(unsigned i=0;i<geometry_layer_count;++i)std::fprintf(file,",layer%u",i);
    std::fprintf(file,"\n");
    std::unordered_set<ID3D11Buffer*> allocations;std::size_t gpu_bytes=0;
    for(auto const& entry:state.tile_geometry_cache){auto const& owner=entry.second;
        auto world=state.topology_cache.retained(state.topology_cache.key(owner.tile_x,owner.tile_y));
        std::fprintf(file,"%llu,%llu,%llu,%llu,%llu,%d,%d,%u,%u,%zu,%llu",
            static_cast<unsigned long long>(state.topology_cache.key(owner.tile_x,owner.tile_y)),
            static_cast<unsigned long long>(owner.binding.generation),static_cast<unsigned long long>(owner.version),
            static_cast<unsigned long long>(owner.signature),static_cast<unsigned long long>(world?state.tile_content_signature(world->appearance):0),
            owner.tile_x,owner.tile_y,unsigned(owner.shared_natural),unsigned(owner.world_ground),owner.byte_count,
            static_cast<unsigned long long>(owner.last_used));
        for(auto value:owner.compile_context)std::fprintf(file,",%llu",static_cast<unsigned long long>(value));
        for(auto const& layer:layers(owner)){
            std::size_t bytes=0;for(auto const& chunk:layer){bytes+=chunk.byte_count;
                for(auto* buffer:{chunk.buffer,chunk.indices})if(buffer && allocations.insert(buffer).second){
                    D3D11_BUFFER_DESC desc={};buffer->GetDesc(&desc);gpu_bytes+=desc.ByteWidth;
                }}
            std::fprintf(file,",%zu",bytes);
        }
        std::fprintf(file,"\n");
    }
    bool ok=!std::ferror(file);ok=std::fclose(file)==0 && ok;
    for(unsigned i=0;i<geometry_layer_count;++i){
        auto const& records=state.geometry_vertex_buffers[i];if(records.empty())continue;
        float low[3]={1e9f,1e9f,1e9f},high[3]={-1e9f,-1e9f,-1e9f};
        for(auto const& record:records)for(unsigned a=0;a<3;++a){
            low[a]=std::min(low[a],record.content().world_bounds.low[a]);
            high[a]=std::max(high[a],record.content().world_bounds.high[a]);
        }
        std::printf("WORLD_LAYER layer=%u records=%zu low=%.6f,%.6f,%.6f high=%.6f,%.6f,%.6f\n",
            i,records.size(),low[0],low[1],low[2],high[0],high[1],high[2]);
    }
    std::printf("WORLD_CONTENT epoch=%llu owners=%zu cache_bytes=%zu retired_and_selection_bytes=%zu retired_peak_bytes=%zu selected_generations=%zu registry_bytes=%zu budget_bytes=%zu unique_gpu_allocations=%zu gpu_bytes=%zu built=%u reused=%u evicted=%u upload_bytes=%zu\n",
        static_cast<unsigned long long>(state.tile_geometry_epoch),state.tile_geometry_cache.size(),state.tile_geometry_cache_bytes,
        retired,peak,leases,state.resident_content.bytes(),state.tile_geometry_runtime_budget,allocations.size(),gpu_bytes,
        state.frame_tiles_built,state.frame_tiles_reused,state.frame_tiles_evicted,state.frame_upload_bytes);
    return ok?0:2;
}
