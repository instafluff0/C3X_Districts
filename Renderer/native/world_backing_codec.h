#pragma once
#include "world_preparation.h"
#include <stdexcept>
#include <type_traits>

namespace c3x_renderer {
// Session-local backing contains values and exact dependency proofs, never
// native pointers, COM resources, borrowed assets or cached validation receipts.
// The format is private to this DLL lifetime; it is not a distributable pack.
class WorldBackingCodec {
    static constexpr std::size_t limit=16u*1024u*1024u;
    struct Writer {
        std::vector<unsigned char> bytes;
        template<class T> void pod(T const& value){
            static_assert(std::is_trivially_copyable<T>::value,"wire value");
            append(&value,sizeof(value));
        }
        void append(void const* source,std::size_t size){
            if(size>limit-bytes.size())throw std::length_error("world backing size");
            if(size){auto p=static_cast<unsigned char const*>(source);bytes.insert(bytes.end(),p,p+size);}
        }
        template<class T> void vector(std::vector<T> const& values){
            static_assert(std::is_trivially_copyable<T>::value,"wire values");
            pod(unsigned(values.size()));append(values.data(),values.size()*sizeof(T));
        }
        template<class Map> void map(Map const& values){pod(unsigned(values.size()));for(auto const& p:values){pod(p.first);pod(p.second);}}
    };
    struct Reader {
        unsigned char const* current;std::size_t left;
        template<class T> void pod(T& value){
            static_assert(std::is_trivially_copyable<T>::value,"wire value");
            if(sizeof(value)>left)throw std::length_error("world backing truncated");
            std::memcpy(&value,current,sizeof(value));current+=sizeof(value);left-=sizeof(value);
        }
        unsigned count(){unsigned n=0;pod(n);if(n>65536u)throw std::length_error("world backing count");return n;}
        template<class T> void vector(std::vector<T>& values){
            static_assert(std::is_trivially_copyable<T>::value,"wire values");
            unsigned n=0;pod(n);if(n>left/sizeof(T))throw std::length_error("world backing vector");
            values.resize(n);auto bytes=std::size_t(n)*sizeof(T);
            if(bytes)std::memcpy(values.data(),current,bytes);current+=bytes;left-=bytes;
        }
        template<class Map> void map(Map& values){auto n=count();for(unsigned i=0;i<n;++i){
            typename Map::key_type key{};typename Map::mapped_type value{};pod(key);pod(value);
            if(!values.emplace(key,value).second)throw std::length_error("world backing duplicate");}}
    };
    static void write_mesh(Writer& w,render_core::PreparedMesh const& m){
        w.pod(m.vertex_stride);w.pod(m.index_stride);w.pod(m.index_count);w.pod(m.shared_grid);
        w.pod(m.bounds);w.pod(m.world_low);w.pod(m.world_high);w.pod(m.projected_bounds);
        w.vector(m.vertices);w.vector(m.indices);
    }
    static void read_mesh(Reader& r,render_core::PreparedMesh& m){
        r.pod(m.vertex_stride);r.pod(m.index_stride);r.pod(m.index_count);r.pod(m.shared_grid);
        r.pod(m.bounds);r.pod(m.world_low);r.pod(m.world_high);r.pod(m.projected_bounds);
        r.vector(m.vertices);r.vector(m.indices);
        if(!m.vertices.empty() && (!m.vertex_stride || m.vertices.size()%m.vertex_stride ||
            (m.index_stride!=2 && m.index_stride!=4) || m.index_count!=m.indices.size()/m.index_stride))
            throw std::length_error("world backing mesh");
    }
    template<class Proof> static void write_rivers(Writer& w,Proof const& proof){
        w.pod(unsigned(proof.size()));
        for(auto const& p:proof){w.pod(p.first);w.vector(p.second->values);
            auto const& inputs=*p.second->inputs;w.pod(unsigned(inputs.values.size()));
            for(auto const& input:inputs.values){w.pod(input.first);w.pod(input.second);}w.vector(inputs.flow);}
    }
    static fidelity::NaturalWorld::CellProof read_rivers(Reader& r){
        fidelity::NaturalWorld::CellProof proof;auto n=r.count();
        for(unsigned i=0;i<n;++i){fidelity::NaturalWorld::CellKey key{};r.pod(key);
            auto cell=std::make_shared<fidelity::NaturalWorld::CellContent>();r.vector(cell->values);
            cell->inputs=std::make_shared<fidelity::NaturalWorld::PageInputs>();auto count=r.count();
            for(unsigned j=0;j<count;++j){std::size_t index=0;std::uint32_t value=0;r.pod(index);r.pod(value);cell->inputs->values.emplace_back(index,value);}
            r.vector(cell->inputs->flow);proof.emplace_back(key,std::move(cell));
        }return proof;
    }
    static void write_part(Writer& w,objects::PreparedPart const& p){
        write_mesh(w,p.mesh);w.pod(p.material);w.pod(p.environment);w.pod(p.terrain_conforming);w.pod(p.atlas);
    }
    static void read_part(Reader& r,objects::PreparedPart& p){
        read_mesh(r,p.mesh);r.pod(p.material);r.pod(p.environment);r.pod(p.terrain_conforming);r.pod(p.atlas);
    }
public:
    static std::vector<unsigned char> encode(PreparedWorld const& source){
        if(!source.ground || !source.terrain || !source.objects ||
           !source.ground->pending_grids.empty() || !source.ground->legacy_shadow.empty())return {};
        Writer w;w.pod(std::uint32_t(2));
        auto const& g=*source.ground;auto const& t=*source.terrain;auto const& o=*source.objects;
        for(auto const& m:g.meshes)write_mesh(w,m);
        w.pod(g.water_coverage);w.map(g.world);w.map(g.coast);w.map(g.topology);write_rivers(w,g.rivers);
        for(auto const& m:t.meshes)write_mesh(w,m);
        w.map(t.world);w.map(t.coast);write_rivers(w,t.rivers);
        for(auto const& p:o.layers)write_part(w,p);
        w.vector(o.rigid);
        w.pod(unsigned(o.city.size()));for(auto const& p:o.city)write_part(w,p);
        bool lighting=!o.city.empty() && bool(o.city.front().lighting);w.pod(lighting);
        if(lighting){w.vector(o.city.front().lighting->lights);w.vector(o.city.front().lighting->blockers);}
        w.pod(o.composition);w.pod(o.instances);w.pod(o.routes);
        w.map(o.world);w.map(o.coast);w.map(o.topology);write_rivers(w,o.rivers);
        return std::move(w.bytes);
    }
    static std::unique_ptr<PreparedWorld> decode(std::vector<unsigned char> const& bytes){
        if(bytes.empty() || bytes.size()>limit)return {};
        try {
            Reader r{bytes.data(),bytes.size()};unsigned version=0;r.pod(version);if(version!=2)return {};
            auto result=std::make_unique<PreparedWorld>();
            result->ground=std::make_unique<fidelity::PreparedGround>();
            result->terrain=std::make_unique<fidelity::TerrainSurfaces>();
            result->objects=std::make_unique<objects::PreparedObjects>();
            auto& g=*result->ground;auto& t=*result->terrain;auto& o=*result->objects;
            for(auto& m:g.meshes)read_mesh(r,m);
            r.pod(g.water_coverage);r.map(g.world);r.map(g.coast);r.map(g.topology);
            auto ground_rivers=read_rivers(r);g.rivers.insert(ground_rivers.begin(),ground_rivers.end());
            for(auto& m:t.meshes)read_mesh(r,m);
            r.map(t.world);r.map(t.coast);t.rivers=read_rivers(r);
            for(auto& p:o.layers)read_part(r,p);
            r.vector(o.rigid);
            for(auto const& instance:o.rigid)if(instance.family>=objects::family_count || instance.layer>=objects::layer_count)return {};
            o.city.resize(r.count());for(auto& p:o.city)read_part(r,p);
            bool lighting=false;r.pod(lighting);
            if(lighting){auto light=std::make_shared<city_fidelity::Lighting>();r.vector(light->lights);r.vector(light->blockers);
                for(auto& p:o.city)p.lighting=light;}
            else if(!o.city.empty())return {};
            r.pod(o.composition);r.pod(o.instances);r.pod(o.routes);
            r.map(o.world);r.map(o.coast);r.map(o.topology);o.rivers=read_rivers(r);
            fidelity::NaturalWorld accounting;
            t.proof_bytes=accounting.proof_bytes(t.rivers);o.proof_bytes=accounting.proof_bytes(o.rivers);
            if(r.left || result->bytes()>32u*1024u*1024u)return {};
            return result;
        }catch(...){return {};}
    }
};
}
