#pragma once
#include <array>
#include <cstdint>
#include <cstddef>
#include <stdexcept>

namespace c3x_renderer { namespace render_core {
// Producer-owned, resource-free invalidation metadata. Consumers retain exact
// dependency keys; an unrelated edit cannot invalidate their completed pixels.
// Falling behind the bounded window requires the ordinary complete proof.
class RasterDependencyRevisions {
public:
    enum class Domain : unsigned { appearance,semantic,visibility,coast,world,flow,barrier };
    struct Key {
        Domain domain=Domain::barrier;std::uint64_t id=0;
        bool operator==(Key const& other)const{return domain==other.domain&&id==other.id;}
    };
    struct Hash {std::size_t operator()(Key const& key)const{
        return std::size_t((key.id^(std::uint64_t(key.domain)*0x9e3779b97f4a7c15ull))*1099511628211ull);
    }};
    struct Checkpoint {RasterDependencyRevisions const* owner=nullptr;std::uint64_t sequence=0;};
    static constexpr std::size_t capacity=4096;
private:
    std::array<Key,capacity> changes{};
    std::uint64_t sequence=0;
public:
    RasterDependencyRevisions()=default;
    RasterDependencyRevisions(RasterDependencyRevisions const&)=delete;
    RasterDependencyRevisions& operator=(RasterDependencyRevisions const&)=delete;
    void touch(Domain domain,std::uint64_t id){
        if(sequence==UINT64_MAX)throw std::length_error("raster dependency sequence exhausted");
        changes[sequence%capacity]={domain,id};++sequence;
    }
    void invalidate(){touch(Domain::barrier,0);}
    Checkpoint checkpoint()const{return {this,sequence};}
    template<class Keys>bool unchanged(Checkpoint prior,Keys const& dependencies,std::uint64_t& visits)const{
        if(prior.owner!=this||prior.sequence>sequence||sequence-prior.sequence>capacity)return false;
        for(auto i=prior.sequence;i<sequence;++i){auto const& changed=changes[i%capacity];++visits;
            if(changed.domain==Domain::barrier||dependencies.count(changed))return false;
        }return true;
    }
};
} }
