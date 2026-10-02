#pragma once
#include "geometry_draws.h"
#include "resident_content.h"
#include <memory>
#include <algorithm>

namespace c3x_renderer { namespace render_core {
// Replaces the mutable selected-record array and its separately copied content
// lease. A publication freezes both together. Consumers borrow one generation;
// resetting selection leaves old camera/native consumers valid and charged.
template<class Chunk,std::size_t Layers> class SceneMembership {
public:
    using Records=typename GeometryDrawView<Chunk,Layers>::Records;
    struct Generation {
        Records records;
        ResidentSelection content;
        ResidentRetirementToken charge;
        std::uint64_t revision=0;
        explicit Generation(std::shared_ptr<ResidentRetirement> const& ledger):content(ledger){}
        std::size_t bytes()const{
            std::size_t result=sizeof(Generation);
            for(auto const& layer:records)result+=layer.capacity()*sizeof(typename Records::value_type::value_type);
            return result;
        }
    };
    using Lease=std::shared_ptr<Generation const>;
private:
    std::shared_ptr<ResidentRetirement> ledger;
    std::shared_ptr<Generation> selected;
    std::uint64_t serial=0,order_serial=0;
    void writable(){
        if(!selected){selected=std::make_shared<Generation>(ledger);selected->revision=++serial;}
        else if(selected.use_count()>1){
            auto next=std::make_shared<Generation>(ledger);
            next->records=selected->records;next->content=selected->content;
            next->revision=++serial;selected=std::move(next);
        }
    }
public:
    explicit SceneMembership(std::shared_ptr<ResidentRetirement> budget={}):ledger(std::move(budget)){writable();}
    // A full assembly may rebuild the same contributor set in a new native
    // order without calling the boundary sort. Retire old pixel order too.
    void clear(){selected.reset();writable();++order_serial;}
    // An entering/leaving strip mutates one current generation. Saved pass and
    // native readers keep the prior records and their exact resource leases.
    template<class Keep>void retain_occurrences(Keep keep){
        auto revision=selected->revision;writable();bool changed=false;
        for(auto& layer:selected->records){auto size=layer.size();
            layer.erase(std::remove_if(layer.begin(),layer.end(),[&](auto const& draw){return !keep(draw);}),layer.end());
            changed=changed || layer.size()!=size;
        }
        if(changed && selected->revision==revision)selected->revision=++serial;
        ResidentSelection retained(ledger);
        for(auto const& layer:selected->records)for(auto const& draw:layer){
            if(!retained.retain(draw.owner,selected->content.get(draw.owner)))throw std::bad_alloc();
        }
        selected->content=std::move(retained);
    }
    template<class Order>bool order_occurrences(Order order){
        auto before=[&](auto const& a,auto const& b){return order(a)<order(b);};
        bool changed=false;
        for(auto const& layer:selected->records)
            if(!std::is_sorted(layer.begin(),layer.end(),before)){changed=true;break;}
        if(!changed)return false;
        auto revision=selected->revision;writable();
        for(auto& layer:selected->records)
            std::stable_sort(layer.begin(),layer.end(),before);
        if(selected->revision==revision)selected->revision=++serial;
        ++order_serial;return true;
    }
    auto& edit(std::size_t layer){writable();return selected->records[layer];}
    auto const& operator[](std::size_t layer)const{return selected->records[layer];}
    auto begin()const{return selected->records.begin();}
    auto end()const{return selected->records.end();}
    operator Records const&()const{return selected->records;}
    Records const& records()const{return selected->records;}
    bool retain(ContentHandle h,std::shared_ptr<void> owner){writable();return selected->content.retain(h,std::move(owner));}
    Lease publish(){
        if(ledger){selected->charge.retire(ledger,selected->bytes());selected->charge.resize(selected->bytes());}
        return selected;
    }
    std::uint64_t revision()const{return selected->revision;}
    std::uint64_t order_revision()const{return order_serial;}
    std::size_t content_size()const{return selected->content.size();}
    std::size_t bytes()const{return selected->bytes();}
};
} }
