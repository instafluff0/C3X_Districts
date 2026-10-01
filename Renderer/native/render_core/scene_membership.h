#pragma once
#include "geometry_draws.h"
#include "resident_content.h"
#include <memory>

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
    std::uint64_t serial=0;
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
    void clear(){selected.reset();writable();}
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
    std::size_t content_size()const{return selected->content.size();}
    std::size_t bytes()const{return selected->bytes();}
};
} }
