#pragma once
#include "content_preparation.h"

namespace c3x_renderer { namespace render_core {
// One current-tile task on the existing bounded preparation machinery. The
// scope is the read lease: destroy/join it BEFORE any captured local or borrowed
// world/asset owner can change. No job or callback survives the scope, including
// early returns, cancellation, allocation failure and exceptions.
template<class Result> class ScopedPreparation {
public:
    using Compile=std::function<std::unique_ptr<Result>(std::atomic<bool> const&)>;
    using Queue=ContentPreparation<unsigned,Compile,Result>;
private:
    Queue* queue=nullptr;
    std::unique_ptr<Result> synchronous;
public:
    ScopedPreparation(Queue& owner,Compile compile,bool concurrent) {
        if(!concurrent){std::atomic<bool> stop{false};synchronous=compile(stop);return;}
        queue=&owner;
        try {
            owner.configure({{1,std::move(compile),true}},
                [](Compile const& job,std::atomic<bool> const& stop,unsigned){return job(stop);},1,{1});
            owner.resume();
        }catch(...){owner.clear();throw;}
    }
    ScopedPreparation(ScopedPreparation const&)=delete;
    ~ScopedPreparation(){if(queue)queue->clear();}
    std::unique_ptr<Result> take(){
        if(!queue)return std::move(synchronous);
        auto result=queue->take(1);
        queue->clear();queue=nullptr;
        return result;
    }
};
}}
