#pragma once
#include <tuple>
#include <utility>

namespace c3x_renderer { namespace render_core {
// One completed view, forked by value with shared immutable mesh membership.
// Only the immediate-context owner borrows it, between preparation chunks.
// Swapping back on every exit protects the suspended compiler's mutable view.
template<class... Values> class RetainedSceneView {
    std::tuple<Values...> values;
    template<std::size_t... I>void exchange(std::tuple<Values&...> fields,std::index_sequence<I...>){
        using std::swap;(swap(std::get<I>(fields),std::get<I>(values)),...);
    }
    void exchange(std::tuple<Values&...> fields){exchange(fields,std::index_sequence_for<Values...>{});}
public:
    using References=std::tuple<Values&...>;
    explicit RetainedSceneView(References fields):values(fields){}
    class Borrow {
        RetainedSceneView* owner=nullptr;
        References fields;
    public:
        Borrow(RetainedSceneView* view,References input):owner(view),fields(input){
            if(owner)owner->exchange(fields);
        }
        Borrow(Borrow const&)=delete;
        Borrow& operator=(Borrow const&)=delete;
        ~Borrow(){if(owner)owner->exchange(fields);}
    };
    Borrow borrow(References fields){return Borrow(this,fields);}
};
template<class... Values>auto retain_scene_view(std::tuple<Values&...> fields){
    return RetainedSceneView<Values...>(fields);
}
} }
