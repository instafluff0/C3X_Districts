#ifndef C3X_UNIT_FRAME_PREPARATION_H
#define C3X_UNIT_FRAME_PREPARATION_H

#include <algorithm>
#include <cstring>
#include <deque>
#include "../c3x_renderer_api.h"

namespace c3x_renderer { namespace render_core {
// Caller observations authorize a finite next pose. No autonomous animation clock,
// native pointer, canvas, or prediction of the next action enters this queue.
class UnitFramePreparation {
    struct Seen {c3x_renderer_unit_v1 request;};
    std::deque<Seen> seen;
    std::deque<c3x_renderer_unit_v1> pending;
    static bool same(c3x_renderer_unit_v1 a,c3x_renderer_unit_v1 b) {
        a.body_x=b.body_x=a.body_y=b.body_y=0;
        a.presentation_time_ticks=b.presentation_time_ticks=0;
        a.presentation_frequency=b.presentation_frequency=0;
        return !std::memcmp(&a,&b,sizeof(a));
    }
public:
    unsigned long long offered=0,replaced=0;
    void observe(c3x_renderer_unit_v1 request,bool loop,bool advancing=true,unsigned step=1) {
        if(!advancing) {
            pending.erase(std::remove_if(pending.begin(),pending.end(),[&](auto const& x){return x.unit_id==request.unit_id;}),pending.end());
            seen.erase(std::remove_if(seen.begin(),seen.end(),[&](auto const& x){return x.request.unit_id==request.unit_id;}),seen.end());
            return;
        }
        if(request.frame_count<=1 || request.action_cursor<0 || request.unit_key[63])return;
        request.action_cursor=loop?request.action_cursor%request.frame_count:
            std::min(request.action_cursor,request.frame_count-1);
        auto old=std::find_if(seen.begin(),seen.end(),[&](auto const& x){return x.request.unit_id==request.unit_id;});
        if(old!=seen.end() && same(old->request,request))return;
        if(old!=seen.end())seen.erase(old);
        if(seen.size()==128)seen.pop_front();
        seen.push_back({request});
        for(auto i=pending.begin();i!=pending.end();) {
            if(i->unit_id==request.unit_id){i=pending.erase(i);++replaced;}else ++i;
        }
        if(!loop && request.action_cursor==request.frame_count-1)return;
        request.action_cursor=loop?(request.action_cursor+int(step%unsigned(request.frame_count)))%request.frame_count:
            std::min(request.action_cursor+int(std::min(step,65536u)),request.frame_count-1);
        if(pending.size()==32){pending.pop_front();++replaced;}
        pending.push_back(request);++offered;
    }
    unsigned take(c3x_renderer_unit_v1* out,unsigned limit) {
        unsigned count=0;
        while(count<limit && !pending.empty()){out[count++]=pending.front();pending.pop_front();}
        return count;
    }
    // Keep unready predictions until content completion or a newer observation;
    // one slow CPU pose must not prevent adoption of other ready unit poses.
    template<class Ready> unsigned take_ready(c3x_renderer_unit_v1* out,unsigned limit,Ready ready) {
        unsigned count=0;
        for(auto it=pending.begin();it!=pending.end() && count<limit;){
            if(ready(*it)){out[count++]=*it;it=pending.erase(it);}else ++it;
        }
        return count;
    }
    bool empty() const {return pending.empty();}
    void clear(){pending.clear();seen.clear();}
    void forget(int id){
        pending.erase(std::remove_if(pending.begin(),pending.end(),[&](auto const& r){return r.unit_id==id;}),pending.end());
        seen.erase(std::remove_if(seen.begin(),seen.end(),[&](auto const& r){return r.request.unit_id==id;}),seen.end());
    }
};
}}
#endif
