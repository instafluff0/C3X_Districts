#pragma once
namespace c3x_renderer { namespace render_core {
// Opt-in accounting only. A moving view never emits; one completed stable
// scene emits once per owner lifetime. No frame timing or GPU synchronization.
template<class Key> struct SubmissionCensus {
    Key previous{};
    unsigned matches=0;
    bool reported=false;
    bool observe(bool enabled,Key const& key){
        if(reported)return false;
        if(!enabled){matches=0;return false;}
        matches=matches && previous==key?matches+1:1;
        previous=key;
        if(matches<3)return false;
        reported=true;return true;
    }
};
}}
