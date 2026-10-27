#pragma once
#include "c3x_renderer_api.h"
#include <mutex>
#include <unordered_map>
namespace c3x_native_images {
// Process-lifetime observation precedes scene/configuration loading. No native
// image is inspected, copied or taken over here. Missing evidence never admits.
class Lifetimes {
    // Object -> demanded. Civ III's per-pixel loops (Military and Cultural
    // advisors) ask here hundreds of thousands of times per screen, mostly
    // about escaped images; a scan of every slot per access stalled them.
    static constexpr std::size_t capacity=1024;
    std::unordered_map<void*,bool> entries;std::mutex mutex;unsigned owner=0;
public:
    bool observe(int operation,void* object,int context,unsigned thread,bool* revoked=nullptr,bool* early_escape=nullptr){
        std::lock_guard<std::mutex> lock(mutex);
        if(revoked)*revoked=false;
        if(early_escape)*early_escape=false;
        if(operation==C3X_NATIVE_VERIFY&&!object){entries.clear();owner=thread;return true;}
        if(!owner)owner=thread;
        if(!object)return false;
        auto found=entries.find(object);
        if(operation==C3X_NATIVE_DESTROY||operation==C3X_NATIVE_IMAGE_REINIT){if(found!=entries.end())entries.erase(found);return false;}
        if(operation==C3X_NATIVE_INIT){
            // Forgetting an escape frees a slot; only a new INIT can reinsert it.
            if(found!=entries.end())entries.erase(found);
            found=thread==owner&&entries.size()<capacity?entries.emplace(object,false).first:entries.end();
        }else if(found!=entries.end()){
            // These audited JGL bodies return scalar results and release their
            // private leases. Their pointers/DCs do not escape to game callers.
            bool private_lease=context==C3X_NATIVE_COPY||context==C3X_NATIVE_FILL||
                context==C3X_NATIVE_IMAGE_DRAW||context==C3X_NATIVE_SPRITE||
                (context==C3X_NATIVE_HIT_PIXEL&&operation==C3X_NATIVE_BITS)||
                context==C3X_NATIVE_TEXT||context==C3X_NATIVE_IMAGE_CLIP||context==C3X_NATIVE_IMAGE_PALETTE||context==C3X_NATIVE_IMAGE_TEXT_STATE||
                // Graphsy's final native transfer borrows only a DC, never a
                // pixel pointer. This changes evidence, not CPU/GPU barriers.
                (context==C3X_NATIVE_IMAGE_PRESENT&&operation==C3X_NATIVE_DC);
            if(thread!=owner||(!private_lease&&(operation==C3X_NATIVE_PIXEL||operation==C3X_NATIVE_BITS||operation==C3X_NATIVE_DC))){
                if(revoked)*revoked=found->second;
                if(early_escape)*early_escape=!found->second;
                entries.erase(found);found=entries.end();
            }
        }
        if(found!=entries.end()&&operation==C3X_NATIVE_MAP)found->second=true;
        return found!=entries.end()&&thread==owner;
    }
};
}
