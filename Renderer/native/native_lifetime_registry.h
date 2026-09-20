#pragma once
#include "c3x_renderer_api.h"
#include <array>
#include <mutex>
namespace c3x_native_images {
// Process-lifetime observation precedes scene/configuration loading. No native
// image is inspected, copied or taken over here. Missing evidence never admits.
class Lifetimes {
    struct Entry {void* object=nullptr;bool demanded=false;};
    std::array<Entry,1024> entries={};std::mutex mutex;unsigned owner=0;
public:
    bool observe(int operation,void* object,int context,unsigned thread,bool* revoked=nullptr){
        std::lock_guard<std::mutex> lock(mutex);
        if(revoked)*revoked=false;
        if(operation==C3X_NATIVE_VERIFY&&!object){entries={};owner=thread;return true;}
        if(!owner)owner=thread;
        if(!object)return false;
        Entry* found=nullptr;for(auto& e:entries)if(e.object==object){found=&e;break;}
        if(operation==C3X_NATIVE_DESTROY||operation==C3X_NATIVE_IMAGE_REINIT){if(found)*found={};return false;}
        if(operation==C3X_NATIVE_INIT){
            if(!found)for(auto& e:entries)if(!e.object){found=&e;break;}
            // Forgetting an escape frees a slot; only a new INIT can reinsert it.
            if(found)*found={thread==owner?object:nullptr};
        }else if(found){
            // These audited JGL bodies return scalar results and release their
            // private leases. Their pointers/DCs do not escape to game callers.
            bool private_lease=context==C3X_NATIVE_COPY||context==C3X_NATIVE_FILL||
                context==C3X_NATIVE_IMAGE_DRAW||context==C3X_NATIVE_SPRITE||
                context==C3X_NATIVE_TEXT||context==C3X_NATIVE_IMAGE_CLIP||context==C3X_NATIVE_IMAGE_PALETTE||context==C3X_NATIVE_IMAGE_TEXT_STATE;
            if(thread!=owner||(!private_lease&&(operation==C3X_NATIVE_PIXEL||operation==C3X_NATIVE_BITS||operation==C3X_NATIVE_DC))){
                if(revoked)*revoked=found->demanded;
                *found={};
            }
        }
        if(found&&found->object&&operation==C3X_NATIVE_MAP)found->demanded=true;
        return found&&found->object&&thread==owner;
    }
};
}
