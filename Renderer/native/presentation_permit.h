#pragma once
#include <windows.h>
#include <stdexcept>

namespace c3x_renderer {
// The DXGI signal grants one presentation. Keep that grant across a static
// no-op frame; querying an auto-reset signal again would lose it. Polling has
// zero timeout so a busy compositor cannot hold the renderer command queue.
class PresentationPermit {
    HANDLE signal=nullptr;
    bool admitted=false;
public:
    ~PresentationPermit(){reset();}
    void reset(HANDLE next=nullptr){
        if(signal)CloseHandle(signal);
        signal=next;admitted=false;
    }
    bool ready(){
        if(admitted)return true;
        if(!signal)return false;
        DWORD result=WaitForSingleObject(signal,0);
        if(result==WAIT_FAILED)throw std::runtime_error("presentation signal failed");
        admitted=result==WAIT_OBJECT_0;return admitted;
    }
    void presented(){admitted=false;}
};
}
