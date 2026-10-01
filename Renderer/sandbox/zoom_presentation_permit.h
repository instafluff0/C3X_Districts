#pragma once
#include <windows.h>
#include <stdexcept>

namespace c3x_renderer::sandbox {

enum class ZoomPresentationWait { granted, message, timeout, failed, unsupported };

// Own the DXGI frame-latency handle and retain its auto-reset grant across
// no-op frames. Call wait before sampling input or rendering, outside all
// renderer/input locks. On message wake, the caller drains its message queue
// and retries; this helper neither pumps messages nor sleeps/polls for them.
class ZoomPresentationPermit {
    HANDLE signal = nullptr;
    bool admitted = false;
    DWORD error = ERROR_SUCCESS;

public:
    ZoomPresentationPermit() = default;
    ZoomPresentationPermit(const ZoomPresentationPermit&) = delete;
    ZoomPresentationPermit& operator=(const ZoomPresentationPermit&) = delete;
    ~ZoomPresentationPermit() { reset(); }

    void reset(HANDLE next = nullptr) {
        if (signal && signal != next) CloseHandle(signal);
        signal = next;
        admitted = false;
        error = ERROR_SUCCESS;
    }

    bool valid() const { return signal != nullptr; }
    DWORD last_error() const { return error; }

    // Preserve PresentationPermit's nonblocking query and failure exception.
    bool ready() {
        error = ERROR_SUCCESS;
        if (admitted) return true;
        if (!signal) return false;
        const DWORD result = WaitForSingleObject(signal, 0);
        if (result == WAIT_OBJECT_0) return admitted = true;
        if (result == WAIT_TIMEOUT) return false;
        error = result == WAIT_FAILED ? GetLastError() : ERROR_INVALID_DATA;
        throw std::runtime_error("zoom presentation signal failed");
    }

    ZoomPresentationWait wait(DWORD timeout) {
        error = ERROR_SUCCESS;
        if (admitted) return ZoomPresentationWait::granted;
        if (!signal) {
            error = ERROR_NOT_SUPPORTED;
            return ZoomPresentationWait::unsupported;
        }
        const DWORD result = MsgWaitForMultipleObjectsEx(
            1, &signal, timeout, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
        if (result == WAIT_OBJECT_0) {
            // The wait consumed the auto-reset signal. Latch it directly;
            // querying the signal again here would lose the granted frame.
            admitted = true;
            return ZoomPresentationWait::granted;
        }
        if (result == WAIT_OBJECT_0 + 1) return ZoomPresentationWait::message;
        if (result == WAIT_TIMEOUT) return ZoomPresentationWait::timeout;
        error = result == WAIT_FAILED ? GetLastError() : ERROR_INVALID_DATA;
        return ZoomPresentationWait::failed;
    }

    // Only an actual S_OK Present consumes a grant. Failure and positive
    // status results (including occlusion) leave the admitted frame intact.
    bool presented(HRESULT result = S_OK) {
        if (result != S_OK || !admitted) return false;
        admitted = false;
        return true;
    }
};

}
