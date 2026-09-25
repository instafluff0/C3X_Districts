#pragma once
#include <windows.h>
#include <cstdint>

// Shared only by the sandbox's two executables. Plain 32-bit slots and Win32
// interlocked operations have the same layout in x86 and x64 processes.
namespace sandbox_exchange {
constexpr wchar_t mapping_name[] = L"Local\\C3XRendererSandboxSceneV1";
constexpr LONG ready_magic = 0x43535831;
constexpr unsigned action_capacity = 32;
struct Snapshot {
    LONG generation;
    LONG viewer;
    LONG scene;
    LONG unit_incarnation;
    LONG unit_x;
    LONG unit_y;
    LONG unit_visible;
    LONG combat_serial;
};
struct Action {
    LONG serial;
    LONG seen_generation;
    LONG viewer;
    LONG unit_incarnation;
    LONG target_x;
    LONG target_y;
};
struct Exchange {
    volatile LONG ready;
    volatile LONG host_heartbeat;
    volatile LONG client_heartbeat;
    volatile LONG publication_sequence;
    Snapshot snapshot;
    volatile LONG camera_sequence;
    LONG camera_x, camera_y;
    volatile LONG action_write;
    volatile LONG action_read;
    volatile LONG action_ack;
    volatile LONG action_result;
    volatile LONG reconcile_required;
    volatile LONG host_publication_p95_us;
    volatile LONG host_publication_worst_us;
    Action actions[action_capacity];
};
inline LONG acquire(volatile LONG* value) {
    return InterlockedCompareExchange(value, 0, 0);
}
inline void publish(Exchange* exchange, Snapshot const& snapshot) {
    InterlockedIncrement(&exchange->publication_sequence);
    exchange->snapshot = snapshot;
    MemoryBarrier();
    InterlockedIncrement(&exchange->publication_sequence);
}
inline bool observe(Exchange* exchange, Snapshot& snapshot) {
    for (int retry = 0; retry < 4; ++retry) {
        LONG before = acquire(&exchange->publication_sequence);
        if (before & 1) continue;
        snapshot = exchange->snapshot;
        MemoryBarrier();
        if (before == acquire(&exchange->publication_sequence)) return true;
    }
    return false;
}
inline bool send_action(Exchange* exchange, Action const& action) {
    LONG write = acquire(&exchange->action_write);
    if (write - acquire(&exchange->action_read) >= LONG(action_capacity)) {
        InterlockedExchange(&exchange->reconcile_required, 1);
        return false;
    }
    exchange->actions[unsigned(write) % action_capacity] = action;
    MemoryBarrier();
    InterlockedExchange(&exchange->action_write, write + 1);
    return true;
}
}
