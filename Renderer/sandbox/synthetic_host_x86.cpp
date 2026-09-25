#include "exchange.h"
#include <algorithm>
#include <cstdio>
#include <vector>

int main() {
    using namespace sandbox_exchange;
    HANDLE mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
        sizeof(Exchange), mapping_name);
    if (!mapping) return 1;
    bool fresh_mapping = GetLastError() != ERROR_ALREADY_EXISTS;
    auto* exchange = static_cast<Exchange*>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(Exchange)));
    if (!exchange) { CloseHandle(mapping); return 2; }
    if (fresh_mapping) {
        ZeroMemory(exchange, sizeof(Exchange));
        InterlockedExchange(&exchange->ready, ready_magic);
    }
    while (acquire(&exchange->ready) != ready_magic) Sleep(1);
    auto start = GetTickCount64();
    LONG accepted = 0, rejected = 0, published_during_client_pause = 0;
    LONG previous_client = acquire(&exchange->client_heartbeat);
    Snapshot snapshot = {0, 1, 1, 7, 19, 47, 1, 0};
    bool host_paused = false;
    LARGE_INTEGER frequency{}; QueryPerformanceFrequency(&frequency);
    std::vector<LONG> publication_us;
    while (GetTickCount64() - start < 36000) {
        auto elapsed = GetTickCount64() - start;
        if (elapsed >= 9000 && !host_paused) {
            host_paused = true;
            std::printf("HOST_PAUSE_BEGIN generation=%ld client_frames=%ld\n", snapshot.generation,
                acquire(&exchange->client_heartbeat));
            Sleep(2000);
            std::printf("HOST_PAUSE_END generation=%ld client_frames=%ld\n", snapshot.generation,
                acquire(&exchange->client_heartbeat));
        }
        LONG write = acquire(&exchange->action_write);
        while (acquire(&exchange->action_read) < write) {
            LONG read = acquire(&exchange->action_read);
            Action action = exchange->actions[unsigned(read) % action_capacity];
            bool valid = action.viewer == snapshot.viewer &&
                action.unit_incarnation == snapshot.unit_incarnation &&
                action.seen_generation <= snapshot.generation &&
                action.target_x == snapshot.unit_x + 1 &&
                action.target_y == snapshot.unit_y + 1;
            if (valid) {
                snapshot.unit_x = action.target_x;
                snapshot.unit_y = action.target_y;
                ++accepted;
            } else ++rejected;
            InterlockedExchange(&exchange->action_result, valid ? 1 : -1);
            InterlockedExchange(&exchange->action_ack, action.serial);
            InterlockedExchange(&exchange->action_read, read + 1);
        }
        if (acquire(&exchange->reconcile_required)) {
            // The latest authoritative snapshot covers actions that could not
            // enter the bounded queue. Already published actions keep order.
            InterlockedExchange(&exchange->reconcile_required, 0);
        }
        if(elapsed>=17500)snapshot.combat_serial=1;
        snapshot.viewer=elapsed>=34000 && elapsed<35000?2:1;
        snapshot.unit_visible=snapshot.viewer==1;
        if (elapsed>=35000) snapshot.unit_incarnation=8;
        ++snapshot.generation;
        LARGE_INTEGER publication_begin{}, publication_end{};
        QueryPerformanceCounter(&publication_begin);
        publish(exchange, snapshot);
        QueryPerformanceCounter(&publication_end);
        publication_us.push_back(LONG((publication_end.QuadPart - publication_begin.QuadPart) *
            1000000 / frequency.QuadPart));
        InterlockedIncrement(&exchange->host_heartbeat);
        LONG client = acquire(&exchange->client_heartbeat);
        if (client == previous_client && elapsed >= 12000 && elapsed < 14000)
            ++published_during_client_pause;
        previous_client = client;
        Sleep(16);
    }
    std::sort(publication_us.begin(), publication_us.end());
    if (!publication_us.empty()) {
        InterlockedExchange(&exchange->host_publication_p95_us,
            publication_us[std::size_t(.95 * double(publication_us.size()-1))]);
        InterlockedExchange(&exchange->host_publication_worst_us, publication_us.back());
    }
    std::printf("HOST_RESULT generations=%ld client_frames=%ld during_client_pause=%ld accepted=%ld rejected=%ld ack=%ld unit=%ld,%ld\n",
        snapshot.generation, acquire(&exchange->client_heartbeat), published_during_client_pause,
        accepted, rejected, acquire(&exchange->action_ack), snapshot.unit_x, snapshot.unit_y);
    UnmapViewOfFile(exchange); CloseHandle(mapping);
    return 0;
}
