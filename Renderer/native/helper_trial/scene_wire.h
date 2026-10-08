#pragma once
#include <cstdint>
#include "../camera_completion.h"

namespace c3x_helper_trial {
constexpr unsigned wire_magic=0x32483343,wire_version=16,wire_capacity=16*1024*1024;
struct Wire {
    unsigned magic,version,sequence,kind,subtype,size,reply_size,status,code,shared_raw,expected_code,executed,live;
    unsigned width,height,rendered,fallback,hash[4],gpu_hash[4],gpu_hash_valid;
    std::uint64_t service_us,private_bytes,shared_handle;
    std::int64_t recorded_ticket,recorded_image,result_image;
    std::int64_t resident_bytes,uploads,commands,readbacks;
    std::int64_t clock_ticks,clock_frequency;
    unsigned replay_clock;
    unsigned result_pixels;
    int bounds[4];
    unsigned consumer_pid;
    alignas(4) volatile std::int32_t visual_frames;
    alignas(4) volatile std::int32_t presented_zoom_q16;
    // Advisory admission pressure, independent of the ordered command payload.
    alignas(4) volatile std::int32_t native_queue_records;
    // Receipt-only cancellation mailbox. No reliable resource/action command
    // is displaced; the ordered request wire still owns admission/adoption.
    alignas(8) volatile std::int64_t obsolete_camera_through;
    char error[128];
    unsigned char payload[wire_capacity];
    // Independent of the ordered request/reply slot. Its named mutex protects
    // one immutable inspection result; overwriting it never adopts a GPU map.
    alignas(4) volatile std::int32_t camera_receiver_thread;
    alignas(4) volatile std::int32_t camera_completion_available;
    c3x_remote_scene::CameraCompletionSlot camera_completion;
};
}
