#pragma once
#include <cstdint>

namespace c3x_helper_trial {
constexpr unsigned wire_magic=0x32483343,wire_version=8,wire_capacity=16*1024*1024;
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
    char error[128];
    unsigned char payload[wire_capacity];
};
}
