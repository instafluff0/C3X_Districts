// Compose independent source-independent packet modules in explicit graph
// order.
#include "../contracts/packet_v1.h"
#include <cstdlib>
#include <utility>
int main(int argc, char **argv) {
  try {
    if (argc < 4)
      throw std::runtime_error(
          "usage: compose output.packet input.packet input.packet [...]");
    labv2::Packet result;
    for (int i = 2; i < argc; i++) {
      auto packet = labv2::read_packet(argv[i]);
      if (i == 2) {
        result.width = packet.width;
        result.height = packet.height;
        result.downsample = packet.downsample;
        result.color_branch=packet.color_branch;result.valid_rect=packet.valid_rect;result.exposure=packet.exposure;result.shader_count=0;
      }
      if (packet.width != result.width || packet.height != result.height ||
          packet.downsample != result.downsample)
        throw std::runtime_error("module viewport contract mismatch");
      if(packet.color_branch!=result.color_branch || packet.valid_rect!=result.valid_rect || packet.exposure!=result.exposure) throw std::runtime_error("module color/validity/exposure mismatch");
      result.geometry_contract=(std::max)(result.geometry_contract,packet.geometry_contract);
      result.binding_contract=(std::max)(result.binding_contract,packet.binding_contract);
      auto textures = uint32_t(result.textures.size()),
           buffers = uint32_t(result.buffers.size());
      for (auto &draw : packet.draws) {
        if(result.color_branch) draw.shader_index+=result.shader_count;
        draw.vertex_buffer += buffers;
        draw.constant_buffer += buffers;
        if(draw.frame_buffer!=UINT32_MAX) draw.frame_buffer+=buffers;
        for (auto &texture : draw.textures)
          if (texture)
            texture += textures;
        if (!result.draws.empty())
          draw.clear_depth = 0;
        result.draws.push_back(draw);
      }
      result.shader_count+=packet.shader_count;
      for (auto &texture : packet.textures)
        result.textures.push_back(std::move(texture));
      for (auto &buffer : packet.buffers)
        result.buffers.push_back(std::move(buffer));
    }
    return labv2::write_packet(argv[1], result) ? 0 : 1;
  } catch (const std::exception &e) {
    fprintf(stderr, "lab-v2 compose: %s\n", e.what());
    return 1;
  }
}
