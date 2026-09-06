#include "../contracts/packet_v1.h"
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("usage: validate_packet packet");
    auto p = labv2::read_packet(argv[1]);
    printf("PASS packet: %ux%u, %zu textures, %zu draws\n", p.width, p.height,
           p.textures.size(), p.draws.size());
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "lab-v2 packet: %s\n", e.what());
    return 1;
  }
}
