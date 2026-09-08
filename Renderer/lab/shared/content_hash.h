#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include <windows.h>

// BCrypt requires the Windows base types before its declarations.
#include <bcrypt.h>
#else
#include <CommonCrypto/CommonDigest.h>
#endif
namespace labv2 {
inline std::string content_sha256(const void *bytes, size_t size) {
  if (size > 512u * 1024u * 1024u)
    throw std::runtime_error("content hash size exceeds limit");
  unsigned char output[32];
#ifdef _WIN32
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr,
                                  0) < 0)
    throw std::runtime_error("SHA256 provider unavailable");
  auto status =
      BCryptHash(algorithm, nullptr, 0, (PUCHAR)bytes, ULONG(size), output, 32);
  BCryptCloseAlgorithmProvider(algorithm, 0);
  if (status < 0)
    throw std::runtime_error("SHA256 failed");
#else
  CC_SHA256(bytes, CC_LONG(size), output);
#endif
  const char *hex = "0123456789abcdef";
  std::string result;
  for (auto value : output) {
    result += hex[value >> 4];
    result += hex[value & 15];
  }
  return result;
}
} // namespace labv2
