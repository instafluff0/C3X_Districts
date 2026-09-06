#pragma once
// Versioned, little-endian replay packet. No API pointers or source-format IDs.
#include "../shared/content_hash.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
namespace labv2 {
#ifdef _WIN32
inline std::wstring native_path(const std::string& path) {
  int n=MultiByteToWideChar(CP_UTF8,0,path.c_str(),-1,nullptr,0);
  if(n<=0)throw std::runtime_error("invalid UTF8 path");
  std::vector<wchar_t> value(n);MultiByteToWideChar(CP_UTF8,0,path.c_str(),-1,value.data(),n);
  for(auto& c:value)if(c==L'/')c=L'\\';
  DWORD needed=GetFullPathNameW(value.data(),0,nullptr,nullptr);
  std::vector<wchar_t> full(needed+1);
  if(!GetFullPathNameW(value.data(),DWORD(full.size()),full.data(),nullptr))throw std::runtime_error("invalid absolute path");
  std::wstring result=full.data();
  if(result.rfind(L"\\\\?\\",0)==0)return result;
  if(result.rfind(L"\\\\",0)==0)return L"\\\\?\\UNC\\"+result.substr(2);
  return L"\\\\?\\"+result;
}
#endif
inline FILE* open_path(const std::string& path,const char* mode) {
#ifdef _WIN32
  std::wstring wm;for(auto c=mode;*c;c++)wm+=wchar_t(*c);
  return _wfopen(native_path(path).c_str(),wm.c_str());
#else
  return fopen(path.c_str(),mode);
#endif
}
constexpr uint32_t packet_version = 2;
struct Mip {
  uint32_t pitch = 0;
  std::vector<uint8_t> bytes;
};
struct Texture {
  uint32_t width = 0, height = 0, format = 0;
  std::vector<Mip> mips;
};
struct Attribute {
  uint32_t components = 0, offset = 0;
};
struct Draw {
  uint32_t vertex_buffer = 0, constant_buffer = 0, count = 0, stride = 0,
           feature = 0, depth = 0, clear_depth = 0;
  // Wire v3 only: explicit semantics, independent of the legacy depth bit.
  uint32_t depth_mode = 2, blend_mode = 0, shader_index = 0; // off/read/read-write; off/over/additive
  uint32_t frame_buffer = UINT32_MAX; // wire5 optional shared b1, 16-byte aligned
  uint32_t world_attribute=UINT32_MAX, normal_attribute=UINT32_MAX,
           uv_attribute=UINT32_MAX, alpha_texture_slot=UINT32_MAX,
           geometry_flags=0; // wire6: caster1, receiver2, cutout4; explicit attributes
  float alpha_cutoff=.5f;
  std::vector<Attribute> attributes;
  std::array<uint32_t, 128>
      textures{}; // 0 denotes explicit unbound/zero texture.
};
struct Packet {
  uint32_t width = 0, height = 0, downsample = 1;
  uint32_t color_branch = 0; // 0 frozen v2; 1 Q6 scene-linear premultiplied v1
  std::array<uint32_t, 4> valid_rect{}; // half-open final-pixel rectangle
  uint32_t geometry_contract = 0; // 1 selects wire6 explicit generic draw semantics
  uint32_t binding_contract = 1; // 2 selects wire5 optional per-draw b1
  uint32_t shader_count = 1; // wire4 composed shader namespace count
  float exposure = 1; // frame * profile, applied once after reconstruction
  std::vector<Texture> textures;
  std::vector<std::vector<uint8_t>> buffers;
  std::vector<Draw> draws;
};
struct Stream {
  FILE *f;
  bool reading;
  std::string resource_directory;
  uint64_t allocated_bytes = 0;
  void bytes(void *p, size_t n) {
    if (n && (reading ? fread(p, 1, n, f) : fwrite(p, 1, n, f)) != n)
      throw std::runtime_error("packet truncated or write failed");
  }
  void u32(uint32_t &v) { bytes(&v, 4); }
  void blob(std::vector<uint8_t> &v) {
    uint32_t n = uint32_t(v.size());
    u32(n);
    bool external = reading && (n & 0x80000000u);
    n &= 0x7fffffffu;
    if (n > 512 * 1024 * 1024 ||
        (allocated_bytes += n) > 2ull * 1024 * 1024 * 1024)
      throw std::runtime_error("packet allocation exceeds limit");
    if (reading)
      v.resize(n);
    if (!external) {
      bytes(v.data(), n);
      return;
    }
    char name[65] = {};
    bytes(name, 64);
    for (unsigned i = 0; i < 64; i++)
      if (!((name[i] >= '0' && name[i] <= '9') ||
            (name[i] >= 'a' && name[i] <= 'f')))
        throw std::runtime_error("invalid packet content reference");
    FILE *resource = open_path(resource_directory + "/" + name, "rb");
    if (!resource)
      throw std::runtime_error("packet content resource missing");
    bool ok = fread(v.data(), 1, n, resource) == n && fgetc(resource) == EOF;
    fclose(resource);
    if (!ok || content_sha256(v.data(), v.size()) != name)
      throw std::runtime_error("corrupt packet content resource");
  }
  template <class T, class F>
  void list(std::vector<T> &v, F fn, uint32_t limit) {
    uint32_t n = uint32_t(v.size());
    u32(n);
    if (n > limit)
      throw std::runtime_error("packet count exceeds limit");
    if (reading)
      v.resize(n);
    for (auto &x : v)
      fn(x);
  }
};
inline void transfer(Stream &s, Packet &p) {
  if(!s.reading && (p.color_branch>1 || p.binding_contract<1 || p.binding_contract>2 || p.geometry_contract>1 || (!p.color_branch && (p.binding_contract!=1 || p.geometry_contract))))throw std::runtime_error("unsupported packet extension contract");
  uint32_t magic = 0x32514c43, version = p.color_branch ? (p.geometry_contract==1?6:p.binding_contract==2?5:p.shader_count>1?4:3) : packet_version;
  s.u32(magic);
  s.u32(version);
  if (magic != 0x32514c43 || (version != 1 && version != 2 && version != 3 && version != 4 && version != 5 && version != 6))
    throw std::runtime_error("render packet contract drift");
  s.u32(p.width);
  s.u32(p.height);
  s.u32(p.downsample);
  if (!p.width || !p.height || p.width > 8192 || p.height > 8192 ||
      !p.downsample || p.downsample > 4 || p.width % p.downsample ||
      p.height % p.downsample)
    throw std::runtime_error("invalid packet viewport");
  if (version >= 3) {
    s.u32(p.color_branch);
    for (auto &v : p.valid_rect) s.u32(v);
    s.bytes(&p.exposure, 4);
    if (version >= 4) s.u32(p.shader_count);
    if (version >= 5) p.binding_contract=2;
    if (version >= 6) p.geometry_contract=1;
    if (!p.shader_count || p.shader_count>64) throw std::runtime_error("invalid shader namespace count");
    if (p.color_branch != 1 || !std::isfinite(p.exposure) || p.exposure <= 0 ||
        p.valid_rect[0] >= p.valid_rect[2] || p.valid_rect[1] >= p.valid_rect[3] ||
        p.valid_rect[2] > p.width / p.downsample || p.valid_rect[3] > p.height / p.downsample)
      throw std::runtime_error("invalid scene-linear attachment contract");
  }
  s.list(
      p.textures,
      [&](Texture &t) {
        s.u32(t.width);
        s.u32(t.height);
        s.u32(t.format);
        if (!t.width || !t.height || t.width > 16384 || t.height > 16384)
          throw std::runtime_error("invalid texture dimensions");
        s.list(
            t.mips,
            [&](Mip &m) {
              s.u32(m.pitch);
              s.blob(m.bytes);
            },
            15);
      },
      512);
  s.list(p.buffers, [&](std::vector<uint8_t> &b) { s.blob(b); }, 256);
  s.list(
      p.draws,
      [&](Draw &d) {
        s.u32(d.vertex_buffer);
        s.u32(d.constant_buffer);
        s.u32(d.count);
        s.u32(d.stride);
        s.u32(d.feature);
        s.u32(d.depth);
        s.u32(d.clear_depth);
        if (version >= 3) {
          s.u32(d.depth_mode); s.u32(d.blend_mode);
          if(version>=4) s.u32(d.shader_index);
          if(version>=5) s.u32(d.frame_buffer);
          if(version>=6) {
            s.u32(d.world_attribute);s.u32(d.normal_attribute);s.u32(d.uv_attribute);
            s.u32(d.alpha_texture_slot);s.u32(d.geometry_flags);s.bytes(&d.alpha_cutoff,4);
          }
          if(d.shader_index>=p.shader_count) throw std::runtime_error("missing draw shader namespace");
          if (d.depth_mode > 2 || d.blend_mode > 2 || d.clear_depth)
            throw std::runtime_error("invalid v3 depth/blend or category depth clear");
        }
        s.list(
            d.attributes,
            [&](Attribute &a) {
              s.u32(a.components);
              s.u32(a.offset);
            },
            31);
        for (auto &t : d.textures)
          s.u32(t);
      },
      1024);
  for (auto &t : p.textures) {
    if (t.mips.empty())
      throw std::runtime_error("missing texture mip chain");
    uint32_t w = t.width, h = t.height;
    for (auto &m : t.mips) {
      uint32_t pitch = 0, rows = h;
      if (t.format == 71 || t.format == 72 || t.format == 80) {
        pitch = ((w + 3) / 4) * 8;
        rows = (h + 3) / 4;
      } else if (t.format == 77 || t.format == 78 || t.format == 83) {
        pitch = ((w + 3) / 4) * 16;
        rows = (h + 3) / 4;
      } else if (t.format == 61)
        pitch = w;
      else if (t.format == 10 || t.format == 11)
        pitch = w * 8;
      else if (t.format == 35)
        pitch = w * 4;
      else
        throw std::runtime_error("unsupported packet texture format");
      if (m.pitch != pitch || m.bytes.size() != uint64_t(pitch) * rows)
        throw std::runtime_error("invalid texture mip byte layout");
      w = (std::max)(1u, w / 2);
      h = (std::max)(1u, h / 2);
    }
  }
  for (auto &d : p.draws) {
    if(d.frame_buffer!=UINT32_MAX && (p.binding_contract!=2 || d.frame_buffer>=p.buffers.size() || p.buffers[d.frame_buffer].empty() || p.buffers[d.frame_buffer].size()%16 || p.buffers[d.frame_buffer].size()>65536)) throw std::runtime_error("invalid shared frame buffer");
    if (d.vertex_buffer >= p.buffers.size() ||
        d.constant_buffer >= p.buffers.size() || !d.stride ||
        uint64_t(d.count) * d.stride > p.buffers[d.vertex_buffer].size() ||
        d.count % 3 || d.feature > 1 || d.depth > 1 || d.clear_depth > 1)
      throw std::runtime_error("invalid draw bounds");
    for (auto &a : d.attributes)
      if (!a.components || a.components > 4 ||
          a.offset + 4 * a.components > d.stride)
        throw std::runtime_error("invalid vertex layout");
    if(d.geometry_flags || d.world_attribute!=UINT32_MAX) {
      auto attr=[&](unsigned i,unsigned n){return i<d.attributes.size() && d.attributes[i].components==n;};
      if(p.geometry_contract!=1 || d.geometry_flags>7 || !attr(d.world_attribute,4) ||
         (d.normal_attribute!=UINT32_MAX && !attr(d.normal_attribute,3)) ||
         (d.uv_attribute!=UINT32_MAX && !attr(d.uv_attribute,2))) throw std::runtime_error("invalid generic geometry semantics");
      if((d.geometry_flags&4) && (!attr(d.uv_attribute,2) || d.alpha_texture_slot>=128 || !d.textures[d.alpha_texture_slot] || !std::isfinite(d.alpha_cutoff) || d.alpha_cutoff<0 || d.alpha_cutoff>1))throw std::runtime_error("cutout geometry requires actual alpha binding");
    }
    for (auto t : d.textures)
      if (t > p.textures.size())
        throw std::runtime_error("missing packet texture");
  }
}
inline Packet read_packet(const char *path) {
  FILE *f = open_path(path, "rb");
  if (!f)
    throw std::runtime_error("packet missing");
  Packet p;
  try {
    Stream s{f, true, std::string(path) + ".blobs"};
    transfer(s, p);
    if (fgetc(f) != EOF)
      throw std::runtime_error("packet trailing data");
  } catch (...) {
    fclose(f);
    throw;
  }
  fclose(f);
  return p;
}
inline bool write_packet(const char *path, Packet &p) {
  FILE *f = open_path(path, "wb");
  if (!f)
    return false;
  try {
    Stream s{f, false, {}};
    transfer(s, p);
  } catch (...) {
    fclose(f);
    throw;
  }
  return fclose(f) == 0;
}
inline bool write_bmp(const char *path, const std::vector<uint8_t> &bgra,
                      uint32_t w, uint32_t h, uint32_t scale) {
  if(!scale || w%scale || h%scale || bgra.size()!=uint64_t(w)*h*4) return false;
  std::string pending=std::string(path)+".pending";
  FILE *f = open_path(pending, "wb");
  if (!f)
    return false;
  uint32_t tw = w / scale, th = h / scale;
  uint8_t header[54] = {};
  auto put = [&](int off, uint32_t v) {
    for (int i = 0; i < 4; i++)
      header[off + i] = uint8_t(v >> (8 * i));
  };
  header[0] = 'B';
  header[1] = 'M';
  put(2, 54 + tw * th * 4);
  put(10, 54);
  put(14, 40);
  put(18, tw);
  put(22, uint32_t(-int(th)));
  header[26] = 1;
  header[28] = 32;
  if(fwrite(header,1,54,f)!=54){fclose(f);std::remove(pending.c_str());return false;}
  std::vector<uint8_t> row(tw * 4);
  for (uint32_t y = 0; y < th; y++) {
    for (uint32_t x = 0; x < tw; x++)
      for (uint32_t c = 0; c < 4; c++) {
        uint32_t sum = 0;
        for (uint32_t dy = 0; dy < scale; dy++)
          for (uint32_t dx = 0; dx < scale; dx++)
            sum += bgra[((y * scale + dy) * w + x * scale + dx) * 4 + c];
        row[x * 4 + c] = uint8_t(sum / (scale * scale));
      }
    if (fwrite(row.data(), 1, row.size(), f) != row.size()) {
      fclose(f);std::remove(pending.c_str());
      return false;
    }
  }
  if(fclose(f)!=0){std::remove(pending.c_str());return false;}
#ifdef _WIN32
  return MoveFileExW(native_path(pending).c_str(),native_path(path).c_str(),MOVEFILE_REPLACE_EXISTING)!=0;
#else
  return std::rename(pending.c_str(),path)==0;
#endif
}
} // namespace labv2
