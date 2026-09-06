// Headless packet replay. No NSWindow, CAMetalLayer, or second presenter.
#include "../contracts/packet_v1.h"
#include "../shared/color_response.h"
#include <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
using namespace labv2;
MTLPixelFormat pixel_format(uint32_t f) {
  switch (f) {
  case 61:
    return MTLPixelFormatR8Unorm;
  case 71:
    return MTLPixelFormatBC1_RGBA;
  case 72:
    return MTLPixelFormatBC1_RGBA_sRGB;
  case 77:
    return MTLPixelFormatBC3_RGBA;
  case 78:
    return MTLPixelFormatBC3_RGBA_sRGB;
  case 80:
    return MTLPixelFormatBC4_RUnorm;
  case 83:
    return MTLPixelFormatBC5_RGUnorm;
  case 10:
    return MTLPixelFormatRGBA16Float;
  case 11:
    return MTLPixelFormatRGBA16Unorm;
  case 35:
    return MTLPixelFormatRG16Unorm;
  default:
    throw std::runtime_error("unsupported packet texture format " +
                             std::to_string(f));
  }
}
void check(bool value, const char *text) {
  if (!value)
    throw std::runtime_error(text);
}
std::string read_text(std::string p) {
  std::ifstream f(p);
  check(bool(f), "missing shader module");
  return std::string(std::istreambuf_iterator<char>(f), {});
}
struct Shader {
  id<MTLFunction> fn;
  std::set<unsigned> ids;
};
Shader compile_shader(id<MTLDevice> dev, std::string dir, std::string entry) {
  auto source = read_text(dir + "/" + entry + ".msl");
  NSError *error = nil;
  MTLCompileOptions *options = [MTLCompileOptions new];
  options.mathMode = MTLMathModeSafe;
  options.languageVersion = MTLLanguageVersion2_2;
  id<MTLLibrary> lib =
      [dev newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                        options:options
                          error:&error];
  if (!lib)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  Shader out;
  out.fn =
      [lib newFunctionWithName:[NSString stringWithUTF8String:entry.c_str()]];
  check(out.fn != nil, "shader entry missing");
  std::regex r("\\[\\[id\\(([0-9]+)\\)\\]\\]");
  for (auto i = std::sregex_iterator(source.begin(), source.end(), r);
       i != std::sregex_iterator(); ++i)
    out.ids.insert(unsigned(std::stoul((*i)[1])));
  return out;
}
std::map<std::string, Shader> shader_cache;
Shader shader(id<MTLDevice> dev, std::string dir, std::string entry) {
  auto key = dir + "/" + entry;
  auto it = shader_cache.find(key);
  if (it != shader_cache.end())
    return it->second;
  return shader_cache.emplace(key, compile_shader(dev, dir, entry))
      .first->second;
}
std::string content_key(const void *data, size_t n) {
  unsigned char hash[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(data, CC_LONG(n), hash);
  std::string result;
  const char *hex = "0123456789abcdef";
  for (auto b : hash) {
    result += hex[b >> 4];
    result += hex[b & 15];
  }
  return result;
}
std::map<std::string, id<MTLTexture>> texture_cache;
std::map<std::string, id<MTLBuffer>> buffer_cache;
id<MTLDevice> shared_device;
struct GPUScene {
  std::vector<id<MTLTexture>> textures;
  std::vector<id<MTLBuffer>> buffers;
};
GPUScene upload(id<MTLDevice> dev, const Packet &p) {
  GPUScene g;
  MTLTextureDescriptor *z = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:1
                                  height:1
                               mipmapped:NO];
  z.storageMode = MTLStorageModeShared;
  auto zero = [dev newTextureWithDescriptor:z];
  uint32_t black = 0;
  [zero replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
          mipmapLevel:0
            withBytes:&black
          bytesPerRow:4];
  g.textures.push_back(zero);
  for (auto &t : p.textures) {
    std::string key = std::to_string(t.width) + ":" + std::to_string(t.height) +
                      ":" + std::to_string(t.format);
    for (auto &m : t.mips)
      key += ":" + content_key(m.bytes.data(), m.bytes.size());
    auto found = texture_cache.find(key);
    if (found != texture_cache.end()) {
      g.textures.push_back(found->second);
      continue;
    }
    auto d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:pixel_format(t.format)
                                     width:t.width
                                    height:t.height
                                 mipmapped:t.mips.size() > 1];
    d.mipmapLevelCount = t.mips.size();
    d.storageMode = MTLStorageModeShared;
    auto texture = [dev newTextureWithDescriptor:d];
    check(texture != nil, "Metal texture creation failed");
    for (size_t i = 0; i < t.mips.size(); i++) {
      auto &m = t.mips[i];
      [texture replaceRegion:MTLRegionMake2D(0, 0, std::max(1u, t.width >> i),
                                             std::max(1u, t.height >> i))
                 mipmapLevel:i
                   withBytes:m.bytes.data()
                 bytesPerRow:m.pitch];
    }
    g.textures.push_back(texture);
    texture_cache.emplace(key, texture);
  }
  for (auto &b : p.buffers) {
    auto key = content_key(b.data(), b.size());
    auto found = buffer_cache.find(key);
    if (found != buffer_cache.end()) {
      g.buffers.push_back(found->second);
      continue;
    }
    auto buffer = [dev newBufferWithBytes:b.data()
                                   length:b.size()
                                  options:MTLResourceStorageModeShared];
    check(buffer != nil, "Metal buffer allocation failed");
    g.buffers.push_back(buffer);
    buffer_cache.emplace(key, buffer);
  }
  return g;
}
struct Settings {
  unsigned samples = 1, anisotropy = 8, scale = 1;
  double offset_x = 0, offset_y = 0;
};
int render_job(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (!shared_device)
        shared_device = MTLCreateSystemDefaultDevice();
      id<MTLDevice> dev = shared_device;
      check(dev != nil, "Metal device unavailable");
      if (argc == 2 && std::string(argv[1]) == "--capabilities") {
        std::string fingerprint =
            std::string(dev.name.UTF8String) + "/" +
            std::string([NSProcessInfo processInfo]
                            .operatingSystemVersionString.UTF8String);
        fingerprint = content_key(fingerprint.data(), fingerprint.size());
        printf("{\"device_driver_fingerprint\":\"%s\",", fingerprint.c_str());
        printf("\"schema\":\"c3x.lab_v2.capabilities.v1\",\"backend\":"
               "\"metal\",\"sample_counts\":[1");
        for (unsigned n : {2, 4, 8})
          if ([dev supportsTextureSampleCount:n])
            printf(",%u", n);
        printf("],\"max_anisotropy\":16,\"max_buffer_bytes\":%llu,\"argument_"
               "buffers_tier\":%lu,\"bc_textures\":%s,\"color_target\":\"bgra8_"
               "unorm\",\"blend\":\"straight_alpha\",\"packet_read_versions\":[1,2,3,4,5,6],\"linear_target\":\"rgba16float_premultiplied\",\"validity_target\":\"r8unorm\"}\n",
               (unsigned long long)dev.maxBufferLength,
               (unsigned long)dev.argumentBuffersSupport,
               dev.supportsBCTextureCompression ? "true" : "false");
        return 0;
      }
      check(argc >= 5,
            "usage: metal <packet> <shader-dir> <output.bmp> <metrics.json> "
            "[samples anisotropy scale offset-x offset-y repeat]");
      Settings s;
      if (argc > 5)
        s.samples = std::stoul(argv[5]);
      if (argc > 6)
        s.anisotropy = std::stoul(argv[6]);
      if (argc > 7)
        s.scale = std::stoul(argv[7]);
      if (argc > 8)
        s.offset_x = std::stod(argv[8]);
      if (argc > 9)
        s.offset_y = std::stod(argv[9]);
      unsigned repeats = argc > 10 ? std::stoul(argv[10]) : 1;
      check([dev supportsTextureSampleCount:s.samples] && s.samples > 0,
            "unsupported sample count");
      check(s.anisotropy >= 1 && s.anisotropy <= 16, "unsupported anisotropy");
      check(s.scale == 1 || s.scale == 2 || s.scale == 4,
            "unsupported integer render scale");
      check(repeats > 0 && repeats <= 100, "invalid batch size");
      auto started = std::chrono::steady_clock::now();
      uint64_t allocation_high_water = dev.currentAllocatedSize;
      Packet p = read_packet(argv[1]);
      bool linear = p.color_branch == 1;
      auto target_format = linear ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm;
      if (linear && argc > 11 && std::string(argv[11]) != "box")
        check(argc > 12 && std::string(argv[12]) == "2", "linear branch requires postprocess contract 2");
      auto uploads_before = texture_cache.size();
      GPUScene g = upload(dev, p);
      allocation_high_water =
          std::max(allocation_high_water, uint64_t(dev.currentAllocatedSize));
      auto queue = [dev newCommandQueue];
      std::vector<Shader> vs,ps;
      for(unsigned module=0;module<p.shader_count;module++) {
        std::string dir=argv[2];if(p.shader_count>1)dir+="/m"+std::to_string(module);
        vs.push_back(shader(dev,dir,"VSMain"));vs.push_back(shader(dev,dir,"VSFeature"));
        ps.push_back(shader(dev,dir,"PSMain"));ps.push_back(shader(dev,dir,"PSFeature"));
      }
      id<MTLSamplerState> samplers[2];
      for (unsigned i = 0; i < 2; i++) {
        auto d = [MTLSamplerDescriptor new];
        d.minFilter = MTLSamplerMinMagFilterLinear;
        d.magFilter = MTLSamplerMinMagFilterLinear;
        d.mipFilter = MTLSamplerMipFilterLinear;
        d.maxAnisotropy = s.anisotropy;
        d.sAddressMode = d.tAddressMode = d.rAddressMode =
            i ? MTLSamplerAddressModeClampToEdge : MTLSamplerAddressModeRepeat;
        d.supportArgumentBuffers = YES;
        samplers[i] = [dev newSamplerStateWithDescriptor:d];
      }
      auto depthDesc = [MTLDepthStencilDescriptor new];
      depthDesc.depthCompareFunction = MTLCompareFunctionLessEqual;
      depthDesc.depthWriteEnabled = YES;
      auto depthState = [dev newDepthStencilStateWithDescriptor:depthDesc];
      depthDesc.depthWriteEnabled = NO;
      auto depthRead = [dev newDepthStencilStateWithDescriptor:depthDesc];
      depthDesc.depthCompareFunction = MTLCompareFunctionAlways;
      auto depthOff = [dev newDepthStencilStateWithDescriptor:depthDesc];
      std::vector<id<MTLRenderPipelineState>> pipelines;
      std::vector<id<MTLBuffer>> arguments;
      for (auto &d : p.draws) {
        auto vd = [MTLVertexDescriptor vertexDescriptor];
        for (unsigned i = 0; i < d.attributes.size(); i++) {
          vd.attributes[i].format = MTLVertexFormat(
              MTLVertexFormatFloat + d.attributes[i].components - 1);
          vd.attributes[i].offset = d.attributes[i].offset;
          vd.attributes[i].bufferIndex = 30;
        }
        vd.layouts[30].stride = d.stride;
        auto pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = vs[d.shader_index*2+d.feature].fn;
        pd.fragmentFunction = ps[d.shader_index*2+d.feature].fn;
        pd.vertexDescriptor = vd;
        pd.rasterSampleCount = s.samples;
        pd.colorAttachments[0].pixelFormat = target_format;
        if (linear) {
          pd.colorAttachments[1].pixelFormat = MTLPixelFormatR8Unorm;
          auto v = pd.colorAttachments[1];
          v.blendingEnabled = YES; v.rgbBlendOperation = MTLBlendOperationMax;
          v.sourceRGBBlendFactor = v.destinationRGBBlendFactor = MTLBlendFactorOne;
          v.writeMask = d.blend_mode != 0 ? MTLColorWriteMaskNone : MTLColorWriteMaskRed;
        }
        pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
        auto c = pd.colorAttachments[0];
        c.blendingEnabled = !linear || d.blend_mode != 0;
        c.sourceRGBBlendFactor = linear ? MTLBlendFactorOne : MTLBlendFactorSourceAlpha;
        c.destinationRGBBlendFactor = linear && d.blend_mode == 2 ? MTLBlendFactorOne : MTLBlendFactorOneMinusSourceAlpha;
        c.sourceAlphaBlendFactor = MTLBlendFactorOne;
        c.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        if (linear && d.blend_mode == 2) c.writeMask = MTLColorWriteMaskRed | MTLColorWriteMaskGreen | MTLColorWriteMaskBlue;
        NSError *error = nil;
        auto pipeline = [dev newRenderPipelineStateWithDescriptor:pd
                                                            error:&error];
        if (!pipeline)
          throw std::runtime_error(error.localizedDescription.UTF8String);
        pipelines.push_back(pipeline);
        id<MTLArgumentEncoder> encoder =
            ps[d.shader_index*2+d.feature].ids.empty()
                ? nil
                : [ps[d.shader_index*2+d.feature].fn newArgumentEncoderWithBufferIndex:0];
        auto args = [dev
            newBufferWithLength:std::max<NSUInteger>(1, encoder.encodedLength)
                        options:MTLResourceStorageModeShared];
        [encoder setArgumentBuffer:args offset:0];
        for (auto binding : ps[d.shader_index*2+d.feature].ids) {
          if (binding < 128)
            [encoder setTexture:g.textures[d.textures[binding]]
                        atIndex:binding];
          else if (binding < 130)
            [encoder setSamplerState:samplers[binding - 128] atIndex:binding];
          else if (binding == 130)
            [encoder setBuffer:g.buffers[d.constant_buffer]
                        offset:0
                       atIndex:130];
          else if(binding==131 && d.frame_buffer!=UINT32_MAX)
            [encoder setBuffer:g.buffers[d.frame_buffer] offset:0 atIndex:131];
          else
            throw std::runtime_error("missing or unsupported shader binding");
        }
        arguments.push_back(args);
      }
      uint32_t w = p.width * s.scale, h = p.height * s.scale;
      check(w <= 8192 && h <= 8192, "scaled viewport exceeds limit");
      auto td = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:target_format
                                       width:w
                                      height:h
                                   mipmapped:NO];
      td.storageMode = MTLStorageModeShared;
      td.usage = MTLTextureUsageRenderTarget;
      td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
      auto resolved = [dev newTextureWithDescriptor:td];
      id<MTLTexture> color = resolved;
      if (s.samples > 1) {
        td.textureType = MTLTextureType2DMultisample;
        td.sampleCount = s.samples;
        td.storageMode = MTLStorageModePrivate;
        color = [dev newTextureWithDescriptor:td];
      }
      id<MTLTexture> validity = nil, valid_resolved = nil;
      if (linear) {
        auto vd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:w height:h mipmapped:NO];
        vd.storageMode = MTLStorageModeShared; vd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        valid_resolved = [dev newTextureWithDescriptor:vd]; validity = valid_resolved;
        if (s.samples > 1) { vd.textureType = MTLTextureType2DMultisample; vd.sampleCount = s.samples; vd.storageMode = MTLStorageModePrivate; validity = [dev newTextureWithDescriptor:vd]; }
        check(validity && valid_resolved, "validity attachment allocation failed");
      }
      td.pixelFormat = MTLPixelFormatDepth32Float;
      td.storageMode = MTLStorageModePrivate;
      auto depth = [dev newTextureWithDescriptor:td];
      check(color && depth && resolved, "render target allocation failed");
      double gpu_ms = 0;
      allocation_high_water =
          std::max(allocation_high_water, uint64_t(dev.currentAllocatedSize));
      for (unsigned frame = 0; frame < repeats; frame++) {
        auto cb = [queue commandBuffer];
        for (size_t i = 0; i < p.draws.size(); i++) {
          auto &d = p.draws[i];
          auto pass = [MTLRenderPassDescriptor renderPassDescriptor];
          pass.colorAttachments[0].texture = color;
          pass.colorAttachments[0].loadAction =
              i ? MTLLoadActionLoad : MTLLoadActionClear;
          pass.colorAttachments[0].clearColor =
              linear ? MTLClearColorMake(0,0,0,0) : MTLClearColorMake(0.035, 0.035, 0.035, 1);
          pass.colorAttachments[0].storeAction =
              s.samples > 1 && (!linear || i+1==p.draws.size()) ? MTLStoreActionStoreAndMultisampleResolve
                            : MTLStoreActionStore;
          if (s.samples > 1 && (!linear || i+1==p.draws.size()))
            pass.colorAttachments[0].resolveTexture = resolved;
          if (linear) {
            auto v = pass.colorAttachments[1]; v.texture = validity;
            v.loadAction = i ? MTLLoadActionLoad : MTLLoadActionClear;
            v.clearColor = MTLClearColorMake(0,0,0,0);
            v.storeAction = s.samples > 1 && i+1==p.draws.size() ? MTLStoreActionStoreAndMultisampleResolve : MTLStoreActionStore;
            if (s.samples > 1 && i+1==p.draws.size()) v.resolveTexture = valid_resolved;
          }
          pass.depthAttachment.texture = depth;
          pass.depthAttachment.loadAction = (i == 0 || d.clear_depth)
                                                ? MTLLoadActionClear
                                                : MTLLoadActionLoad;
          pass.depthAttachment.storeAction = MTLStoreActionStore;
          pass.depthAttachment.clearDepth = 1;
          auto e = [cb renderCommandEncoderWithDescriptor:pass];
          [e setRenderPipelineState:pipelines[i]];
          [e setCullMode:MTLCullModeNone];
          if (linear) [e setDepthStencilState:d.depth_mode == 2 ? depthState : d.depth_mode == 1 ? depthRead : depthOff];
          else if (d.depth) [e setDepthStencilState:depthState];
          [e setViewport:MTLViewport{s.offset_x * s.scale * p.downsample,
                                     s.offset_y * s.scale * p.downsample,
                                     double(w), double(h), 0, 1}];
          [e setVertexBuffer:g.buffers[d.vertex_buffer] offset:0 atIndex:30];
          [e setFragmentBuffer:arguments[i] offset:0 atIndex:0];
          for (auto t : g.textures)
            [e useResource:t
                     usage:MTLResourceUsageRead
                    stages:MTLRenderStageFragment];
          [e useResource:g.buffers[d.constant_buffer]
                   usage:MTLResourceUsageRead
                  stages:MTLRenderStageFragment];
          if(d.frame_buffer!=UINT32_MAX) [e useResource:g.buffers[d.frame_buffer] usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
          [e drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0
                vertexCount:d.count];
          [e endEncoding];
        }
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError)
          throw std::runtime_error(cb.error.localizedDescription.UTF8String);
        gpu_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1000;
        uint32_t read_width = w, read_height = h,
                 read_scale = p.downsample * s.scale;
        id<MTLTexture> read_texture = resolved, read_validity = valid_resolved;
        if (linear && frame == 0) {
          std::vector<uint16_t> raw(size_t(w)*h*4); std::vector<uint8_t> mask(size_t(w)*h);
          [resolved getBytes:raw.data() bytesPerRow:w*8 fromRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0];
          [valid_resolved getBytes:mask.data() bytesPerRow:w fromRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0];
          write_attachment(std::string(argv[3])+".linear.rgba16f",raw.data(),raw.size()*2);
          write_attachment(std::string(argv[3])+".validity.r8",mask.data(),mask.size());
        }
        if (argc > 11 && std::string(argv[11]) != "box") {
          std::string path = argv[11];
          auto slash = path.find_last_of('/');
          auto post = shader(dev, path.substr(0, slash), "CSPost");
          NSError *err = nil;
          auto pipeline = [dev newComputePipelineStateWithFunction:post.fn
                                                             error:&err];
          if (!pipeline)
            throw std::runtime_error(err.localizedDescription.UTF8String);
          read_width = p.width / p.downsample;
          read_height = p.height / p.downsample;
          read_scale = 1;
          auto pd = [MTLTextureDescriptor
              texture2DDescriptorWithPixelFormat:target_format
                                           width:read_width
                                          height:read_height
                                       mipmapped:NO];
          pd.storageMode = MTLStorageModeShared;
          pd.usage = MTLTextureUsageShaderWrite;
          read_texture = [dev newTextureWithDescriptor:pd];
          check(read_texture != nil, "postprocess target allocation failed");
          if (linear) {
            pd.pixelFormat = MTLPixelFormatR8Unorm; read_validity = [dev newTextureWithDescriptor:pd];
            check(read_validity != nil, "postprocess validity allocation failed");
          }
          auto post_cb = [queue commandBuffer];
          auto e = [post_cb computeCommandEncoder];
          [e setComputePipelineState:pipeline];
          [e setTexture:resolved atIndex:0];
          [e setTexture:read_texture atIndex:1];
          if (linear) { [e setTexture:valid_resolved atIndex:3]; [e setTexture:read_validity atIndex:4]; }
          uint32_t sizes[] = {w, h, read_width, read_height,p.valid_rect[0],p.valid_rect[1],p.valid_rect[2],p.valid_rect[3]};
          [e setBytes:sizes length:linear?sizeof(sizes):16 atIndex:2];
          [e dispatchThreads:MTLSizeMake(read_width, read_height, 1)
              threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
          [e endEncoding];
          [post_cb commit];
          [post_cb waitUntilCompleted];
          if (post_cb.status == MTLCommandBufferStatusError)
            throw std::runtime_error(
                post_cb.error.localizedDescription.UTF8String);
          gpu_ms += (post_cb.GPUEndTime - post_cb.GPUStartTime) * 1000;
        }
        allocation_high_water =
            std::max(allocation_high_water, uint64_t(dev.currentAllocatedSize));
        std::vector<uint8_t> pixels(size_t(read_width) * read_height * 4);
        if (linear) {
          std::vector<uint16_t> rgba(size_t(read_width)*read_height*4);
          std::vector<uint8_t> valid(size_t(read_width)*read_height);
          [read_texture getBytes:rgba.data() bytesPerRow:read_width*8 fromRegion:MTLRegionMake2D(0,0,read_width,read_height) mipmapLevel:0];
          [read_validity getBytes:valid.data() bytesPerRow:read_width fromRegion:MTLRegionMake2D(0,0,read_width,read_height) mipmapLevel:0];
          pixels = display_pixels(rgba, valid, read_width, read_height, read_scale, p);
          read_width /= read_scale; read_height /= read_scale; read_scale = 1;
        } else
        [read_texture getBytes:pixels.data()
                   bytesPerRow:read_width * 4
                    fromRegion:MTLRegionMake2D(0, 0, read_width, read_height)
                   mipmapLevel:0];
        std::string out = argv[3];
        if (frame == 0 && read_scale > 1) {
          check(write_bmp((out + ".resolved.bmp").c_str(), pixels, read_width,
                          read_height, 1),
                "raw resolve write failed");
        }
        if (frame)
          out += ".repeat" + std::to_string(frame) + ".bmp";
        check(
            write_bmp(out.c_str(), pixels, read_width, read_height, read_scale),
            "image write failed");
      }
      double wall_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - started)
                           .count();
      FILE *metrics = fopen(argv[4], "wb");
      check(metrics != nullptr, "metrics write failed");
      fprintf(metrics,
              "{\"schema\":\"c3x.lab_v2.cost.v1\",\"backend\":\"metal\",\"gpu_"
              "ms_mean\":%.6f,\"gpu_scope\":\"render and optional postprocess, "
              "excludes "
              "uploads/"
              "readback/CPU output conversion\",\"wall_ms\":%.6f,\"allocation_high_water_sampled_"
              "bytes\":%llu,\"allocation_scope\":\"sampled at upload, target "
              "creation and completed frame; not driver transient "
              "peak\",\"allocated_bytes\":%llu,\"texture_count\":%zu,\"draw_"
              "count\":%zu,\"batch_frames\":%u,\"new_texture_uploads\":%zu,"
              "\"samples\":%u,\"anisotropy\":%u,\"render_scale\":%u,\"output_"
              "size\":[%u,%u],\"camera_offset\":[%.6f,%.6f]}\n",
              gpu_ms / repeats, wall_ms,
              (unsigned long long)allocation_high_water,
              (unsigned long long)dev.currentAllocatedSize, p.textures.size(),
              p.draws.size(), repeats, texture_cache.size() - uploads_before,
              s.samples, s.anisotropy, s.scale, p.width / p.downsample,
              p.height / p.downsample, s.offset_x, s.offset_y);
      fclose(metrics);
      return 0;
    } catch (const std::exception &e) {
      fprintf(stderr, "lab-v2 Metal: %s\n", e.what());
      return 1;
    }
  }
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc == 3 && std::string(argv[1]) == "--batch") {
      NSData *data = [NSData
          dataWithContentsOfFile:[NSString stringWithUTF8String:argv[2]]];
      NSError *error = nil;
      id jobs = data ? [NSJSONSerialization JSONObjectWithData:data
                                                       options:0
                                                         error:&error]
                     : nil;
      if (![jobs isKindOfClass:[NSArray class]] || [jobs count] > 128) {
        fprintf(stderr, "invalid batch manifest\n");
        return 2;
      }
      for (id job in jobs) {
        @autoreleasepool {
          if (![job isKindOfClass:[NSArray class]])
            return 2;
          std::vector<std::string> strings{"metal"};
          for (id arg in job) {
            if (![arg isKindOfClass:[NSString class]])
              return 2;
            strings.emplace_back([arg UTF8String]);
          }
          std::vector<char *> args;
          for (auto &v : strings)
            args.push_back(v.data());
          int status = render_job(int(args.size()), args.data());
          if (status)
            return status;
        }
      }
      return 0;
    }
    return render_job(argc, argv);
  }
}
