// Included after the Metal resource/shader helpers. The native caster shader
// writes physical light depth; the ordinary material shaders read this atlas.
struct MetalWorldShadow {
    struct Command {unsigned draw;id<MTLBuffer> settings,arguments;};
    Packet const&packet;
    ShadowPlan const&plan;
    GPUScene const&scene;
    std::vector<id<MTLRenderPipelineState>> pipelines;
    std::vector<id<MTLBuffer>> pixel_arguments;
    std::vector<std::vector<Command>> commands;
    id<MTLTexture> field=nil;
    unsigned draw_count=0;
    MetalWorldShadow(id<MTLDevice> device,Packet const&p,ShadowPlan const&selection,GPUScene&resources,
                     std::vector<Shader> const&vertices,std::vector<Shader> const&pixels)
        :packet(p),plan(selection),scene(resources){
        if(!p.shadow.enabled())return;
        auto descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
            width:1024 height:1024 mipmapped:NO];
        descriptor.textureType=MTLTextureType2DArray;descriptor.arrayLength=32;
        descriptor.storageMode=MTLStorageModePrivate;
        descriptor.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
        field=[device newTextureWithDescriptor:descriptor];check(field!=nil,"source shadow allocation failed");
        resources.textures[p.shadow.texture]=field;
        pipelines.resize(p.draws.size());pixel_arguments.resize(p.draws.size());
        std::map<std::array<unsigned,6>,id<MTLRenderPipelineState>> compiled;
        for(unsigned index=0;index<p.draws.size();index++){
            auto const&draw=p.draws[index];if(!(draw.geometry_flags&1))continue;
            auto const&vs=vertices[p.shadow.shader_index*2+draw.shadow_variant];
            auto const&ps=pixels[p.shadow.shader_index*2+draw.shadow_variant];
            unsigned uv=draw.attributes[draw.uv_attribute].offset,world=draw.attributes[draw.world_attribute].offset;
            std::array<unsigned,6> key{{draw.shadow_variant,draw.stride,uv,world,draw.shadow_material,draw.shadow_coverage}};
            auto found=compiled.find(key);
            if(found==compiled.end()){
                auto layout=[MTLVertexDescriptor vertexDescriptor];
                unsigned offsets[]={uv,draw.shadow_material,world,draw.shadow_coverage};
                MTLVertexFormat formats[]={MTLVertexFormatFloat2,MTLVertexFormatFloat,MTLVertexFormatFloat4,MTLVertexFormatFloat};
                for(unsigned i=0;i<4;i++){
                    layout.attributes[i].format=formats[i];layout.attributes[i].offset=offsets[i];layout.attributes[i].bufferIndex=30;
                }
                layout.layouts[30].stride=draw.stride;
                auto pipeline=[MTLRenderPipelineDescriptor new];pipeline.vertexFunction=vs.fn;pipeline.fragmentFunction=ps.fn;
                pipeline.vertexDescriptor=layout;pipeline.rasterSampleCount=1;
                auto target=pipeline.colorAttachments[0];target.pixelFormat=MTLPixelFormatR32Float;target.blendingEnabled=YES;
                target.rgbBlendOperation=target.alphaBlendOperation=MTLBlendOperationMax;
                target.sourceRGBBlendFactor=target.destinationRGBBlendFactor=MTLBlendFactorOne;
                target.sourceAlphaBlendFactor=target.destinationAlphaBlendFactor=MTLBlendFactorOne;
                NSError*error=nil;auto compiled_pipeline=[device newRenderPipelineStateWithDescriptor:pipeline error:&error];
                if(!compiled_pipeline)throw std::runtime_error(error.localizedDescription.UTF8String);
                found=compiled.emplace(key,compiled_pipeline).first;
            }
            pipelines[index]=found->second;
            auto encoder=ps.ids.empty()?nil:[ps.fn newArgumentEncoderWithBufferIndex:0];
            auto arguments=[device newBufferWithLength:std::max<NSUInteger>(1,encoder.encodedLength) options:MTLResourceStorageModeShared];
            [encoder setArgumentBuffer:arguments offset:0];
            for(unsigned binding:ps.ids){
                check(binding<128,"unsupported caster pixel binding");unsigned texture=0;
                for(auto pair:draw.shadow_textures)if(pair[0]==binding)texture=pair[1];
                [encoder setTexture:resources.textures[texture] atIndex:binding];
            }
            pixel_arguments[index]=arguments;
        }
        for(auto const&page:plan.pages){
            commands.emplace_back();
            for(unsigned index:page.casters){
                auto const&caster=plan.casters[index];auto const&draw=p.draws[caster.draw];
                auto const&vs=vertices[p.shadow.shader_index*2+draw.shadow_variant];
                std::array<float,20> constants{};std::copy(plan.basis.begin(),plan.basis.end(),constants.begin());
                constants[12]=float(page.x);constants[13]=float(page.y);
                std::copy(caster.offset.begin(),caster.offset.end(),constants.begin()+16);
                auto settings=[device newBufferWithBytes:constants.data() length:sizeof(constants) options:MTLResourceStorageModeShared];
                auto encoder=[vs.fn newArgumentEncoderWithBufferIndex:0];
                auto arguments=[device newBufferWithLength:std::max<NSUInteger>(1,encoder.encodedLength) options:MTLResourceStorageModeShared];
                [encoder setArgumentBuffer:arguments offset:0];
                check(vs.ids==std::set<unsigned>{130},"unsupported caster vertex binding");
                [encoder setBuffer:settings offset:0 atIndex:130];
                commands.back().push_back({caster.draw,settings,arguments});++draw_count;
            }
        }
    }
    void encode(id<MTLCommandBuffer> buffer){
        for(unsigned page=0;page<plan.pages.size();page++){
            auto pass=[MTLRenderPassDescriptor renderPassDescriptor];auto target=pass.colorAttachments[0];
            target.texture=field;target.slice=plan.pages[page].slot;target.loadAction=MTLLoadActionClear;
            target.storeAction=MTLStoreActionStore;target.clearColor=MTLClearColorMake(-1e6,-1e6,-1e6,-1e6);
            auto encoder=[buffer renderCommandEncoderWithDescriptor:pass];
            [encoder setCullMode:MTLCullModeNone];[encoder setDepthClipMode:MTLDepthClipModeClamp];
            [encoder setViewport:MTLViewport{0,0,1024,1024,0,1}];
            for(auto const&command:commands[page]){
                auto const&draw=packet.draws[command.draw];
                [encoder setRenderPipelineState:pipelines[command.draw]];
                [encoder setVertexBuffer:scene.buffers[draw.vertex_buffer] offset:0 atIndex:30];
                [encoder setVertexBuffer:command.arguments offset:0 atIndex:0];
                [encoder setFragmentBuffer:pixel_arguments[command.draw] offset:0 atIndex:0];
                [encoder useResource:command.settings usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
                [encoder useResource:scene.textures[0] usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
                for(auto binding:draw.shadow_textures)
                    [encoder useResource:scene.textures[binding[1]] usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
                [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:draw.count];
            }
            [encoder endEncoding];
        }
    }
};
