// Generic optional coastal breaker material. The crest atlas supplies the shape;
// the embedded auxiliary texture supplies the connected fine foam. Distances,
// contrast and texture footprints below are C3X calibration, not recovered code.
Texture2D coastal_crest : register(t0);
Texture2D coastal_foam : register(t1);
Texture2D<float> coastal_delay : register(t2);
cbuffer CoastalWaveFrame : register(b7) { float4 coastal_time; };
float4 PSCoastalWave(PixelInput input):SV_Target {
    clip(input.shape_visibility.y-.999);
    float seed=input.base_terrain,seed2=input.real_terrain;
    float scale=lerp(.30,1.,seed);
    float duration=lerp(15.,30.,seed2),restart=4.*seed;
    float age=fmod(coastal_time.x+seed*37.,duration+restart)/duration;
    float life=smoothstep(0,.1,age)*(1-smoothstep(.85,1.,age));
    float d=input.uv.x,v=input.uv.y;
    float center=lerp(40.,2.,saturate(age))/64.;
    float u=(d-center)/(20./64.*scale*1.4)+.2;
    clip(min(min(u,1-u),min(v,1-v)));clip(life-.001);
    uint page=min(15,(uint)(seed2*16));
    float delay=coastal_delay.Load(int3(min(511,(uint)(v*512)),page,0));
    float marked=step(delay,.99);
    delay=min(delay,.99);
    // Inactive delay rows still contain authored feathering (RGB up to 32/255).
    // Preserve that coverage instead of imposing the old rectangular row cut.
    float2 phase=float2(seed*7.,seed2*13.);
    float2 foam_uv=float2(u*.16,v*.65)+phase+float2(-age*.12,age*.025);
    float foam=coastal_foam.Sample(material_sampler,foam_uv).r;
    float broad=coastal_foam.Sample(material_sampler,float2(u*.05,v*.18)+phase).r;
    float warped_u=u+(broad-.20)*.035;
    float2 origin=float2(page%8,page/8);
    float2 uv=(origin+clamp(float2(warped_u,v),float2(.5/128.,.5/512.),float2(1.-.5/128.,1.-.5/512.)))/float2(8,2);
    float crest=coastal_crest.SampleLevel(material_sampler,uv,0).r;
    // Map a useful part of the source foam onto a small screen-space breaker.
    // Mapping the entire 512-wide texture into a few pixels erased its veins.
    float filament=pow(saturate(foam),.60);
    float body=pow(max(crest,0),.72)*(.16+1.65*filament);
    float edge=exp(-pow((u-delay)/.04,2))*crest*marked*(.08+.28*filament);
    float coverage=(1-exp(-(body+edge)*1.5))*life;
    coverage*=smoothstep(.002,.022,d)*smoothstep(0,.06,v)*(1-smoothstep(.94,1,v));
    clip(coverage-.002);
    float3 lit=float3(.74,.80,.83)*q6_receiver_illumination(input,float3(0,0,1),1,1);
    return float4(lit*coverage,coverage);
}
