// Generic optional coastal breaker material. RGB is crest intensity; authored
// alpha is preserved in the pack but is not the opacity of the foam.
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
    float u=(d-center)/(20./64.*scale)+.2;
    clip(min(min(u,1-u),min(v,1-v)));clip(life-.001);
    uint page=min(15,(uint)(seed2*16));
    float delay=coastal_delay.Load(int3(min(511,(uint)(v*512)),page,0));
    clip(.99-delay);
    float2 origin=float2(page%8,page/8);
    float2 uv=(origin+clamp(float2(u,v),float2(.5/128.,.5/512.),float2(1.-.5/128.,1.-.5/512.)))/float2(8,2);
    float crest=coastal_crest.SampleLevel(material_sampler,uv,0).r;
    float auxiliary=coastal_foam.Sample(material_sampler,float2(u,2*v+coastal_time.x*.5)).r;
    float trail=smoothstep(0,.06,u-delay)*(1-smoothstep(.08,.48,u-delay));
    float breaking=1-smoothstep(8./64.,20./64.,center);
    float coverage=saturate((crest*1.75+auxiliary*trail*breaking*.24)*life);
    coverage*=smoothstep(.002,.022,d)*smoothstep(0,.05,v)*(1-smoothstep(.95,1,v));
    clip(coverage-.002);
    float3 lit=float3(.74,.80,.83)*q6_receiver_illumination(input,float3(0,0,1),1,1);
    return float4(lit,coverage);
}
