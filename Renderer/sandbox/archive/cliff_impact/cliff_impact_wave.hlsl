// A coast-relative breaker moves from open water to the rock foot, then
// leaves a short-lived irregular foam burst. Placement and water clipping
// come from the authoritative coast; these are sandbox-only timing/art rules.
float4 PSCliffImpactWave(PixelInput input) : SV_Target {
    float mask=input.shape_visibility.y;
    clip(mask-.001);
    float seed=input.base_terrain,seed2=input.real_terrain;
    float duration=lerp(8.,20.,seed2),restart=lerp(1.5,3.,seed);
    float age=fmod(coastal_time.x+seed*duration*.72,duration+restart)/duration;
    clip(1.-age);
    float d=input.uv.x,v=input.uv.y;
    float life=smoothstep(0,.055,age)*(1-smoothstep(.68,.82,age));
    float2 phase=float2(seed*7.,seed2*13.);
    if(input.surface_kind>8.5) {
        float impact=smoothstep(.43,.54,age)*(1-smoothstep(.62,.76,age));
        float foam=coastal_foam.Sample(material_sampler,
            float2(d*1.10+seed*8.,v*1.45+seed2*9.+age*.08)).r;
        uint page=min(15,(uint)(seed2*16));
        float2 atlas=(float2(page%8,page/8)+
            clamp(float2(d,v),float2(.5/128.,.5/512.),
                float2(1.-.5/128.,1.-.5/512.)))/float2(8,2);
        float crest=coastal_crest.SampleLevel(material_sampler,atlas,0).r;
        float height=.58+.22*foam+.09*sin(v*43.+seed*19.);
        float profile=smoothstep(0,.10,d)*
            (1-smoothstep(height-.20,height,d));
        float fleck=.20+.65*pow(saturate(foam),.8)+crest*.25;
        float coverage=saturate(impact*life*mask*profile*fleck*.95)*
            smoothstep(0,.15,v)*(1-smoothstep(.85,1.,v));
        clip(coverage-.006);
        float3 lit=float3(.76,.83,.87)*
            q6_receiver_illumination(input,float3(0,0,1),1,1);
        return float4(lit*coverage,coverage);
    }
    if(input.surface_kind>7.5) {
        float impact=smoothstep(.43,.53,age)*(1-smoothstep(.59,.76,age));
        float grow=smoothstep(.43,.57,age);
        float foam=coastal_foam.Sample(material_sampler,
            float2(d*.22+seed*8.,v*.34+seed2*9.+age*.04)).r;
        uint page=min(15,(uint)(seed2*16));
        float2 atlas=(float2(page%8,page/8)+
            clamp(float2(d,v),float2(.5/128.,.5/512.),
                float2(1.-.5/128.,1.-.5/512.)))/float2(8,2);
        float crest=coastal_crest.SampleLevel(material_sampler,atlas,0).r;
        float irregular=.025*sin(v*37.+seed*19.)+
            .025*sin(v*79.+seed2*37.)+(foam-.5)*.055;
        float reach=lerp(.25,.90,grow);
        float sheet=1-smoothstep(reach-.22,reach+.05,d+irregular);
        float silhouette=smoothstep(0,.14,d)*
            smoothstep(0,.12,v)*(1-smoothstep(.86,1.,v));
        float filament=pow(saturate(foam),.60);
        float body=pow(max(crest,0),.72)*(.16+1.65*filament);
        float front=exp(-pow((d+irregular-reach)/.075,2));
        float water_sheet=sheet*(.12+.38*foam*foam+.20*body);
        float crest_edge=front*(.18+.54*filament+.18*crest);
        float coverage=saturate(water_sheet+crest_edge)*
            impact*life*mask*silhouette;
        clip(coverage-.006);
        float3 lit=lerp(float3(.38,.57,.67),float3(.74,.80,.83),
            saturate(front*.65+crest*.35))*
            q6_receiver_illumination(input,float3(0,0,1),1,1);
        return float4(lit*coverage,coverage);
    }
    float scale=lerp(.30,1.,seed);
    float center=lerp(.89,.022,smoothstep(.03,.58,age));
    float u=(d-center)/(.31*scale*1.4)+.2;
    clip(min(min(u,1-u),min(v,1-v)));
    uint page=min(15,(uint)(seed2*16));
    float delay=coastal_delay.Load(int3(min(511,(uint)(v*512)),page,0));
    float marked=step(delay,.99);
    delay=min(delay,.99);
    float2 foam_uv=float2(u*.16,v*.65)+phase+float2(-age*.12,age*.025);
    float foam=coastal_foam.Sample(material_sampler,foam_uv).r;
    float broad=coastal_foam.Sample(material_sampler,float2(u*.05,v*.18)+phase).r;
    float warped_u=u+(broad-.20)*.035;
    float2 origin=float2(page%8,page/8);
    float2 atlas=(origin+clamp(float2(warped_u,v),
        float2(.5/128.,.5/512.),float2(1.-.5/128.,1.-.5/512.)))/float2(8,2);
    float crest=coastal_crest.SampleLevel(material_sampler,atlas,0).r;
    float filament=pow(saturate(foam),.60);
    float body=pow(max(crest,0),.72)*(.16+1.65*filament);
    float edge=exp(-pow((u-delay)/.04,2))*crest*marked*(.08+.28*filament);
    float coverage=(1-exp(-(body+edge)*1.5))*life*mask;
    coverage*=smoothstep(.002,.023,d)*smoothstep(0,.055,v)*
        (1-smoothstep(.945,1,v));
    clip(coverage-.003);
    float3 lit=float3(.74,.80,.83)*
        q6_receiver_illumination(input,float3(0,0,1),1,1);
    return float4(lit*coverage,coverage);
}
