"""Upgrade a city shader's light transport without changing material equations."""
import re


def upgrade(source):
    if 'cbuffer NativeCityLights' not in source:return source
    if 'StructuredBuffer<float4> CityLightData' in source:return source
    start=source.index('cbuffer NativeCityLights')
    end=source.index('};',start)+2
    source=source[:start]+'''cbuffer NativeCityLights : register(b6) {
 float4 CityLightCounts;
 float4 Q8LocalEnvelopeLow4;float4 Q8LocalEnvelopeHigh4;
};
StructuredBuffer<float4> CityLightData : register(t127);
'''+source[end:]
    offsets={'Q8LocalPositionRange':('3*',0),'Q8LocalColorIntensity':('3*',1),
             'Q8LocalDirectionOwner':('3*',2),'Q8LocalBoxLow':('3*int(CityLightCounts.x)+2*',0),
             'Q8LocalBoxHigh':('3*int(CityLightCounts.x)+2*',1)}
    for name,(base,offset) in offsets.items():
        source=re.sub(r'\b'+name+r'\[([^\]]+)\]',lambda m:'CityLightData['+base+'('+m[1]+')+'+str(offset)+']',source)
    source=source.replace('Texture2D city_base_texture_3 : register(t127);','')
    # The retired four-texture city branch no longer has any draw records. Mine
    # and farm emissives still use slots 124/125; the new city pack uses 124.
    start=source.find('    else if (input.material_index < 29.5)')
    if start>=0:
        end=source.index('    float4 mine_sample',start)
        source=source[:start]+'    else { albedo=0; emissive=0; }\n'+source[end:]
    if 'city_base_texture_3' in source:raise ValueError('Unremoved legacy city sampler')
    return source
