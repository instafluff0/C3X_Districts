"""Upgrade a city shader's light transport without changing material equations."""
import re


def upgrade(source):
    if 'cbuffer NativeCityLights' not in source:return source
    if 'StructuredBuffer<float4> CityLightData' in source:return spatial_upgrade(source)
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
    return spatial_upgrade(source)


def spatial_upgrade(source):
    """Conservative candidate loops over the unchanged full-scan equations."""
    if 'Q8LocalGridInfo' in source:return source
    source=source.replace(' float4 Q8LocalEnvelopeLow4;float4 Q8LocalEnvelopeHigh4;',
        ' float4 Q8LocalEnvelopeLow4;float4 Q8LocalEnvelopeHigh4;\n float4 Q8LocalGrid;float4 Q8LocalGridInfo;')
    marker='float3 q8_local_irradiance('
    start=source.index(marker)
    helper="""// Grid/list offsets share t127 with the complete light/blocker field.
// Zero mode retains the exact original loops (failure and diagnostic reference).
int q8_candidate_index(int offset,int entry) {
 int scalar=offset+entry;
 int base=int(Q8LocalGridInfo.z)+int(CityLightCounts.x);
 return int(CityLightData[base+scalar/4][scalar%4]);
}
"""
    source=source[:start]+helper+source[start:]
    source=source.replace(' [loop]for(int i=0;i<Q8_LOCAL_LIGHT_COUNT;i++) {',""" int light_offset=0,light_count=Q8_LOCAL_LIGHT_COUNT;
 bool indexed=Q8LocalGridInfo.w>.5;
 if(indexed) {
  int2 cell=int2(floor((receiver_position.xy-Q8LocalGrid.xy)*Q8LocalGrid.z));
  if(any(cell<0) || cell.x>=int(Q8LocalGrid.w) || cell.y>=int(Q8LocalGridInfo.x))return 0;
  float2 list=CityLightData[int(Q8LocalGridInfo.y)+cell.y*int(Q8LocalGrid.w)+cell.x].xy;
  light_offset=int(list.x);light_count=int(list.y);
 }
 [loop]for(int candidate=0;candidate<light_count;candidate++) {
  int i=indexed?q8_candidate_index(light_offset,candidate):candidate;""")
    source=source.replace('  [loop]for(int j=0;j<Q8_LOCAL_BLOCKER_COUNT;j++) {',"""  int blocker_offset=0,blocker_count=Q8_LOCAL_BLOCKER_COUNT;
  if(indexed) {
   float2 list=CityLightData[int(Q8LocalGridInfo.z)+i].xy;
   blocker_offset=int(list.x);blocker_count=int(list.y);
  }
  [loop]for(int blocker=0;blocker<blocker_count;blocker++) {
   int j=indexed?q8_candidate_index(blocker_offset,blocker):blocker;""")
    if 'int i=indexed?' not in source or 'int j=indexed?' not in source:
        raise ValueError('Missing full-scan city loops in selected shader')
    return source
