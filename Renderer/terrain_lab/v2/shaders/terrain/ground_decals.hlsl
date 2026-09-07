// Generic local triangle/UV data is supplied by the candidate fixture.
// Placement density and source-to-C3X scale are explicit Lab adaptations.
#ifndef Q2_GROUND_CELL_COUNT
#define Q2_GROUND_CELL_COUNT 50
#endif
#ifndef Q2_GROUND_PATCH_SCALE
#define Q2_GROUND_PATCH_SCALE .32
#endif
float q2_ground_hash(float2 cell,float salt) {
 float2 raw=float2(cell.x+cell.y,cell.x-cell.y);
 raw.x-=floor(raw.x/(2*Q2_GROUND_CELL_COUNT))*(2*Q2_GROUND_CELL_COUNT);
 uint x=asuint(int(raw.x)),y=asuint(int(raw.y));
 uint h=x*1664525u+y*1013904223u+uint(salt)*374761393u;
 h=(h^(h>>16))*2246822519u;h=(h^(h>>13))*3266489917u;
 return float((h^(h>>16))&0xffffffu)/16777216.0;
}
float q2_cross2(float2 a,float2 b) {return a.x*b.y-a.y*b.x;}
bool q2_ground_uv(int id,float2 q,out float2 uv,out float2 du,out float2 dv) {
 uv=0;du=0;dv=0;
 int first=ground_decal_ranges[id].x,count=ground_decal_ranges[id].y;
 [loop] for(int j=0;j<count;j+=3) {
  float4 a=ground_decal_vertices[first+j],b=ground_decal_vertices[first+j+1],c=ground_decal_vertices[first+j+2];
  float2 ab=b.xy-a.xy,ac=c.xy-a.xy,aq=q-a.xy;
  float det=q2_cross2(ab,ac);
  float s=q2_cross2(aq,ac)/det,t=q2_cross2(ab,aq)/det;
  if(s>=0&&t>=0&&s+t<=1) {
   uv=a.zw+s*(b.zw-a.zw)+t*(c.zw-a.zw);
   du=((b.zw-a.zw)*ac.y-(c.zw-a.zw)*ab.y)/det;
   dv=(-(b.zw-a.zw)*ac.x+(c.zw-a.zw)*ab.x)/det;
   return true;
  }
 }
 return false;
}
float4 q2_ground_field(Texture2D atlas,Texture2D coverage,float2 world_position,bool plains,bool height) {
 float cell_size=(Q3_MATERIAL_WRAP_WIDTH*.5)/Q2_GROUND_CELL_COUNT;
 float2 p=(world_position+float2(Q3_MATERIAL_ORIGIN_X,Q3_MATERIAL_ORIGIN_Y))/cell_size;
 float2 dx=ddx(p),dy=ddy(p),cell=floor(p);
 float4 accum=0;
 float total=0;
 [loop] for(int id=0;id<GROUND_DECAL_COUNT;id++)
  if(ground_decal_ranges[id].w==int(plains))total+=ground_decal_ranges[id].z;
 [loop] for(int oy=-1;oy<=1;oy++) [loop] for(int ox=-1;ox<=1;ox++) {
  float2 k=cell+float2(ox,oy);
  float choice=q2_ground_hash(k,plains?41:7)*total;
  int selected=-1;
  [loop] for(int id=0;id<GROUND_DECAL_COUNT;id++) {
   if(ground_decal_ranges[id].w!=int(plains))continue;
   choice-=ground_decal_ranges[id].z;
   if(choice<0){selected=id;break;}
  }
  if(selected<0)continue;
  float2 center=k+.5+(float2(q2_ground_hash(k,11),q2_ground_hash(k,13))-.5)*.42;
  float angle=q2_ground_hash(k,17)*6.2831853;
  float cs=cos(angle),sn=sin(angle);
  float4 info=ground_decal_placement[selected];
  float2 size=info.xy*info.z*(Q2_GROUND_PATCH_SCALE/cell_size)*(1+(q2_ground_hash(k,19)*2-1)*info.w);
  float2 rel=p-center;
  float2 q=float2(cs*rel.x+sn*rel.y,-sn*rel.x+cs*rel.y)/size+.5;
  if(any(q<0)||any(q>1))continue;
  float2 uv,du,dv;
  if(!q2_ground_uv(selected,q,uv,du,dv))continue;
  float2 qdx=float2(cs*dx.x+sn*dx.y,-sn*dx.x+cs*dx.y)/size;
  float2 qdy=float2(cs*dy.x+sn*dy.y,-sn*dy.x+cs*dy.y)/size;
  float2 ux=du*qdx.x+dv*qdx.y,uy=du*qdy.x+dv*qdy.y;
  float4 sampled=atlas.SampleGrad(decal_sampler,uv,ux,uy);
  float alpha=coverage.SampleGrad(decal_sampler,uv,ux,uy).a;
  if(height)sampled.rgb=(sampled.r-.5).xxx;
  accum.rgb=lerp(accum.rgb,sampled.rgb,alpha);
  accum.a=alpha+accum.a*(1-alpha);
 }
 // Return straight color to the existing material compositor.
 if(!height)accum.rgb/=max(accum.a,.00001);
 return accum;
}
