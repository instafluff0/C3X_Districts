#pragma once
// Explicit standalone fixture; never compiled into the injected game hooks.
// Coordinates are fixed in the world, so scrolling does not move the inputs.
inline void city_border_fixture(std::vector<c3x_renderer_tile_v1>& tiles,int map_width){
 char setting[160]={};
 if(!GetEnvironmentVariableA("C3X_RENDERER_CITY_BORDER_FIXTURE",setting,sizeof(setting)))return;
 int x=10,y=10,culture=0,era=0,size=0,capital=0,wall=0,seed=0;
 if(sscanf_s(setting,"%d,%d,%d,%d,%d,%d,%d,%d",&x,&y,&culture,&era,&size,&capital,&wall,&seed)!=8)return;
 int c=(x+y)/2,r=(x-y)/2;
 auto owner=[&](int column,int row){return column>=c-2&&column<=c+2&&row>=r-2&&row<=r+2?1:0;};
 for(auto& t:tiles){
  int tx=(t.tile_x%map_width+map_width)%map_width,ty=t.tile_y;
  int column=(tx+ty)/2,row=(tx-ty)/2;
  t.city_id=-1;t.resource_id=-1;t.improvement_flags=0;t.road_mask=t.railroad_mask=0;
  t.territory_owner_id=owner(column,row);t.territory_edge_mask=0;t.territory_color_rgb=0x22bfa5;
  if(t.territory_owner_id){
   if(!owner(column-1,row))t.territory_edge_mask|=1;
   if(!owner(column,row+1))t.territory_edge_mask|=2;
   if(!owner(column,row-1))t.territory_edge_mask|=4;
   if(!owner(column+1,row))t.territory_edge_mask|=8;
  }
  if(tx==x&&ty==y){
   t.city_id=71;t.city_owner_id=1;t.city_culture_group=culture;t.city_era=era;t.city_size=size;
   t.city_population=size==0?5:size==1?10:20;t.variant_seed=unsigned(seed);
   t.city_flags=(capital?C3X_RENDERER_CITY_CAPITAL:0)|(wall?C3X_RENDERER_CITY_WALLED:0);
   t.resource_id=1;t.resource_class=0;strcpy_s(t.resource_name,"Iron");
  }
 }
}
