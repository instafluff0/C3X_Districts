#pragma once
// Q3 local proposal v1. Coordinates are tile-center lattice units; positive shore
// distance is land. No clock, backend calls, source asset names or gameplay state.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace hydro {
constexpr unsigned contract_version=1;
struct P { double x=0,y=0; P operator+(P b)const{return{x+b.x,y+b.y};} P operator-(P b)const{return{x-b.x,y-b.y};} P operator*(double s)const{return{x*s,y*s};} };
inline double dot(P a,P b){return a.x*b.x+a.y*b.y;}
inline double length(P p){return std::sqrt(dot(p,p));}
inline double sat(double x){return std::clamp(x,0.,1.);}
inline double smooth(double a,double b,double x){x=sat((x-a)/(b-a));return x*x*(3-2*x);}
inline uint32_t hash(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
inline int mod(int a,int b){return (a%b+b)%b;}
struct Tile {int c=0,r=0,raw_x=0,raw_y=0,base=2,real=2;unsigned river=0;};
inline bool water(int base){return base>=11&&base<=13;}
struct Segment {P a,b;double rocky=0;};
struct River {uint64_t id=0;P a,b;std::vector<P> points;double width=.095;};
struct Crossing {uint64_t stable_id=0,edge_id=0;P point,tangent;double width=0,bed_height=0;};
struct ExclusionCapsule {uint64_t edge_id=0;P a,b;double water_radius=0,bank_radius=0,clearance_radius=0;};
struct Sample {double shore_distance=0,rocky=0,beach_width=0,wetness=0,depth=0,river_distance=1e6,river_width=0,height=0;};
struct Field {
 int cols=0,rows=0,map_width=0,map_height=0,origin_x=0,origin_y=0;bool wraps=false;
 std::map<std::pair<int,int>,Tile> tiles;
 std::vector<Segment> coast;
 std::vector<River> rivers;
 int shoreline_profile=0; // 0 preserves the original Q3 candidate.
 Field()=default;
 explicit Field(std::string const& path){load(path);build();}
 void load(std::string const& path){
  std::ifstream in(path);if(!in)throw std::runtime_error("missing hydrology tile fixture");
  std::string line;std::getline(in,line);std::replace(line.begin(),line.end(),',',' ');std::istringstream h(line);std::string magic;int count=0,halo_count=0;
  h>>magic>>cols>>rows>>count>>origin_x>>origin_y>>map_width>>map_height>>halo_count;
  if(magic!="C3X_BIQ_TERRAIN_WINDOW_V2"||cols<1||rows<1||count!=cols*rows||map_width<1)throw std::runtime_error("invalid hydrology fixture header");
  while(std::getline(in,line)){if(line.empty()||line[0]=='#')continue;std::replace(line.begin(),line.end(),',',' ');std::istringstream s(line);Tile t;unsigned bonus,overlay;
   if(!(s>>t.c>>t.r>>t.raw_x>>t.raw_y>>t.base>>t.real>>bonus>>overlay>>t.river))throw std::runtime_error("invalid hydrology tile");
   if(!tiles.emplace(std::make_pair(t.c,t.r),t).second)throw std::runtime_error("duplicate tile");
  }
  // Missing crop neighbors are not assumed water. A two-tile authoritative halo
  // is required; this rejects crop-edge ponds instead of hiding absent input.
  if(tiles.size()!=size_t(count+halo_count))throw std::runtime_error("tile/halo count mismatch");
  origin_x=tile(0,0).raw_x;origin_y=tile(0,0).raw_y;
  for(int y=-2;y<rows+2;y++)for(int x=-2;x<cols+2;x++)if(!tiles.count({x,y}))throw std::runtime_error("hydrology requires two-tile halo");
 }
 Tile const& tile(int x,int y)const {
  auto it=tiles.find({x,y});if(it==tiles.end())throw std::runtime_error("sample beyond supplied halo");return it->second;
 }
 double random(int x,int y,unsigned salt)const{
  // Smooth world fields use canonical raw coordinates, stable across crop lifts.
  int rx=origin_x+x+y,ry=origin_y+x-y;
  if(wraps)rx=mod(rx,map_width);
  return double(hash(uint32_t(rx)*73856093u^uint32_t(ry)*19349663u^salt))/4294967295.;
 }
 double noise(P p,unsigned salt)const{
  int x=int(std::floor(p.x)),y=int(std::floor(p.y));double u=smooth(0,1,p.x-x),v=smooth(0,1,p.y-y);
  return (1-v)*((1-u)*random(x,y,salt)+u*random(x+1,y,salt))+v*((1-u)*random(x,y+1,salt)+u*random(x+1,y+1,salt));
 }
 double occupancy(P p,bool hill=false)const{
  // Compact tensor cubic B-spline hides diamond ownership while retaining islands.
  auto kernel=[](double t){t=std::abs(t);return t<1?(4-6*t*t+3*t*t*t)/6:(t<2?std::pow(2-t,3)/6:0);};
  double sum=0,total=0;
  int x=int(std::floor(p.x)),y=int(std::floor(p.y));
  for(int j=y-1;j<=y+2;j++)for(int i=x-1;i<=x+2;i++){
   auto it=tiles.find({i,j});if(it==tiles.end())continue;
   double w=kernel((p.x-i)*1.25)*kernel((p.y-j)*1.25);total+=w;
   sum+=w*(hill?it->second.real==5:!water(it->second.base));
  }
  if(total<=0)throw std::runtime_error("empty sampling support");return sum/total;
 }
 double shore_rockiness(P p)const{return smooth(.10,.80,occupancy(p,true)/std::max(.001,occupancy(p)));}
 double world_noise(P p,double frequency,unsigned salt)const {
  // Canonical lattice origin makes every crop see the same field. Quantized
  // frequency closes exactly across the raw-X wrap; never seed per tile/crop.
  if(wraps)frequency=std::max(1.,std::round(frequency*map_width*.5))/(map_width*.5);
  double gx=(p.x+(origin_x+origin_y)*.5)*frequency;
  double gy=(p.y+(origin_x-origin_y)*.5)*frequency;
  int ix=int(std::floor(gx)),iy=int(std::floor(gy));
  double u=smooth(0,1,gx-ix),v=smooth(0,1,gy-iy);
  auto value=[&](int x,int y){int rx=x+y,ry=x-y;
   if(wraps)rx=mod(rx,int(std::round(map_width*frequency)));
   return double(hash(uint32_t(rx)*73856093u^uint32_t(ry)*19349663u^salt))/4294967295.;};
  return (1-v)*((1-u)*value(ix,iy)+u*value(ix+1,iy))+
         v*((1-u)*value(ix,iy+1)+u*value(ix+1,iy+1));
 }
 double signed_coverage(P p)const{
  if(shoreline_profile>=1){
   auto displacement=[&](unsigned seed){return
    (world_noise(p,.24,seed)-.5)*(shoreline_profile==2?.68:.42)+
    (world_noise(p,.82,seed+1)-.5)*(shoreline_profile==2?.38:.22)+
    (world_noise(p,2.40,seed+2)-.5)*(shoreline_profile==2?.16:.095);};
   P q=p+P{displacement(781),displacement(2183)};
   double shape=occupancy(q)-(.525+(world_noise(p,.38,8191)-.5)*.08);
   // Reserve a stable domain around authoritative centers. Irregular coast
   // cannot erase a playable land/water center or create an island offshore.
   int x=int(std::floor(p.x+.5)),y=int(std::floor(p.y+.5));
   auto t=tiles.find({x,y});
   if(t!=tiles.end()){
    double core=1-smooth(.16,.36,length(p-P{double(x),double(y)}));
    shape=shape*(1-core)+(water(t->second.base)?-.475:.475)*core;
   }
   return shape;
  }
  P q=p+P{noise(p,127)-.5,noise(p,923)-.5}*.18;
  return occupancy(q)-.525; // bounded erosion separates corner-only land contacts
 }
 uint64_t edge_id(P a,P b)const{
  auto node=[&](P p){int x=int(std::llround(2*(origin_x+p.x+p.y))),y=int(std::llround(2*(origin_y+p.x-p.y)));if(wraps)x=mod(x,2*map_width);return uint64_t(uint32_t(x))*0x100000000ull+uint32_t(y);};
  uint64_t x=node(a),y=node(b);if(x>y)std::swap(x,y);
  uint64_t id=14695981039346656037ull;for(uint64_t v:{x,y})for(int i=0;i<8;i++){id^=(v>>(i*8))&255;id*=1099511628211ull;}return id;
 }
 void build(){
  coast.clear();rivers.clear();double step=.125;
  // Marching triangles has no ambiguous saddle case; only one shared zero contour.
  auto tri=[&](P a,P b,P c){P p[3]={a,b,c},cut[3];double v[3]={signed_coverage(a),signed_coverage(b),signed_coverage(c)};int n=0;
   for(int i=0;i<3;i++){int j=(i+1)%3;if((v[i]>0)!=(v[j]>0))cut[n++]=p[i]+(p[j]-p[i])*(v[i]/(v[i]-v[j]));}
   if(n==2){P mid=(cut[0]+cut[1])*.5;double land=occupancy(mid);double hill=occupancy(mid,true)/std::max(.001,land);coast.push_back({cut[0],cut[1],smooth(.10,.80,hill)});}
  };
  for(double y=-1;y<rows;y+=step)for(double x=-1;x<cols;x+=step){P a{x,y},b{x+step,y},c{x+step,y+step},d{x,y+step};tri(a,b,c);tri(a,c,d);}
  std::map<uint64_t,River> unique;
  for(auto const& kv:tiles){auto t=kv.second;if(t.c<-1||t.c>cols||t.r<-1||t.r>rows)continue;
   unsigned bits[]={2,8,32,128};
   for(unsigned bit:bits)if(t.river&bit){P a,b;
    if(bit==2){a={t.c-.5,t.r+.5};b={t.c+.5,t.r+.5};}
    if(bit==8){a={t.c+.5,t.r-.5};b={t.c+.5,t.r+.5};}
    if(bit==32){a={t.c-.5,t.r-.5};b={t.c+.5,t.r-.5};}
    if(bit==128){a={t.c-.5,t.r-.5};b={t.c-.5,t.r+.5};}
    uint64_t id=edge_id(a,b);if(unique.count(id))continue;River r;r.id=id;r.a=a;r.b=b;
    // Single asymmetric Bezier bow, never a repeated two-frequency S template.
    double bend=(double(hash(uint32_t(id)))/4294967295.-.5)*.28;
    P d=b-a,n={-d.y,d.x};P p1=a+d*.33+n*bend,p2=a+d*.67+n*bend*.63;
    for(int i=0;i<=24;i++){double u=i/24.,v=1-u;r.points.push_back(a*(v*v*v)+p1*(3*v*v*u)+p2*(3*v*u*u)+b*(u*u*u));}
    unique[id]=r;
   }
  }
  for(auto const& kv:unique)rivers.push_back(kv.second);
  // Degree-two node tangents bisect the incident directions. Both incident
  // Beziers meet with a common tangent instead of exposing tile-corner elbows.
  for(auto& r:rivers){
   auto tangent=[&](P node,P toward){P other;int degree=0;
    for(auto const& e:rivers){if(length(e.a-node)<1e-9){degree++;if(e.id!=r.id)other=e.b;}if(length(e.b-node)<1e-9){degree++;if(e.id!=r.id)other=e.a;}}
    P t=toward-node;if(degree==2)t=t-(other-node);return t*(1/std::max(1e-12,length(t)));
   };
   P ta=tangent(r.a,r.b),tb=tangent(r.b,r.a),d=r.b-r.a,n{-d.y,d.x};
   double bend=(double(hash(uint32_t(r.id)))/4294967295.-.5)*.10;
   P c1=r.a+ta*.34,c2=r.b+tb*.34;
   r.points.clear();for(int i=0;i<=32;i++){double u=i/32.,v=1-u;
    r.points.push_back(r.a*(v*v*v)+c1*(3*v*v*u)+c2*(3*v*u*u)+r.b*(u*u*u)+n*(bend*16*u*u*v*v));}
  }
 }
 double segment_distance(P p,P a,P b,P* nearest=nullptr)const{P ab=b-a;double t=sat(dot(p-a,ab)/std::max(1e-15,dot(ab,ab)));P q=a+ab*t;if(nearest)*nearest=q;return length(p-q);}
 Sample sample(P p)const{
  Sample s;double nearest=1e6;P foot=p;
  for(auto const& e:coast){P q;double d=segment_distance(p,e.a,e.b,&q);if(d<nearest){nearest=d;foot=q;s.rocky=e.rocky;}}
  s.rocky=shore_rockiness(foot);
  s.shore_distance=nearest*(signed_coverage(p)>=0?1:-1);
  // Width varies ALONG the contour; offshore depth cannot reverse due to noise.
  s.beach_width=(.16+.20*noise(foot,628))*(1-s.rocky);
  if(shoreline_profile>=1)
   s.beach_width=(.065+.18*world_noise(foot,.66,628))*(1-s.rocky);
  double offshore=std::max(0.,-s.shore_distance);
  s.depth=.46*(1-std::exp(-offshore/.85));
  s.wetness=1-smooth(-.035,.13,s.shore_distance);
  s.height=s.shore_distance>0?.08*smooth(0,.35,s.shore_distance):-s.depth;
  for(auto const& r:rivers)for(size_t i=1;i<r.points.size();i++){
   double d=segment_distance(p,r.points[i-1],r.points[i]);if(d<s.river_distance){s.river_distance=d;s.river_width=r.width;}
  }
  // Union of coast and rivers prevents a sand/terrain plug at mouths/junctions.
  double rd=s.river_distance-s.river_width*.5;
  if(rd<.16&&s.shore_distance>-.18){double carve=1-smooth(-.012,.16,rd);s.height=s.height*(1-carve)-.045*carve;}
  return s;
 }
 std::vector<ExclusionCapsule> exclusions(double margin=.04)const {
  if(margin<0)throw std::runtime_error("negative clearance margin");
  std::vector<ExclusionCapsule> out;
  // Match the shader's outer water antialias and bank material support exactly.
  for(auto const&r:rivers)for(size_t i=1;i<r.points.size();i++)
   out.push_back({r.id,r.points[i-1],r.points[i],r.width*.5+.009,r.width*.5+.043,r.width*.5+.043+margin});
  return out;
 }
 bool intersects_footprint(std::vector<P> const& polygon,double margin=.04)const {
  if(polygon.size()<3)throw std::runtime_error("footprint needs convex boundary vertices");
  auto cross=[](P a,P b){return a.x*b.y-a.y*b.x;};
  auto inside=[&](P q){bool pos=false,neg=false;for(size_t i=0;i<polygon.size();i++){double v=cross(polygon[(i+1)%polygon.size()]-polygon[i],q-polygon[i]);pos|=v>1e-10;neg|=v< -1e-10;}return !(pos&&neg);};
  for(auto const& cap:exclusions(margin)){
   if(inside(cap.a)||inside(cap.b))return true;
   for(size_t i=0;i<polygon.size();i++){
    P a=polygon[i],b=polygon[(i+1)%polygon.size()],d=b-a,e=cap.b-cap.a;
    double det=cross(d,e);
    if(std::abs(det)>1e-12){double t=cross(cap.a-a,e)/det,u=cross(cap.a-a,d)/det;if(t>=0&&t<=1&&u>=0&&u<=1)return true;}
    double distance=std::min({segment_distance(a,cap.a,cap.b),segment_distance(b,cap.a,cap.b),segment_distance(cap.a,a,b),segment_distance(cap.b,a,b)});
    if(distance<=cap.clearance_radius+1e-9)return true;
   }
  }return false;
 }
 std::vector<Crossing> crossings(P a,P b)const{
  std::vector<Crossing> out;P d=b-a;auto cross=[](P x,P y){return x.x*y.y-x.y*y.x;};
  for(auto const& r:rivers)for(size_t i=1;i<r.points.size();i++){
   P p=r.points[i-1],e=r.points[i]-p;double det=cross(d,e);if(std::abs(det)<1e-12)continue;
   double t=cross(p-a,e)/det,u=cross(p-a,d)/det;
   if(t>=0&&t<=1&&u>=0&&u<1){P point=a+d*t;bool duplicate=false;for(auto const& c:out)if(c.edge_id==r.id&&length(c.point-point)<1e-7)duplicate=true;
    if(!duplicate)out.push_back({r.id^edge_id(a,b),r.id,point,e*(1/length(e)),r.width,-.045});}
  }return out;
 }
};
}
