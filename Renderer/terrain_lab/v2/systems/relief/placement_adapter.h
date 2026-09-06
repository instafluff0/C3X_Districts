#pragma once
// Q4 consumer for Q0 PlacementHooksV1. Inputs are all actual posed vertices.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <vector>
namespace q4_placement {
struct P {double x,y;};
struct Envelope {double margin,wx,wy;std::vector<P> ring;};
inline std::vector<Envelope> envelopes;
inline double cross(P a,P b,P c){return (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);}
inline double distance(P p,P a,P b){double x=b.x-a.x,y=b.y-a.y,n=x*x+y*y,t=n?std::clamp(((p.x-a.x)*x+(p.y-a.y)*y)/n,0.,1.):0;return std::hypot(p.x-a.x-t*x,p.y-a.y-t*y);}
inline bool inside(P p,std::vector<P> const& ring){bool hit=false;for(size_t i=0,j=ring.size()-1;i<ring.size();j=i++) {auto a=ring[i],b=ring[j];if((a.y>p.y)!=(b.y>p.y) && p.x<(b.x-a.x)*(p.y-a.y)/(b.y-a.y)+a.x)hit=!hit;}return hit;}
inline bool segments(P a,P b,P c,P d){return cross(a,b,c)*cross(a,b,d)<=0 && cross(c,d,a)*cross(c,d,b)<=0 && std::max(std::min(a.x,b.x),std::min(c.x,d.x))<=std::min(std::max(a.x,b.x),std::max(c.x,d.x)) && std::max(std::min(a.y,b.y),std::min(c.y,d.y))<=std::min(std::max(a.y,b.y),std::max(c.y,d.y));}
inline std::vector<P> hull(std::vector<P> p){
 std::sort(p.begin(),p.end(),[](P a,P b){return a.x<b.x || (a.x==b.x && a.y<b.y);});
 p.erase(std::unique(p.begin(),p.end(),[](P a,P b){return a.x==b.x && a.y==b.y;}),p.end());
 if(p.size()<3)return p;std::vector<P> h;
 for(auto v:p){while(h.size()>1 && cross(h[h.size()-2],h.back(),v)<=0)h.pop_back();h.push_back(v);}
 size_t lower=h.size();for(size_t i=p.size()-1;i-->0;){auto v=p[i];while(h.size()>lower && cross(h[h.size()-2],h.back(),v)<=0)h.pop_back();h.push_back(v);}h.pop_back();return h;
}
inline void initialize(char const*,char const* fixture_json){
 envelopes.clear();std::ifstream f(fixture_json);std::string text((std::istreambuf_iterator<char>(f)),{});std::smatch match;
 std::regex index_key("\"path\"\\s*:\\s*\"([^\"]*/q4-clearance\\.json)\"");
 if(!std::regex_search(text,match,index_key))throw std::runtime_error("Q4 placement requires pinned clearance sidecar");
 std::ifstream index(match[1].str());text.assign(std::istreambuf_iterator<char>(index),{});
 std::regex key("\"relief_clearance\"\\s*:\\s*\\{\\s*\"path\"\\s*:\\s*\"([^\"]+)\"");
 if(!std::regex_search(text,match,key))throw std::runtime_error("Q4 clearance cache reference missing");
 std::ifstream input(match[1].str());std::string line;std::getline(input,line);
 if(line!="C3X_Q4_CLEARANCE_V1")throw std::runtime_error("Q4 clearance input invalid");
 while(std::getline(input,line)){
  if(line.empty())continue;std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);Envelope e;unsigned n;
  if(!(row>>e.margin>>e.wx>>e.wy>>n)||n<3||n>100000)throw std::runtime_error("Q4 clearance polygon invalid");
  for(unsigned i=0;i<n;i++){P p;if(!(row>>p.x>>p.y)||!std::isfinite(p.x)||!std::isfinite(p.y))throw std::runtime_error("Q4 clearance vertex invalid");e.ring.push_back(p);}
  if(!std::isfinite(e.margin)||e.margin<0||!std::isfinite(e.wx)||!std::isfinite(e.wy))throw std::runtime_error("Q4 clearance extent invalid");
  envelopes.push_back(e);
 }
}
inline bool accept_vegetation(char const*,char const*,unsigned,unsigned,float const* xyz,unsigned count){
 if(!xyz||count<3)throw std::runtime_error("Q4 placement needs actual geometry");
 std::vector<P> points;for(unsigned i=0;i<count;i++){P p{xyz[i*3],xyz[i*3+1]};if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(xyz[i*3+2]))throw std::runtime_error("Q4 source geometry nonfinite");points.push_back(p);}points=hull(points);
 for(auto const&e:envelopes)for(int wrap=-1;wrap<=1;wrap++){
  if(wrap && e.wx==0 && e.wy==0)continue;std::vector<P> ring=e.ring;for(auto&p:ring){p.x+=wrap*e.wx;p.y+=wrap*e.wy;}
  for(auto p:points)if(inside(p,ring))return false;for(auto p:ring)if(inside(p,points))return false;
  for(size_t i=0;i<points.size();i++)for(size_t j=0;j<ring.size();j++){
   P a=points[i],b=points[(i+1)%points.size()],c=ring[j],d=ring[(j+1)%ring.size()];
   if(segments(a,b,c,d)||std::min({distance(a,c,d),distance(b,c,d),distance(c,a,b),distance(d,a,b)})<=e.margin)return false;
  }
 }
 return true;
}
}
