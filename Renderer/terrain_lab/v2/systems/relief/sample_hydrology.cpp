// Read-only consumer of Q3's declared field contract; no hydrology implementation.
#include "../hydrology/field.h"
#include <cstdio>
int main(int argc,char**argv){
 if(argc!=3)return 2;
 hydro::Field field(argv[1]);FILE*f=fopen(argv[2],"wb");if(!f)return 3;
 for(int y=0;y<257;y++)for(int x=0;x<257;x++){
  hydro::P p{-.5+x/256.*6-1,-.5+y/256.*6-1};auto s=field.sample(p);
  float a[]={float(s.shore_distance),float(s.rocky),float(s.height),float(s.river_distance),float(s.river_width)};
  fwrite(a,sizeof(float),5,f);
 }
 fclose(f);
 std::string path=std::string(argv[2])+".coast";f=fopen(path.c_str(),"w");if(!f)return 5;
 for(auto const&e:field.coast)fprintf(f,"%.9f,%.9f,%.9f,%.9f,%.9f\n",e.a.x,e.a.y,e.b.x,e.b.y,e.rocky);
 fclose(f);path=std::string(argv[2])+".exclusions";f=fopen(path.c_str(),"w");if(!f)return 6;
 for(auto const&e:field.exclusions())fprintf(f,"%.9f,%.9f,%.9f,%.9f,%.9f\n",e.a.x,e.a.y,e.b.x,e.b.y,e.clearance_radius);
 return fclose(f)==0?0:4;
}
