"""Source provenance and executable numerical contracts for the r13 port."""
from pathlib import Path
import hashlib,json,subprocess,tempfile,unittest
from .prepare import ROOT,HERE,LAB,function,terrain_boundaries
from Renderer.lab.test_natural import NaturalInputs
class Contract(unittest.TestCase):
    def test_rock_crevices_preserve_flat_material_and_follow_local_depth(self):
        shader=(LAB/'shaders/relief/beauty_mountain.hlsl').read_text()
        source='''
#include <algorithm>
#include <cassert>
#include <cmath>
float saturate(float x) {return std::clamp(x,0.f,1.f);}
'''+function(shader,'rock_crevice_visibility')+'''
int main() {
    for(float offset:{0.f,.2f,.5f}) {
        assert(rock_crevice_visibility(offset,offset)==1);
        assert(rock_crevice_visibility(offset+.1f,offset)==1);
        float previous=1;
        for(int i=0;i<=100;i++) {
            float depth=i/1000.f;
            float visibility=rock_crevice_visibility(offset,offset+depth);
            assert(visibility>0 && visibility<=previous+.00001f);
            assert(std::abs(visibility-rock_crevice_visibility(0,depth))<.00001f);
            previous=visibility;
        }
        assert(previous<.8f);
    }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            cpp=Path(directory)/'crevices.cpp';binary=Path(directory)/'crevices'
            cpp.write_text(source)
            subprocess.run(['c++','-std=c++17',str(cpp),'-o',str(binary)],
                           check=True,capture_output=True,text=True)
            subprocess.run([str(binary)],check=True,capture_output=True,text=True)

    def test_mountain_material_coverage_is_independent_of_source_mask_and_face(self):
        # Execute the production scalar mask, not a second copy of its formula.
        # Equal rise must give equal coverage for differing source footprints,
        # slope normals and underlying hill elevations.
        shader=(LAB/'shaders/relief/beauty_mountain.hlsl').read_text()
        start=shader.index('    float specular_map;',shader.index('Output shade('))
        end=shader.index('#ifdef BEAUTY_TERRAIN_TRANSITIONS\n    // 42',start)
        mask=shader[start:end]
        source='''
#include <algorithm>
#include <cassert>
#include <cmath>
float max(float a,float b) {return std::max(a,b);}
float saturate(float x) {return std::clamp(x,0.f,1.f);}
float lerp(float a,float b,float t) {return a+(b-a)*t;}
float smoothstep(float a,float b,float x) {
    float t=std::clamp((x-a)/(b-a),0.f,1.f);return t*t*(3-2*t);
}
struct Vector {float x=0,y=0,z=0,w=0;};
Vector normalize(Vector v) {return v;}
struct Pixel {Vector world,normal,material;float base_relief;};
float coverage(float rise,float footprint,float normal_z,float base) {
    Pixel input{};input.world.z=base+2.5f/112.f+rise;
    input.base_relief=base;input.material.z=footprint;
    input.normal={std::sqrt(1-normal_z*normal_z),0,normal_z,0};
    Vector geometric=input.normal;
#define BEAUTY_TERRAIN_TRANSITIONS 1
'''+mask+'''
    assert(std::abs(rock_albedo_coverage-rock_detail_coverage)<0.00001f);
    return rock_albedo_coverage;
}
int main() {
    float previous=-1;
    for(int height=0;height<=120;height++) {
        float rise=height/100.f;
        float expected=coverage(rise,0,.5f,0);
        assert(expected>=previous && expected>=0 && expected<=1);
        previous=expected;
        for(int blend=0;blend<=10;blend++)
            for(int face=1;face<=10;face++)
                for(float base:{0.f,.25f,.75f})
                    assert(std::abs(coverage(rise,blend/10.f,face/10.f,base)-expected)<0.00001f);
    }
    assert(coverage(0,1,.1f,0)==0);
    assert(coverage(.14f,1,.1f,0)>0 && coverage(.14f,1,.1f,0)<1);
    assert(coverage(1,0,.9f,0)==1);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            cpp=Path(directory)/'coverage.cpp';binary=Path(directory)/'coverage'
            cpp.write_text(source)
            subprocess.run(['c++','-std=c++17',str(cpp),'-o',str(binary)],
                           check=True,capture_output=True,text=True)
            subprocess.run([str(binary)],check=True,capture_output=True,text=True)

    def test_selected_provider_equations(self):
        for name,category in [('terrain','relief'),('mountain','relief'),('objects','objects')]:
            selected=(LAB/f'shaders/{category}/beauty_{name}.hlsl').read_text()
            if name=='terrain':selected=terrain_boundaries(selected)
            adapted=(HERE/f'{name}.hlsl').read_text()
            self.assertEqual(function(selected,'shade'),function(adapted,'shade'))
            self.assertIn('float3 light_direction = ShadowL.xyz;',adapted)
            self.assertNotIn('#include',adapted)
    def test_selected_source_pins(self):
        p=json.loads((HERE/'provenance.json').read_text())
        self.assertEqual(p['authority'],'source-fidelity-r13/inland')
        self.assertEqual((p['trees'],p['recipes'],p['count_weight']),(22,25,180))
        for relative,expected in p['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(),expected,relative)
    def test_executable_light_and_river_contract(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'probe'
            subprocess.run(['c++','-std=c++17','-O2',str(HERE/'contract_probe.cpp'),str(HERE.parent/'environment_runtime.cpp'),'-o',str(out)],check=True,capture_output=True)
            r=subprocess.run([str(out),str(LAB/'hydrology/contract-terrain.csv')],check=True,capture_output=True,text=True)
            self.assertIn('samples=9216 exact equality',r.stdout)
            self.assertIn('phases=24 noon up-right',r.stdout)
if __name__=='__main__':unittest.main()
