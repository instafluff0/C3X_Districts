"""Opt-in native Lab profile integrating the independent unit studio findings.

The case builds a pinned private DLL. There is no production staging path.
"""
from pathlib import Path
import argparse,hashlib,json
from PIL import Image,ImageDraw
from . import material_study as material
from .prepare import ROOT,read
HERE=Path(__file__).resolve().parent
OUT=ROOT/'Renderer/lab/out/units/studio-native'
PACK=ROOT/'Renderer/packs/UnitStudioLab'


def prepare():
    material.ROSTER=True;material.OUT=OUT;material.SNAP=OUT/'source/Renderer';material.PACK=PACK
    material.FRAME_PACK=ROOT/'Renderer/lab/out/units/source-roster-frame-pack'
    material.prepare_pack();material.snapshot()
    native=material.SNAP/'native';replace=material.replace
    bindings=read(PACK/'bindings.json');production=read(material.SOURCE/'bindings.json');manifest=read(material.SOURCE/'manifest.json')
    for column,(slug,_,_) in enumerate(material.SUBJECTS):
        old=next(v for v in production.values() if isinstance(v,dict) and v.get('key0') in manifest['units']['unit/'+slug]['civ3_ids'])
        for row in range(3):
            record=bindings['unit'+str(row*6+column)];record['studio_profile']=row;record['sample_scale']=1 if row==0 else 4
            if row==0:record.update(scale=old['scale'],offset_z=old['offset_z'])
            if row==1:
                for action in manifest['units']['unit/'+slug]['actions']:
                    for i in range(record[action]['part_count']):record[action]['part'+str(i)].pop('normal_texture',None)
    (PACK/'bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
    body=native/'unit_body_renderer.h';loader=native/'c3x_renderer.cpp';shader=native/'environment_refresh/unit_shader.h';shadow=native/'unit_shadow.h';transfer=native/'render_core/linear_target.h'
    replace(body,'int sample_scale=1;','int sample_scale=1,studio_profile=0;')
    replace(loader,'sample_scale!=1 && sample_scale!=2','sample_scale!=1 && sample_scale!=2 && sample_scale!=4')
    replace(loader,'unit.sample_scale=int(sample_scale);','unit.sample_scale=int(sample_scale);\n            float profile=0;json_number_after(data,"studio_profile",location,profile);unit.studio_profile=int(profile);')
    replace(body,'(samples!=1 && samples!=2) || !ensure(device,w,h,samples)','(samples!=1 && samples!=2 && samples!=4) || !ensure(device,w,h,samples,found->studio_profile?1536:128)')
    replace(body,'bool ensure(ID3D11Device* device,int w,int h,int samples)', 'bool ensure(ID3D11Device* device,int w,int h,int samples,int shadow_extent)')
    replace(body,'UnitShadow shadow;','UnitShadow shadow(found->studio_profile?1536:128);')
    replace(shadow,'static constexpr int extent=128;\n    std::array<float,extent*extent> heights{};', 'int extent;\n    std::vector<float> heights;\n    explicit UnitShadow(int size=128):extent(size),heights(std::size_t(size)*size){}')
    replace(shadow,'float light_x,float light_y)', 'float light_x,float light_y,float light_z=0)')
    replace(shadow,'dy=length>1e-5f?-light_y/length*scale:0;', 'dy=length>1e-5f?-light_y/length*scale:0;\n        if(light_z>0){dx=light_x/light_z;dy=light_y/light_z;}')
    replace(shadow,'heights.fill(-1.f);','std::fill(heights.begin(),heights.end(),-1.f);')
    replace(body,'        context->UpdateSubresource(beauty_frame,0,nullptr,beauty,0,0);','''        if(found->studio_profile) {
            beauty[0]=-0.281581295f;beauty[1]=-0.522936691f;beauty[2]=0.804517987f;beauty[3]=1;
            beauty[4]=3.2f;beauty[5]=3.05f;beauty[6]=2.9f;
            beauty[8]=.42f;beauty[9]=.50f;beauty[10]=.66f;beauty[11]=1;
            beauty[12]=.57922797f;beauty[13]=-.57922797f;beauty[14]=.57357644f;
        }
        beauty[16]=float(shadow.extent);beauty[17]=float(found->studio_profile);
        context->UpdateSubresource(beauty_frame,0,nullptr,beauty,0,0);''')
    replace(body,'shadow.fit(all_points,light[0],light[1])','shadow.fit(all_points,beauty[0],beauty[1],found->studio_profile?beauty[2]:0)')
    replace(body,'UnitShadow::extent*4','shadow.extent*4')
    replace(body,'(x+y)*32*zoom-z*(150.f*128/224)*zoom;','''(x+y)*32*zoom-z*(150.f*128/224)*zoom;
                if(found->studio_profile) {
                    // Physical orthographic source camera: elevation 35, yaw -45.
                    constexpr float studio_pixels=104.637819f; // Z projects to 150*128/224 pixels.
                    sx=float(pose.anchor_x-request.body_x)+(x+y)*.70710678f*studio_pixels*zoom;
                    sy=float(pose.anchor_y-request.body_y)-((-x+y)*.40557979f+z*.81915204f)*studio_pixels*zoom;
                }''')
    replace(body,'                upload[i]={2*sx/w-1,','''                if(found->studio_profile)normal={p.normal[0]*cosine-p.normal[1]*sine,
                    p.normal[0]*sine+p.normal[1]*cosine,p.normal[2]};
                upload[i]={2*sx/w-1,''')
    replace(body,'.5f-(x+y)*.05f-z*.001f,','found->studio_profile?.5f-(x*.57922797f-y*.57922797f+z*.57357644f)*.1f:.5f-(x+y)*.05f-z*.001f,')
    replace(body,'                    for(unsigned axis=0;axis<3;++axis)upload[i][11+basis*3+axis]=direction[axis];','''                    if(found->studio_profile)direction={f[0]*cosine-f[1]*sine,f[0]*sine+f[1]*cosine,f[2]};
                    for(unsigned axis=0;axis<3;++axis)upload[i][11+basis*3+axis]=direction[axis];''')
    replace(body,'context->PSSetSamplers(0,1,&samplers[part.address]);','context->PSSetSamplers(0,1,&samplers[part.address+(found->studio_profile?4:0)]);')
    replace(body,'samplers[4]', 'samplers[8]')
    replace(body,'mode<4 && SUCCEEDED(hr)','mode<8 && SUCCEEDED(hr)')
    replace(body,'s.AddressW=D3D11_TEXTURE_ADDRESS_WRAP;s.MaxLOD=D3D11_FLOAT32_MAX;','s.AddressW=D3D11_TEXTURE_ADDRESS_WRAP;s.MaxLOD=D3D11_FLOAT32_MAX;\n                if(mode>=4){s.Filter=D3D11_FILTER_MIN_MAG_MIP_LINEAR;s.MinLOD=s.MaxLOD=0;}')
    replace(body,'ID3D11Texture2D* shadow_texture=nullptr;', 'int shadow_size=0;ID3D11Texture2D* shadow_texture=nullptr;')
    replace(body,'if(!shadow_texture) {','if(!shadow_texture || shadow_size!=shadow_extent) {\n            release(shadow_view);release(shadow_texture);shadow_size=shadow_extent;')
    replace(body,'d.Width=d.Height=UnitShadow::extent;','d.Width=d.Height=shadow_extent;')
    replace(body,'transfer.draw(context,linear,target,environment.exposure,samples);','transfer.draw(context,linear,target,found->studio_profile?1.f:environment.exposure,samples,nullptr,0,0,found->studio_profile?1.f:0.f);')
    replace(transfer,'rgb/=1+max(rgb.r,max(rgb.g,rgb.b));','if(padding.x<.5)rgb/=1+max(rgb.r,max(rgb.g,rgb.b));')
    replace(transfer,'UINT reconstructed_height=0)', 'UINT reconstructed_height=0,float display_mode=0)')
    replace(transfer,'values={exposure,render_scale,{0,0}};', 'values={exposure,render_scale,{display_mode,0}};')
    replace(body,'unsigned shade=unsigned(255*lighting::c3x_dynamic_shadow_opacity*environment.shadow_strength*fade*shadow.coverage((sx+sy)*.5f,(sy-sx)*.5f));','''float ground_x=(sx+sy)*.5f,ground_y=(sy-sx)*.5f;
            if(found->studio_profile) {
                float sum=(float(x)+.5f-float(pose.anchor_x-request.body_x))/(104.637819f*.70710678f*zoom);
                float difference=(float(y)+.5f-float(pose.anchor_y-request.body_y))/(104.637819f*.40557979f*zoom);
                ground_x=(sum+difference)*.5f;ground_y=(sum-difference)*.5f;
            }
            unsigned shade=unsigned(255*(found->studio_profile?.16f:lighting::c3x_dynamic_shadow_opacity*environment.shadow_strength)*fade*shadow.coverage(ground_x,ground_y));''')
    # The studio ground uses 0.68 linear radiance in shade. Native alpha blits
    # into display-space backgrounds; 0.16 is its neutral-gray display match.
    replace(shader,'float4 PS(Output i):SV_Target {',(HERE/'studio_response.hlsl').read_text()+'\nfloat4 PS(Output i):SV_Target {')
    replace(shader,'i.tangent*xy.x+i.bitangent*xy.y+i.n*z','i.tangent*xy.x+i.bitangent*xy.y+(Quality.y>1.5?n:i.n)*z')
    replace(shader,'i.shadow.yz*128','i.shadow.yz*Quality.x')
    replace(shader,'all(q<128)','all(q<int(Quality.x))')
    replace(shader,'float ao=channels.x>.5?lerp', 'if(Quality.y>1.5)return float4(studio_unit_response(i,albedo,n,1-occluded),1);\n float ao=channels.x>.5?lerp')
    preview=material.SNAP/'lab/native_preview.cpp'
    replace(preview,'"Current material, 2x sampling","Authored normal detail, 2x sampling","Normal detail + source alpha tint, 2x sampling"',
            '"Old Lab: original fit, shader and sampling","Old material: matched anatomy, studio camera and 4x sampling","New Lab: integrated studio materials and rendering"')
    (OUT/'profile.json').write_text(json.dumps({'profile':'native_unit_studio_lab','production_enabled':False,'ground_shadow_adaptation':'0.68 linear studio shade represented as 0.16 display-alpha on neutral backdrop','shadow_extent':1536,'sample_scale':4,'camera_elevation':35,'camera_yaw':-45,'rows':['old_lab','old_material_matched_conditions','studio_native']},indent=2)+'\n')


def comparison(images):
    outputs=[]
    for entry in images:
        path=ROOT/entry['image'];im=Image.open(path).convert('RGB');im.save(path.with_suffix('.png'))
        # Retain source pixels: rearrange rows into paired columns, never enlarge.
        for mode,left_row in [('before-after',0),('matched-material',1)]:
            panel=Image.new('RGB',(1240,610),(47,52,54));draw=ImageDraw.Draw(panel)
            draw.text((15,12),'OLD LAB' if left_row==0 else 'OLD MATERIAL / MATCHED STUDIO CONDITIONS',fill='white')
            draw.text((640,12),'NEW LAB / INTEGRATED STUDIO',fill='white')
            draw.text((15,28),f"{entry['case']} | tile width {entry['zoom']} px | captured pixels at 1:1",fill=(205,215,220))
            for i,(slug,_,_) in enumerate(material.SUBJECTS):
                row,col=divmod(i,3)
                for side,source_row in enumerate((left_row,2)):
                    crop=im.crop((i*200,source_row*290+28,i*200+200,source_row*290+288))
                    panel.paste(crop,(side*620+10+col*200,45+row*280))
            target=path.with_name(path.stem+'-'+mode+'.png');panel.save(target);outputs.append(str(target.relative_to(ROOT)))
    return outputs


def run(cases=('sizing',),zooms=(128,192)):
    from Renderer.lab.platform import native_command_result
    from Renderer.renderer import native_render
    OUT.mkdir(parents=True,exist_ok=True)
    monitored=[ROOT/'Renderer/bin/C3XRenderer.dll',material.SOURCE/'bindings.json',material.SOURCE/'manifest.json']
    before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in monitored}
    # Keep the pre-promotion studio A/B reproducible after the generic game
    # renderer gains the decoder. The pinned snapshot is the reference fixture;
    # current game rendering uses ordinary detail/gameplay category cases.
    promoted='material_model' in (ROOT/'Renderer/native/unit_body_renderer.h').read_text()
    if promoted:
        if not (OUT/'source/Renderer/native/C3XRenderer.dll').is_file():
            raise ValueError('The retained pre-promotion studio snapshot is missing; use current detail/gameplay Lab cases')
        material.SNAP=OUT/'source/Renderer'
    else:prepare()
    result=native_command_result('Renderer/lab','call out\\units\\studio-native\\build.bat',timeout_seconds=120)
    if result['status']!='pass':raise ValueError('Native studio Lab build failed: '+str(result))
    images=[]
    for case in cases:
        for zoom in zooms:
            images.append(native_render('units',case,12,zoom,OUT/case,candidate=material.SNAP/'native/C3XRenderer.dll',preview=material.SNAP/'lab/native_preview.cpp.exe'))
    panels=comparison(images)
    unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in before.items())
    receipt={'images':images,'comparisons':panels,'production_before':before,'production_unchanged':unchanged,'profile':read(OUT/'profile.json')}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not unchanged:raise ValueError('Production inputs changed during the isolated Lab run; see receipt.json')
    print('PASS native Lab studio comparison; production unchanged:',unchanged)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-only',action='store_true');p.add_argument('--all-cases',action='store_true');a=p.parse_args()
    if a.prepare_only:prepare()
    else:run(('sizing','sizing-gameplay','sizing-move') if a.all_cases else ('sizing',))
