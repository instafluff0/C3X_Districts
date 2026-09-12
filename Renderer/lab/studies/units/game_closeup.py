"""Compare the installed game DLL with the retained independent Warrior studio.

Only private geometry magnification/canvas changes; production is read-only.
Run with the local renderer Python and Windows VM available.
"""
from pathlib import Path
import copy, hashlib, json, os, shutil, struct
from .prepare import ROOT, read
OUT=ROOT/'Renderer/lab/out/units/game-closeup'
PACK=ROOT/'Renderer/packs/UnitGameCloseup'
SOURCE=ROOT/'Renderer/packs/UnitAnimationFidelity'
REFERENCE=ROOT/'Renderer/lab/out/units/studio/warrior-e35-a-45-s1400.png'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def magnify(blob, factor=4):
    """Uniformly scale positions and palette translations, preserving all other bytes."""
    result=bytearray(blob)
    version,n,ni,nb,nf=struct.unpack_from('<5I',blob,8)
    assert version in (1,2)
    stride=88 if version==2 else 64
    offsets=[32+i*stride for i in range(n)]
    offsets += [32+n*stride+ni*4+i*64+48 for i in range(nb*nf)]
    for offset in offsets:
        values=struct.unpack_from('<3f',blob,offset)
        struct.pack_into('<3f',result,offset,*(v*factor for v in values))
    # Power-of-two scaling is reversible for these finite source values.
    inverse=bytearray(result)
    for offset in offsets:
        struct.pack_into('<3f',inverse,offset,*(v/factor for v in struct.unpack_from('<3f',result,offset)))
    assert bytes(inverse)==blob
    return bytes(result)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);PACK.mkdir(parents=True,exist_ok=True)
    source=next(v for v in read(SOURCE/'bindings.json').values() if isinstance(v,dict) and v.get('key0')=='PRTO_Warrior')
    record=copy.deepcopy(source)
    # 4x geometric units * 1.25x authored scale * 2x native projection = 10x.
    record.update(scale=source['scale']*1.25,offset_z=source['offset_z']*4,minimum_canvas=512)
    consumed={};payloads={}
    for action in record.values():
        if not isinstance(action,dict) or 'part_count' not in action:continue
        for i in range(action['part_count']):
            part=action['part'+str(i)]
            for key,value in list(part.items()):
                if key not in ('mesh','texture') and not key.endswith('_texture'):continue
                src=SOURCE/value;consumed[str(src.relative_to(ROOT))]=sha(src)
                if key=='mesh':
                    if value not in payloads:
                        blob=magnify(src.read_bytes());name='clips/'+hashlib.sha256(blob).hexdigest()+'.bin'
                        dst=PACK/name;dst.parent.mkdir(exist_ok=True);dst.write_bytes(blob);payloads[value]=name
                    part[key]=payloads[value]
                else:
                    dst=PACK/value;dst.parent.mkdir(exist_ok=True);shutil.copyfile(src,dst)
    (PACK/'bindings.json').write_text(json.dumps({'unit_count':1,'unit0':record},indent=2)+'\n')
    witness=(ROOT/'Renderer/lab/native_preview.cpp').read_text()
    start=witness.index('bool lab_compose_units(HMODULE',witness.index('#undef main'))
    end=witness.index('\nint main(',start)
    body=r'''bool lab_compose_units(HMODULE module, char const* image_path, int hour, int,
                       c3x_renderer_output_v1 const& terrain) {
    auto draw=reinterpret_cast<c3x_renderer_unit_draw_expanded_fn>(GetProcAddress(module,"c3x_renderer_unit_draw_expanded"));
    HDC dc=CreateCompatibleDC(nullptr);BITMAPINFO info={};
    info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);info.bmiHeader.biWidth=terrain.width;
    info.bmiHeader.biHeight=-terrain.height;info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;
    void* pixels=nullptr;HBITMAP bitmap=CreateDIBSection(dc,&info,DIB_RGB_COLORS,&pixels,nullptr,0);
    if(!dc || !bitmap || !pixels || !draw)return false;
    auto previous=SelectObject(dc,bitmap);
    std::fill_n(static_cast<unsigned*>(pixels),std::size_t(terrain.width)*terrain.height,0xff8b9796u);
    c3x_renderer_unit_v1 unit={};unit.struct_size=sizeof(unit);unit.unit_id=1;
    strcpy_s(unit.unit_key,"PRTO_Warrior");unit.action=1;unit.action_cursor=0;unit.frame_count=16;
    unit.direction=4;unit.hour=hour;unit.display_color_rgb=0x205bdd;
    unit.sprite_width=unit.sprite_height=512;unit.projection_scale_milli=2000;
    unit.body_x=terrain.width/2-512;unit.body_y=720-512;
    unit.presentation_frequency=1000;unit.presentation_time_ticks=0;
    int bounds[4]={};bool ok=true;
    for(int direction=1;direction<=8 && ok;++direction){
    unit.direction=direction;
    std::fill_n(static_cast<unsigned*>(pixels),std::size_t(terrain.width)*terrain.height,0xff8b9796u);
    ok=draw(&unit,dc,dc,bounds)==C3X_RENDERER_RESULT_OK;GdiFlush();
    if(ok){c3x_renderer_output_v1 output={};output.width=terrain.width;output.height=terrain.height;
        output.stride_bytes=terrain.width*4;output.bgra_pixels=pixels;
        char path[2048]={};sprintf_s(path,"%s-direction%d.bmp",image_path,direction);ok=write_bmp(path,output);
        if(direction==4 && ok)ok=write_bmp(image_path,output);}
    }
    SelectObject(dc,previous);DeleteObject(bitmap);DeleteDC(dc);
    std::printf("%s category unit study: 8 magnified Warrior headings through installed DLL; native body API\n",ok?"PASS":"FAIL");
    return ok;
}
'''
    witness=witness[:start]+body+witness[end:]
    for name in ('c3x_renderer_api.h','biq_preview.cpp'):
        witness=witness.replace('../native/'+name,os.path.relpath(ROOT/'Renderer/native'/name,OUT))
    witness=witness.replace('int main(int argc, char** argv) {','int main(int argc, char** argv) {\n    SetEnvironmentVariableA("C3X_RENDERER_UNIT_PACK","UnitGameCloseup");')
    (OUT/'preview.cpp').write_text(witness)
    setup=(ROOT/'Renderer/lab/build_native_preview.bat').read_text().split('cl /nologo')[0]
    (OUT/'build.bat').write_text(setup+'cl /nologo /std:c++17 /EHsc /O2 /W4 /WX preview.cpp /Fo:preview.obj /Fe:preview.exe /link /LARGEADDRESSAWARE gdi32.lib user32.lib\nexit /b %errorlevel%\n')
    return {'dll_sha256':sha(ROOT/'Renderer/bin/C3XRenderer.dll'),'production_bindings_sha256':sha(SOURCE/'bindings.json'),
            'reference_sha256':sha(REFERENCE),'inputs':consumed,'magnification':10,
            'private_overrides':['uniform geometric magnification','minimum canvas 512','native projection 2x'],
            'preserved':['installed DLL','all texture bytes','all non-position vertex fields','rotation/scale animation matrices','sampling 4x','native noon light','native display transfer','native isometric projection']}


def run():
    from Renderer.lab.platform import native_command_result
    from Renderer.renderer import native_render
    record=prepare()
    assert native_command_result(str(OUT.relative_to(ROOT)),'call build.bat')['status']=='pass'
    render=native_render('units','sizing',12,192,OUT/'native',candidate=ROOT/'Renderer/bin/C3XRenderer.dll',preview=OUT/'preview.exe',diagnostics=True)
    assert record['dll_sha256']==sha(ROOT/'Renderer/bin/C3XRenderer.dll')==render['dll_sha256']
    assert record['production_bindings_sha256']==sha(SOURCE/'bindings.json')
    for path,digest in record['inputs'].items():assert sha(ROOT/path)==digest
    record.update(render=render,production_unchanged=True)
    (OUT/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    from PIL import Image
    Image.open(ROOT/render['image']).save(OUT/'game-closeup.png')



def comparison():
    """Keep native pixels 1:1; reduce the original reference to the same subject height."""
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    game_path=OUT/'native/sizing-h12-z192.bmp-direction5.bmp'
    game=Image.open(game_path).convert('RGB');reference=Image.open(REFERENCE).convert('RGB')
    def subject_box(im):
        a=np.asarray(im).astype(int)
        # Both neutral backdrops and ground shadows are nearly achromatic.
        mask=(a.max(axis=2)-a.min(axis=2))>24
        ys,xs=np.nonzero(mask)
        return (int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1))
    gb=subject_box(game);rb=subject_box(reference)
    factor=(gb[3]-gb[1])/(rb[3]-rb[1]);assert 0<factor<1
    # Reduce radiance rather than averaging display-encoded RGB.
    a=np.asarray(reference).astype(np.float32)/255
    linear=np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
    size=tuple(round(v*factor) for v in reference.size)
    small=np.stack([np.asarray(Image.fromarray(linear[:,:,i]).resize(size,Image.Resampling.BOX)) for i in range(3)],axis=-1)
    rgb=np.where(small<=.0031308,small*12.92,1.055*np.maximum(small,0)**(1/2.4)-.055)
    reference=Image.fromarray(np.uint8(np.clip(np.round(rgb*255),0,255)))
    rb2=tuple(round(v*factor) for v in rb)
    height=gb[3]-gb[1];panel=Image.new('RGB',(1200,height+158),(139,151,150));draw=ImageDraw.Draw(panel)
    font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',23)
    small_font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',15)
    for side,(im,box) in enumerate(((reference,rb2),(game,gb))):
        center=(box[0]+box[2])//2
        crop=im.crop((center-290,box[1]-22,center+290,box[1]+height+45))
        panel.paste(crop,(side*600+10,78))
    draw.rectangle((0,0,1199,77),fill='#242b2d')
    draw.text((24,13),'ORIGINAL STUDIO REFERENCE',font=font,fill='white')
    draw.text((624,13),'CURRENT GAME RENDERER',font=font,fill='white')
    draw.text((24,47),'Original image reduced to matching character height',font=small_font,fill='#d5dedd')
    draw.text((624,47),'Installed DLL · fresh 10× magnification · native pixels',font=small_font,fill='#d5dedd')
    draw.line((599,0,599,panel.height),fill='#242b2d',width=2)
    panel.save(OUT/'comparison.png')
    Image.open(game_path).save(OUT/'game-front.png')
    evidence={'reference':str(REFERENCE.relative_to(ROOT)),'reference_sha256':sha(REFERENCE),
        'game':str(game_path.relative_to(ROOT)),'game_sha256':sha(game_path),'game_direction':5,
        'reference_reduction':factor,'game_resized':False,'approximate_colored_subject_height_px':height,
        'reference_subject_box':rb,'game_subject_box':gb,
        'limits':['Game projection, lighting and output transfer intentionally differ from studio.',
                  'Geometric magnification is a private diagnostic beyond the playable 2x zoom, not a game screenshot.',
                  'Comparison crops focus on the unit; see full native capture for ground shadow.']}
    (OUT/'comparison.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':
    run()
    comparison()
