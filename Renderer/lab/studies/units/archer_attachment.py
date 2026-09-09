"""Native before/after proof for the archer's parented bow rig and arrow socket."""
from pathlib import Path
import copy,json,os,shutil
from .prepare import ROOT,read
OUT=ROOT/'Renderer/lab/out/units/archer-attachment'
PACK=ROOT/'Renderer/packs/ArcherAttachmentComparison'
BEFORE=ROOT/'Renderer/packs/ArcherAttachmentBefore'

def capture_before():
    """Retain the old attachment result across later production rebuilds."""
    if (BEFORE/'bindings.json').exists():return
    source=ROOT/'Renderer/packs/UnitAnimationFidelity'
    archer=read(source/'manifest.json')['units']['unit/archer']
    if any(p.get('attachment_binding') for p in archer['actions']['idle']['parts']):
        raise ValueError('Old attachment snapshot is missing; corrected art cannot be labeled before')
    unit=next(v for v in read(source/'bindings.json').values() if isinstance(v,dict) and v.get('key0')=='PRTO_Archer')
    for action in unit.values():
        if not isinstance(action,dict) or 'part_count' not in action:continue
        for i in range(action['part_count']):
            for key,value in action['part'+str(i)].items():
                if key in ('mesh','texture') or key.endswith('_texture'):
                    dst=BEFORE/value;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/value,dst)
    (BEFORE/'bindings.json').write_text(json.dumps({'unit_count':1,'unit0':unit},indent=2)+'\n')

def prepare():
    from Renderer.native.environment_refresh.prepare_units import build_pack
    from Renderer.tools.asset_compiler.build_unit_animation_runtime import build
    OUT.mkdir(parents=True,exist_ok=True)
    build([ROOT/'Renderer/packs/UnitFamilyLab'],ROOT/'Renderer/packs/ArcherAttachmentRuntime')
    candidate=ROOT/'Renderer/packs/ArcherAttachmentFidelity'
    build_pack(candidate,ROOT/'Renderer/packs/ArcherAttachmentRuntime')
    capture_before()
    bindings={'unit_count':18};names=['warrior','spearman','pikeman','archer','settler','worker']
    for row,variant in enumerate(['baseline','anatomy','ssaa']):
        source=BEFORE if row==0 else candidate
        unit=next(v for v in read(source/'bindings.json').values() if isinstance(v,dict) and v.get('key0')=='PRTO_Archer')
        for col,name in enumerate(names):
            record=copy.deepcopy(unit)
            for k in list(record):
                if k.startswith('key'):del record[k]
            record.update(key_count=1,key0='PRTO_Lab_'+variant+'_'+name)
            for action in record.values():
                if not isinstance(action,dict) or 'part_count' not in action:continue
                for i in range(action['part_count']):
                    for key,value in action['part'+str(i)].items():
                        if key=='mesh' or key=='texture' or key.endswith('_texture'):
                            dst=PACK/value;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/value,dst)
            bindings['unit'+str(row*6+col)]=record
    (PACK/'bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
    s=(ROOT/'Renderer/lab/native_preview.cpp').read_text()
    for name in ['c3x_renderer_api.h','biq_preview.cpp']:
        s=s.replace('../native/'+name,os.path.relpath(ROOT/'Renderer/native'/name,OUT))
    s=s.replace('int main(int argc, char** argv) {','int main(int argc, char** argv) {\n    SetEnvironmentVariableA("C3X_RENDERER_UNIT_PACK","ArcherAttachmentComparison");')
    s=s.replace('unit.action_cursor=unit.action==1?0:7;','unit.direction=column+1;\n            unit.action_cursor=unit.action==1?0:7;')
    s=s.replace('265+row*290-half','220+row*290-half')
    s=s.replace('"Current fit / 1x material sampling", "Anatomy fit / 1x material sampling", "Anatomy fit / 2x material sampling"','"Before: detached bow rig", "After: bow follows hand, arrow uses its source animated socket", "After: additional view"')
    s=s.replace('study_keys[column],int(std::strlen(study_keys[column]))','"Archer",6')
    (OUT/'native_preview.cpp').write_text(s)
    setup=(ROOT/'Renderer/lab/build_native_preview.bat').read_text().split('cl /nologo')[0]
    setup=setup.replace('pushd "%~dp0"','pushd "%~dp0"')
    # The command runs inside the output directory; setup only locates MSVC.
    (OUT/'build.bat').write_text(setup+'cl /nologo /std:c++17 /EHsc /O2 /W4 /WX native_preview.cpp /Fo:preview.obj /Fe:preview.exe /link /LARGEADDRESSAWARE gdi32.lib user32.lib\nexit /b %errorlevel%\n')


def run():
    from Renderer.lab.platform import native_command_result
    from Renderer.renderer import native_render
    from PIL import Image,ImageDraw
    prepare()
    if native_command_result(str(OUT.relative_to(ROOT)),'call build.bat')['status']!='pass':raise ValueError('Archer witness build failed')
    images=[]
    for case in ('sizing','sizing-move','sizing-gameplay'):
        images.append(native_render('units',case,12,192,OUT/case,preview=OUT/'preview.exe'))
    panel=Image.new('RGB',(820,330),'#303638');d=ImageDraw.Draw(panel)
    for state,e in enumerate(images[:2]):
        im=Image.open(ROOT/e['image']);im.save((ROOT/e['image']).with_suffix('.png'))
        for side in range(2):
            # Direction 3 matches the reported running capture; native pixels.
            panel.paste(im.crop((400,side*290+28,600,side*290+288)),(state*410+side*200,45))
            d.text((state*410+side*200+10,15),('STANDING' if state==0 else 'RUNNING')+' / '+('BEFORE' if side==0 else 'AFTER'),fill='white')
    panel.save(OUT/'before-after.png')
    context=Image.open(ROOT/images[2]['image'])
    panel=Image.new('RGB',(400,280));d=ImageDraw.Draw(panel)
    for side in range(2):
        panel.paste(context.crop((400,side*290+28,600,side*290+288)),(side*200,20))
        d.text((side*200+10,3),'BEFORE' if side==0 else 'AFTER',fill='white')
    panel.save(OUT/'gameplay-before-after.png')
    (OUT/'renders.json').write_text(json.dumps(images,indent=2)+'\n')

if __name__=='__main__':run()
