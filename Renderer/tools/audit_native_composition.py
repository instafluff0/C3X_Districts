"""Read-only byte and patch-table audit of the GOG native transfer seam."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct


def audit(path):
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!='838df6f8b3518d5f5f7ff50c7c7add715628ffabbd37221d0bcc7f080afc2746':
        raise ValueError('expected preserved original GOG executable')
    pe=struct.unpack_from('<I',data,0x3c)[0];count=struct.unpack_from('<H',data,pe+6)[0]
    optional=pe+24;base=struct.unpack_from('<I',data,optional+28)[0]
    sections=optional+struct.unpack_from('<H',data,pe+20)[0]
    def read(address,size):
        for n in range(count):
            _,start,length,raw=struct.unpack_from('<4I',data,sections+n*40+8)
            rva=address-base
            if start<=rva and rva+size<=start+length:return data[raw+rva-start:raw+rva-start+size]
        raise ValueError('address outside file-backed image')
    checks={0x606780:'a12cd3ca00', # complete first instruction; no relative relocation
            0x6067a2:'b930d0ca00', # retained PCX screen canvas for final tooltip
            0x606997:'6830d0ca00e83f19ffff', # cursor Sprite_draw to same canvas
            0x6069a1:'8b0dccb2ca0085c974098b1156ff92a40000005ec3'} # final Graphsy slot41; cdecl return
    for address,expected in checks.items():
        if read(address,len(expected)//2).hex()!=expected:raise ValueError(f'byte mismatch: {address:#x}')
    root=Path(__file__).resolve().parents[2]
    wanted={'JGL_present_screen':('inlead',0x606780,'void (__cdecl *) (RECT * rect)'),
            'p_jgl_screen_canvas':('define',0xcad030,'PCX_Image *')}
    found={}
    for row in csv.reader((root/'civ_prog_objects.csv').read_text().splitlines(),skipinitialspace=True):
        row=[s.strip() for s in row]
        if len(row)!=6 or row[4] not in wanted:continue
        if row[4] in found:raise ValueError('duplicate native composition symbol')
        if (row[0],int(row[1],0),row[5])!=wanted[row[4]] or any(int(v,0) for v in row[2:4]):
            raise ValueError('unexpected symbol definition or unverified other-build address')
        found[row[4]]=row
    if set(found)!=set(wanted):raise ValueError('missing native composition symbol')
    return {'status':'pass','executable_sha256':hashlib.sha256(data).hexdigest(),
            'checks':{hex(k):v for k,v in checks.items()},'symbols':found,'executed_game':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable',type=Path,default=Path('Renderer/native/build/unit-audit-original.exe'))
    p.add_argument('--output',type=Path,default=Path('Renderer/native/build/native-observation/byte-audit.json'))
    args=p.parse_args();result=audit(args.executable);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print('Native composition byte/ABI audit: pass')
