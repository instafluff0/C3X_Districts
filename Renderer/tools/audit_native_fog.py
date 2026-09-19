"""Read-only GOG fog function entry and ABI audit; installed hooks elsewhere are irrelevant."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def audit(path):
    data=path.read_bytes();pe=struct.unpack_from('<I',data,0x3c)[0]
    if data[:2]!=b'MZ' or data[pe:pe+4]!=b'PE\0\0' or struct.unpack_from('<H',data,pe+4)[0]!=0x14c:
        raise ValueError('expected x86 PE image')
    count=struct.unpack_from('<H',data,pe+6)[0];optional=pe+24
    base=struct.unpack_from('<I',data,optional+28)[0]
    sections=optional+struct.unpack_from('<H',data,pe+20)[0]
    def read(address,size):
        for i in range(count):
            _,start,length,raw=struct.unpack_from('<4I',data,sections+i*40+8)
            rva=address-base
            if start<=rva and rva+size<=start+length:return data[raw+rva-start:raw+rva-start+size]
        raise ValueError('address outside file-backed image')
    checks={0x4c4ef0:'83ec44538bd9',0x4c4f21:'8b74245c',0x4c5510:'e82bf9ffff',0x4c555d:'c20c00'}
    for address,value in checks.items():
        if read(address,len(value)//2).hex()!=value:raise ValueError(f'byte mismatch at {address:#x}')
    return {'schema':'c3x.native_fog_audit.v1','status':'pass',
            'executable_sha256':hashlib.sha256(data).hexdigest(),
            'symbol':'Map_Renderer_draw_fog','gog_address':'0x4C4EF0','stack_arguments':3,
            'checks':{f'0x{a:08X}':v for a,v in checks.items()},'executed_game':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.executable)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print('Native fog byte/ABI audit: pass')
