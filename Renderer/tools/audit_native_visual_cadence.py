"""Read-only GOG animation split, timer transport and ABI evidence."""
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
    checks={
        # Advisor entry/one stack argument, page construction, modal dialog and return.
        0x49d070:'568bf1e8b8e4fdff',0x49d095:'8b7c240c',
        0x49d1ae:'ff927c010000',0x49d205:'ff9010010000',0x49d239:'5ec20400',
        0x4de5c0:'a1c4c27200',0x4de5d0:'a1bc37cc00',
        0x4de6c0:'6a426a426a0168c0e54d00b900659f00',
        0x6205d0:'8b4424108b542408',0x6205f6:'c21000',
        0x405fc0:'d9442404d81db4536600',0x406112:'c21000',
        # Four arguments: effects, count, units, elapsed. Includes tile effects.
        0x4ef168:'8dbedc1800008d4e0857505152b940da7300',
        # Animator visual gate and native elapsed accumulator read/commit.
        0x4eec57:'8b9ee01a00008b4f048b96e41a0000',
        0x4eec9b:'8a861819000084c00f8451070000',
        0x4ef3d7:'8996e01a00008b5424348986e41a0000',
        0x620681:'83f832',0x62068f:'6830086200',
        # Multimedia callback skips direct callback unless callback_fn_2 exists;
        # normal native Timer posts the same coalesced WM_USER+1 as timer_proc.
        0x620839:'8b460485c07406ffd0',0x620853:'8b462485c07521',
        0x620863:'6801040000',0x62086e:'ff158c526600c7462401000000',
    }
    for address,value in checks.items():
        if read(address,len(value)//2).hex()!=value:raise ValueError(f'byte mismatch at {address:#x}')
    calls={0x4de6d0:0x6205d0,0x6205e9:0x620560,0x6205f0:0x620660,0x4ef17a:0x405fc0}
    for address,target in calls.items():
        code=read(address,5)
        if code[0]!=0xe8 or address+5+struct.unpack_from('<i',code,1)[0]!=target:
            raise ValueError(f'call mismatch at {address:#x}')
    return {'schema':'c3x.native_visual_cadence_audit.v1','status':'pass',
        'executable_sha256':hashlib.sha256(data).hexdigest(),
        'checks':{f'0x{a:08X}':v for a,v in checks.items()},
        'calls':{f'0x{a:08X}':f'0x{t:08X}' for a,t in calls.items()},
        'advance_stack_arguments':4,'timer_stack_arguments':4,
        'native_timer_transport':'coalesced WM_USER+1; callback_fn_2 must remain null',
        'supported_build':'GOG; other addresses unverified','executed_game':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable',type=Path,default=Path('Renderer/native/build/unit-audit-original.exe'))
    p.add_argument('--output',type=Path,default=Path('Renderer/native/build/native-visual-cadence/byte-audit.json'))
    args=p.parse_args();result=audit(args.executable);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print('Native visual cadence byte/ABI audit: pass')
