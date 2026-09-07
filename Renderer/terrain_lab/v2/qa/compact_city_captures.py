"""Compact city captures into the Lab's existing shared-resource packet format.

Verify that expanding each packet produces its original byte hash. Keep images,
reports and source fixtures. Failed, unrendered packet intermediates are removed.
"""
import hashlib
import json
from pathlib import Path
import struct
import sys

V2=Path(__file__).resolve().parents[1];ROOT=V2.parents[2]
sys.path.insert(0,str(V2/'app'))
from cache import file_hash
from packet_store import compact_packet
from packet_archive import restore

def expanded_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        def copy(n):
            b=f.read(n)
            if len(b)!=n:raise ValueError('truncated compact packet')
            digest.update(b);return b
        def number():return struct.unpack('<I',copy(4))[0]
        def blob():
            size=struct.unpack('<I',f.read(4))[0]
            if size&0x80000000:
                size&=0x7fffffff;key=f.read(64).decode();digest.update(struct.pack('<I',size))
                part=Path(str(path)+'.blobs')/key
                assert part.stat().st_size==size and file_hash(part)==key
                with part.open('rb') as resource:
                    for chunk in iter(lambda:resource.read(1024*1024),b''):digest.update(chunk)
            else:digest.update(struct.pack('<I',size));copy(size)
        assert number()==0x32514c43
        version=number();assert version in (2,3,4,5,6)
        copy(12)
        if version>=3:copy(28 if version>=4 else 24)
        for _ in range(number()):
            copy(12)
            for _ in range(number()):copy(4);blob()
        for _ in range(number()):blob()
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def main():
    out=V2/'audits/beauty/out';rows=[];discarded=0
    record=V2/'audits/beauty/CITY_STORAGE_CLEANUP.json'
    if record.exists():rows=json.loads(record.read_text())['packets']
    for root in sorted(out.glob('city-scene-r*')):
        for case in sorted(p for p in root.iterdir() if p.is_dir()):
            packets=set(case.glob('combined-*.packet'))
            packets.update(p.with_name(p.name.replace('.storage.json','')) for p in case.glob('combined-*.packet.storage.json'))
            complete=(case/'combined/report.json').exists()
            for packet in sorted(packets):
                if not complete:
                    if packet.exists():discarded+=packet.stat().st_size;packet.unlink()
                    for suffix in ('.gz','.storage.json'):
                        extra=Path(str(packet)+suffix)
                        if extra.exists():discarded+=extra.stat().st_size;extra.unlink()
                    continue
                identity=packet.relative_to(ROOT).as_posix()
                if any(r['packet']==identity for r in rows):continue
                if not packet.exists():restore(packet)
                original=file_hash(packet);size=packet.stat().st_size
                compact_packet(packet,V2/'app/.cache/content')
                assert expanded_hash(packet)==original,'packet resources changed during compaction'
                row={'packet':identity,'original_sha256':original,'expanded_sha256':original,
                     'compact_sha256':file_hash(packet),'original_bytes':size,'compact_header_bytes':packet.stat().st_size}
                rows.append(row)
                record.write_text(json.dumps({'preserved':'all completed images, reports, source fixtures and byte-equivalent replay data',
                    'packet_format':'existing shared-resource packet; direct replay supported',
                    'failed_intermediate_bytes_removed':discarded,'packets':rows},indent=2)+'\n')
                for suffix in ('.gz','.storage.json'):
                    extra=Path(str(packet)+suffix)
                    if extra.exists():extra.unlink()
                print(identity,flush=True)

if __name__=='__main__':main()
