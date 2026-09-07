"""Lossless storage for bulky Lab replay packets; restore before direct replay."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

def digest(path,compressed=False):
    h=hashlib.sha256()
    with (gzip.open(path,'rb') if compressed else path.open('rb')) as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def archive(path):
    target=path.with_suffix(path.suffix+'.gz');record=path.with_suffix(path.suffix+'.storage.json')
    if target.exists():raise ValueError('packet archive already exists')
    original=digest(path);size=path.stat().st_size
    with path.open('rb') as source,gzip.open(target,'wb',compresslevel=1) as dest:shutil.copyfileobj(source,dest)
    assert digest(target,True)==original,'lossless archive verification failed'
    record.write_text(json.dumps({'sha256':original,'bytes':size,'archive_bytes':target.stat().st_size,'compression':'gzip'},indent=2)+'\n')
    path.unlink()

def restore(path):
    target=path.with_suffix(path.suffix+'.gz');record=json.loads(path.with_suffix(path.suffix+'.storage.json').read_text())
    if path.exists():assert digest(path)==record['sha256'];return
    with gzip.open(target,'rb') as source,path.open('wb') as dest:shutil.copyfileobj(source,dest)
    assert digest(path)==record['sha256'],'restored packet hash mismatch'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['archive','restore']);parser.add_argument('packet',type=Path)
    a=parser.parse_args();(archive if a.action=='archive' else restore)(a.packet)
