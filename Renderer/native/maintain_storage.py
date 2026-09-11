"""Preview-first maintenance of old generated output; never touches source packs or Git."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = (Path('Renderer/native/build'), Path('Renderer/lab/out'))
INTERMEDIATES = {'.obj', '.exp', '.lib'}


def digest(path, compressed=False):
    h = hashlib.sha256()
    with (gzip.open(path, 'rb') if compressed else path.open('rb')) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def protected(relative):
    # Snapshot roots may contain unique source art. Recent/current fixtures and
    # the acceptance/rollback evidence stay directly usable without restoration.
    parts = relative.parts
    return any(p in {'root', 'source', 'packs', 'promotion', 'candidate', 'verified',
                     'approved', 'maintenance', 'references'} or
               'current' in p or 'latest' in p or 'rollback' in p or
               'joined-skin' in p for p in parts)


def allowed(path):
    relative = path.relative_to(ROOT)
    if relative.is_absolute() or '..' in relative.parts:
        return False
    if not any(relative.is_relative_to(root) for root in OUTPUTS) or protected(relative):
        return False
    # A symlink anywhere along the path must not escape the allowed trees.
    return all(not parent.is_symlink() for parent in (path, *path.parents) if parent != ROOT.parent)


def snapshot(path):
    st = path.stat()
    return {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns,
            'inode': st.st_ino, 'device': st.st_dev, 'links': st.st_nlink}


def plan(days):
    entries = []
    cutoff = time.time() - days * 86400
    tracked = set()
    import subprocess
    tracked.update(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT)
                   .decode().split('\0'))
    for relative in OUTPUTS:
        for folder, dirs, files in os.walk(ROOT / relative, followlinks=False):
            dirs[:] = [d for d in dirs if allowed(Path(folder) / d)]
            for name in sorted(files):
                path = Path(folder) / name
                rel = path.relative_to(ROOT).as_posix()
                if rel in tracked or not allowed(path) or not path.is_file():
                    continue
                info = snapshot(path)
                if info['links'] != 1 or info['mtime_ns'] / 1e9 > cutoff:
                    continue
                action = ('gzip' if path.suffix.lower() == '.bmp' else
                          'remove_intermediate' if path.suffix.lower() in INTERMEDIATES else None)
                if action:
                    entries.append(dict(path=rel, action=action, **info))
    return {'schema': 1, 'created_unix': time.time(), 'minimum_age_days': days,
            'policy': 'old unshared generated BMPs and compiler intermediates only',
            'entries': entries}


def apply(record, receipt_path):
    age = record.get('minimum_age_days', 0)
    if record.get('schema') != 1 or not math.isfinite(age) or age < 1:
        raise ValueError('Unsupported plan or minimum age')
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError('Receipt already exists; preserve it and create a fresh plan')
    import subprocess
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    results = []
    receipt = {'schema': 1, 'started_unix': time.time(), 'results': results,
               'free_before': shutil.disk_usage(ROOT).free}
    def save():
        temp = receipt_path.with_suffix('.tmp')
        temp.write_text(json.dumps(receipt, indent=2) + '\n')
        temp.replace(receipt_path)
    save()
    for entry in record['entries']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() in tracked:
            raise ValueError('Disallowed planned path')
        path = ROOT / entry['path']
        if not allowed(path):
            raise ValueError('Disallowed planned path')
        if not path.is_file():
            continue
        expected = {k: entry[k] for k in snapshot(path)}
        if snapshot(path) != expected:
            raise ValueError('Planned file changed: ' + entry['path'])
        if expected['links'] != 1 or expected['mtime_ns'] / 1e9 > time.time() - record['minimum_age_days'] * 86400:
            raise ValueError('File is shared or too recent')
        action = entry['action']
        if action == 'gzip' and path.suffix.lower() == '.bmp':
            archive = path.with_suffix(path.suffix + '.gz')
            before = digest(path)
            if archive.exists() or archive.is_symlink():
                if archive.is_symlink() or digest(archive, True) != before:
                    raise ValueError('Existing archive differs: ' + entry['path'])
            else:
                temp = archive.with_suffix('.gz.partial')
                with temp.open('xb') as out:
                    with gzip.GzipFile(filename='', fileobj=out, mode='wb', compresslevel=6, mtime=0) as stream:
                        with path.open('rb') as source:
                            shutil.copyfileobj(source, stream, 1024 * 1024)
                if digest(temp, True) != before:
                    raise ValueError('Archive verification failed: ' + entry['path'])
                if temp.stat().st_size >= entry['bytes']:
                    temp.unlink()
                    continue
                temp.replace(archive)
            if snapshot(path) != expected or digest(path) != before:
                raise ValueError('Source changed during archival: ' + entry['path'])
            result = dict(path=entry['path'], action=action, sha256=before,
                          original_bytes=entry['bytes'], archive_bytes=archive.stat().st_size)
        elif action == 'remove_intermediate' and path.suffix.lower() in INTERMEDIATES:
            result = dict(path=entry['path'], action=action, sha256=digest(path),
                          original_bytes=entry['bytes'])
            if snapshot(path) != expected:
                raise ValueError('Intermediate changed during cleanup')
        else:
            raise ValueError('Disallowed action')
        # Persist intent before unlink; a crash cannot erase the provenance.
        result['removed'] = False
        results.append(result)
        save()
        path.unlink()
        result['removed'] = True
        save()
    receipt.update(finished_unix=time.time(), free_after=shutil.disk_usage(ROOT).free)
    save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True, help='Ignored local JSON manifest')
    parser.add_argument('--apply', action='store_true', help='Apply the reviewed manifest, validating file identity')
    parser.add_argument('--minimum-age-days', type=float, default=2)
    args = parser.parse_args()
    if not math.isfinite(args.minimum_age_days) or args.minimum_age_days < 1:
        parser.error('Minimum age is one day; current work is protected')
    manifest = args.plan.resolve()
    manifest.relative_to(ROOT / 'Renderer/lab/out/maintenance')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.apply:
        receipt = apply(json.loads(manifest.read_text()), manifest.with_suffix('.receipt.json'))
        print(json.dumps({'processed': len(receipt['results']),
                          'volume_free_delta': receipt['free_after'] - receipt['free_before']}))
    else:
        if manifest.exists():
            parser.error('Plan already exists; choose a fresh filename to preserve the audit trail')
        record = plan(args.minimum_age_days)
        manifest.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'mode': 'preview', 'files': len(record['entries']),
                          'logical_bytes': sum(e['bytes'] for e in record['entries']),
                          'actions': {a: sum(e['bytes'] for e in record['entries'] if e['action'] == a)
                                      for a in ('gzip', 'remove_intermediate')}}))


if __name__ == '__main__':
    main()
