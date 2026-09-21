"""Real D3D recording/replay, negative controls and bounded-file validation."""
import json
import os
import struct
import unittest
import uuid

from Renderer.lab.platform import ROOT, native_command_result, windows_root


def run(command, timeout=90):
    # Child completion is authoritative even when Parallels loses its transport
    # result. Never mistake a failed dispatch for the expected negative control.
    nonce = uuid.uuid4().hex
    relative = 'Renderer/native/build/composition-replay/invocations'
    folder = ROOT / relative
    folder.mkdir(parents=True, exist_ok=True)
    target = (ROOT if os.name == 'nt' else windows_root()) / relative
    log = target / (nonce + '.log')
    done = target / (nonce + '.done')
    batch = folder / (nonce + '.bat')
    batch.write_text('@echo off\nsetlocal\n' + f'{command} >"{log}" 2>&1\n'
                     + f'>"{done}" echo {nonce} %errorlevel%\nexit /b 0\n')
    transport = native_command_result('Renderer/native', f'call "{target / batch.name}"', timeout_seconds=timeout)
    receipt = folder / (nonce + '.done')
    if not receipt.is_file():
        raise AssertionError(f'Native completion unconfirmed; inspect {relative}/{nonce}.log before retry: {transport}')
    identity, code = receipt.read_text().split()
    if identity != nonce:
        raise AssertionError('Unmatched native completion receipt')
    output = (folder / (nonce + '.log')).read_text(errors='replace')
    print(output[-4000:], flush=True)
    return {'status': 'pass' if code == '0' else 'fail', 'returncode': int(code),
            'output_tail': output[-4000:], 'invocation': nonce,
            'transport_returncode': transport['returncode']}


class CompositionRecordingTests(unittest.TestCase):
    def test_record_replay_and_reject_wrong_capacity_or_corruption(self):
        folder = ROOT / 'Renderer/native/build/composition-replay'
        folder.mkdir(parents=True, exist_ok=True)
        result = run('call BUILD.bat composition-replay', timeout=180)
        self.assertEqual(result['status'], 'pass', result)
        exe = r'build\composition-replay\replay_composition.exe'
        path = r'build\composition-replay\contract.c3xr'
        result = run(f'{exe} --generate {path}')
        self.assertEqual(result['status'], 'pass', result)
        cases = []
        for suffix in ('', ' --paced', ' --budget-mib 128'):
            result = run(f'{exe} {path}{suffix}')
            expected = 'fail' if '128' in suffix else 'pass'
            self.assertEqual(result['status'], expected, result)
            if expected == 'pass':
                line = next(line for line in result['output_tail'].splitlines() if line.startswith('{'))
                report = json.loads(line)
                self.assertEqual(report['status'], 'verified_composition')
                self.assertEqual(report['pixel_checks'], 6)
                self.assertEqual(report['display_boundaries'], 2)
                self.assertGreaterEqual(report['external_snapshots'], 4)
            else:
                self.assertIn('image admission differs', result['output_tail'])
            cases.append({'option': suffix, 'expected': expected, **result})
        original = (folder / 'contract.c3xr').read_bytes()
        at = 16
        records = []
        while at < len(original):
            marker, kind, size = struct.unpack_from('<III', original, at)
            self.assertEqual(marker, 0x31523343)
            records.append((at, kind, size))
            at += 40 + size
        self.assertEqual(at, len(original))
        self.assertEqual(records[-1][1], 14)
        mutations = {
            'truncated': original[:-1],
            'bad-checksum': original[:60] + bytes([original[60] ^ 1]) + original[61:],
            'oversized': original[:24] + struct.pack('<I', 0xffffffff) + original[28:],
            'missing-footer': original[:records[-1][0]],
            'after-footer': original + original[16:records[1][0]],
        }
        for name, data in mutations.items():
            (folder / f'{name}.c3xr').write_bytes(data)
            result = run(f'{exe} build\\composition-replay\\{name}.c3xr')
            if name == 'missing-footer':
                self.assertEqual(result['status'], 'pass', result)
                self.assertIn('"status":"verified_prefix"', result['output_tail'])
                self.assertIn('"footer":false', result['output_tail'])
            else:
                self.assertEqual(result['status'], 'fail', result)
            cases.append({'mutation': name, **result})
        (folder / 'test-results.json').write_text(json.dumps(cases, indent=2) + '\n')


if __name__ == '__main__':
    unittest.main()
