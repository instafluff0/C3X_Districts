"""Portable input protocol checks; these do not qualify a live gameplay capture."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / 'Renderer/native'


def members(source, name):
    body = re.search(r'struct\s+' + name + r'\s*\{(.*?)\};', source, re.S).group(1)
    body = re.sub(r'/\*.*?\*/|//[^\n]*', '', body, flags=re.S)
    result = set()
    for declaration in body.split(';'):
        if not declaration.strip():
            continue
        for field in declaration.split(','):
            match = re.search(r'(\w+)\s*(?:\[[^]]+\])?\s*$', field.strip())
            if not match:
                raise AssertionError(f'Unparsed ABI declaration: {declaration}')
            result.add(match.group(1))
    return result


class InputRecordingTests(unittest.TestCase):
    def test_every_consumed_abi_field_is_explicit(self):
        api = (NATIVE / 'c3x_renderer_api.h').read_text()
        gpu = (NATIVE / 'gpu_frame_api.h').read_text()
        codec = (NATIVE / 'input_recording/codec.h').read_text()
        contracts = (
            (api, 'c3x_renderer_tile_v1', 'c3x_renderer_tile_v1_fields', set()),
            (api, 'c3x_renderer_unit_v1', 'c3x_renderer_unit_v1_fields', {'struct_size'}),
            (api, 'c3x_renderer_camera_identity_v1', 'c3x_renderer_camera_identity_v1_fields', set()),
            (api, 'c3x_renderer_frame_v1', 'frame_fields', {'api_version', 'struct_size', 'tile_count', 'tiles', 'world_topology_count', 'world_topology'}),
            (gpu, 'c3x_renderer_gpu_unit_v1', 'target_fields', {'struct_size'}),
            (gpu, 'c3x_renderer_gpu_command_v1', 'command_fields', set()),
            (gpu, 'c3x_renderer_gpu_images_v1', 'image_fields', {'struct_size', 'pixels', 'commands', 'command_count', 'command_struct_size'}),
        )
        for source, structure, visitor, excluded in contracts:
            with self.subTest(structure=structure):
                body = re.search(r'void ' + visitor + r'\(.*?\)\{(.*?)\n\}', codec, re.S).group(1)
                used = set(re.findall(r'\bv\.(\w+)', body))
                self.assertEqual(members(source, structure) - excluded, used)

    def test_storage_and_owned_input_mutations(self):
        compiler = shutil.which('clang++') or shutil.which('g++')
        if compiler is None:
            self.skipTest('portable C++ compiler unavailable')
        with tempfile.TemporaryDirectory(prefix='c3x-input-tests-') as folder:
            exe = Path(folder) / 'test-input-recording'
            subprocess.run([compiler, '-std=c++17', '-O2', '-pthread', str(NATIVE / 'test_input_recording.cpp'), '-o', str(exe)], check=True)
            subprocess.run([str(exe), str(Path(folder) / 'captures')], check=True)


if __name__ == '__main__':
    unittest.main()
