"""Read compiled source shader instructions through the approved native dispatcher."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'Renderer/tools'))
import renderer_dev

if __name__ == '__main__':
    result = renderer_dev.native_command_result(
        'Renderer/terrain_lab/v2/qa',
        'powershell -NoProfile -ExecutionPolicy Bypass -File disassemble_ground_source.ps1')
    if result['status'] != 'pass' and 'cannot find the drive specified' in result.get('output_tail', ''):
        # Noninteractive VM sessions can lose Y: while the same share remains available.
        os.environ['C3X_RENDERER_WINDOWS_ROOT'] = str(renderer_dev.windows_live_target())
        result = renderer_dev.native_command_result(
            'Renderer/terrain_lab/v2/qa',
            'powershell -NoProfile -ExecutionPolicy Bypass -File disassemble_ground_source.ps1')
    raise SystemExit(0 if result['status'] == 'pass' else 1)
