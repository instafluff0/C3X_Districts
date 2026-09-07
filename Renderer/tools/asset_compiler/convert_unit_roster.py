#!/usr/bin/env python3
"""Convert the roster's explicitly selected clips on Windows without game UI."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler.unit_member_resolver import ASSETS_ROOT
from Renderer.tools.asset_compiler.unit_family_asset_importer import SourceDataRoots
from Renderer.tools.renderer_dev import windows_command_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', type=Path, default=Path('Renderer/tools/asset_compiler/unit_roster_strategy.json'))
    parser.add_argument('--pack', type=Path, default=Path('Renderer/packs/UnitRosterLab'))
    args = parser.parse_args()
    strategy = json.loads(args.strategy.read_text())
    if "compositions" in strategy:
        strategy["units"] = []
        for composition in strategy["compositions"]:
            for node in [composition["parent"], *composition["children"]]:
                strategy["units"].append({"slug": composition["slug"] + "/" + node["id"],
                    "source_content": composition.get("source_content", strategy["source_content"]),
                    "actions": {action: {"source": record["nodes"][node["id"]]}
                        for action, record in composition["actions"].items() if "nodes" in record},
                    "additional_actions": {}})
    pack = args.pack
    if pack.parent != Path('Renderer/packs') or not pack.name.isalnum():
        raise ValueError('conversion pack must be a simple Renderer/packs child')
    root = ASSETS_ROOT / 'Base/Platforms/Windows/BLPs/SHARED_DATA'
    cache = pack / 'conversion_sources.json'
    previous = json.loads(cache.read_text()) if cache.exists() else {}
    current = {}
    lines = ['@echo off', 'setlocal',
        'set "TOOLS_DIR=%~dp0..\\..\\..\\tools\\asset_compiler\\"',
        'set "SOURCE=\\\\Mac\\Home\\Library\\Application Support\\Steam\\steamapps\\common\\Sid Meier\'s Civilization VI\\Civ6.app\\Contents\\Assets\\Base\\Platforms\\Windows\\BLPs\\SHARED_DATA"',
        'set "PACK=%TOOLS_DIR%..\\..\\packs\\UnitRosterLab"',
        'set "CONVERTER=%TOOLS_DIR%..\\..\\preview\\out\\animation_tools\\export_civ6_animation.exe"',
        'set "CIVNEXUS=%TOOLS_DIR%..\\..\\third_party\\CivNexus6\\bin\\Release\\CivNexus6.exe"',
        'call "%TOOLS_DIR%BUILD_CIV6_ANIMATION_CONVERTER.bat"',
        'if errorlevel 1 exit /b %errorlevel%']
    for unit in strategy['units']:
        if not re.fullmatch(r'[a-z0-9_]+(?:/[a-z0-9_]+)?',unit['slug']):
            raise ValueError('unsafe conversion unit path')
        root = SourceDataRoots(ASSETS_ROOT, unit.get('source_content', strategy['source_content']))
        for action, record in {**unit['actions'], **unit['additional_actions']}.items():
            if 'source' not in record:
                continue
            if not re.fullmatch(r'[a-z_]+',action) or not re.fullmatch(r'ANIMATION_[A-Za-z0-9_]+',record['source']):
                raise ValueError('unsafe conversion action or source name')
            key = f"{unit['slug']}/{action}"
            current[key] = hashlib.sha256((root / record['source']).read_bytes()).hexdigest()
            if previous.get(key) == current[key] and (pack / 'animations/unit' / (key+'.c3anim')).is_file():
                continue
            content = (root / record['source']).relative_to(ASSETS_ROOT).parent.as_posix().replace('/', '\\')
            if not re.fullmatch(r'[A-Za-z0-9_ .\\-]+',content):
                raise ValueError('unsafe conversion content path')
            lines += [f"set \"SOURCE=\\\\Mac\\Home\\Library\\Application Support\\Steam\\steamapps\\common\\Sid Meier's Civilization VI\\Civ6.app\\Contents\\Assets\\{content}\""]
            slug = unit["slug"].replace('/', '\\')
            lines += [f'call "%TOOLS_DIR%CONVERT_UNIT_FAMILY_ANIMATION_ONE.bat" "{slug}" {action} {record["source"]}',
                      'if errorlevel 1 exit /b %errorlevel%']
    lines.insert(8, f'set "PACK=%TOOLS_DIR%..\\..\\packs\\{pack.name}"')
    output = Path('Renderer/preview/out/units/convert_roster.bat')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(('\r\n'.join(lines+['exit /b 0'])+'\r\n').encode())
    result = windows_command_result('Renderer/preview/out/units', 'call convert_roster.bat')
    print(result['status'], result.get('output_tail', '')[-1600:])
    if result['status'] != 'pass':
        return 1
    cache.write_text(json.dumps(current, indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
