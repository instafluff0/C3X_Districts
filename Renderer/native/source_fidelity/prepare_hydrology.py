#!/usr/bin/env python3
"""Apply the existing native ABI adapter to the r13 retained hydrology closure."""
from pathlib import Path
import re
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
LAB=ROOT/'Renderer/lab/shared'
def main():
    import sys
    sys.path.insert(0,str(ROOT))
    from Renderer.native.profile_v2.generate_shaders import generate
    generate(LAB,HERE,'hydrology.hlsl',complete_rock_channels=True)
    for name in ['field.h','river_corridor.h']:
        source=(LAB/'hydrology'/name).read_text()
        if name=='river_corridor.h':
            source=source.replace('void build(char const* csv,Height height)', 'void build(hydro::Field const& field,Height height)')
            source=source.replace('        hydro::Field field;field.load(csv);field.wraps=true;\n','')
        source=re.sub(r'\bnear\b','query_point',source) # Windows SDK near macro
        (HERE/name).write_text(source)
if __name__=='__main__':main()
