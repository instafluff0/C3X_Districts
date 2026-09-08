#!/usr/bin/env python3
"""Apply the existing native ABI adapter to the r13 retained hydrology closure."""
from pathlib import Path
import re
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
LAB=ROOT/'Renderer/terrain_lab/v2'
def main():
    original=(HERE.parent/'profile_v2/generate_shaders.py').read_text()
    original=original.replace("ROOT = Path(__file__).resolve().parent", "ROOT = Path(__file__).resolve().parent\nLAB = ROOT.parents[2] / 'Renderer/terrain_lab/v2'\nSOURCE = LAB")
    original=original.replace("ROOT / 'reference/shaders'", "SOURCE / 'shaders'").replace("ROOT / 'reference'", "SOURCE")
    original=original.replace("(ROOT / 'shadow_receiver.hlsl')", "(ROOT.parent / 'profile_v2/shadow_receiver.hlsl')")
    original=original.replace("#define Q6_TEXEL_RECEIVER_OFFSET 1", "#define Q4_COMPLETE_ROCK_CHANNELS 1\n#define Q4_COHERENT_ROCK_CHANNELS 1\n#define Q6_TEXEL_RECEIVER_OFFSET 1")
    original=original.replace("'integrated_v2.hlsl'", "'hydrology.hlsl'").replace("'lab_v2_terrain_lighting_r1'", "'source-fidelity-r13-retained-hydrology'")
    # Preserve relative include closure into generated/, including shared frozen
    # foundations, without enabling any city/Warrior experiments.
    namespace={'__file__':str(HERE/'prepare_hydrology.py'),'__name__':'fidelity_generator'}
    exec(compile(original,'native_hydrology_adapter','exec'),namespace)
    namespace['generate']()
    for name in ['field.h','river_corridor.h']:
        source=(LAB/'systems/hydrology'/name).read_text()
        if name=='river_corridor.h':
            source=source.replace('void build(char const* csv,Height height)', 'void build(hydro::Field const& field,Height height)')
            source=source.replace('        hydro::Field field;field.load(csv);field.wraps=true;\n','')
        source=re.sub(r'\bnear\b','query_point',source) # Windows SDK near macro
        (HERE/name).write_text(source)
if __name__=='__main__':main()
