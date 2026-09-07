"""Bounded DXBC arithmetic decoder, checked against local D3DDisassemble output.

This is an inspection aid, not a shader translator. Unsupported operand forms
fail explicitly. Raw shader text remains in ignored, locally derived output.
"""
from pathlib import Path
import hashlib
import json
import re
import struct

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'Renderer/terrain_lab/v2/audits/beauty/out/ground-shader-source'
OPS = {0: 'add', 14: 'div', 27: 'ftoi', 45: 'ld', 50: 'mad',
       54: 'mov', 56: 'mul', 62: 'ret', 65: 'round_ni', 69: 'sample'}


def chunks(path):
    data = path.read_bytes()
    count = struct.unpack_from('<I', data, 28)[0]
    return {data[o:o+4].decode(): data[o+8:o+8+struct.unpack_from('<I', data, o+4)[0]]
            for o in struct.unpack_from('<'+'I'*count, data, 32)}


def instructions(path):
    parts = chunks(path)
    data = parts.get('SHEX', parts.get('SHDR'))
    words = struct.unpack('<'+'I'*(len(data)//4), data)
    i = 2
    while i < len(words):
        n = (words[i] >> 24) & 127
        if not n or i+n > len(words):
            raise ValueError('unsupported or truncated instruction')
        yield words[i:i+n]
        i += n


def operand(words, pos):
    token = words[pos]
    pos += 1
    modifier = 0
    if token >> 31:
        extension = words[pos]
        pos += 1
        if extension & 63 != 1 or extension >> 31:
            raise ValueError('unsupported operand extension')
        modifier = (extension >> 6) & 255
    count, mode, kind = token & 3, (token >> 2) & 3, (token >> 12) & 255
    dimensions = (token >> 20) & 3
    if kind == 4:
        n = 1 if count == 1 else 4 if count == 2 else 0
        if not n:
            raise ValueError('unsupported literal')
        floats = struct.unpack('<'+'f'*n, struct.pack('<'+'I'*n, *words[pos:pos+n]))
        pos += n
        value = 'l(' + ', '.join(f'{f:.6f}' for f in floats) + ')'
    else:
        indices = []
        for index in range(dimensions):
            if (token >> (22+3*index)) & 7:
                raise ValueError('relative indexing outside bounded decoder')
            indices.append(words[pos])
            pos += 1
        prefix = {0: 'r', 1: 'v', 2: 'o', 6: 's', 7: 't', 8: 'cb'}.get(kind)
        if prefix is None or not indices:
            raise ValueError('unsupported register kind')
        value = prefix + str(indices[0]) + ''.join(f'[{i}]' for i in indices[1:])
        if count == 2:
            if mode == 0:
                suffix = ''.join('xyzw'[i] for i in range(4) if token & (1 << (4+i)))
            elif mode == 1:
                suffix = ''.join('xyzw'[(token >> (4+2*i)) & 3] for i in range(4))
            elif mode == 2:
                suffix = 'xyzw'[(token >> 4) & 3]
            else:
                raise ValueError('unsupported selection mode')
            if suffix:
                value += '.' + suffix
    if modifier == 1:
        value = '-' + value
    elif modifier:
        raise ValueError('unsupported modifier')
    return value, pos


def decode(inst):
    opcode = inst[0] & 2047
    name = OPS[opcode]
    if inst[0] & (1 << 13):
        name += '_sat'
    values = []
    pos = 1
    extended = inst[0] >> 31
    while extended:
        extension = inst[pos]
        if extension & 63 not in (2, 3):
            raise ValueError('instruction offset extension outside bounded decoder')
        extended = extension >> 31
        pos += 1
    while pos < len(inst):
        value, pos = operand(inst, pos)
        values.append(value)
    return name + (' ' + ', '.join(values) if values else '')


def resources(path):
    data = chunks(path)['RDEF']
    count, offset = struct.unpack_from('<II', data, 8)
    result = []
    for i in range(count):
        name, kind, _, _, _, slot, length, _ = struct.unpack_from('<8I', data, offset+32*i)
        end = data.index(b'\0', name)
        result.append({'name': data[name:end].decode(), 'kind': kind, 'slot': slot, 'count': length})
    return result


def main():
    validated = 0
    counts = {name: 0 for name in OPS.values()}
    skipped = []
    # Validate arithmetic/texture operand decoding against the actual Microsoft
    # output already obtained for these terrain variants. Do not infer opcodes
    # by aligning text when the instruction counts differ.
    for asm_path in sorted(OUT.glob('*.asm')):
        path = asm_path.with_suffix('.dxbc')
        stem = path.stem
        parts = re.split(r'(?m)^(?:ps|vs|cs)_\d_\d\n', asm_path.read_text().replace('\0', ''))
        if len(parts) != 2:
            skipped.append(stem)
            continue
        asm = parts[1]
        lines = [s.strip() for s in asm.splitlines() if s.strip() and not s.strip().startswith('//')]
        try:
            program = list(instructions(path))
        except ValueError:
            skipped.append(stem)
            continue
        if len(program) != len(lines):
            raise ValueError('instruction/text count mismatch')
        for inst, actual in zip(program, lines):
            if inst[0] & 2047 not in OPS:
                continue
            try:
                decoded = decode(inst)
            except ValueError:
                continue
            actual = re.sub(r'^(sample|ld)_indexable\(texture2d(?:array)?\)\((?:float|uint),(?:float|uint),(?:float|uint),(?:float|uint)\)', r'\1', actual)
            # D3DDisassemble uses integer-looking zero vectors for mov literals.
            normalize = lambda s: re.sub(r'(?<![A-Za-z_])(-?\d+)\.0+(?!\d)', r'\1', s).replace(' ', '')
            if normalize(decoded) != normalize(actual):
                raise ValueError((stem, decoded, actual))
            validated += 1
            counts[OPS[inst[0] & 2047]] += 1
    if not all(counts.values()):
        raise ValueError(('unvalidated instruction kind', counts))
    records = []
    for stem in ['shader-0041c5fe', 'shader-0041d8a6', 'shader-004203b6',
                 'shader-00421d66', 'shader-00422efe', 'shader-00424096', 'shader-0042524e']:
        path = OUT / (stem + '.dxbc')
        lines = []
        for inst in instructions(path):
            opcode = inst[0] & 2047
            if opcode in OPS:
                lines.append(decode(inst))
            elif opcode < 88:
                raise ValueError(('unexpected executable opcode', opcode))
        output = path.with_suffix('.bounded.txt')
        output.write_text(json.dumps(resources(path), indent=2) + '\n' + '\n'.join(lines) + '\n')
        records.append({'source': path.relative_to(ROOT).as_posix(),
                        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'decoded': output.relative_to(ROOT).as_posix(),
                        'decoded_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                        'resources': resources(path)})
    evidence = {'classification': 'bounded bytecode decode validated against Microsoft disassembly; runtime state still unproven',
                'matching_instructions': validated, 'validated_operations': counts,
                'unsupported_reference_containers': skipped, 'records': records}
    (OUT.parent.parent / 'GROUND_BAKE_DECODE_EVIDENCE.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({'matching_instructions': validated, 'decoded_variants': len(records)}))


if __name__ == '__main__':
    main()
