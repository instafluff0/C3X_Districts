"""Copy local Civ III HUD pixels into an ignored native replay input, never a runtime asset."""
import hashlib
import struct
from pathlib import Path

PAIRS = (("box right color.pcx", "box right alpha.pcx"),
         ("box left color.pcx", "box left alpha.pcx"),
         ("box trans color.pcx", "box trans alpha.pcx"),
         ("menuButtons.pcx", "menuButtonsAlpha.pcx"))


def pcx(path):
    data = path.read_bytes()
    if data[:4] != bytes((10, 5, 1, 8)) or data[65] != 1 or data[-769] != 12:
        raise ValueError(f"Unsupported indexed PCX: {path.name}")
    x, y, right, bottom = struct.unpack_from('<4H', data, 4)
    width, height = right-x+1, bottom-y+1
    stride = struct.unpack_from('<H', data, 66)[0]
    decoded = bytearray()
    pos = 128
    while len(decoded) < stride*height:
        value = data[pos]; pos += 1
        count = 1
        if value >= 192:
            count = value & 63; value = data[pos]; pos += 1
        decoded.extend(bytes((value,))*count)
    if len(decoded) != stride*height or stride < width:
        raise ValueError('Invalid PCX scanline extent')
    return width, height, b''.join(decoded[y*stride:y*stride+width] for y in range(height)), data[-768:]


def prepare(root, output, inputs):
    art = root.parent.parent / 'Art' / 'interface'
    records = []
    for color, alpha in PAIRS:
        sources = []
        for name in (color, alpha):
            path = art / name
            inputs['../../Art/interface/' + name] = hashlib.sha256(path.read_bytes()).hexdigest()
            sources.append(pcx(path))
        # Main_Screen_Form slices three distinct native menu/alpha pairs.
        crops = ((1, 1, 36, 30), (37, 1, 36, 30), (74, 1, 34, 30)) if color == 'menuButtons.pcx' else ((0, 0, sources[0][0], sources[0][1]),)
        for x, y, width, height in crops:
            pair = []
            for sw, sh, pixels, palette in sources:
                if x+width > sw or y+height > sh:
                    raise ValueError('HUD slice exceeds local source')
                pair.append(palette + b''.join(pixels[(y+r)*sw+x:(y+r)*sw+x+width] for r in range(height)))
            records.append(struct.pack('<II', width, height) + b''.join(pair))
    output.write_bytes(struct.pack('<I', len(records)) + b''.join(records))
    inputs[output.relative_to(root).as_posix()] = hashlib.sha256(output.read_bytes()).hexdigest()
