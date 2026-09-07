"""Offline packed direction decoding; source profile/offset selection is separate."""
import math
import struct


def decode_octahedral_snorm8(data, offset=0):
    x,y=(max(-1.,v/127.) for v in struct.unpack_from('<2b',data,offset))
    z=1-abs(x)-abs(y)
    if z<0:
        x,y=(1-abs(y))*(1 if x>=0 else -1),(1-abs(x))*(1 if y>=0 else -1)
    length=math.sqrt(x*x+y*y+z*z)
    return [x/length,y/length,z/length]
