"""Losslessly repack BC4 UNORM blocks into the alpha channel of BC3 blocks."""
import struct


def bc4_blocks_to_bc3_alpha(payload):
    if len(payload)%8:raise ValueError('incomplete BC4 block')
    white=struct.pack('<HHI',0xffff,0,0)
    return b''.join(payload[i:i+8]+white for i in range(0,len(payload),8))
