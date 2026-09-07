"""Check packed buffer capacity and equivalence to frozen shader literals."""
import struct
import unittest
from city_light_buffer_probe import payload, LIGHTS, BLOCKERS


class LightBuffer(unittest.TestCase):
    def test_float32_literals_and_nonoverlapping_array_offsets(self):
        data={'gain':1.234567891,'z_metric':.648267129381,
              'lights':[{'position':[1.234567891234,2,3],'range':.32,'color_linear':[.1,.2,.3],
                         'intensity':8,'direction':[-1,0,0],'owner':0}],
              'blockers':[{'low':[1,2,3],'high':[4,5,6]}]}
        raw=payload(data);rows=list(struct.iter_unpack('<4f',raw))
        self.assertEqual(len(raw),(3+LIGHTS*3+BLOCKERS*2)*16)
        f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
        self.assertEqual(rows[0],(1,1,f32(float('1.23456789')),f32(float('.6482671294'))))
        self.assertEqual(rows[3],tuple(f32(float(f'{v:.10f}')) for v in [1.234567891234,2,3,.32]))
        self.assertEqual(rows[3+LIGHTS*2],(-1,0,0,0))
        self.assertEqual(rows[3+LIGHTS*3],(1,2,3,0))
        self.assertEqual(rows[3+LIGHTS*3+BLOCKERS],(4,5,6,0))
        self.assertTrue(all(row==(0,0,0,0) for row in rows[4:3+LIGHTS]))

    def test_capacity_rejected_before_serialization(self):
        with self.assertRaisesRegex(ValueError,'capacity'):
            payload({'lights':[{}]*129,'blockers':[{}]})


if __name__=='__main__':unittest.main()
