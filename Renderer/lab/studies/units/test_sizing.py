"""Executable anatomy-fit, ground clipping and production isolation checks."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .prepare import anatomy_fit, visible_points, build


class UnitSizingTests(unittest.TestCase):
    def test_weapon_length_cannot_change_anatomy_scale_or_ground(self):
        anatomy = [[0,0,-.01],[0,0,.12]]
        self.assertEqual(anatomy_fit(anatomy, .52, 1), (4., .01))
        # Weapon geometry is deliberately outside the explicit reference set.
        for weapon_top in (.2, .4, 1.):
            kit = dict(anatomy=anatomy, weapon=[[0,0,-.1],[0,0,weapon_top]])
            self.assertEqual(anatomy_fit(kit['anatomy'], .52, 1), (4., .01))
        self.assertAlmostEqual(anatomy_fit(anatomy,.52,1.1/1.22)[0]/4,1.1/1.22)

    def test_clipping_keeps_crossing_edges_and_discards_buried_inventory(self):
        points=np.array([[0.,0.,1.],[4.,0.,-1.],[0.,4.,-1.], [100.,100.,-1.]])
        result=visible_points(points,np.array([0,1,2,1,2,3]))
        self.assertEqual({tuple(v) for v in result},{(0.,0.,1.),(2.,0.,0.),(0.,2.,0.)})

    def test_rejects_degenerate_reference_and_production_output(self):
        with self.assertRaises(ValueError):anatomy_fit([[0,0,1],[0,1,1]],1,1)
        with self.assertRaises(ValueError):anatomy_fit([[0,0,0],[0,0,1]],1,float('nan'))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(ValueError,'overlap'):build(root,root)
            with self.assertRaisesRegex(ValueError,'overlap'):build(root,root/'nested')
            with self.assertRaisesRegex(ValueError,'overlap'):build(root/'nested',root)

    def test_study_has_explicit_sampling_and_keeps_original_aliases(self):
        root=Path(__file__).resolve().parents[4]
        pack=root/'Renderer/packs/UnitQualityStudy'
        if not pack.exists():self.skipTest('Local study assets not prepared')
        build() # derive the fixture from current production inputs, not stale ignored output
        original=json.loads((root/'Renderer/packs/UnitAnimationFidelity/bindings.json').read_text())
        study=json.loads((pack/'bindings.json').read_text())
        self.assertEqual(study['unit_count'],18)
        for entry in study.values():
            if not isinstance(entry,dict):continue
            key=entry['key0'];self.assertTrue(key.startswith('PRTO_Lab_'))
            self.assertEqual(entry['sample_scale'],2 if '_ssaa_' in key else 1)
            name=key.rsplit('_',1)[1]
            before=next(v for v in original.values() if isinstance(v,dict) and
                        v.get('key0','').lower()=='prto_'+name)
            if '_baseline_' in key:
                self.assertEqual(entry['scale'],before['scale'])
                self.assertEqual(entry['offset_z'],before['offset_z'])
            for action,data in before.items():
                if isinstance(data,dict):self.assertEqual(entry[action],data)


if __name__=='__main__':unittest.main()
