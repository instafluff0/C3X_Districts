"""Object reuse requires transitive inputs, recipe, tier and intact artifacts."""
import json
from pathlib import Path
import tempfile
import unittest

from Renderer.native.record_renderer_build import unit_inputs, reusable_units, compiler_timing
from Renderer.native.record_navigation_evidence import digest


class RendererBuildTests(unittest.TestCase):
    def test_transitive_headers_and_unchanged_unrelated_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();native=root/'Renderer/native';native.mkdir(parents=True)
            (native/'a.cpp').write_text('#include "one.h"\n')
            (native/'one.h').write_text('#include "two.h"\n')
            (native/'two.h').write_text('// first\n')
            (native/'b.cpp').write_text('#include <vector>\n')
            a,b=unit_inputs('a',root),unit_inputs('b',root)
            (native/'two.h').write_text('// changed\n')
            self.assertNotEqual(a,unit_inputs('a',root));self.assertEqual(b,unit_inputs('b',root))
            (native/'two.h').unlink()
            with self.assertRaisesRegex(ValueError,'Unresolved'):unit_inputs('a',root)

    def test_changed_flags_and_tampered_object_reject_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);obj=root/'a.obj';obj.write_bytes(b'object')
            closure={'a':{'source':'hash'}}
            record={'returncode':0,'sources_unchanged':True,'tier':'normal','build_recipe':'recipe',
                    'unit_inputs':closure,'objects':{'a':digest(obj)}}
            (root/'build-evidence.json').write_text(json.dumps(record))
            self.assertEqual(['a'],reusable_units(root,closure,'normal','recipe'))
            self.assertEqual([],reusable_units(root,closure,'384','recipe'))
            self.assertEqual([],reusable_units(root,closure,'normal','changed'))
            obj.write_bytes(b'tampered')
            self.assertEqual([],reusable_units(root,closure,'normal','recipe'))

    def test_compiler_clock_midnight_and_missing_markers(self):
        lines='\n'.join(f'BUILD_TIMING phase={p} clock={t}' for p,t in zip(
            ('begin','compiler_begin','dll_done','preview_done'),
            ('23:59:59.00','23:59:59.50','00:00:01.00','00:00:02.00')))
        report=compiler_timing(lines)
        self.assertEqual(1500,report['dll_compile_link_ms'])
        self.assertEqual('unmeasured',compiler_timing('')['status'])


if __name__=='__main__':unittest.main()
