"""Exact inherited shader preservation and failed-admission transaction tests."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

from Renderer.tools import prepare_resident_submission_shaders as tool


class ResidentSubmissionShaderTests(unittest.TestCase):
    def test_staging_guard_matches_runtime_entries_and_rejects_old_input_bindings(self):
        batch = (tool.ROOT/'Renderer/native/BUILD_RENDERER64.bat').read_text()
        entries = re.findall(r'call :require_resident_entry "([^"]+)" "([^"]+)"', batch)
        source_bytes = {name: (tool.ROOT/path).read_bytes() for name, path in tool.SOURCES.items()}
        expected = {(name, entry) for name, (_, _, added) in tool.adapters(source_bytes).items() for entry in added}
        actual = {('Renderer/native/' + name.replace('\\', '/'), entry) for name, entry in entries}
        self.assertEqual(expected, actual)
        self.assertEqual(len(expected), len(entries))
        guard = batch[batch.index(':require_resident_entry\n'):batch.index(':resident_entry_missing\n')]
        tokens = re.findall(r'findstr /l /c:"([^"]+)"', guard)
        self.assertEqual(['%~2(', 'uint selection:TEXCOORD1;',
                          'StructuredBuffer<ResidentPlacement> C3XResidentPlacements:register(t15);'], tokens)
        self.assertLess(batch.index('if /i "%~1"=="no-stage" goto done'), batch.index('call :require_resident_entry'))
        self.assertLess(batch.index('call :require_resident_entry'), batch.index('copy /y "build\\candidate'))
        self.assertIn('call BUILD.bat candidate-compile no-stage', batch)
        consumers = ['Renderer/native/rigid_object_gpu.h', 'Renderer/native/source_fidelity/runtime.h',
                     'Renderer/native/environment_refresh/reflection.h', 'Renderer/native/render_core/source_shadow.h',
                     'Renderer/sandbox/fresh_pipeline.h']
        compiled = set()
        for name in consumers:
            compiled.update(re.findall(r'"(VSResident\w+)"', (tool.ROOT/name).read_text()))
        self.assertEqual(compiled, {entry for _, entry in actual})
        def admitted(directory):
            for name, entry in actual:
                path = directory/name
                if not path.is_file():
                    return False
                text = path.read_text()
                if not all(token.replace('%~2', entry) in text for token in tokens):
                    return False
            return True
        baseline = tool.ROOT/'Renderer/packs/Renderer64CutoverControl'
        self.assertFalse(admitted(baseline))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'candidate'
            tool.prepare(baseline, output)
            self.assertTrue(admitted(output))
            shader = output/'Renderer/native/city_fidelity/rigid_caster.hlsl'
            original = shader.read_text()
            for old in ('VSResidentPlacedCaster(', 'uint selection:TEXCOORD1;',
                        'StructuredBuffer<ResidentPlacement> C3XResidentPlacements:register(t15);'):
                shader.write_text(original.replace(old, 'incompatible binding'))
                self.assertFalse(admitted(output), old)

    def baseline(self, directory):
        baseline = Path(directory) / "baseline"
        baseline.mkdir()
        source_bytes = {name: (tool.ROOT / path).read_bytes() for name, path in tool.SOURCES.items()}
        for name, (_, required, _) in tool.adapters(source_bytes).items():
            target = baseline / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("// Immutable selected material shader.\n" +
                              "\n".join("P " + entry + "(InstanceInput input) { return input; }" for entry in required))
        (baseline / "unchanged.bin").write_bytes(bytes(range(256)))
        return baseline

    def test_actual_pinned_pack_keeps_exact_prefix_and_unrelated_files(self):
        baseline = tool.ROOT / "Renderer/packs/Renderer64CutoverControl"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            before = tool.inventory(baseline)
            receipt = tool.prepare(baseline, output)
            self.assertEqual(before, tool.inventory(baseline))
            self.assertEqual(6, len(receipt["changed"]))
            self.assertEqual(receipt, json.loads((output / "resident-submission-overlay.json").read_text()))
            for name, expected in before.items():
                inherited = (baseline / name).read_bytes()
                candidate = (output / name).read_bytes()
                if name not in receipt["changed"]:
                    self.assertEqual(inherited, candidate, name)
                    continue
                self.assertTrue(candidate.startswith(inherited), name)
                self.assertEqual(expected, receipt["changed"][name]["baseline_sha256"])
                self.assertEqual(hashlib.sha256(candidate).hexdigest(), receipt["changed"][name]["candidate_sha256"])
                text = candidate.decode()
                for entry in receipt["changed"][name]["entries"]:
                    self.assertEqual(1, len(re.findall(r"\b" + entry + r"\s*\([^)]*\)\s*\{", text)), name)
                self.assertEqual(1, len(re.findall(r"struct ResidentInstanceInput\s*\{", text)), name)
            self.assertNotIn(str(Path(directory)), json.dumps(receipt))

    def test_missing_entry_and_duplicate_namespace_leave_no_candidate(self):
        for kind in ("missing", "duplicate"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                baseline = self.baseline(directory)
                target = baseline / "Renderer/native/city_fidelity/objects.hlsl"
                target.write_text("P Other(InstanceInput input) { return input; }" if kind == "missing" else
                                  target.read_text() + "\nstruct ResidentPlacement {};\n")
                before = tool.inventory(baseline)
                output = Path(directory) / "candidate"
                with self.assertRaises(ValueError):
                    tool.prepare(baseline, output)
                self.assertFalse(output.exists())
                self.assertEqual(before, tool.inventory(baseline))

    def test_input_mutation_during_copy_is_rejected_without_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = self.baseline(directory)
            output = Path(directory) / "candidate"
            original = shutil.copytree
            def mutate(source, target, *args, **kwargs):
                result = original(source, target, *args, **kwargs)
                (baseline / "unchanged.bin").write_bytes(b"changed after capture")
                return result
            with patch.object(tool.shutil, "copytree", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    tool.prepare(baseline, output)
            self.assertFalse(output.exists())
            self.assertEqual(["baseline"], sorted(path.name for path in Path(directory).iterdir()))

    def test_existing_output_and_nested_output_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = self.baseline(directory)
            output = Path(directory) / "candidate"
            output.mkdir()
            (output / "owned").write_bytes(b"preserve")
            with self.assertRaisesRegex(ValueError, "already exists"):
                tool.prepare(baseline, output)
            self.assertEqual(b"preserve", (output / "owned").read_bytes())
            with self.assertRaisesRegex(ValueError, "separate directories"):
                tool.prepare(baseline, baseline / "nested")


if __name__ == "__main__":
    unittest.main()
