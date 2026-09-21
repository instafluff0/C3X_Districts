"""Exercise the real elevated capture launcher using a harmless native stand-in."""
import json
import unittest
from Renderer.lab.platform import ROOT, windows_root
from Renderer.native.test_composition_recording import run
from Renderer.tools.inspect_composition_recording import inspect


class CaptureLaunchTests(unittest.TestCase):
    def launch(self, name, compile_option=''):
        folder = ROOT / f'Renderer/native/build/{name}'
        folder.mkdir(parents=True, exist_ok=True)
        result = run(fr'..\..\tcc\tcc.exe -m32 {compile_option} -o build\{name}\Civ3Conquests.exe test_capture_launch.c')
        self.assertEqual(result['status'], 'pass', result)
        captures = ROOT / 'Renderer/native/build/live-captures'
        before = set(captures.iterdir())
        target = windows_root() / f'Renderer/native/build/{name}'
        result = run('powershell.exe -NoProfile -ExecutionPolicy Bypass -File ..\\tools\\capture_game.ps1 '
                     f'-ConquestsDirectory "{target}"', timeout=120)
        added = set(captures.iterdir()) - before
        self.assertEqual(len(added), 1, added)
        capture = added.pop()
        session = json.loads((capture / 'session.json').read_text(encoding='utf-8-sig'))
        # These are synthetic launcher witnesses, never the newest user run.
        capture = capture.rename(folder / capture.name)
        return folder, result, capture, session

    def test_diagnostics_survive_capture_host_elevation(self):
        folder, result, capture, session = self.launch('capture-launch')
        self.assertEqual(result['status'], 'pass', result)
        self.assertTrue(session['capture_host_elevated'])
        self.assertEqual(session['game_exit_code'], 0, session)
        self.assertEqual(session['result'], 'game-exited', session)
        self.assertTrue(session['recording_present'], session)
        self.assertGreater(session['recording_bytes'], 100)
        self.assertTrue((capture / 'renderer-runtime.log').is_file())
        report = inspect(capture / 'composition.c3xr')
        self.assertFalse(report['truncated_tail'], report)
        (folder / 'receipt.json').write_text(json.dumps({
            'status': 'pass', 'game_launched': False,
            'capture': capture.name, 'session': session, 'recording': report,
        }, indent=2) + '\n')
        print('PASS real capture launcher: elevated host, inherited settings, staged DLL journal; no game launched')
        # Do not dispatch another native process after unconfirmed transport.
        self.check_missing_recording_is_not_success()

    def check_missing_recording_is_not_success(self):
        folder, result, capture, session = self.launch('capture-launch-missing', '-DC3X_CAPTURE_SKIP_RECORDING')
        self.assertEqual(result['returncode'], 1, result)
        self.assertEqual(session['game_exit_code'], 0, session)
        self.assertEqual(session['result'], 'recording-missing', session)
        self.assertFalse(session['recording_present'], session)
        (folder / 'receipt.json').write_text(json.dumps({
            'status': 'pass', 'game_launched': False, 'capture': capture.name,
            'expected_failure': 'recording-missing', 'session': session,
        }, indent=2) + '\n')


if __name__ == '__main__':
    unittest.main()
