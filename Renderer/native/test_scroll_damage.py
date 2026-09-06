"""Compile the platform-independent damage planner and check pixel coverage."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ScrollDamageTests(unittest.TestCase):
    def test_coverage_and_translation_guards(self):
        compiler = shutil.which('c++') or shutil.which('g++')
        if compiler is None:
            self.skipTest('portable C++ compiler unavailable; native smoke also exercises damage')
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / 'scroll_damage_test'
            subprocess.run([compiler, '-std=c++17', '-O2', '-I', str(Path(__file__).parent),
                            str(Path(__file__).with_suffix('.cpp')), '-o', str(executable)], check=True)
            subprocess.run([str(executable)], check=True)


if __name__ == '__main__':
    unittest.main()
