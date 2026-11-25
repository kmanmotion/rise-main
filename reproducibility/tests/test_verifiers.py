"""Regression tests for the numerical replay entry points (Python 3.12, NumPy, SciPy)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "evidence" / "scripts"


class ReplayTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.scripts = Path(directory.name) / "reproducibility" / "evidence" / "scripts"
        self.scripts.mkdir(parents=True)
        for script in SCRIPTS.glob("verify_*.py"):
            (self.scripts / script.name).write_bytes(script.read_bytes())

    def replay(self, name, optimized=False, fault=""):
        args = [sys.executable]
        if optimized:
            args.append("-O")
        if fault:
            args += ["-c", textwrap.dedent(fault) + "\nimport runpy, sys\nrunpy.run_path(sys.argv[1], run_name='__main__')"]
        process = subprocess.run(args + [str(self.scripts / name)], capture_output=True, text=True)
        self.assertTrue(process.stdout.strip(), process.stderr)
        return process.returncode, json.loads(process.stdout)

    def test_published_examples(self):
        for script in ("verify_reduced_examples.py", "verify_joint_witness.py"):
            with self.subTest(script=script):
                code, result = self.replay(script)
                self.assertEqual((code, result["status"]), (0, "PASS"), result)

    def test_wrong_reference(self):
        path = self.scripts / "verify_reduced_examples.py"
        source = path.read_text(encoding="utf-8")
        original = "9.270995112117777"
        self.assertEqual(source.count(original), 1)
        path.write_text(source.replace(original, "99.0"), encoding="utf-8")
        for optimized in (False, True):
            with self.subTest(optimized=optimized):
                code, result = self.replay(path.name, optimized)
                self.assertNotEqual(code, 0)
                self.assertEqual(result["status"], "FAIL")
                self.assertIn("Figure 5", result["check"])
                self.assertFalse((self.scripts.parent / "results" / "REDUCED_EXAMPLES.json").exists())

    def test_failed_reference(self):
        fault = """
            from types import SimpleNamespace
            import scipy.optimize
            scipy.optimize.root = lambda *a, **k: SimpleNamespace(success=False, message='test')
        """
        code, result = self.replay("verify_joint_witness.py", True, fault)
        self.assertNotEqual(code, 0)
        self.assertEqual((result["status"], result["check"]), ("FAIL", "reduced reference"))

    def test_incomplete_integrations(self):
        cases = [(1, False, "equilibration"), (2, True, "intact support"), (3, False, "blocked support")]
        for which, success, check in cases:
            with self.subTest(check=check):
                fault = f"""
                    from types import SimpleNamespace
                    import numpy as np
                    import scipy.integrate
                    real_solve = scipy.integrate.solve_ivp
                    calls = 0
                    def solve(*args, **kwargs):
                        global calls
                        calls += 1
                        if calls == {which}:
                            return SimpleNamespace(success={success}, t=np.array([0.05]),
                                                   y=np.zeros((51, 1)), message="injected partial integration")
                        return real_solve(*args, **kwargs)
                    scipy.integrate.solve_ivp = solve
                """
                code, result = self.replay("verify_joint_witness.py", True, fault)
                self.assertNotEqual(code, 0)
                self.assertEqual((result["status"], result["check"]), ("FAIL", check))


if __name__ == "__main__":
    unittest.main(verbosity=2)
