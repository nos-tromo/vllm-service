"""Regression test: the real CLI import path for eval.sweep must resolve src/.

Every other test file in this package inserts ``src/`` onto ``sys.path`` at
import time and imports ``diarize_pipeline`` before ``eval.sweep`` ever runs,
which caches the module in ``sys.modules`` and masks a missing module-level
path insertion in ``eval/sweep.py`` itself. This test runs a fresh
interpreter — with none of that side effect — to exercise the same import
sequence a bare ``python -m eval.sweep`` invocation would.
"""

import subprocess
import sys
from pathlib import Path


def test_importing_eval_sweep_makes_src_importable() -> None:
    """Running ``python -m eval.sweep`` must resolve src/ modules without relying on test side effects."""
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", "from eval.sweep import run_sweep; import diarize_pipeline; print('ok')"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
