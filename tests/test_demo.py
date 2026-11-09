import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_demo(*args):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "demo.py"), *args],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )


def test_default_example_shows_the_silent_failure():
    proc = run_demo()
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    for stage in ("Normalization:", "Tokens", "BM25 ranking", "n-gram ranking", "Fused ranking"):
        assert stage in out
    assert "NORMALIZATION OFF" in out
    # BM25 matches nothing without normalization, and says so instead of erroring.
    assert "nothing matched, and no error was raised" in out
    assert "expected passage at rank 1" in out


def test_accepts_a_query_that_is_not_labelled():
    proc = run_demo("قانون العمل", "--top", "3")
    assert proc.returncode == 0, proc.stderr
    assert "Not a labelled query" in proc.stdout


def test_query_with_nothing_to_match_does_not_crash():
    proc = run_demo("???")
    assert proc.returncode == 0, proc.stderr
