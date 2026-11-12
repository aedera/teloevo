from pathlib import Path
import shutil
import subprocess

import pytest


def test_packed_ranges_and_overlapping_conversions(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("No C++ compiler available")

    source = Path(__file__).parent / "cpp" / "test_packed_ranges.cpp"
    executable = tmp_path / "test_packed_ranges"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
