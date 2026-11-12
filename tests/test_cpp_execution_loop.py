import csv
from pathlib import Path
import shutil
import subprocess

import pytest

from teloevo.symbolic_centromere_cpp import compile_cpp_backend


@pytest.fixture(scope="module")
def compiler():
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("No C++ compiler available")
    return compiler


def test_nonprofiling_path_preserves_sequences_and_rng(tmp_path, compiler):
    source = Path(__file__).parent / "cpp" / "test_execution_loop.cpp"
    executable = tmp_path / "test_execution_loop"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", str(source), "-o", str(executable)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize("start,n_generations,requested,expected", [
    (0, 10, [10, 0, 5, 5, 0, 11, -1], [0, 5, 10]),
    (100, 10, [0, 99, 100, 110, 105, 105, 999], [100, 105, 110]),
    (100, 0, [100, 100, 99, 101], [100]),
    (0, 10, [], []),
    (100, 10, [99, 111], []),
    (0, 10, [1, 5], [1, 5]),
])
def test_backend_checkpoint_cursor(
    tmp_path, compiler, profile, start, n_generations, requested, expected
):
    # Exercise the raw backend because the Python wrapper already sorts and
    # deduplicates checkpoints, and rejects requests outside the run interval.
    config = {
        "n_generations": n_generations,
        "start_generation": start,
        "checkpoints": ",".join(map(str, requested)) or "-",
        "profile": int(profile),
        "profile_interval": 4 if profile else 0,
    }
    (tmp_path / "config").write_text("".join(f"{k} {v}\n" for k, v in config.items()))
    (tmp_path / "symbols").write_text("0 1 0\n")
    (tmp_path / "lookup").write_text("AAAA\nCCCC\n")
    outputs = ["checkpoints", "final", "lookup_out", "events", "profile"]
    subprocess.run(
        [str(compile_cpp_backend(compiler=compiler)),
         *[str(tmp_path / name) for name in ["config", "symbols", "lookup", *outputs]]],
        check=True, capture_output=True, text=True,
    )
    lines = (tmp_path / "checkpoints").read_text().splitlines()
    assert [int(line.split("\t")[0]) for line in lines] == expected
    assert (tmp_path / "final").read_text().split() == ["0", "1", "0"]
    if profile:
        with (tmp_path / "profile").open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        assert [int(row["generation_end"]) for row in rows] == [
            start + min(offset, n_generations)
            for offset in range(4, n_generations + 4, 4)
        ]
    else:
        assert not (tmp_path / "profile").exists()
