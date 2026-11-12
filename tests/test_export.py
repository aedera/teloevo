from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

import teloevo.export as exporter


def save_example(path, dtype="S4", **overrides):
    arrays = {
        "generations": np.asarray([0, 10]),
        "lookup_sequences": np.asarray(["ACGT", "TTAA", "CCCC"], dtype=dtype),
        "symbols_000000": np.asarray([0, 1]),
        "symbols_000001": np.asarray([1, 2, 0]),
    }
    arrays.update(overrides)
    np.savez_compressed(path, **arrays)


def read_outputs(fasta, gff):
    lines = fasta.read_text().splitlines()
    sequence = "".join(lines[1:])
    features = [line.split("\t") for line in gff.read_text().splitlines() if not line.startswith("#")]
    return lines, sequence, features


@pytest.mark.parametrize("dtype", ["S4", "U4"])
def test_last_checkpoint_preserves_sequence_order_and_annotation_coordinates(tmp_path, dtype):
    source = tmp_path / "trajectory.npz"
    save_example(source, dtype)
    fasta, gff = exporter.export_checkpoint(source, tmp_path / "result", seqid="CEN2")
    lines, sequence, features = read_outputs(fasta, gff)
    assert lines[0] == ">CEN2 generation=10"
    assert sequence == "TTAACCCCACGT"
    assert [(int(f[3]), int(f[4])) for f in features] == [(1, 4), (5, 8), (9, 12)]
    for index, (feature, expected_symbol) in enumerate(zip(features, [1, 2, 0])):
        attributes = dict(item.split("=", 1) for item in feature[8].split(";"))
        assert feature[:3] == ["CEN2", "teloevo", "satellite_DNA"]
        assert feature[6] == "+"
        assert attributes == {"ID": f"satellite_{index + 1}", "SymbolID": str(expected_symbol), "Length": "4"}
        assert sequence[int(feature[3]) - 1:int(feature[4])] == ["TTAA", "CCCC", "ACGT"][index]


def test_explicit_generation_and_fasta_wrapping(tmp_path):
    source = tmp_path / "trajectory.npz"
    save_example(source, symbols_000000=np.asarray([0] * 23))
    fasta, gff = exporter.export_checkpoint(source, tmp_path / "generation0", generation=0)
    lines, sequence, features = read_outputs(fasta, gff)
    assert lines[0] == ">centromere generation=0"
    assert sequence == "ACGT" * 23
    assert [len(line) for line in lines[1:]] == [80, 12]
    assert len(features) == 23
    assert int(features[-1][4]) == len(sequence)


def test_exporter_runs_as_an_independent_copied_script(tmp_path):
    source = tmp_path / "trajectory.npz"
    save_example(source)
    script = tmp_path / "export.py"
    shutil.copyfile(Path(exporter.__file__), script)
    subprocess.run([sys.executable, str(script), str(source), "--generation", "0",
                    "--output-prefix", str(tmp_path / "standalone"), "--seqid", "Chr1"],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    lines, sequence, features = read_outputs(tmp_path / "standalone.fasta", tmp_path / "standalone.gff3")
    assert lines[0] == ">Chr1 generation=0"
    assert sequence == "ACGTTTAA"
    assert len(features) == 2


@pytest.mark.parametrize("existing_suffix", [".fasta", ".gff3"])
def test_existing_outputs_are_preserved(tmp_path, existing_suffix):
    source = tmp_path / "trajectory.npz"
    save_example(source)
    existing = tmp_path / ("result" + existing_suffix)
    existing.write_text("preserve this file\n")
    with pytest.raises(FileExistsError):
        exporter.export_checkpoint(source, tmp_path / "result")
    assert existing.read_text() == "preserve this file\n"
    assert not (tmp_path / ("result" + (".gff3" if existing_suffix == ".fasta" else ".fasta"))).exists()


@pytest.mark.parametrize("overrides, message", [
    ({"symbols_000001": np.asarray([-1])}, "outside the sequence lookup"),
    ({"symbols_000001": np.asarray([3])}, "outside the sequence lookup"),
    ({"symbols_000001": np.asarray([1.0])}, "integer vector"),
    ({"symbols_000001": np.asarray([], dtype=int)}, "integer vector"),
    ({"lookup_sequences": np.asarray(["ACGT", "NAAA", "CCCC"])}, "A/C/G/T"),
    ({"lookup_sequences": np.asarray(["ACGT", "AAA", "CCCC"])}, "same length"),
    ({"generations": np.asarray([10, 10])}, "distinct nonnegative integers"),
])
def test_invalid_snapshots_do_not_create_outputs(tmp_path, overrides, message):
    source = tmp_path / "invalid.npz"
    save_example(source, **overrides)
    with pytest.raises(ValueError, match=message):
        exporter.export_checkpoint(source, tmp_path / "result")
    assert not (tmp_path / "result.fasta").exists()
    assert not (tmp_path / "result.gff3").exists()


def test_missing_generation_and_summary_only_inputs_are_explained(tmp_path):
    source = tmp_path / "trajectory.npz"
    save_example(source)
    with pytest.raises(ValueError, match="Generation 5 is not saved"):
        exporter.export_checkpoint(source, tmp_path / "missing", generation=5)
    summary = tmp_path / "summary.npz"
    np.savez(summary, generations=[0, 10], total_monomers=[2, 3])
    with pytest.raises(ValueError, match="summary-only"):
        exporter.export_checkpoint(summary, tmp_path / "summary")
    assert not list(tmp_path.glob("*.fasta"))
    assert not list(tmp_path.glob("*.gff3"))


def test_plain_npy_input_is_explained(tmp_path):
    source = tmp_path / "summary.npy"
    np.save(source, np.zeros((2, 8), dtype=int))
    with pytest.raises(ValueError, match="NPZ trajectory"):
        exporter.export_checkpoint(source, tmp_path / "result")
    assert not list(tmp_path.glob("*.fasta"))
    assert not list(tmp_path.glob("*.gff3"))
