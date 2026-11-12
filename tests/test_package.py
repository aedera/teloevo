import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from teloevo import DEFAULT_MONOMER, cen178_parameters, example_data_path, load_example_centromere
from teloevo.symbolic_centromere import SymbolicDongSimulator, _prepare_snp_model


def test_bundled_resources_and_model_are_consistent():
    provenance = json.loads(example_data_path("provenance.json").read_text())
    for name, expected in provenance["bundled_sha256"].items():
        assert hashlib.sha256(example_data_path(name).read_bytes()).hexdigest() == expected
    consensus = "".join(line for line in example_data_path("cen178_consensus.fasta").read_text().splitlines()
                        if not line.startswith(">"))
    assert DEFAULT_MONOMER == consensus
    assert len(consensus) == 178
    state = load_example_centromere()
    assert state.n_monomers == 12_746
    assert state.length_bp == 2_268_788
    assert state.monomer_length == 178
    parameters = cen178_parameters()
    assert parameters.min_array_bp is None
    assert parameters.max_array_bp == 100_000_000
    rates, _ = _prepare_snp_model(parameters, state.monomer_length)
    assert rates.shape == (178,)
    assert np.count_nonzero(rates == 1e-10) == 139
    assert rates.mean() == pytest.approx(3.059627787446573e-10)
    # The corrected default also works directly with the 178-position model.
    SymbolicDongSimulator.homogeneous(10, parameters=parameters, seed=846)


def test_gff_matches_the_local_fasta_and_preserves_genomic_coordinates():
    sequence = "".join(line for line in example_data_path("CEN1.fasta").read_text().splitlines()
                       if not line.startswith(">"))
    assert len(sequence) == 2_287_856
    ids = set()
    for line in example_data_path("CEN1.gff3").read_text().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        start, end = map(int, fields[3:5])
        attributes = dict(item.split("=", 1) for item in fields[8].split(";"))
        assert fields[0] == "CEN1"
        assert 1 <= start <= end <= len(sequence)
        assert int(attributes["OriginalStart"]) == start + 14_841_109
        assert int(attributes["OriginalEnd"]) == end + 14_841_109
        assert int(attributes["Length"]) == end - start + 1
        assert attributes["ID"] not in ids
        ids.add(attributes["ID"])
    assert len(ids) == 12_746


def test_installed_cli_saves_full_seeded_checkpoints_without_overwriting(tmp_path):
    output = tmp_path / "example.npz"
    command = [sys.executable, "-m", "teloevo", "--backend", "python",
               "--generations", "20", "--checkpoints", "0,10,20", "--seed", "846",
               "--output", str(output)]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    with np.load(output, allow_pickle=False) as result:
        assert result["generations"].tolist() == [0, 10, 20]
        assert result["total_monomers"][0] == 12_746
        initial = load_example_centromere()
        sequences = result["lookup_sequences"][result["symbols_000000"]].astype(str).tolist()
        assert sequences == [initial.lookup.sequence(s) for s in initial.symbols]
        metadata = json.loads(result["metadata_json"].item())
        assert metadata["parameters"]["snp_model"] == "cen178"
        assert metadata["backend"] == "python"
        simulator = SymbolicDongSimulator(state=initial, parameters=cen178_parameters(), seed=846)
        expected = simulator.run(20, checkpoints=[0, 10, 20])
        for i, checkpoint in enumerate(expected.checkpoints):
            expected_sequences = [expected.lookup[s] for s in checkpoint.symbols]
            assert result["lookup_sequences"][result[f"symbols_{i:06d}"]].astype(str).tolist() == expected_sequences
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    rerun = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert rerun.returncode != 0
    assert "already exists" in rerun.stderr
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("extra", [
    ["--fasta", "missing.fasta"],
    ["--checkpoints", "-1,20"],
    ["--seed", str(2**64)],
])
def test_cli_rejects_inconsistent_arguments_without_output(tmp_path, extra):
    output = tmp_path / "invalid.npz"
    run = subprocess.run([sys.executable, "-m", "teloevo", "--output", str(output),
                          "--generations", "20", *extra], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode != 0
    assert not output.exists()
