import json
import shutil

import numpy as np
import pytest

from teloevo.symbolic_centromere import (
    CentromereState,
    SimulationParameters,
    SymbolicDongSimulator,
    dong_parameters,
)
from teloevo.symbolic_centromere_cpp import (
    CppBlockedSymbolDongSimulator,
    CppSymbolicDongSimulator,
)


def _write_cen178_inputs(tmp_path, rates):
    rates_path = tmp_path / "rates.npy"
    spectrum_path = tmp_path / "spectrum.json"

    np.save(rates_path, np.asarray(rates, dtype=float))

    spectrum = {
        "A>C": 0.0,
        "A>G": 1.0,
        "A>T": 0.0,
        "C>A": 0.0,
        "C>G": 0.0,
        "C>T": 1.0,
        "G>A": 1.0,
        "G>C": 0.0,
        "G>T": 0.0,
        "T>A": 0.0,
        "T>C": 1.0,
        "T>G": 0.0,
    }

    spectrum_path.write_text(json.dumps(spectrum))
    return rates_path, spectrum_path


def _cen178_parameters(rates_path, spectrum_path):
    return SimulationParameters(
        snp_rate_per_bp=1.97e-8,
        snp_model="cen178",
        snp_position_rates_path=str(rates_path),
        snp_spectrum_path=str(spectrum_path),
        snp_background_rate=None,
        ins_rate_per_bp=0.0,
        del_rate_per_bp=0.0,
        large_del_rate_per_bp=0.0,
        conversion_rate_per_bp=0.0,
        min_array_bp=None,
        max_array_bp=None,
    )


def _require_cpp_compiler():
    if shutil.which("g++") is None and shutil.which("clang++") is None:
        pytest.skip("No C++ compiler available")


def test_dong_parameters_selects_snp_model(tmp_path):
    rates_path, spectrum_path = _write_cen178_inputs(
        tmp_path,
        [0.0, 1e-8, 0.0, 0.0],
    )

    default = dong_parameters()
    assert default.snp_model == "dong"

    cen178 = dong_parameters(
        snp_model="cen178",
        snp_position_rates=str(rates_path),
        snp_spectrum=str(spectrum_path),
    )
    assert cen178.snp_model == "cen178"

    disabled = dong_parameters(
        include_snps=False,
        snp_model="cen178",
    )
    assert disabled.snp_model == "none"


def test_python_cen178_snp_uses_position_and_spectrum(tmp_path):
    rates_path, spectrum_path = _write_cen178_inputs(
        tmp_path,
        [0.0, 1.0, 0.0, 0.0],
    )

    params = _cen178_parameters(rates_path, spectrum_path)

    sim = SymbolicDongSimulator.homogeneous(
        n_monomers=8,
        monomer_sequence="AAAA",
        parameters=params,
        seed=7,
        record_events=True,
    )

    for generation in range(1, 21):
        assert sim.apply_snp(generation=generation)

    assert all(event.phase == 1 for event in sim.events)

    for index in range(sim.state.n_monomers):
        sequence = sim.state.monomer_sequence(index)
        assert sequence[0] == "A"
        assert sequence[2:] == "AA"


def test_cen178_rate_length_must_match_monomer_length(tmp_path):
    rates_path, spectrum_path = _write_cen178_inputs(
        tmp_path,
        [1e-8, 1e-8, 1e-8],
    )

    params = _cen178_parameters(rates_path, spectrum_path)

    with pytest.raises(ValueError, match="must match the monomer length"):
        SymbolicDongSimulator.homogeneous(
            n_monomers=4,
            monomer_sequence="AAAA",
            parameters=params,
            seed=1,
        )


def test_cpp_cen178_blocked_matches_vector_trajectory(tmp_path):
    _require_cpp_compiler()

    rates_path, spectrum_path = _write_cen178_inputs(
        tmp_path,
        [0.0, 0.08, 0.0, 0.0],
    )

    params = _cen178_parameters(rates_path, spectrum_path)

    common = dict(
        n_monomers=20,
        monomer_sequence="AAAA",
        parameters=params,
        seed=11,
        record_events=True,
    )

    vector_sim = CppSymbolicDongSimulator.homogeneous(
        **common,
        symbol_storage="vector",
    )

    blocked_sim = CppBlockedSymbolDongSimulator.homogeneous(
        **common,
        symbol_block_size=8,
    )

    vector_result = vector_sim.run(
        n_generations=10,
        checkpoints=[0, 5, 10],
    )

    blocked_result = blocked_sim.run(
        n_generations=10,
        checkpoints=[0, 5, 10],
    )

    assert blocked_result.lookup == vector_result.lookup
    assert [
        checkpoint.symbols
        for checkpoint in blocked_result.checkpoints
    ] == [
        checkpoint.symbols
        for checkpoint in vector_result.checkpoints
    ]
    assert blocked_sim.events == vector_sim.events

    assert blocked_sim.events
    assert all(event.phase == 1 for event in blocked_sim.events)
