from pathlib import Path
import shutil
import subprocess

import pytest

from teloevo.symbolic_centromere import (
    CentromereState,
    SequenceLookup,
    SimulationParameters,
    SymbolicDongSimulator,
)
from teloevo.symbolic_centromere_cpp import (
    CppBlockedSymbolDongSimulator,
    CppSymbolicDongSimulator,
    compile_cpp_backend,
)


def no_event_parameters():
    return SimulationParameters(
        snp_rate_per_bp=0.0,
        ins_rate_per_bp=0.0,
        del_rate_per_bp=0.0,
        large_del_rate_per_bp=0.0,
        conversion_rate_per_bp=0.0,
        min_array_bp=None,
        max_array_bp=None,
    )


def require_cpp_compiler():
    if shutil.which("g++") is None and shutil.which("clang++") is None:
        pytest.skip("No C++ compiler available")


def test_backend_labels():
    assert SymbolicDongSimulator.backend == "python"
    assert CppSymbolicDongSimulator.backend == "cpp"
    assert CppBlockedSymbolDongSimulator.backend == "cpp-blocked"


def test_cpp_backend_compiles():
    require_cpp_compiler()
    executable = compile_cpp_backend()
    assert executable.exists()


def test_cached_poisson_counts_preserve_draws_and_rng_state(tmp_path):
    require_cpp_compiler()
    compiler = shutil.which("g++") or shutil.which("clang++")
    source = Path(__file__).parent / "cpp" / "test_poisson_counts.cpp"
    executable = tmp_path / "test_poisson_counts"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)


def test_symbol_array_cursor_and_insertion_buffer(tmp_path):
    require_cpp_compiler()
    compiler = shutil.which("g++") or shutil.which("clang++")
    source = Path(__file__).parent / "cpp" / "test_symbol_array_cursor.cpp"
    executable = tmp_path / "test_symbol_array_cursor"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", str(source), "-o", str(executable)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)


def test_cpp_zero_event_run_matches_python_state():
    require_cpp_compiler()

    params = no_event_parameters()

    python_sim = SymbolicDongSimulator.homogeneous(
        n_monomers=20,
        monomer_sequence="AAAA",
        parameters=params,
        seed=1,
    )

    cpp_sim = CppSymbolicDongSimulator.homogeneous(
        n_monomers=20,
        monomer_sequence="AAAA",
        parameters=params,
        seed=1,
    )

    python_result = python_sim.run(
        n_generations=10,
        checkpoints=[0, 5, 10],
    )

    cpp_result = cpp_sim.run(
        n_generations=10,
        checkpoints=[0, 5, 10],
    )

    assert [
        checkpoint.generation
        for checkpoint in cpp_result.checkpoints
    ] == [0, 5, 10]

    assert [
        checkpoint.symbols
        for checkpoint in cpp_result.checkpoints
    ] == [
        checkpoint.symbols
        for checkpoint in python_result.checkpoints
    ]

    assert cpp_result.lookup == python_result.lookup
    assert cpp_sim.state.symbols == python_sim.state.symbols
    assert cpp_sim.state.lookup.as_dict() == python_sim.state.lookup.as_dict()



def test_cpp_profile_output():
    require_cpp_compiler()

    params = no_event_parameters()

    sim = CppSymbolicDongSimulator.homogeneous(
        n_monomers=20,
        monomer_sequence="AAAA",
        parameters=params,
        seed=1,
        profile=True,
        profile_interval=4,
    )

    sim.run(
        n_generations=10,
    )

    assert [row["generation_end"] for row in sim.profile] == [4, 8, 10]
    assert [row["generation_start"] for row in sim.profile] == [1, 5, 9]
    assert [row["n_generations"] for row in sim.profile] == [4, 4, 2]

    for row in sim.profile:
        assert row["n_monomers"] == 20
        assert row["n_states"] == 1
        assert row["n_symbol_blocks"] == 1
        assert row["SNP"] == 0
        assert row["INS"] == 0
        assert row["DEL"] == 0
        assert row["Large_DEL"] == 0
        assert row["Conversion"] == 0
        assert row["time_total"] >= 0.0
        assert row["time_mutations"] == 0.0
        assert row["time_other"] >= 0.0
        assert row["intern_calls"] == 0
        assert row["new_states"] == 0
        assert row["rehashes"] == 0
        assert row["time_intern"] == 0.0
        assert row["time_conversion_extract"] == 0.0
        assert row["time_conversion_replace"] == 0.0

    assert sim.profile_io["time_cpp_process"] >= 0.0
    assert sim.profile_io["time_cpp_profiled_loop"] >= 0.0
    assert sim.profile_io["time_cpp_unprofiled_overhead"] >= 0.0
    assert sim.profile_io["time_read_lookup"] >= 0.0
    assert sim.profile_io["time_compact_outputs"] >= 0.0
    assert sim.profile_io["time_rebuild_state"] >= 0.0
    assert sim.profile_io["n_output_states"] == 1



def test_cpp_prunes_and_remaps_unused_lookup_states():
    require_cpp_compiler()

    lookup = SequenceLookup(4)
    lookup.intern("AAAA")
    used_symbol = lookup.intern("CCCC")

    state = CentromereState(
        symbols=[used_symbol] * 20,
        lookup=lookup,
    )

    sim = CppSymbolicDongSimulator(
        state=state,
        parameters=no_event_parameters(),
        seed=1,
    )

    result = sim.run(
        n_generations=1,
        checkpoints=[1],
    )

    # The unused historical state ("AAAA", old symbol 0) must not be
    # returned.  The only required old symbol (1) is compacted to 0.
    assert result.lookup == {0: "CCCC"}
    assert result.checkpoints[0].symbols == [0] * 20
    assert sim.state.symbols == [0] * 20
    assert sim.state.lookup.as_dict() == {0: "CCCC"}



def test_cpp_identical_conversions_use_fast_noop_path():
    require_cpp_compiler()

    params = SimulationParameters(
        snp_rate_per_bp=0.0,
        ins_rate_per_bp=0.0,
        del_rate_per_bp=0.0,
        large_del_rate_per_bp=0.0,
        conversion_rate_per_bp=0.25,
        conversion_geometric_p=1.0,
        min_array_bp=None,
        max_array_bp=None,
    )

    sim = CppSymbolicDongSimulator.homogeneous(
        n_monomers=20,
        monomer_sequence="AAAA",
        parameters=params,
        seed=1,
        profile=True,
        profile_interval=1,
    )

    sim.run(n_generations=1)

    row = sim.profile[0]

    assert row["Conversion"] > 0
    assert row["conversion_fast_path"] == row["Conversion"]
    assert row["conversion_noop"] == row["Conversion"]

    # Identical conversions are still counted as successful Dong-style
    # conversions, but they no longer construct/intern unchanged monomers.
    assert row["intern_calls"] == 0
    assert row["new_states"] == 0
    assert sim.state.lookup.as_dict() == {0: "AAAA"}



def test_cpp_packed_lookup_roundtrip_preserves_acgt_sequences():
    require_cpp_compiler()

    lookup = SequenceLookup(8)
    symbols = [
        lookup.intern("ACGTACGT"),
        lookup.intern("TTTTCCCC"),
        lookup.intern("GGGGAAAA"),
    ]

    state = CentromereState(
        symbols=[symbols[0], symbols[1], symbols[2], symbols[0]],
        lookup=lookup,
    )

    sim = CppSymbolicDongSimulator(
        state=state,
        parameters=no_event_parameters(),
        seed=1,
    )

    result = sim.run(
        n_generations=1,
        checkpoints=[1],
    )

    assert result.lookup == {
        0: "ACGTACGT",
        1: "TTTTCCCC",
        2: "GGGGAAAA",
    }
    assert result.checkpoints[0].symbols == [0, 1, 2, 0]
    assert sim.state.lookup.as_dict() == result.lookup
    assert sim.state.symbols == [0, 1, 2, 0]


def test_cpp_packed_backend_rejects_monomers_over_256_bp():
    require_cpp_compiler()

    sequence = "A" * 257
    lookup = SequenceLookup(len(sequence))
    symbol = lookup.intern(sequence)

    state = CentromereState(
        symbols=[symbol] * 4,
        lookup=lookup,
    )

    sim = CppSymbolicDongSimulator(
        state=state,
        parameters=no_event_parameters(),
        seed=1,
    )

    with pytest.raises(RuntimeError, match="up to 256 bp"):
        sim.run(n_generations=0)



def test_cpp_insertion_profile_breakdown():
    require_cpp_compiler()

    params = SimulationParameters(
        snp_rate_per_bp=0.0,
        ins_rate_per_bp=0.20,
        del_rate_per_bp=0.0,
        large_del_rate_per_bp=0.0,
        conversion_rate_per_bp=0.0,
        ins_span_lambda=2.0,
        min_array_bp=None,
        max_array_bp=None,
    )

    sim = CppSymbolicDongSimulator.homogeneous(
        n_monomers=20,
        monomer_sequence="ACGT",
        parameters=params,
        seed=1,
        profile=True,
        profile_interval=1,
    )

    sim.run(n_generations=1)

    row = sim.profile[0]

    assert row["INS"] > 0
    assert row["ins_inserted_symbols"] >= row["INS"]
    assert row["ins_shifted_symbols"] >= row["INS"]
    assert row["ins_reallocations"] >= 0
    assert row["time_ins_chimera"] >= 0.0
    assert row["time_ins_prepare"] >= 0.0
    assert row["time_ins_vector_insert"] >= 0.0

    # The sub-timers are components of total insertion time.
    assert (
        row["time_ins_chimera"]
        + row["time_ins_prepare"]
        + row["time_ins_vector_insert"]
        <= row["time_ins"] + 1e-9
    )



def test_cpp_blocked_storage_matches_vector_trajectory():
    require_cpp_compiler()

    params = SimulationParameters(
        snp_rate_per_bp=1.0e-3,
        ins_rate_per_bp=1.0e-3,
        del_rate_per_bp=8.0e-4,
        large_del_rate_per_bp=0.0,
        conversion_rate_per_bp=1.0e-3,
        ins_span_lambda=2.0,
        del_span_lambda=2.0,
        conversion_geometric_p=0.5,
        min_array_bp=None,
        max_array_bp=None,
    )

    common = dict(
        n_monomers=40,
        monomer_sequence="ACGTACGT",
        parameters=params,
        seed=17,
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

    checkpoints = [0, 25, 50]

    vector_result = vector_sim.run(
        n_generations=50,
        checkpoints=checkpoints,
    )

    blocked_result = blocked_sim.run(
        n_generations=50,
        checkpoints=checkpoints,
    )

    assert blocked_result.lookup == vector_result.lookup
    assert [
        checkpoint.symbols
        for checkpoint in blocked_result.checkpoints
    ] == [
        checkpoint.symbols
        for checkpoint in vector_result.checkpoints
    ]
    assert blocked_sim.state.symbols == vector_sim.state.symbols
    assert blocked_sim.state.lookup.as_dict() == vector_sim.state.lookup.as_dict()
    assert blocked_sim.events == vector_sim.events


def test_cpp_blocked_profile_reports_local_symbol_movement():
    require_cpp_compiler()

    params = SimulationParameters(
        snp_rate_per_bp=0.0,
        ins_rate_per_bp=0.02,
        del_rate_per_bp=0.0,
        large_del_rate_per_bp=0.0,
        conversion_rate_per_bp=0.0,
        ins_span_lambda=2.0,
        min_array_bp=None,
        max_array_bp=None,
    )

    sim = CppBlockedSymbolDongSimulator.homogeneous(
        n_monomers=64,
        monomer_sequence="ACGT",
        parameters=params,
        seed=3,
        profile=True,
        profile_interval=1,
        symbol_block_size=8,
    )

    sim.run(n_generations=1)

    row = sim.profile[0]

    assert row["INS"] > 0
    assert row["n_symbol_blocks"] > 1
    assert row["ins_shifted_symbols"] >= 0
    assert row["ins_block_splits"] >= 0
    assert row["time_ins_vector_insert"] >= 0.0
