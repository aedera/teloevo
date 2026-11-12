import numpy as np

from teloevo.symbolic_centromere import (
    CentromereState,
    SequenceLookup,
    SimulationParameters,
    SymbolicDongSimulator,
)


def no_limits():
    return SimulationParameters(
        snp_rate_per_bp=0.0,
        ins_rate_per_bp=0.0,
        del_rate_per_bp=0.0,
        large_del_rate_per_bp=0.0,
        conversion_rate_per_bp=0.0,
        min_array_bp=None,
        max_array_bp=None,
    )


def test_lookup_reuses_identical_sequence():
    lookup = SequenceLookup(4)
    a = lookup.intern("AAAA")
    b = lookup.intern("AAAA")
    assert a == b
    assert len(lookup) == 1


def test_phase_insertion_creates_expected_chimera():
    state = CentromereState.from_sequences(["AAAA", "CCCC"])
    sim = SymbolicDongSimulator(state, no_limits(), seed=1)

    ok = sim.apply_phase_insertion(
        start_monomer=0,
        span_monomers=1,
        phase=2,
    )

    assert ok
    seqs = [state.monomer_sequence(i) for i in range(state.n_monomers)]
    assert seqs == ["AAAA", "CCAA", "CCCC"]


def test_phase_deletion_creates_expected_chimera():
    state = CentromereState.from_sequences(["AAAA", "CCCC"])
    sim = SymbolicDongSimulator(state, no_limits(), seed=1)

    ok = sim.apply_phase_deletion(
        start_monomer=0,
        span_monomers=1,
        phase=2,
    )

    assert ok
    seqs = [state.monomer_sequence(i) for i in range(state.n_monomers)]
    assert seqs == ["AACC"]


def test_homogeneous_phase_insertion_reuses_symbol():
    state = CentromereState.homogeneous(4, "AAAA")
    sim = SymbolicDongSimulator(state, no_limits(), seed=1)

    ok = sim.apply_phase_insertion(0, 2, 1)

    assert ok
    assert state.n_monomers == 6
    assert len(state.lookup) == 1
    assert len(set(state.symbols)) == 1


def test_homogeneous_phase_deletion_reuses_symbol():
    state = CentromereState.homogeneous(5, "AAAA")
    sim = SymbolicDongSimulator(state, no_limits(), seed=1)

    ok = sim.apply_phase_deletion(1, 2, 3)

    assert ok
    assert state.n_monomers == 3
    assert len(state.lookup) == 1
    assert len(set(state.symbols)) == 1


def test_zero_rates_leave_array_unchanged():
    state = CentromereState.homogeneous(20, "AAAA")
    sim = SymbolicDongSimulator(state, no_limits(), seed=1)

    summary = sim.step_generation(1)

    assert summary.snp == 0
    assert summary.ins == 0
    assert summary.deletion == 0
    assert summary.large_deletion == 0
    assert summary.conversion == 0
    assert state.n_monomers == 20


def test_seeded_snp_is_reproducible():
    params = no_limits()
    state1 = CentromereState.homogeneous(5, "AAAA")
    state2 = CentromereState.homogeneous(5, "AAAA")

    sim1 = SymbolicDongSimulator(state1, params, seed=42)
    sim2 = SymbolicDongSimulator(state2, params, seed=42)

    sim1.apply_snp()
    sim2.apply_snp()

    assert state1.symbols == state2.symbols
    assert state1.lookup.as_dict() == state2.lookup.as_dict()



def test_run_defaults_to_initial_and_final_checkpoints():
    state = CentromereState.homogeneous(5, "AAAA")
    sim = SymbolicDongSimulator(state, no_limits(), seed=1)

    result = sim.run(3)

    assert [x.generation for x in result.checkpoints] == [0, 3]
    assert result.checkpoints[0].symbols == [0, 0, 0, 0, 0]
    assert result.checkpoints[1].symbols == [0, 0, 0, 0, 0]
    assert result.lookup == {0: "AAAA"}


def test_run_stores_requested_checkpoints_only():
    state = CentromereState.homogeneous(5, "AAAA")
    sim = SymbolicDongSimulator(state, no_limits(), seed=1)

    result = sim.run(
        5,
        checkpoints=[0, 2, 4, 5],
    )

    assert [x.generation for x in result.checkpoints] == [0, 2, 4, 5]
    assert all(x.symbols == [0, 0, 0, 0, 0] for x in result.checkpoints)


def test_final_lookup_reconstructs_earlier_checkpoint():
    params = no_limits()
    state = CentromereState.homogeneous(2, "AAAA")
    sim = SymbolicDongSimulator(state, params, seed=1)

    # Create a second sequence state before storing the final lookup.
    sim.apply_snp(generation=1)
    checkpoint_symbols = state.symbols.copy()
    final_lookup = state.lookup.as_dict()

    reconstructed = [final_lookup[s] for s in checkpoint_symbols]
    assert len(reconstructed) == 2
    assert all(len(seq) == 4 for seq in reconstructed)
