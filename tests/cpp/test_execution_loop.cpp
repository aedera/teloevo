#define main symbolic_centromere_backend_main
#include "../../src/teloevo/cpp/symbolic_centromere_backend.cpp"
#undef main

#include <cassert>

static void test_execution_paths_match() {
    const vector<string> lookup = {
        string(178, 'A'), string(178, 'C'), string(178, 'G'), string(178, 'T')
    };
    vector<int> symbols;
    for (int i = 0; i < 80; ++i) symbols.push_back(i % lookup.size());

    for (int storage : {0, 1}) {
        for (int snp_model : {0, 1, 2}) {
            for (int seed : {7, 30, 79}) {
                Params params;
                params.seed = seed;
                params.symbol_storage = storage;
                params.symbol_block_size = 8;
                params.snp_model = snp_model;
                params.snp_rate = 0.001;
                params.snp_position_rates.assign(178, 0.001);
                params.snp_spectrum = {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0};
                params.ins_rate = 0.0008;
                params.del_rate = 0.0006;
                params.large_del_rate = 0.00002;
                params.conversion_rate = 0.001;
                params.ins_span_lambda = 2;
                params.del_span_lambda = 2;
                params.large_del_span_lambda = 8;
                params.conversion_p = 1.0 / 356;
                params.min_array_bp = 178 * 20;
                params.max_array_bp = 178 * 300;

                Simulator plain(params, symbols, lookup, "");
                params.profile = true;
                Simulator profiled(params, symbols, lookup, "");
                for (int generation = 1; generation <= 30; ++generation) {
                    plain.step_generation_unprofiled(generation);
                    profiled.step_generation(generation);
                    assert(plain.rng == profiled.rng);
                    assert(plain.symbols.to_vector() == profiled.symbols.to_vector());
                    assert(plain.symbol_to_sequence == profiled.symbol_to_sequence);
                    assert(plain.active_profile == nullptr);
                    assert(profiled.active_profile == nullptr);
                }
            }
        }
    }
}

static void test_length_guards() {
    for (bool too_short : {false, true}) {
        for (bool profile : {false, true}) {
            Params params;
            params.profile = profile;
            Simulator sim(params, {0}, {string(178, 'A')}, "");
            if (too_short) sim.p.min_array_bp = 179;
            else sim.p.max_array_bp = 177;
            const auto before = sim.rng;
            bool rejected = false;
            try {
                if (profile) sim.step_generation(1);
                else sim.step_generation_unprofiled(1);
            } catch (const runtime_error& error) {
                rejected = true;
                assert(string(error.what()) == (too_short
                    ? "Array is below min_array_bp" : "Array is above max_array_bp"));
            }
            assert(rejected);
            assert(sim.rng == before);
        }
    }
}

static void test_insertion_stops_at_capacity() {
    for (int storage : {0, 1}) {
        for (int headroom : {0, 1, 177}) {
            Params params;
            params.seed = 846;
            params.symbol_storage = storage;
            params.max_array_bp = 20 * 178 + headroom;
            Simulator sim(params, vector<int>(20, 0), {string(178, 'A')}, "");
            const auto before = sim.rng;
            assert(sim.run_rejection_step(10, 50, 1, 1) == 0);
            assert(sim.rng == before);
            assert(sim.symbols.size() == 20);
        }
    }
}

static void test_insertion_rechecks_capacity_and_resumes_after_deletion() {
    for (int storage : {0, 1}) {
        Params params;
        params.seed = 846;
        params.symbol_storage = storage;
        params.max_array_bp = 21 * 178;
        // Use short spans to exercise an exact one-monomer gap.
        params.ins_span_lambda = 1e-12;
        params.del_span_lambda = 1e-12;
        Simulator sim(params, vector<int>(20, 0), {string(178, 'A')}, "");
        Simulator single(params, vector<int>(20, 0), {string(178, 'A')}, "");
        assert(single.run_rejection_step(1, 50, 1, 1) == 1);
        // The second requested insertion must stop without consuming draws.
        assert(sim.run_rejection_step(2, 50, 1, 1) == 1);
        assert(sim.rng == single.rng);
        assert(sim.length_bp() == params.max_array_bp);

        assert(sim.run_rejection_step(1, 100, 1, 2) == 1);
        assert(sim.length_bp() == params.max_array_bp - 178);
        assert(sim.run_rejection_step(1, 50, 2, 1) == 1);
        assert(sim.length_bp() == params.max_array_bp);
    }
}

int main() {
    test_execution_paths_match();
    test_length_guards();
    test_insertion_stops_at_capacity();
    test_insertion_rechecks_capacity_and_resumes_after_deletion();
}
