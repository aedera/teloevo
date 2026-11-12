// Compare cached count draws with the original fresh-distribution behavior.
#define main symbolic_centromere_backend_main
#include "../../src/teloevo/cpp/symbolic_centromere_backend.cpp"
#undef main

#include <cassert>

static void resize_array(Simulator& sim, size_t size) {
    if (size > sim.symbols.size()) {
        sim.symbols.insert(sim.symbols.size(), vector<int>(size - sim.symbols.size(), 0));
    } else {
        sim.symbols.erase(size, sim.symbols.size());
    }
}

static void test_poisson_draws() {
    for (int seed : {1, 17, 30, 79}) {
        Params params;
        params.seed = seed;
        Simulator sim(params, {0}, {string(178, 'A')}, "");
        mt19937_64 expected_rng(seed);
        poisson_distribution<long long> cached;

        // Repeated means and changes across GCC's mean=12 algorithm boundary.
        // The large-mean cases detect retained normal-distribution state.
        for (double mean : {1.0, 1e-8, 0.01, 0.5, 11.999, 12.0, 12.001,
                            100.0, 10000.0, 1e6, 12.0, 0.01}) {
            for (int draw = 0; draw < 1000; ++draw) {
                poisson_distribution<long long> fresh(mean);
                const auto expected = fresh(expected_rng);
                assert(sim.draw_poisson_event_count(mean, cached) == expected);
                assert(sim.rng == expected_rng);
            }
        }
    }
}

static void test_event_count_caches() {
    for (int storage : {0, 1}) {
        Params params;
        params.seed = 71;
        params.symbol_storage = storage;
        params.symbol_block_size = 4;
        Simulator sim(params, vector<int>(80, 0), {string(178, 'A')}, "");
        mt19937_64 expected_rng(params.seed);
        array<poisson_distribution<long long>*, 5> caches = {
            &sim.snp_count_distribution,
            &sim.ins_count_distribution,
            &sim.del_count_distribution,
            &sim.large_del_count_distribution,
            &sim.conversion_count_distribution
        };
        const array<double, 6> rates = {0.0, -1.0, 1e-8, 0.01, 0.1, 1.0};

        for (int iteration = 0; iteration < 2000; ++iteration) {
            // Growth/shrinkage can occur between event classes, not just at
            // generation boundaries. Rates may also change at fixed lengths.
            resize_array(sim, 5 + (iteration / 37) % 80);
            const double rate = rates[(iteration / 23) % rates.size()];
            long long expected = 0;
            if (rate > 0.0) {
                poisson_distribution<long long> fresh(
                    static_cast<double>(sim.length_bp()) * rate
                );
                expected = fresh(expected_rng);
            }
            assert(sim.draw_event_count(rate, *caches[iteration % caches.size()]) == expected);
            assert(sim.rng == expected_rng);
        }
    }
}

static void test_snp_count_modes() {
    Params params;
    params.seed = 42;
    params.snp_model = 1;
    params.snp_position_rates.assign(178, 0.01);
    params.snp_spectrum = {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0};
    Simulator sim(params, vector<int>(5, 0), {string(178, 'A')}, "");
    mt19937_64 expected_rng(params.seed);

    for (int iteration = 0; iteration < 3000; ++iteration) {
        resize_array(sim, 5 + (iteration / 41) % 12);
        sim.p.snp_model = (iteration / 29) % 3;
        sim.p.snp_rate = (iteration / 19) % 2 ? 0.02 : 0.0;
        long long expected = 0;
        if (sim.p.snp_model == 1) {
            poisson_distribution<long long> fresh(
                static_cast<double>(sim.symbols.size()) * sim.snp_position_rate_sum
            );
            expected = fresh(expected_rng);
        } else if (sim.p.snp_model == 0 && sim.p.snp_rate > 0.0) {
            poisson_distribution<long long> fresh(
                static_cast<double>(sim.length_bp()) * sim.p.snp_rate
            );
            expected = fresh(expected_rng);
        }
        assert(sim.draw_snp_event_count() == expected);
        assert(sim.rng == expected_rng);
    }
}

int main() {
    test_poisson_draws();
    test_event_count_caches();
    test_snp_count_modes();
}
