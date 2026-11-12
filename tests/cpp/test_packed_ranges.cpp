// Exercise the private packed helpers and conversion against string oracles.
#define main symbolic_centromere_backend_main
#include "../../src/teloevo/cpp/symbolic_centromere_backend.cpp"
#undef main

#include <cassert>

static string random_sequence(int length, mt19937_64& rng) {
    string sequence(static_cast<size_t>(length), 'A');
    for (char& base : sequence) base = "ACGT"[rng() % 4];
    return sequence;
}

static void check_range(
    const string& source,
    const string& destination,
    int source_start,
    int destination_start,
    int length
) {
    const PackedMonomer packed_source = pack_sequence(source);
    PackedMonomer packed_destination = pack_sequence(destination);
    const bool equal = source.compare(
        source_start, length, destination, destination_start, length
    ) == 0;
    assert(packed_ranges_equal(
        packed_source, source_start, packed_destination, destination_start, length
    ) == equal);

    string expected = destination;
    expected.replace(destination_start, length, source, source_start, length);
    copy_packed_range(
        packed_destination, destination_start, packed_source, source_start, length
    );
    // Compare all eight words, including untouched bases and unused padding.
    assert(packed_destination == pack_sequence(expected));
    assert(packed_ranges_equal(
        packed_source, source_start, packed_destination, destination_start, length
    ));
    if (length > 0) {
        const int position = destination_start + length / 2;
        set_base_code(
            packed_destination, position,
            (get_base_code(packed_destination, position) + 1) % 4
        );
        assert(!packed_ranges_equal(
            packed_source, source_start, packed_destination, destination_start, length
        ));
    }
}

static void test_ranges() {
    mt19937_64 rng(13);
    for (int size : {1, 31, 32, 33, 64, 65, 178, 256}) {
        const string source = random_sequence(size, rng);
        const string destination = random_sequence(size, rng);
        // Every same-phase interval, including empty intervals and the end.
        for (int start = 0; start <= size; ++start) {
            for (int length = 0; length <= size - start; ++length) {
                check_range(source, destination, start, start, length);
            }
        }
        // Different phases and word indices, including the last packed word.
        for (int trial = 0; trial < 5000; ++trial) {
            const int source_start = rng() % (size + 1);
            const int destination_start = rng() % (size + 1);
            const int available = size - max(source_start, destination_start);
            const int length = rng() % (available + 1);
            check_range(source, destination, source_start, destination_start, length);
        }
    }
    // Every relative word phase with exact and partial word-sized tracts.
    const string source = random_sequence(256, rng);
    const string destination = random_sequence(256, rng);
    for (int left = 0; left < 32; ++left) {
        for (int right = 0; right < 32; ++right) {
            for (int length : {0, 1, 31, 32, 33, 63, 64, 65, 178, 224}) {
                check_range(source, destination, left, right, length);
            }
        }
    }
}

static void test_conversions() {
    mt19937_64 input_rng(29);
    int overlapping = 0;
    int cross_boundary = 0;
    int rejected = 0;
    for (int size : {8, 31, 32, 33, 178, 256}) {
        for (bool homogeneous : {false, true}) {
            vector<string> lookup;
            vector<int> symbols;
            string before;
            for (int index = 0; index < 12; ++index) {
                const string sequence = homogeneous
                    ? string(size, 'A') : random_sequence(size, input_rng);
                lookup.push_back(sequence);
                symbols.push_back(index);
                before += sequence;
            }
            for (int storage : {0, 1}) {
                for (int seed = 1; seed <= 100; ++seed) {
                    Params params;
                    params.seed = seed;
                    params.conversion_p = 1.0 / (2 * size);
                    params.symbol_storage = storage;
                    params.symbol_block_size = 4;
                    params.profile = true;
                    Simulator sim(params, symbols, lookup, "");
                    StepProfile profile;
                    sim.active_profile = &profile;

                    // Reproduce only the original donor/length draws; compare
                    // RNG states to detect extra draws even for rejected events.
                    mt19937_64 expected_rng(seed);
                    uniform_int_distribution<long long> donor(0, before.size() - 1);
                    geometric_distribution<int> tract(params.conversion_p);
                    const long long start = donor(expected_rng);
                    const int length = 1 + tract(expected_rng);
                    const long long end = start + length;
                    const bool accepted = end + size < static_cast<long long>(before.size());
                    string expected = before;
                    if (accepted) {
                        expected.replace(start + size, length, before.substr(start, length));
                        overlapping += length > size;
                        cross_boundary += start % size + length > size;
                    } else {
                        ++rejected;
                    }

                    assert(sim.apply_conversion(1) == accepted);
                    assert(sim.rng == expected_rng);
                    string actual;
                    for (int symbol : sim.symbols.to_vector()) {
                        actual += unpack_sequence(sim.symbol_to_sequence[symbol], size);
                    }
                    assert(actual == expected);
                    assert(profile.conversion_noop == (accepted && expected == before));
                }
            }
        }
    }
    assert(overlapping > 0 && cross_boundary > 0 && rejected > 0);
}

int main() {
    test_ranges();
    test_conversions();
}
