#define main symbolic_centromere_backend_main
#include "../../src/teloevo/cpp/symbolic_centromere_backend.cpp"
#undef main

#include <cassert>

static_assert(is_same_v<decltype(*declval<SymbolArray::Cursor>()), int&>);
static_assert(is_same_v<decltype(*declval<SymbolArray::ConstCursor>()), const int&>);

static void check_array(const SymbolArray& array, const vector<int>& expected) {
    assert(array.size() == expected.size());
    assert(array.to_vector() == expected);
    for (size_t start : {size_t(0), expected.size() / 2, expected.size()}) {
        auto cursor = array.cursor(start);
        for (size_t index = start; index < expected.size(); ++index, ++cursor) {
            assert(cursor);
            assert(*cursor == expected[index]);
            assert(array[index] == expected[index]);
        }
        assert(!cursor);
    }
    bool rejected = false;
    try { array.cursor(array.size() + 1); }
    catch (const out_of_range&) { rejected = true; }
    assert(rejected);
}

static void test_cursor_after_edits() {
    for (bool blocked : {false, true}) {
        for (size_t block_size : {1, 2, 8, 512}) {
            mt19937_64 rng(47);
            vector<int> expected;
            SymbolArray array(expected, blocked, block_size);
            check_array(array, expected);
            for (int trial = 0; trial < 1000; ++trial) {
                if (expected.empty() || (rng() % 2 && expected.size() < 3000)) {
                    const size_t start = rng() % (expected.size() + 1);
                    // Include insertions large enough to split a block repeatedly.
                    vector<int> added(rng() % (3 * block_size + 1));
                    for (int& symbol : added) symbol = rng() % 1000;
                    array.insert(start, added);
                    expected.insert(expected.begin() + start, added.begin(), added.end());
                } else {
                    const size_t first = rng() % (expected.size() + 1);
                    const size_t last = first + rng() % (expected.size() - first + 1);
                    array.erase(first, last);
                    expected.erase(expected.begin() + first, expected.begin() + last);
                }
                check_array(array, expected);
                const size_t start = expected.size() / 3;
                auto cursor = array.cursor(start);
                const auto saved = cursor;
                for (size_t index = start; index < expected.size(); ++index, ++cursor) {
                    *cursor += 1;
                    expected[index] += 1;
                }
                // Copies advance independently and observe in-place replacement.
                if (saved) assert(*saved == expected[start]);
                assert(!cursor);
                check_array(array, expected);
            }
            array.erase(0, array.size());
            check_array(array, {});
            array.insert(0, {1, 2, 3});
            check_array(array, {1, 2, 3});
        }
    }
}

static void test_insertion_buffer_reused() {
    Params params;
    params.seed = 30;
    params.ins_span_lambda = 40;
    params.symbol_storage = 1;
    params.symbol_block_size = 8;
    Simulator sim(params, vector<int>(1000, 0), {string(178, 'A')}, "");
    while (!sim.apply_insertion(1)) {}
    const auto capacity = sim.insertion_buffer.capacity();
    const auto data = sim.insertion_buffer.data();
    assert(capacity > 10);
    sim.p.ins_span_lambda = 1;
    for (int generation = 2; generation <= 30; ++generation) {
        while (!sim.apply_insertion(generation)) {}
        assert(sim.insertion_buffer.capacity() == capacity);
        assert(sim.insertion_buffer.data() == data);
    }
}

int main() {
    test_cursor_after_edits();
    test_insertion_buffer_reused();
}
