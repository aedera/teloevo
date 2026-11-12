#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;


static constexpr size_t MAX_PACKED_WORDS = 8;
static constexpr int BASES_PER_PACKED_WORD = 32;
static constexpr int MAX_PACKED_MONOMER_BP =
    static_cast<int>(MAX_PACKED_WORDS) * BASES_PER_PACKED_WORD;

struct PackedMonomer {
    array<uint64_t, MAX_PACKED_WORDS> words{};

    bool operator==(const PackedMonomer& other) const noexcept {
        return words == other.words;
    }
};

struct PackedMonomerHash {
    size_t operator()(const PackedMonomer& sequence) const noexcept {
        uint64_t h = 0x9e3779b97f4a7c15ULL;

        for (uint64_t word : sequence.words) {
            uint64_t x = word + 0x9e3779b97f4a7c15ULL;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            x ^= (x >> 31);

            h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }

        return static_cast<size_t>(h);
    }
};

static uint8_t encode_base(char base) {
    if (base == 'A') return 0;
    if (base == 'C') return 1;
    if (base == 'G') return 2;
    if (base == 'T') return 3;

    throw runtime_error(
        "The packed C++ backend supports only uppercase A/C/G/T bases"
    );
}

static char decode_base(uint8_t code) {
    static const char bases[4] = {'A', 'C', 'G', 'T'};
    return bases[code & 3U];
}

static uint8_t get_base_code(
    const PackedMonomer& sequence,
    int position
) {
    const size_t word =
        static_cast<size_t>(position / BASES_PER_PACKED_WORD);

    const int shift =
        2 * (position % BASES_PER_PACKED_WORD);

    return static_cast<uint8_t>(
        (sequence.words[word] >> shift) & 3ULL
    );
}

static void set_base_code(
    PackedMonomer& sequence,
    int position,
    uint8_t code
) {
    const size_t word =
        static_cast<size_t>(position / BASES_PER_PACKED_WORD);

    const int shift =
        2 * (position % BASES_PER_PACKED_WORD);

    const uint64_t mask = 3ULL << shift;

    sequence.words[word] =
        (sequence.words[word] & ~mask) |
        ((static_cast<uint64_t>(code) & 3ULL) << shift);
}

static PackedMonomer pack_sequence(const string& sequence) {
    if (sequence.empty()) {
        throw runtime_error("Monomer sequence must not be empty");
    }

    if (
        sequence.size() >
        static_cast<size_t>(MAX_PACKED_MONOMER_BP)
    ) {
        throw runtime_error(
            "Packed C++ backend supports monomers up to " +
            to_string(MAX_PACKED_MONOMER_BP) +
            " bp"
        );
    }

    PackedMonomer packed;

    for (
        int position = 0;
        position < static_cast<int>(sequence.size());
        ++position
    ) {
        set_base_code(
            packed,
            position,
            encode_base(sequence[static_cast<size_t>(position)])
        );
    }

    return packed;
}

static string unpack_sequence(
    const PackedMonomer& sequence,
    int monomer_length
) {
    string output(
        static_cast<size_t>(monomer_length),
        'A'
    );

    for (int position = 0; position < monomer_length; ++position) {
        output[static_cast<size_t>(position)] =
            decode_base(get_base_code(sequence, position));
    }

    return output;
}

static bool packed_ranges_equal(
    const PackedMonomer& left,
    int left_start,
    const PackedMonomer& right,
    int right_start,
    int length
) {
    while (length > 0) {
        const int left_phase = left_start % BASES_PER_PACKED_WORD;
        const int right_phase = right_start % BASES_PER_PACKED_WORD;
        const int take = min({
            length,
            BASES_PER_PACKED_WORD - left_phase,
            BASES_PER_PACKED_WORD - right_phase
        });
        const uint64_t left_word =
            left.words[left_start / BASES_PER_PACKED_WORD];
        const uint64_t right_word =
            right.words[right_start / BASES_PER_PACKED_WORD];

        if (take == BASES_PER_PACKED_WORD) {
            if (left_word != right_word) return false;
        } else {
            // take < 32 here, so the mask never shifts by 64 bits.
            const uint64_t mask = (1ULL << (2 * take)) - 1;
            if (
                ((left_word >> (2 * left_phase)) & mask) !=
                ((right_word >> (2 * right_phase)) & mask)
            ) {
                return false;
            }
        }

        left_start += take;
        right_start += take;
        length -= take;
    }

    return true;
}

static void copy_packed_range(
    PackedMonomer& destination,
    int destination_start,
    const PackedMonomer& source,
    int source_start,
    int length
) {
    // Callers use a separate destination monomer. Overlapping array tracts
    // must retain their original donor symbols before replacing recipients.
    while (length > 0) {
        const int destination_phase = destination_start % BASES_PER_PACKED_WORD;
        const int source_phase = source_start % BASES_PER_PACKED_WORD;
        const int take = min({
            length,
            BASES_PER_PACKED_WORD - destination_phase,
            BASES_PER_PACKED_WORD - source_phase
        });
        uint64_t& destination_word =
            destination.words[destination_start / BASES_PER_PACKED_WORD];
        const uint64_t source_word =
            source.words[source_start / BASES_PER_PACKED_WORD];

        if (take == BASES_PER_PACKED_WORD) {
            destination_word = source_word;
        } else {
            const int shift = 2 * destination_phase;
            const uint64_t mask = ((1ULL << (2 * take)) - 1) << shift;
            destination_word =
                (destination_word & ~mask) |
                (((source_word >> (2 * source_phase)) << shift) & mask);
        }

        destination_start += take;
        source_start += take;
        length -= take;
    }
}

struct Params {
    long long n_generations = 0;
    long long start_generation = 0;
    uint64_t seed = 1;

    double snp_rate = 0.0;

    // SNP model: 0 = Dong uniform, 1 = CEN178 position/spectrum, 2 = none.
    int snp_model = 0;
    vector<double> snp_position_rates;
    vector<double> snp_spectrum;

    double ins_rate = 0.0;
    double del_rate = 0.0;
    double large_del_rate = 0.0;
    double conversion_rate = 0.0;

    double ins_span_lambda = 10.51;
    double del_span_lambda = 5.19;
    double large_del_span_lambda = 2809.0;
    double conversion_p = 0.05;

    long long min_array_bp = -1;
    long long max_array_bp = -1;

    int ins_attempt_multiplier = 50;
    int del_attempt_multiplier = 100;
    int large_del_attempt_multiplier = 200;
    int conversion_attempt_multiplier = 100;

    bool record_events = false;
    bool profile = false;
    long long profile_interval = 100000;

    // 0 = contiguous std::vector<int>, 1 = blocked symbol array.
    int symbol_storage = 0;
    int symbol_block_size = 512;

    vector<long long> checkpoints;
};

struct StepProfile {
    long long snp = 0;
    long long ins = 0;
    long long deletion = 0;
    long long large_deletion = 0;
    long long conversion = 0;
    long long conversion_fast_path = 0;
    long long conversion_noop = 0;

    long long ins_inserted_symbols = 0;
    long long ins_shifted_symbols = 0;
    long long ins_reallocations = 0;
    long long ins_block_splits = 0;

    double time_snp = 0.0;
    double time_ins = 0.0;
    double time_del = 0.0;
    double time_large_del = 0.0;
    double time_conversion = 0.0;

    long long intern_calls = 0;
    long long new_states = 0;
    long long rehashes = 0;

    double time_intern = 0.0;
    double time_intern_lookup = 0.0;
    double time_intern_insert = 0.0;

    double time_intern_snp = 0.0;
    double time_intern_ins = 0.0;
    double time_intern_del = 0.0;
    double time_intern_conversion = 0.0;

    double time_conversion_extract = 0.0;
    double time_conversion_replace = 0.0;

    double time_ins_chimera = 0.0;
    double time_ins_prepare = 0.0;
    double time_ins_vector_insert = 0.0;

    void add(const StepProfile& other) {
        snp += other.snp;
        ins += other.ins;
        deletion += other.deletion;
        large_deletion += other.large_deletion;
        conversion += other.conversion;
        conversion_fast_path += other.conversion_fast_path;
        conversion_noop += other.conversion_noop;

        ins_inserted_symbols += other.ins_inserted_symbols;
        ins_shifted_symbols += other.ins_shifted_symbols;
        ins_reallocations += other.ins_reallocations;
        ins_block_splits += other.ins_block_splits;

        time_snp += other.time_snp;
        time_ins += other.time_ins;
        time_del += other.time_del;
        time_large_del += other.time_large_del;
        time_conversion += other.time_conversion;

        intern_calls += other.intern_calls;
        new_states += other.new_states;
        rehashes += other.rehashes;

        time_intern += other.time_intern;
        time_intern_lookup += other.time_intern_lookup;
        time_intern_insert += other.time_intern_insert;

        time_intern_snp += other.time_intern_snp;
        time_intern_ins += other.time_intern_ins;
        time_intern_del += other.time_intern_del;
        time_intern_conversion += other.time_intern_conversion;

        time_conversion_extract += other.time_conversion_extract;
        time_conversion_replace += other.time_conversion_replace;

        time_ins_chimera += other.time_ins_chimera;
        time_ins_prepare += other.time_ins_prepare;
        time_ins_vector_insert += other.time_ins_vector_insert;
    }

    double mutation_time() const {
        return (
            time_snp +
            time_ins +
            time_del +
            time_large_del +
            time_conversion
        );
    }
};

static vector<long long> parse_csv_ll(const string& value) {
    vector<long long> out;
    if (value.empty() || value == "-") return out;

    string item;
    stringstream ss(value);
    while (getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(stoll(item));
    }
    return out;
}

static vector<double> parse_csv_double(const string& value) {
    vector<double> out;
    if (value.empty() || value == "-") return out;

    string item;
    stringstream ss(value);

    while (getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(stod(item));
        }
    }

    return out;
}

static Params read_config(const string& path) {
    Params p;
    ifstream in(path);
    if (!in) throw runtime_error("Cannot open config: " + path);

    string key, value;
    while (in >> key >> value) {
        if (key == "n_generations") p.n_generations = stoll(value);
        else if (key == "start_generation") p.start_generation = stoll(value);
        else if (key == "seed") p.seed = stoull(value);

        else if (key == "snp_rate") p.snp_rate = stod(value);
        else if (key == "snp_model") p.snp_model = stoi(value);
        else if (key == "snp_position_rates") p.snp_position_rates = parse_csv_double(value);
        else if (key == "snp_spectrum") p.snp_spectrum = parse_csv_double(value);
        else if (key == "ins_rate") p.ins_rate = stod(value);
        else if (key == "del_rate") p.del_rate = stod(value);
        else if (key == "large_del_rate") p.large_del_rate = stod(value);
        else if (key == "conversion_rate") p.conversion_rate = stod(value);

        else if (key == "ins_span_lambda") p.ins_span_lambda = stod(value);
        else if (key == "del_span_lambda") p.del_span_lambda = stod(value);
        else if (key == "large_del_span_lambda") p.large_del_span_lambda = stod(value);
        else if (key == "conversion_p") p.conversion_p = stod(value);

        else if (key == "min_array_bp") p.min_array_bp = stoll(value);
        else if (key == "max_array_bp") p.max_array_bp = stoll(value);

        else if (key == "ins_attempt_multiplier") p.ins_attempt_multiplier = stoi(value);
        else if (key == "del_attempt_multiplier") p.del_attempt_multiplier = stoi(value);
        else if (key == "large_del_attempt_multiplier") p.large_del_attempt_multiplier = stoi(value);
        else if (key == "conversion_attempt_multiplier") p.conversion_attempt_multiplier = stoi(value);

        else if (key == "record_events") p.record_events = (stoi(value) != 0);
        else if (key == "profile") p.profile = (stoi(value) != 0);
        else if (key == "profile_interval") p.profile_interval = stoll(value);
        else if (key == "symbol_storage") p.symbol_storage = stoi(value);
        else if (key == "symbol_block_size") p.symbol_block_size = stoi(value);
        else if (key == "checkpoints") p.checkpoints = parse_csv_ll(value);
        else throw runtime_error("Unknown config key: " + key);
    }

    return p;
}


class SymbolArray {
public:
    struct InsertStats {
        long long shifted_symbols = 0;
        bool reallocated = false;
        long long block_splits = 0;
    };

private:
    bool blocked_ = false;
    size_t target_block_size_ = 512;
    size_t total_size_ = 0;

    vector<int> contiguous_;
    vector<vector<int>> blocks_;
    vector<size_t> prefix_end_;

    void rebuild_prefix() {
        prefix_end_.clear();
        prefix_end_.reserve(blocks_.size());

        size_t running = 0;

        for (const auto& block : blocks_) {
            running += block.size();
            prefix_end_.push_back(running);
        }

        total_size_ = running;
    }

    pair<size_t, size_t> locate(size_t index) const {
        if (index >= total_size_) {
            throw out_of_range("SymbolArray index out of range");
        }

        auto found =
            upper_bound(prefix_end_.begin(), prefix_end_.end(), index);

        const size_t block_index =
            static_cast<size_t>(found - prefix_end_.begin());

        const size_t block_start =
            block_index == 0 ? 0 : prefix_end_[block_index - 1];

        return {
            block_index,
            index - block_start
        };
    }

    pair<size_t, size_t> locate_for_insert(size_t index) const {
        if (index > total_size_) {
            throw out_of_range("SymbolArray insertion index out of range");
        }

        if (blocks_.empty()) {
            return {0, 0};
        }

        if (index == total_size_) {
            return {
                blocks_.size() - 1,
                blocks_.back().size()
            };
        }

        return locate(index);
    }

    long long split_oversized_block(size_t block_index) {
        long long splits = 0;
        size_t index = block_index;

        while (index < blocks_.size()) {
            if (
                blocks_[index].size() <=
                2 * target_block_size_
            ) {
                ++index;

                // Only the original block and blocks created from it can
                // be oversized, so stop once the next untouched block is
                // reached.
                if (
                    index >= blocks_.size() ||
                    blocks_[index].size() <=
                        2 * target_block_size_
                ) {
                    break;
                }

                continue;
            }

            auto& block = blocks_[index];

            const size_t split_at =
                block.size() / 2;

            vector<int> right(
                block.begin() + static_cast<ptrdiff_t>(split_at),
                block.end()
            );

            block.erase(
                block.begin() + static_cast<ptrdiff_t>(split_at),
                block.end()
            );

            blocks_.insert(
                blocks_.begin() +
                    static_cast<ptrdiff_t>(index + 1),
                std::move(right)
            );

            ++splits;

            // Recheck the left half. Very large insertions may require
            // more than one split before every block is <= 2*target.
        }

        return splits;
    }

    void merge_small_neighbors(size_t block_index) {
        if (blocks_.empty()) return;

        block_index =
            min(block_index, blocks_.size() - 1);

        if (
            block_index > 0 &&
            blocks_[block_index - 1].size() +
                blocks_[block_index].size() <=
                2 * target_block_size_
        ) {
            auto& left = blocks_[block_index - 1];
            auto& right = blocks_[block_index];

            left.insert(
                left.end(),
                right.begin(),
                right.end()
            );

            blocks_.erase(
                blocks_.begin() +
                    static_cast<ptrdiff_t>(block_index)
            );

            --block_index;
        }

        if (
            block_index + 1 < blocks_.size() &&
            blocks_[block_index].size() +
                blocks_[block_index + 1].size() <=
                2 * target_block_size_
        ) {
            auto& left = blocks_[block_index];
            auto& right = blocks_[block_index + 1];

            left.insert(
                left.end(),
                right.begin(),
                right.end()
            );

            blocks_.erase(
                blocks_.begin() +
                    static_cast<ptrdiff_t>(block_index + 1)
            );
        }
    }

public:
    // Locate once, then walk sequentially within and between blocks. Cursors
    // survive symbol replacement (and lookup interning), but insert/erase or
    // moving the SymbolArray invalidates them. Do not advance an end cursor.
    template<bool IsConst>
    class BasicCursor {
        using Array = conditional_t<IsConst, const SymbolArray, SymbolArray>;
        using Pointer = conditional_t<IsConst, const int*, int*>;
        Array* owner_;
        size_t block_index_ = 0;
        Pointer current_ = nullptr;
        Pointer block_end_ = nullptr;

        void select_block(size_t offset = 0) {
            auto& block = owner_->blocks_[block_index_];
            current_ = block.data() + offset;
            block_end_ = block.data() + block.size();
        }

    public:
        BasicCursor(Array& owner, size_t index) : owner_(&owner) {
            if (index > owner.size()) {
                throw out_of_range("SymbolArray cursor index out of range");
            }
            if (index == owner.size()) return;
            if (owner.blocked_) {
                const auto [block_index, offset] = owner.locate(index);
                block_index_ = block_index;
                select_block(offset);
            } else {
                current_ = owner.contiguous_.data() + index;
                block_end_ = owner.contiguous_.data() + owner.contiguous_.size();
            }
        }

        explicit operator bool() const { return current_ != nullptr; }
        auto& operator*() const { return *current_; }

        BasicCursor& operator++() {
            ++current_;
            if (current_ == block_end_) {
                if (owner_->blocked_ && ++block_index_ < owner_->blocks_.size()) {
                    select_block();
                } else {
                    current_ = nullptr;
                }
            }
            return *this;
        }
    };

    using Cursor = BasicCursor<false>;
    using ConstCursor = BasicCursor<true>;

    Cursor cursor(size_t index) { return Cursor(*this, index); }
    ConstCursor cursor(size_t index) const { return ConstCursor(*this, index); }

    SymbolArray() = default;

    SymbolArray(
        vector<int> data,
        bool blocked,
        size_t block_size
    )
        : blocked_(blocked),
          target_block_size_(max<size_t>(1, block_size)),
          total_size_(data.size()) {

        if (!blocked_) {
            contiguous_ = std::move(data);
            return;
        }

        blocks_.reserve(
            max<size_t>(
                1,
                (data.size() + target_block_size_ - 1) /
                    target_block_size_
            )
        );

        for (
            size_t start = 0;
            start < data.size();
            start += target_block_size_
        ) {
            const size_t end =
                min(data.size(), start + target_block_size_);

            blocks_.emplace_back(
                data.begin() + static_cast<ptrdiff_t>(start),
                data.begin() + static_cast<ptrdiff_t>(end)
            );
        }

        if (blocks_.empty()) {
            blocks_.emplace_back();
        }

        rebuild_prefix();
    }

    bool blocked() const {
        return blocked_;
    }

    size_t size() const {
        return blocked_ ? total_size_ : contiguous_.size();
    }

    size_t block_count() const {
        if (!blocked_) {
            return contiguous_.empty() ? 0 : 1;
        }

        return blocks_.size();
    }

    int& operator[](size_t index) {
        if (!blocked_) {
            return contiguous_[index];
        }

        const auto [block_index, offset] =
            locate(index);

        return blocks_[block_index][offset];
    }

    const int& operator[](size_t index) const {
        if (!blocked_) {
            return contiguous_[index];
        }

        const auto [block_index, offset] =
            locate(index);

        return blocks_[block_index][offset];
    }

    InsertStats insert(
        size_t index,
        const vector<int>& values
    ) {
        InsertStats stats;

        if (values.empty()) return stats;

        if (!blocked_) {
            const size_t old_size = contiguous_.size();
            const size_t old_capacity = contiguous_.capacity();

            stats.shifted_symbols =
                static_cast<long long>(old_size - index);

            contiguous_.insert(
                contiguous_.begin() +
                    static_cast<ptrdiff_t>(index),
                values.begin(),
                values.end()
            );

            stats.reallocated =
                contiguous_.capacity() != old_capacity;

            return stats;
        }

        const auto [block_index, offset] =
            locate_for_insert(index);

        if (blocks_.empty()) {
            blocks_.emplace_back();
        }

        auto& block = blocks_[block_index];

        stats.shifted_symbols =
            static_cast<long long>(block.size() - offset);

        const size_t old_capacity = block.capacity();

        block.insert(
            block.begin() + static_cast<ptrdiff_t>(offset),
            values.begin(),
            values.end()
        );

        stats.reallocated =
            block.capacity() != old_capacity;

        stats.block_splits +=
            split_oversized_block(block_index);

        rebuild_prefix();

        return stats;
    }

    void erase(size_t first, size_t last) {
        if (first > last || last > size()) {
            throw out_of_range("Invalid SymbolArray erase interval");
        }

        if (first == last) return;

        if (!blocked_) {
            contiguous_.erase(
                contiguous_.begin() +
                    static_cast<ptrdiff_t>(first),
                contiguous_.begin() +
                    static_cast<ptrdiff_t>(last)
            );

            return;
        }

        size_t remaining = last - first;

        while (remaining > 0) {
            const auto [block_index, offset] =
                locate(first);

            auto& block = blocks_[block_index];

            const size_t take =
                min(remaining, block.size() - offset);

            block.erase(
                block.begin() + static_cast<ptrdiff_t>(offset),
                block.begin() +
                    static_cast<ptrdiff_t>(offset + take)
            );

            remaining -= take;

            if (block.empty() && blocks_.size() > 1) {
                blocks_.erase(
                    blocks_.begin() +
                        static_cast<ptrdiff_t>(block_index)
                );
            } else {
                merge_small_neighbors(block_index);
            }

            rebuild_prefix();
        }

        if (blocks_.empty()) {
            blocks_.emplace_back();
            rebuild_prefix();
        }
    }

    vector<int> to_vector() const {
        if (!blocked_) {
            return contiguous_;
        }

        vector<int> output;
        output.reserve(total_size_);

        for (const auto& block : blocks_) {
            output.insert(
                output.end(),
                block.begin(),
                block.end()
            );
        }

        return output;
    }

    void add_to_set(
        unordered_set<int>& output
    ) const {
        if (!blocked_) {
            for (int symbol : contiguous_) {
                output.insert(symbol);
            }

            return;
        }

        for (const auto& block : blocks_) {
            for (int symbol : block) {
                output.insert(symbol);
            }
        }
    }
};

class Simulator {
public:
    Params p;
    SymbolArray symbols;
    vector<PackedMonomer> symbol_to_sequence;
    unordered_map<PackedMonomer, int, PackedMonomerHash> sequence_to_symbol;
    vector<int> insertion_buffer;  // Retain capacity between accepted insertions.
    int monomer_length = 0;

    mt19937_64 rng;
    ofstream events;
    StepProfile* active_profile = nullptr;

    // Separate caches avoid replacing parameters when switching event classes.
    poisson_distribution<long long> snp_count_distribution;
    poisson_distribution<long long> ins_count_distribution;
    poisson_distribution<long long> del_count_distribution;
    poisson_distribution<long long> large_del_count_distribution;
    poisson_distribution<long long> conversion_count_distribution;

    double snp_position_rate_sum = 0.0;
    discrete_distribution<int> snp_phase_distribution;
    array<array<double, 4>, 4> snp_spectrum_cdf{};

    Simulator(
        Params parameters,
        vector<int> input_symbols,
        vector<string> input_lookup,
        const string& events_path
    )
        : p(std::move(parameters)),
          symbols(
              std::move(input_symbols),
              p.symbol_storage == 1,
              static_cast<size_t>(p.symbol_block_size)
          ),
          rng(p.seed) {

        if (input_lookup.empty()) {
            throw runtime_error("Lookup is empty");
        }

        if (p.symbol_storage != 0 && p.symbol_storage != 1) {
            throw runtime_error("symbol_storage must be 0 or 1");
        }

        if (p.symbol_block_size <= 0) {
            throw runtime_error("symbol_block_size must be > 0");
        }

        monomer_length =
            static_cast<int>(input_lookup[0].size());

        if (monomer_length <= 0) {
            throw runtime_error("Monomer length must be > 0");
        }

        if (monomer_length > MAX_PACKED_MONOMER_BP) {
            throw runtime_error(
                "Packed C++ backend supports monomers up to " +
                to_string(MAX_PACKED_MONOMER_BP) +
                " bp"
            );
        }

        if (p.snp_model < 0 || p.snp_model > 2) {
            throw runtime_error("snp_model must be 0, 1, or 2");
        }

        if (p.snp_model == 1) {
            if (
                p.snp_position_rates.size() !=
                static_cast<size_t>(monomer_length)
            ) {
                throw runtime_error(
                    "CEN178 SNP position-rate count must match "
                    "the monomer length"
                );
            }

            if (p.snp_spectrum.size() != 16) {
                throw runtime_error(
                    "CEN178 SNP spectrum must contain 16 probabilities"
                );
            }

            for (double value : p.snp_position_rates) {
                if (!isfinite(value) || value < 0.0) {
                    throw runtime_error(
                        "CEN178 SNP position rates must be finite and >= 0"
                    );
                }

                snp_position_rate_sum += value;
            }

            if (snp_position_rate_sum <= 0.0) {
                throw runtime_error(
                    "CEN178 SNP position rates must contain "
                    "at least one positive value"
                );
            }

            snp_phase_distribution =
                discrete_distribution<int>(
                    p.snp_position_rates.begin(),
                    p.snp_position_rates.end()
                );

            for (int ref = 0; ref < 4; ++ref) {
                double cumulative = 0.0;

                for (int alt = 0; alt < 4; ++alt) {
                    const double value =
                        p.snp_spectrum[
                            static_cast<size_t>(ref * 4 + alt)
                        ];

                    if (!isfinite(value) || value < 0.0) {
                        throw runtime_error(
                            "CEN178 SNP spectrum probabilities "
                            "must be finite and >= 0"
                        );
                    }

                    cumulative += value;
                    snp_spectrum_cdf[ref][alt] = cumulative;
                }

                if (cumulative <= 0.0) {
                    throw runtime_error(
                        "Each CEN178 SNP spectrum row must "
                        "have positive probability"
                    );
                }

                for (int alt = 0; alt < 4; ++alt) {
                    snp_spectrum_cdf[ref][alt] /= cumulative;
                }

                snp_spectrum_cdf[ref][3] = 1.0;
            }
        }

        sequence_to_symbol.max_load_factor(0.75f);

        const double initial_length_bp =
            static_cast<double>(symbols.size()) *
            static_cast<double>(monomer_length);

        const double effective_snp_rate =
            p.snp_model == 1
                ? (
                    snp_position_rate_sum /
                    static_cast<double>(monomer_length)
                )
                : (
                    p.snp_model == 2
                        ? 0.0
                        : p.snp_rate
                );

        const double expected_intern_rate =
            effective_snp_rate +
            p.ins_rate +
            p.del_rate +
            p.large_del_rate +
            1.5 * p.conversion_rate;

        const double expected_new_states =
            static_cast<double>(max<long long>(0, p.n_generations)) *
            initial_length_bp *
            max(0.0, expected_intern_rate);

        const size_t reserve_hint = static_cast<size_t>(
            min(
                10000000.0,
                static_cast<double>(input_lookup.size()) +
                1.5 * expected_new_states +
                1024.0
            )
        );

        sequence_to_symbol.reserve(
            max(input_lookup.size(), reserve_hint)
        );

        symbol_to_sequence.reserve(
            max(input_lookup.size(), reserve_hint)
        );

        for (
            int i = 0;
            i < static_cast<int>(input_lookup.size());
            ++i
        ) {
            if (
                static_cast<int>(input_lookup[i].size()) !=
                monomer_length
            ) {
                throw runtime_error(
                    "Variable monomer lengths are not supported"
                );
            }

            PackedMonomer packed =
                pack_sequence(input_lookup[i]);

            symbol_to_sequence.push_back(packed);
            sequence_to_symbol.emplace(packed, i);
        }

        for (auto cursor = symbols.cursor(0); cursor; ++cursor) {
            const int symbol = *cursor;

            if (
                symbol < 0 ||
                symbol >= static_cast<int>(symbol_to_sequence.size())
            ) {
                throw runtime_error("Symbol outside lookup range");
            }
        }

        if (p.record_events) {
            events.open(events_path);
            if (!events) {
                throw runtime_error("Cannot open events output");
            }
        }
    }

    long long length_bp() const {
        return static_cast<long long>(symbols.size()) *
               static_cast<long long>(monomer_length);
    }

    int intern(const PackedMonomer& sequence, int context = 0) {
        if (!p.profile || active_profile == nullptr) {
            auto found = sequence_to_symbol.find(sequence);
            if (found != sequence_to_symbol.end()) return found->second;

            const int symbol = static_cast<int>(symbol_to_sequence.size());
            symbol_to_sequence.push_back(sequence);
            sequence_to_symbol.emplace(sequence, symbol);
            return symbol;
        }

        active_profile->intern_calls++;

        const auto total_start = chrono::steady_clock::now();
        const auto lookup_start = total_start;

        auto found = sequence_to_symbol.find(sequence);

        const auto lookup_end = chrono::steady_clock::now();
        const double lookup_time =
            chrono::duration<double>(lookup_end - lookup_start).count();

        active_profile->time_intern_lookup += lookup_time;

        if (found != sequence_to_symbol.end()) {
            const double total_time =
                chrono::duration<double>(
                    chrono::steady_clock::now() - total_start
                ).count();

            active_profile->time_intern += total_time;

            if (context == 1) active_profile->time_intern_snp += total_time;
            else if (context == 2) active_profile->time_intern_ins += total_time;
            else if (context == 3) active_profile->time_intern_del += total_time;
            else if (context == 4) active_profile->time_intern_conversion += total_time;

            return found->second;
        }

        active_profile->new_states++;

        const size_t buckets_before = sequence_to_symbol.bucket_count();
        const auto insert_start = chrono::steady_clock::now();

        const int symbol = static_cast<int>(symbol_to_sequence.size());
        symbol_to_sequence.push_back(sequence);
        sequence_to_symbol.emplace(sequence, symbol);

        const auto insert_end = chrono::steady_clock::now();
        const double insert_time =
            chrono::duration<double>(insert_end - insert_start).count();

        active_profile->time_intern_insert += insert_time;

        if (sequence_to_symbol.bucket_count() != buckets_before) {
            active_profile->rehashes++;
        }

        const double total_time =
            chrono::duration<double>(
                chrono::steady_clock::now() - total_start
            ).count();

        active_profile->time_intern += total_time;

        if (context == 1) active_profile->time_intern_snp += total_time;
        else if (context == 2) active_profile->time_intern_ins += total_time;
        else if (context == 3) active_profile->time_intern_del += total_time;
        else if (context == 4) active_profile->time_intern_conversion += total_time;

        return symbol;
    }

    long long draw_poisson_event_count(
        double lambda,
        poisson_distribution<long long>& distribution
    ) {
        if (distribution.mean() != lambda) {
            distribution.param(
                poisson_distribution<long long>::param_type(lambda)
            );
        }

        // Match a freshly constructed distribution on every draw. In
        // particular, discard GCC's spare normal variate for large means;
        // reset() keeps the precomputed Poisson parameters, not outcomes.
        distribution.reset();
        return distribution(rng);
    }

    long long draw_snp_event_count() {
        if (p.snp_model == 2) {
            return 0;
        }

        if (p.snp_model == 1) {
            const double lambda =
                static_cast<double>(symbols.size()) *
                snp_position_rate_sum;

            return draw_poisson_event_count(lambda, snp_count_distribution);
        }

        return draw_event_count(p.snp_rate, snp_count_distribution);
    }

    long long draw_event_count(
        double rate_per_bp,
        poisson_distribution<long long>& distribution
    ) {
        if (rate_per_bp <= 0.0) return 0;

        const double lambda =
            static_cast<double>(length_bp()) * rate_per_bp;

        return draw_poisson_event_count(lambda, distribution);
    }

    int draw_span(double lambda) {
        poisson_distribution<int> distribution(lambda);
        return max(1, distribution(rng));
    }

    pair<int, int> draw_breakpoint() {
        const long long length = length_bp();
        if (length <= 0) throw runtime_error("Empty centromere");

        uniform_int_distribution<long long> distribution(0, length - 1);
        const long long bp = distribution(rng);

        return {
            static_cast<int>(bp / monomer_length),
            static_cast<int>(bp % monomer_length)
        };
    }

    int draw_conversion_length() {
        geometric_distribution<int> distribution(p.conversion_p);
        return distribution(rng) + 1;
    }

    void record_event(
        long long generation,
        const string& event_type,
        int start_monomer,
        int phase,
        int span_monomers = 0,
        long long size_bp = 0
    ) {
        if (!p.record_events) return;

        events
            << generation << '\t'
            << event_type << '\t'
            << start_monomer << '\t'
            << phase << '\t'
            << span_monomers << '\t'
            << size_bp << '\n';
    }

    bool apply_snp(long long generation) {
        if (p.snp_model == 2) {
            return false;
        }

        int monomer_index = 0;
        int phase = 0;
        uint8_t alternative = 0;

        if (p.snp_model == 1) {
            uniform_int_distribution<int> monomer_distribution(
                0,
                static_cast<int>(symbols.size()) - 1
            );

            monomer_index = monomer_distribution(rng);
            phase = snp_phase_distribution(rng);

            const PackedMonomer& old_sequence =
                symbol_to_sequence[symbols[monomer_index]];

            const uint8_t reference =
                get_base_code(old_sequence, phase);

            uniform_real_distribution<double> probability_distribution(
                0.0,
                1.0
            );

            const double draw =
                probability_distribution(rng);

            alternative = 3;

            for (uint8_t alt = 0; alt < 4; ++alt) {
                if (
                    draw <
                    snp_spectrum_cdf[reference][alt]
                ) {
                    alternative = alt;
                    break;
                }
            }

            // The prepared spectrum has a zero diagonal. This is a
            // defensive check against malformed external inputs.
            if (alternative == reference) {
                throw runtime_error(
                    "CEN178 SNP spectrum selected the reference base"
                );
            }
        } else {
            const auto breakpoint =
                draw_breakpoint();

            monomer_index = breakpoint.first;
            phase = breakpoint.second;

            const PackedMonomer& old_sequence =
                symbol_to_sequence[symbols[monomer_index]];

            const uint8_t reference =
                get_base_code(old_sequence, phase);

            uint8_t alternatives[3];
            int alternative_count = 0;

            for (uint8_t base = 0; base < 4; ++base) {
                if (base != reference) {
                    alternatives[alternative_count++] = base;
                }
            }

            uniform_int_distribution<int> distribution(0, 2);
            alternative = alternatives[distribution(rng)];
        }

        PackedMonomer new_sequence =
            symbol_to_sequence[symbols[monomer_index]];

        set_base_code(
            new_sequence,
            phase,
            alternative
        );

        symbols[monomer_index] =
            intern(new_sequence, 1);

        record_event(
            generation,
            "SNP",
            monomer_index,
            phase,
            0,
            1
        );

        return true;
    }

    bool apply_insertion(long long generation) {
        const auto [start_monomer, phase] = draw_breakpoint();
        const int span_monomers = draw_span(p.ins_span_lambda);
        const int end_monomer = start_monomer + span_monomers;

        if (
            span_monomers < 1 ||
            end_monomer >= static_cast<int>(symbols.size())
        ) {
            return false;
        }

        const long long added_bp =
            static_cast<long long>(span_monomers) * monomer_length;

        if (
            p.max_array_bp >= 0 &&
            length_bp() + added_bp > p.max_array_bp
        ) {
            return false;
        }

        const bool do_profile =
            p.profile && active_profile != nullptr;

        // ----- chimera construction + intern -----
        const auto chimera_start =
            do_profile
                ? chrono::steady_clock::now()
                : chrono::steady_clock::time_point();

        const int left_symbol = symbols[start_monomer];
        const int right_symbol = symbols[end_monomer];

        const PackedMonomer& left_sequence =
            symbol_to_sequence[left_symbol];

        const PackedMonomer& right_sequence =
            symbol_to_sequence[right_symbol];

        PackedMonomer chimera = right_sequence;

        copy_packed_range(
            chimera,
            phase,
            left_sequence,
            phase,
            monomer_length - phase
        );

        const int chimera_symbol =
            intern(chimera, 2);

        if (do_profile) {
            active_profile->time_ins_chimera +=
                chrono::duration<double>(
                    chrono::steady_clock::now() - chimera_start
                ).count();
        }

        // ----- prepare duplicated symbolic span -----
        const auto prepare_start =
            do_profile
                ? chrono::steady_clock::now()
                : chrono::steady_clock::time_point();

        auto& insertion = insertion_buffer;
        insertion.clear();
        insertion.reserve(static_cast<size_t>(span_monomers));

        insertion.push_back(chimera_symbol);

        auto span_cursor = symbols.cursor(start_monomer + 1);
        for (
            int index = start_monomer + 1;
            index < end_monomer;
            ++index, ++span_cursor
        ) {
            insertion.push_back(*span_cursor);
        }

        if (do_profile) {
            active_profile->time_ins_prepare +=
                chrono::duration<double>(
                    chrono::steady_clock::now() - prepare_start
                ).count();
        }

        // ----- middle insertion into the centromere symbol vector -----
        if (do_profile) {
            active_profile->ins_inserted_symbols +=
                static_cast<long long>(insertion.size());
        }

        const auto vector_insert_start =
            do_profile
                ? chrono::steady_clock::now()
                : chrono::steady_clock::time_point();

        const SymbolArray::InsertStats insert_stats =
            symbols.insert(
                static_cast<size_t>(end_monomer),
                insertion
            );

        if (do_profile) {
            active_profile->time_ins_vector_insert +=
                chrono::duration<double>(
                    chrono::steady_clock::now() - vector_insert_start
                ).count();

            active_profile->ins_shifted_symbols +=
                insert_stats.shifted_symbols;

            if (insert_stats.reallocated) {
                active_profile->ins_reallocations++;
            }

            active_profile->ins_block_splits +=
                insert_stats.block_splits;
        }

        record_event(
            generation,
            "INS",
            start_monomer,
            phase,
            span_monomers,
            added_bp
        );

        return true;
    }

    bool apply_deletion(
        long long generation,
        bool large_deletion
    ) {
        const auto [start_monomer, phase] = draw_breakpoint();

        const int span_monomers = draw_span(
            large_deletion
                ? p.large_del_span_lambda
                : p.del_span_lambda
        );

        const int end_monomer =
            start_monomer + span_monomers;

        if (
            span_monomers < 1 ||
            end_monomer >= static_cast<int>(symbols.size())
        ) {
            return false;
        }

        const long long removed_bp =
            static_cast<long long>(span_monomers) * monomer_length;

        if (
            p.min_array_bp >= 0 &&
            length_bp() - removed_bp < p.min_array_bp
        ) {
            return false;
        }

        const PackedMonomer& left_sequence =
            symbol_to_sequence[symbols[start_monomer]];

        const PackedMonomer& right_sequence =
            symbol_to_sequence[symbols[end_monomer]];

        PackedMonomer chimera = left_sequence;

        copy_packed_range(
            chimera,
            phase,
            right_sequence,
            phase,
            monomer_length - phase
        );

        const int chimera_symbol =
            intern(chimera, 3);

        symbols[start_monomer] = chimera_symbol;

        symbols.erase(
            static_cast<size_t>(start_monomer + 1),
            static_cast<size_t>(end_monomer + 1)
        );

        record_event(
            generation,
            large_deletion ? "Large_DEL" : "DEL",
            start_monomer,
            phase,
            span_monomers,
            removed_bp
        );

        return true;
    }

    bool apply_conversion(long long generation) {
        const long long length = length_bp();

        uniform_int_distribution<long long> distribution(
            0,
            length - 1
        );

        const long long donor_start = distribution(rng);
        const int tract_length = draw_conversion_length();
        const long long donor_end =
            donor_start + tract_length;

        if (
            donor_end >= length ||
            donor_end + monomer_length >= length
        ) {
            return false;
        }

        const long long receptor_start =
            donor_start + monomer_length;

        const int donor_monomer =
            static_cast<int>(donor_start / monomer_length);

        const int donor_phase =
            static_cast<int>(donor_start % monomer_length);

        const int receptor_monomer =
            static_cast<int>(receptor_start / monomer_length);

        const int receptor_phase =
            static_cast<int>(receptor_start % monomer_length);

        // Most conversion tracts are much shorter than one monomer.  When
        // the donor tract does not cross a monomer boundary, the receptor
        // tract (one monomer downstream at the same phase) cannot cross
        // either.  Handle this common case directly instead of going
        // through the generic base-coordinate extraction/replacement loops.
        if (donor_phase + tract_length <= monomer_length) {
            if (p.profile && active_profile != nullptr) {
                active_profile->conversion_fast_path++;
            }

            const auto fast_start =
                (p.profile && active_profile != nullptr)
                    ? chrono::steady_clock::now()
                    : chrono::steady_clock::time_point();

            auto cursor = symbols.cursor(donor_monomer);
            const int donor_symbol = *cursor;
            ++cursor;
            int& receptor_symbol = *cursor;
            const PackedMonomer& donor_sequence =
                symbol_to_sequence[donor_symbol];

            const PackedMonomer& receptor_sequence =
                symbol_to_sequence[receptor_symbol];

            // Dong counts an identical donor/receptor tract as a successful
            // conversion.  Preserve that event, but avoid constructing and
            // interning an unchanged monomer.
            if (
                packed_ranges_equal(
                    receptor_sequence,
                    receptor_phase,
                    donor_sequence,
                    donor_phase,
                    tract_length
                )
            ) {
                if (p.profile && active_profile != nullptr) {
                    active_profile->conversion_noop++;
                    active_profile->time_conversion_replace +=
                        chrono::duration<double>(
                            chrono::steady_clock::now() - fast_start
                        ).count();
                }

                record_event(
                    generation,
                    "Conversion",
                    receptor_monomer,
                    receptor_phase,
                    0,
                    tract_length
                );

                return true;
            }

            PackedMonomer new_sequence =
                receptor_sequence;

            copy_packed_range(
                new_sequence,
                receptor_phase,
                donor_sequence,
                donor_phase,
                tract_length
            );

            receptor_symbol = intern(new_sequence, 4);

            if (p.profile && active_profile != nullptr) {
                active_profile->time_conversion_replace +=
                    chrono::duration<double>(
                        chrono::steady_clock::now() - fast_start
                    ).count();
            }

            record_event(
                generation,
                "Conversion",
                receptor_monomer,
                receptor_phase,
                0,
                tract_length
            );

            return true;
        }

        // Snapshot donor symbol IDs before changing any recipient. Lookup
        // sequences are immutable, but long tracts can overlap their donors;
        // retaining IDs also avoids references invalidated by intern().
        const bool do_profile = p.profile && active_profile != nullptr;
        const auto extract_start =
            do_profile
                ? chrono::steady_clock::now()
                : chrono::steady_clock::time_point();
        const int donor_last =
            static_cast<int>((donor_end - 1) / monomer_length);
        const int n_monomers = donor_last - donor_monomer + 1;
        vector<int> donor_symbols;
        donor_symbols.reserve(static_cast<size_t>(n_monomers));
        auto donor_cursor = symbols.cursor(donor_monomer);
        auto receptor_cursor = donor_cursor;
        ++receptor_cursor;
        const auto receptor_begin = receptor_cursor;
        bool identical = true;
        int remaining = tract_length;
        int phase = donor_phase;

        for (int offset = 0; offset < n_monomers; ++offset, ++donor_cursor, ++receptor_cursor) {
            const int donor_symbol = *donor_cursor;
            donor_symbols.push_back(donor_symbol);
            const int take = min(remaining, monomer_length - phase);
            identical = identical && packed_ranges_equal(
                symbol_to_sequence[*receptor_cursor],
                phase,
                symbol_to_sequence[donor_symbol],
                phase,
                take
            );
            remaining -= take;
            phase = 0;
        }

        if (do_profile) {
            active_profile->time_conversion_extract +=
                chrono::duration<double>(
                    chrono::steady_clock::now() - extract_start
                ).count();
        }

        // Identical tracts still count as successful Dong conversions.
        if (identical) {
            if (p.profile && active_profile != nullptr) {
                active_profile->conversion_noop++;
            }

            record_event(
                generation,
                "Conversion",
                receptor_monomer,
                receptor_phase,
                0,
                tract_length
            );

            return true;
        }

        const auto replace_start =
            do_profile
                ? chrono::steady_clock::now()
                : chrono::steady_clock::time_point();
        remaining = tract_length;
        phase = receptor_phase;
        receptor_cursor = receptor_begin;

        for (int offset = 0; offset < n_monomers; ++offset, ++receptor_cursor) {
            const int take = min(remaining, monomer_length - phase);
            PackedMonomer new_sequence =
                symbol_to_sequence[*receptor_cursor];
            copy_packed_range(
                new_sequence,
                phase,
                symbol_to_sequence[donor_symbols[offset]],
                phase,
                take
            );
            // Preserve the original interning order, even for unchanged
            // monomers within an otherwise non-identical tract.
            *receptor_cursor = intern(new_sequence, 4);
            remaining -= take;
            phase = 0;
        }

        if (do_profile) {
            active_profile->time_conversion_replace +=
                chrono::duration<double>(
                    chrono::steady_clock::now() - replace_start
                ).count();
        }

        record_event(
            generation,
            "Conversion",
            receptor_monomer,
            receptor_phase,
            0,
            tract_length
        );

        return true;
    }

    int run_rejection_step(
        long long requested,
        int attempt_multiplier,
        long long generation,
        int event_kind
    ) {
        if (requested <= 0) return 0;

        long long accepted = 0;
        long long attempts = 0;

        const long long max_attempts =
            requested *
            static_cast<long long>(attempt_multiplier);

        while (
            accepted < requested &&
            attempts < max_attempts
        ) {
            if (event_kind == 1 &&
                p.max_array_bp >= 0 &&
                p.max_array_bp - length_bp() < monomer_length) {
                break;
            }

            ++attempts;

            bool success = false;

            if (event_kind == 1) {
                success = apply_insertion(generation);
            } else if (event_kind == 2) {
                success = apply_deletion(generation, false);
            } else if (event_kind == 3) {
                success = apply_deletion(generation, true);
            } else if (event_kind == 4) {
                success = apply_conversion(generation);
            }

            if (success) ++accepted;
        }

        return static_cast<int>(accepted);
    }

    void validate_array_length() const {
        const long long current_length = length_bp();

        if (
            p.min_array_bp >= 0 &&
            current_length < p.min_array_bp
        ) {
            throw runtime_error("Array is below min_array_bp");
        }

        if (
            p.max_array_bp >= 0 &&
            current_length > p.max_array_bp
        ) {
            throw runtime_error("Array is above max_array_bp");
        }
    }

    // Main selects this path once when profiling is disabled. Keep event
    // order and draw each count after the preceding class has changed length.
    void step_generation_unprofiled(long long generation) {
        validate_array_length();
        const long long n_snp = draw_snp_event_count();
        for (long long event = 0; event < n_snp; ++event) {
            apply_snp(generation);
        }
        run_rejection_step(
            draw_event_count(p.ins_rate, ins_count_distribution),
            p.ins_attempt_multiplier, generation, 1
        );
        run_rejection_step(
            draw_event_count(p.del_rate, del_count_distribution),
            p.del_attempt_multiplier, generation, 2
        );
        run_rejection_step(
            draw_event_count(p.large_del_rate, large_del_count_distribution),
            p.large_del_attempt_multiplier, generation, 3
        );
        run_rejection_step(
            draw_event_count(p.conversion_rate, conversion_count_distribution),
            p.conversion_attempt_multiplier, generation, 4
        );
    }

    StepProfile step_generation(long long generation) {
        validate_array_length();

        StepProfile profile_row;
        active_profile = p.profile ? &profile_row : nullptr;

        // Same mutation-class order as the Python/Dong-style implementation.
        const long long n_snp =
            draw_snp_event_count();

        if (p.profile && n_snp > 0) {
            const auto t0 = chrono::steady_clock::now();

            for (long long event = 0; event < n_snp; ++event) {
                apply_snp(generation);
            }

            profile_row.time_snp =
                chrono::duration<double>(
                    chrono::steady_clock::now() - t0
                ).count();
        } else {
            for (long long event = 0; event < n_snp; ++event) {
                apply_snp(generation);
            }
        }

        profile_row.snp = n_snp;

        const long long n_ins =
            draw_event_count(p.ins_rate, ins_count_distribution);

        if (p.profile && n_ins > 0) {
            const auto t0 = chrono::steady_clock::now();

            profile_row.ins = run_rejection_step(
                n_ins,
                p.ins_attempt_multiplier,
                generation,
                1
            );

            profile_row.time_ins =
                chrono::duration<double>(
                    chrono::steady_clock::now() - t0
                ).count();
        } else {
            profile_row.ins = run_rejection_step(
                n_ins,
                p.ins_attempt_multiplier,
                generation,
                1
            );
        }

        const long long n_del =
            draw_event_count(p.del_rate, del_count_distribution);

        if (p.profile && n_del > 0) {
            const auto t0 = chrono::steady_clock::now();

            profile_row.deletion = run_rejection_step(
                n_del,
                p.del_attempt_multiplier,
                generation,
                2
            );

            profile_row.time_del =
                chrono::duration<double>(
                    chrono::steady_clock::now() - t0
                ).count();
        } else {
            profile_row.deletion = run_rejection_step(
                n_del,
                p.del_attempt_multiplier,
                generation,
                2
            );
        }

        const long long n_large_del =
            draw_event_count(p.large_del_rate, large_del_count_distribution);

        if (p.profile && n_large_del > 0) {
            const auto t0 = chrono::steady_clock::now();

            profile_row.large_deletion = run_rejection_step(
                n_large_del,
                p.large_del_attempt_multiplier,
                generation,
                3
            );

            profile_row.time_large_del =
                chrono::duration<double>(
                    chrono::steady_clock::now() - t0
                ).count();
        } else {
            profile_row.large_deletion = run_rejection_step(
                n_large_del,
                p.large_del_attempt_multiplier,
                generation,
                3
            );
        }

        const long long n_conversion =
            draw_event_count(p.conversion_rate, conversion_count_distribution);

        if (p.profile && n_conversion > 0) {
            const auto t0 = chrono::steady_clock::now();

            profile_row.conversion = run_rejection_step(
                n_conversion,
                p.conversion_attempt_multiplier,
                generation,
                4
            );

            profile_row.time_conversion =
                chrono::duration<double>(
                    chrono::steady_clock::now() - t0
                ).count();
        } else {
            profile_row.conversion = run_rejection_step(
                n_conversion,
                p.conversion_attempt_multiplier,
                generation,
                4
            );
        }

        active_profile = nullptr;
        return profile_row;
    }
};

static vector<int> read_symbols(const string& path) {
    ifstream input(path);
    if (!input) throw runtime_error("Cannot open symbols input");

    vector<int> output;
    int value;

    while (input >> value) output.push_back(value);
    return output;
}

static vector<string> read_lookup(const string& path) {
    ifstream input(path);
    if (!input) throw runtime_error("Cannot open lookup input");

    vector<string> output;
    string line;

    while (getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (!line.empty()) output.push_back(line);
    }

    return output;
}

static void write_symbols(
    const string& path,
    const SymbolArray& symbols
) {
    ofstream output(path);
    if (!output) throw runtime_error("Cannot write symbols output");

    auto cursor = symbols.cursor(0);
    for (size_t index = 0; cursor; ++index, ++cursor) {
        if (index) output << ' ';
        output << *cursor;
    }

    output << '\n';
}

static void add_required_symbols(
    unordered_set<int>& required,
    const SymbolArray& symbols
) {
    symbols.add_to_set(required);
}

static void write_lookup_subset(
    const string& path,
    const vector<PackedMonomer>& lookup,
    int monomer_length,
    const unordered_set<int>& required
) {
    ofstream output(path);
    if (!output) throw runtime_error("Cannot write lookup output");

    vector<int> ordered(required.begin(), required.end());
    sort(ordered.begin(), ordered.end());

    for (int symbol : ordered) {
        if (
            symbol < 0 ||
            symbol >= static_cast<int>(lookup.size())
        ) {
            throw runtime_error("Required symbol outside lookup range");
        }

        // Keep the original C++ symbol ID in the file. Python compacts
        // these sparse IDs after loading the much smaller required subset.
        output
            << symbol
            << '\t'
            << unpack_sequence(lookup[symbol], monomer_length)
            << '\n';
    }
}

static void write_checkpoint(
    ofstream& output,
    long long generation,
    const SymbolArray& symbols
) {
    output << generation << '\t';

    auto cursor = symbols.cursor(0);
    for (size_t index = 0; cursor; ++index, ++cursor) {
        if (index) output << ' ';
        output << *cursor;
    }

    output << '\n';
}

static void write_profile_header(ofstream& output) {
    output
        << "generation_start\t"
        << "generation_end\t"
        << "n_generations\t"
        << "n_monomers\t"
        << "n_states\t"
        << "n_symbol_blocks\t"
        << "SNP\t"
        << "INS\t"
        << "DEL\t"
        << "Large_DEL\t"
        << "Conversion\t"
        << "conversion_fast_path\t"
        << "conversion_noop\t"
        << "ins_inserted_symbols\t"
        << "ins_shifted_symbols\t"
        << "ins_reallocations\t"
        << "ins_block_splits\t"
        << "time_snp\t"
        << "time_ins\t"
        << "time_del\t"
        << "time_large_del\t"
        << "time_conversion\t"
        << "intern_calls\t"
        << "new_states\t"
        << "rehashes\t"
        << "time_intern\t"
        << "time_intern_lookup\t"
        << "time_intern_insert\t"
        << "time_intern_snp\t"
        << "time_intern_ins\t"
        << "time_intern_del\t"
        << "time_intern_conversion\t"
        << "time_conversion_extract\t"
        << "time_conversion_replace\t"
        << "time_ins_chimera\t"
        << "time_ins_prepare\t"
        << "time_ins_vector_insert\t"
        << "time_mutations\t"
        << "time_other\t"
        << "time_total\n";
}

static void write_profile_row(
    ofstream& output,
    long long generation_start,
    long long generation_end,
    const Simulator& simulator,
    const StepProfile& profile,
    double time_total
) {
    const double time_mutations = profile.mutation_time();
    const double time_other = max(0.0, time_total - time_mutations);

    output
        << generation_start << '\t'
        << generation_end << '\t'
        << (generation_end - generation_start + 1) << '\t'
        << simulator.symbols.size() << '\t'
        << simulator.symbol_to_sequence.size() << '\t'
        << simulator.symbols.block_count() << '\t'
        << profile.snp << '\t'
        << profile.ins << '\t'
        << profile.deletion << '\t'
        << profile.large_deletion << '\t'
        << profile.conversion << '\t'
        << profile.conversion_fast_path << '\t'
        << profile.conversion_noop << '\t'
        << profile.ins_inserted_symbols << '\t'
        << profile.ins_shifted_symbols << '\t'
        << profile.ins_reallocations << '\t'
        << profile.ins_block_splits << '\t'
        << profile.time_snp << '\t'
        << profile.time_ins << '\t'
        << profile.time_del << '\t'
        << profile.time_large_del << '\t'
        << profile.time_conversion << '\t'
        << profile.intern_calls << '\t'
        << profile.new_states << '\t'
        << profile.rehashes << '\t'
        << profile.time_intern << '\t'
        << profile.time_intern_lookup << '\t'
        << profile.time_intern_insert << '\t'
        << profile.time_intern_snp << '\t'
        << profile.time_intern_ins << '\t'
        << profile.time_intern_del << '\t'
        << profile.time_intern_conversion << '\t'
        << profile.time_conversion_extract << '\t'
        << profile.time_conversion_replace << '\t'
        << profile.time_ins_chimera << '\t'
        << profile.time_ins_prepare << '\t'
        << profile.time_ins_vector_insert << '\t'
        << time_mutations << '\t'
        << time_other << '\t'
        << time_total << '\n';
}

int main(int argc, char** argv) {
    try {
        if (argc != 9) {
            cerr
                << "usage: symbolic_centromere_backend "
                << "CONFIG SYMBOLS LOOKUP CHECKPOINTS_OUT "
                << "FINAL_SYMBOLS_OUT LOOKUP_OUT EVENTS_OUT PROFILE_OUT\n";

            return 2;
        }

        Params parameters = read_config(argv[1]);

        vector<int> symbols = read_symbols(argv[2]);
        vector<string> lookup = read_lookup(argv[3]);

        Simulator simulator(
            parameters,
            std::move(symbols),
            std::move(lookup),
            argv[7]
        );

        vector<long long> checkpoints = parameters.checkpoints;
        sort(checkpoints.begin(), checkpoints.end());
        checkpoints.erase(
            unique(checkpoints.begin(), checkpoints.end()),
            checkpoints.end()
        );

        unordered_set<int> required_output_symbols;
        required_output_symbols.reserve(
            max<size_t>(1024, simulator.symbols.size() * 2)
        );

        ofstream checkpoint_output(argv[4]);

        if (!checkpoint_output) {
            throw runtime_error("Cannot open checkpoint output");
        }

        const long long start_generation =
            parameters.start_generation;

        const long long end_generation =
            start_generation + parameters.n_generations;

        if (parameters.profile && parameters.profile_interval <= 0) {
            throw runtime_error("profile_interval must be > 0");
        }

        ofstream profile_output;

        if (parameters.profile) {
            profile_output.open(argv[8]);

            if (!profile_output) {
                throw runtime_error("Cannot open profile output");
            }

            write_profile_header(profile_output);
        }

        // Skip requests before a resumed run; each requested generation is
        // emitted once, including the initial state when requested.
        auto next_checkpoint = lower_bound(
            checkpoints.begin(), checkpoints.end(), start_generation
        );

        if (
            next_checkpoint != checkpoints.end() &&
            *next_checkpoint == start_generation
        ) {
            ++next_checkpoint;
            write_checkpoint(
                checkpoint_output,
                start_generation,
                simulator.symbols
            );

            add_required_symbols(
                required_output_symbols,
                simulator.symbols
            );
        }

        if (parameters.profile) {
            StepProfile profile_window;
            long long profile_window_start = start_generation + 1;
            auto profile_window_clock = chrono::steady_clock::now();

            for (
                long long generation = start_generation + 1;
                generation <= end_generation;
                ++generation
            ) {
                StepProfile generation_profile =
                    simulator.step_generation(generation);

                profile_window.add(generation_profile);

                if (
                    next_checkpoint != checkpoints.end() &&
                    *next_checkpoint == generation
                ) {
                    ++next_checkpoint;
                    write_checkpoint(
                        checkpoint_output,
                        generation,
                        simulator.symbols
                    );

                    add_required_symbols(
                        required_output_symbols,
                        simulator.symbols
                    );
                }

                if (
                    ((generation - start_generation) % parameters.profile_interval == 0) ||
                    generation == end_generation
                ) {
                    const double time_total =
                        chrono::duration<double>(
                            chrono::steady_clock::now() - profile_window_clock
                        ).count();

                    write_profile_row(
                        profile_output,
                        profile_window_start,
                        generation,
                        simulator,
                        profile_window,
                        time_total
                    );

                    profile_window = StepProfile();
                    profile_window_start = generation + 1;
                    profile_window_clock = chrono::steady_clock::now();
                }
            }
        } else {
            for (
                long long generation = start_generation + 1;
                generation <= end_generation;
                ++generation
            ) {
                simulator.step_generation_unprofiled(generation);

                if (
                    next_checkpoint != checkpoints.end() &&
                    *next_checkpoint == generation
                ) {
                    ++next_checkpoint;
                    write_checkpoint(
                        checkpoint_output,
                        generation,
                        simulator.symbols
                    );

                    add_required_symbols(
                        required_output_symbols,
                        simulator.symbols
                    );
                }
            }
        }

        add_required_symbols(
            required_output_symbols,
            simulator.symbols
        );

        write_symbols(argv[5], simulator.symbols);

        write_lookup_subset(
            argv[6],
            simulator.symbol_to_sequence,
            simulator.monomer_length,
            required_output_symbols
        );

        return 0;

    } catch (const exception& error) {
        cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
