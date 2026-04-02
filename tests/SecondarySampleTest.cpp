#include "secondary/consensus/sample.h"

#include <catch2/catch_test_macros.hpp>
#include <torch/torch.h>

#include <cstdint>
#include <ostream>
#include <vector>

namespace dorado::secondary {
template <typename T>
std::ostream& operator<<(std::ostream& os, const IntervalGeneric<T>& interval) {
    os << '[' << interval.start << ", " << interval.end << ')';
    return os;
}
}  // namespace dorado::secondary

namespace dorado::secondary::sample::tests {

#define TEST_GROUP "[SecondaryConsensus]"

CATCH_TEST_CASE("slice_sample: Basic slicing", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::tensor({{1, 2, 3, 4, 5},
                                     {6, 7, 8, 9, 10},
                                     {11, 12, 13, 14, 15},
                                     {16, 17, 18, 19, 20},
                                     {21, 22, 23, 24, 25},
                                     {26, 27, 28, 29, 30},
                                     {31, 32, 33, 34, 35},
                                     {36, 37, 38, 39, 40},
                                     {41, 42, 43, 44, 45},
                                     {46, 47, 48, 49, 50}},
                                    torch::kInt32);
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    sample.depth =
            torch::tensor({1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1}, torch::kFloat32);

    CATCH_SECTION("Slice middle range") {
        const int64_t idx_start = 2;
        const int64_t idx_end = 7;

        const auto expected_features = torch::tensor({{11, 12, 13, 14, 15},
                                                      {16, 17, 18, 19, 20},
                                                      {21, 22, 23, 24, 25},
                                                      {26, 27, 28, 29, 30},
                                                      {31, 32, 33, 34, 35}},
                                                     torch::kInt32);
        const auto expected_depth = torch::tensor({3.3, 4.4, 5.5, 6.6, 7.7}, torch::kFloat32);
        const std::vector<int64_t> expected_positions_major{2, 3, 4, 5, 6};
        const std::vector<int64_t> expected_positions_minor{12, 13, 14, 15, 16};

        const Sample sliced_sample = slice_sample(sample, idx_start, idx_end);

        CATCH_CHECK(sliced_sample.seq_id == sample.seq_id);
        CATCH_CHECK(sliced_sample.features.equal(expected_features));
        CATCH_CHECK(sliced_sample.depth.equal(expected_depth));
        CATCH_CHECK(sliced_sample.positions_major == expected_positions_major);
        CATCH_CHECK(sliced_sample.positions_minor == expected_positions_minor);
        CATCH_CHECK(std::empty(sliced_sample.read_ids_left));
        CATCH_CHECK(std::empty(sliced_sample.read_ids_right));
    }

    CATCH_SECTION("Slice entire range") {
        const int64_t idx_start = 0;
        const int64_t idx_end = 10;

        const Sample sliced_sample = slice_sample(sample, idx_start, idx_end);

        CATCH_CHECK(sliced_sample.seq_id == sample.seq_id);
        CATCH_CHECK(sliced_sample.features.equal(sample.features));
        CATCH_CHECK(sliced_sample.depth.equal(sample.depth));
        CATCH_CHECK(sliced_sample.positions_major == sample.positions_major);
        CATCH_CHECK(sliced_sample.positions_minor == sample.positions_minor);
        CATCH_CHECK(std::empty(sliced_sample.read_ids_left));
        CATCH_CHECK(std::empty(sliced_sample.read_ids_right));
    }

    CATCH_SECTION("Slice single row") {
        const int64_t idx_start = 4;
        const int64_t idx_end = 5;

        const auto expected_features = torch::tensor({{21, 22, 23, 24, 25}}, torch::kInt32);
        const auto expected_depth = torch::tensor({5.5}, torch::kFloat32);
        const std::vector<int64_t> expected_positions_major{4};
        const std::vector<int64_t> expected_positions_minor{14};

        const Sample sliced_sample = slice_sample(sample, idx_start, idx_end);

        CATCH_CHECK(sliced_sample.seq_id == sample.seq_id);
        CATCH_CHECK(sliced_sample.features.equal(expected_features));
        CATCH_CHECK(sliced_sample.depth.equal(expected_depth));
        CATCH_CHECK(sliced_sample.positions_major == expected_positions_major);
        CATCH_CHECK(sliced_sample.positions_minor == expected_positions_minor);
        CATCH_CHECK(std::empty(sliced_sample.read_ids_left));
        CATCH_CHECK(std::empty(sliced_sample.read_ids_right));
    }
}

CATCH_TEST_CASE("slice_sample: Error conditions", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::rand({10, 5});
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    sample.depth = torch::rand({10});

    CATCH_SECTION("Invalid range: idx_start >= idx_end") {
        CATCH_CHECK_THROWS_AS(slice_sample(sample, 5, 5), std::out_of_range);
        CATCH_CHECK_THROWS_AS(slice_sample(sample, 6, 5), std::out_of_range);
    }

    CATCH_SECTION("Invalid range: idx_start or idx_end out of bounds") {
        CATCH_CHECK_THROWS_AS(slice_sample(sample, -1, 5), std::out_of_range);
        CATCH_CHECK_THROWS_AS(slice_sample(sample, 0, 11), std::out_of_range);
        CATCH_CHECK_THROWS_AS(slice_sample(sample, 10, 11), std::out_of_range);
    }
}

CATCH_TEST_CASE("slice_sample: features tensor in sample not defined", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    sample.depth = torch::rand({10});

    CATCH_CHECK_THROWS_AS(slice_sample(sample, 0, 5), std::runtime_error);
}

CATCH_TEST_CASE("slice_sample: depth tensor in sample not defined", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::rand({10, 5});
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

    CATCH_CHECK_THROWS_AS(slice_sample(sample, 0, 5), std::runtime_error);
}

CATCH_TEST_CASE("slice_sample: error, wrong length of the features tensor.", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::rand({20, 5});
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    sample.depth = torch::rand({10});

    CATCH_CHECK_THROWS_AS(slice_sample(sample, 0, 5), std::runtime_error);
}

CATCH_TEST_CASE("slice_sample: error, wrong length of the depth tensor.", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::rand({10, 5});
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    sample.depth = torch::rand({20});

    CATCH_CHECK_THROWS_AS(slice_sample(sample, 0, 5), std::runtime_error);
}

CATCH_TEST_CASE("slice_sample: error, wrong length of the positions_major vector.", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::rand({10, 5});
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    sample.depth = torch::rand({10});

    CATCH_CHECK_THROWS_AS(slice_sample(sample, 0, 5), std::runtime_error);
}

CATCH_TEST_CASE("slice_sample: error, wrong length of the positions_minor vector.", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::rand({10, 5});
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18};
    sample.depth = torch::rand({10});

    CATCH_CHECK_THROWS_AS(slice_sample(sample, 0, 5), std::runtime_error);
}

CATCH_TEST_CASE("find_max_depth", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::tensor({{1, 2, 3, 4, 5},
                                     {6, 7, 8, 9, 10},
                                     {11, 12, 13, 14, 15},
                                     {16, 17, 18, 19, 20},
                                     {21, 22, 23, 24, 25},
                                     {26, 27, 28, 29, 30},
                                     {31, 32, 33, 34, 35},
                                     {36, 37, 38, 39, 40},
                                     {41, 42, 43, 44, 45},
                                     {46, 47, 48, 49, 50}},
                                    torch::kInt32);
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    sample.depth = torch::tensor({1, 2, 2, 3, 2, 5, 2, 1, 0, 11}, torch::kInt64);

    CATCH_SECTION("Throws. Entire range") {
        const int64_t idx_start = 0;
        const int64_t idx_end = 10;

        constexpr int64_t EXPECTED = 11;

        const int64_t result = sample.find_max_depth(idx_start, idx_end);

        CATCH_CHECK(result == EXPECTED);
    }

    CATCH_SECTION("Subrange") {
        const int64_t idx_start = 6;
        const int64_t idx_end = 9;

        constexpr int64_t EXPECTED = 2;

        const int64_t result = sample.find_max_depth(idx_start, idx_end);

        CATCH_CHECK(result == EXPECTED);
    }

    CATCH_SECTION(
            "Throws. Float depth data type. This should throw because depth should be integral.") {
        Sample sample2 = sample;
        sample2.depth = torch::tensor({1.0f, 2.0f, 2.0f, 3.0f, 2.0f, 5.0f, 2.0f, 1.0f, 0.0f, 11.0f},
                                      torch::kFloat);
        const int64_t idx_start = 0;
        const int64_t idx_end = 10;

        CATCH_CHECK_THROWS(sample2.find_max_depth(idx_start, idx_end));
    }

    CATCH_SECTION("Throws. Start index < 0") {
        const int64_t idx_start = -1;
        const int64_t idx_end = 9;

        CATCH_CHECK_THROWS(sample.find_max_depth(idx_start, idx_end));
    }

    CATCH_SECTION("Throws. Emd index < start index") {
        const int64_t idx_start = 5;
        const int64_t idx_end = 4;

        CATCH_CHECK_THROWS(sample.find_max_depth(idx_start, idx_end));
    }

    CATCH_SECTION("Throws. Emd index == start index") {
        const int64_t idx_start = 5;
        const int64_t idx_end = 5;

        CATCH_CHECK_THROWS(sample.find_max_depth(idx_start, idx_end));
    }

    CATCH_SECTION("Throws. End index > length.") {
        const int64_t idx_start = 5;
        const int64_t idx_end = 11;

        CATCH_CHECK_THROWS(sample.find_max_depth(idx_start, idx_end));
    }
}

CATCH_TEST_CASE("find_sample_intervals: splits on true gaps and excludes low-depth runs",
                TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::rand({8, 5});
    sample.positions_major = {0, 1, 1, 2, 5, 6, 6, 7};
    sample.positions_minor = {0, 0, 1, 0, 0, 0, 1, 0};
    sample.read_ids_left = {"left_0", "left_1"};
    sample.read_ids_right = {"right_0", "right_1"};
    sample.depth = torch::tensor({5, 4, 1, 0, 4, 5, 2, 6}, torch::kInt64);

    CATCH_SECTION("Low-depth runs are excluded from the kept intervals") {
        const std::vector<Interval64> result = find_sample_intervals(sample, true, 3);
        const std::vector<Interval64> expected = {
                {0, 2},
                {4, 6},
                {7, 8},
        };

        CATCH_CHECK(result == expected);
    }

    CATCH_SECTION("True coordinate gaps split intervals even without low coverage") {
        const std::vector<Interval64> result = find_sample_intervals(sample, true, -1);
        const std::vector<Interval64> expected = {
                {0, 4},
                {4, 8},
        };

        CATCH_CHECK(result == expected);
    }
}

CATCH_TEST_CASE("find_sample_intervals: edge cases", TEST_GROUP) {
    CATCH_SECTION("Empty but valid sample returns no intervals") {
        Sample sample;
        sample.seq_id = 11;
        sample.features = torch::empty({0, 5}, torch::kFloat32);
        sample.positions_major = {};
        sample.positions_minor = {};
        sample.depth = torch::empty({0}, torch::kInt64);

        const std::vector<Interval64> result = find_sample_intervals(sample, true, 3);

        CATCH_CHECK(result.empty());
    }

    CATCH_SECTION("Dense input without gaps returns one full interval") {
        Sample sample;
        sample.seq_id = 12;
        sample.features = torch::rand({6, 5});
        sample.positions_major = {10, 10, 11, 11, 12, 13};
        sample.positions_minor = {0, 1, 0, 1, 0, 0};
        sample.depth = torch::tensor({3, 4, 5, 6, 7, 8}, torch::kInt64);

        const std::vector<Interval64> result = find_sample_intervals(sample, true, 3);

        CATCH_REQUIRE(result.size() == 1);
        CATCH_CHECK(result[0].start == 0);
        CATCH_CHECK(result[0].end == 6);
    }

    CATCH_SECTION("Everything masked out returns no intervals") {
        Sample sample;
        sample.seq_id = 13;
        sample.features = torch::rand({4, 5});
        sample.positions_major = {0, 1, 2, 3};
        sample.positions_minor = {0, 0, 0, 0};
        sample.depth = torch::tensor({0, 1, 1, 2}, torch::kInt64);

        const std::vector<Interval64> result = find_sample_intervals(sample, true, 3);

        CATCH_CHECK(result.empty());
    }

    CATCH_SECTION("Coordinate gaps are ignored when splitting is disabled") {
        Sample sample;
        sample.seq_id = 14;
        sample.features = torch::rand({4, 5});
        sample.positions_major = {0, 1, 5, 6};
        sample.positions_minor = {0, 0, 0, 0};
        sample.depth = torch::tensor({5, 5, 5, 5}, torch::kInt64);

        const std::vector<Interval64> result = find_sample_intervals(sample, false, 1);

        CATCH_REQUIRE(result.size() == 1);
        CATCH_CHECK(result[0].start == 0);
        CATCH_CHECK(result[0].end == 4);
    }
}

CATCH_TEST_CASE("split_sample_on_intervals: preserves flanking read IDs and slices correctly",
                TEST_GROUP) {
    Sample sample;
    sample.seq_id = 7;
    sample.features = torch::arange(0, 40, torch::kInt64).view({8, 5});
    sample.positions_major = {0, 1, 1, 2, 5, 6, 6, 7};
    sample.positions_minor = {0, 0, 1, 0, 0, 0, 1, 0};
    sample.depth = torch::tensor({5, 4, 1, 0, 4, 5, 2, 6}, torch::kInt64);
    sample.read_ids_left = {"left_0", "left_1"};
    sample.read_ids_right = {"right_0", "right_1"};

    const std::vector<Interval64> intervals = {
            {0, 2},
            {4, 6},
            {7, 8},
    };

    const std::vector<Sample> result = split_sample_on_intervals(sample, intervals);

    CATCH_REQUIRE(result.size() == 3);

    CATCH_CHECK(result[0].features.equal(
            torch::tensor({{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}}, torch::kInt64)));
    CATCH_CHECK(result[1].features.equal(
            torch::tensor({{20, 21, 22, 23, 24}, {25, 26, 27, 28, 29}}, torch::kInt64)));
    CATCH_CHECK(result[2].features.equal(torch::tensor({{35, 36, 37, 38, 39}}, torch::kInt64)));

    CATCH_CHECK(result[0].positions_major == std::vector<int64_t>{0, 1});
    CATCH_CHECK(result[1].positions_major == std::vector<int64_t>{5, 6});
    CATCH_CHECK(result[2].positions_major == std::vector<int64_t>{7});

    CATCH_CHECK(result[0].read_ids_left == sample.read_ids_left);
    CATCH_CHECK(result[0].read_ids_right ==
                std::vector<std::string>{"__placeholder_0", "__placeholder_1"});
    CATCH_CHECK(result[1].read_ids_left ==
                std::vector<std::string>{"__placeholder_0", "__placeholder_1"});
    CATCH_CHECK(result[1].read_ids_right ==
                std::vector<std::string>{"__placeholder_0", "__placeholder_1"});
    CATCH_CHECK(result[2].read_ids_left ==
                std::vector<std::string>{"__placeholder_0", "__placeholder_1"});
    CATCH_CHECK(result[2].read_ids_right == sample.read_ids_right);
}

CATCH_TEST_CASE("split_sample_on_intervals: edge cases", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 15;
    sample.features = torch::arange(0, 20, torch::kInt64).view({4, 5});
    sample.positions_major = {0, 1, 2, 3};
    sample.positions_minor = {0, 0, 0, 0};
    sample.depth = torch::tensor({5, 5, 5, 5}, torch::kInt64);
    sample.read_ids_left = {"left_0", "left_1"};
    sample.read_ids_right = {"right_0", "right_1"};

    CATCH_SECTION("Empty interval list returns no samples") {
        const std::vector<Sample> result = split_sample_on_intervals(sample, {});

        CATCH_CHECK(result.empty());
    }

    CATCH_SECTION("Full interval returns one sample with original boundary read IDs") {
        const std::vector<Sample> result = split_sample_on_intervals(sample, {{0, 4}});

        CATCH_REQUIRE(result.size() == 1);
        CATCH_CHECK(result[0].features.equal(sample.features));
        CATCH_CHECK(result[0].positions_major == sample.positions_major);
        CATCH_CHECK(result[0].positions_minor == sample.positions_minor);
        CATCH_CHECK(result[0].depth.equal(sample.depth));
        CATCH_CHECK(result[0].read_ids_left == sample.read_ids_left);
        CATCH_CHECK(result[0].read_ids_right == sample.read_ids_right);
    }

    CATCH_SECTION("Negative interval start throws") {
        CATCH_CHECK_THROWS_AS(split_sample_on_intervals(sample, {{-1, 2}}), std::runtime_error);
    }

    CATCH_SECTION("Empty interval throws") {
        CATCH_CHECK_THROWS_AS(split_sample_on_intervals(sample, {{2, 2}}), std::runtime_error);
    }

    CATCH_SECTION("Out of bounds interval throws") {
        CATCH_CHECK_THROWS_AS(split_sample_on_intervals(sample, {{0, 5}}), std::runtime_error);
    }
}

CATCH_TEST_CASE("split_sample_on_discontinuities: splits features and preserves boundary IDs",
                TEST_GROUP) {
    Sample sample;
    sample.seq_id = 16;
    sample.features = torch::arange(0, 40, torch::kInt64).view({8, 5});
    sample.positions_major = {0, 1, 1, 2, 5, 6, 6, 7};
    sample.positions_minor = {0, 0, 1, 0, 0, 0, 1, 0};
    sample.depth = torch::tensor({4, 4, 4, 4, 4, 4, 4, 4}, torch::kInt64);
    sample.read_ids_left = {"left_0", "left_1"};
    sample.read_ids_right = {"right_0", "right_1"};

    const std::vector<Sample> result = split_sample_on_discontinuities(sample);

    CATCH_REQUIRE(result.size() == 2);

    CATCH_CHECK(result[0].features.equal(torch::tensor(
            {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}, {10, 11, 12, 13, 14}, {15, 16, 17, 18, 19}},
            torch::kInt64)));
    CATCH_CHECK(result[1].features.equal(torch::tensor({{20, 21, 22, 23, 24},
                                                        {25, 26, 27, 28, 29},
                                                        {30, 31, 32, 33, 34},
                                                        {35, 36, 37, 38, 39}},
                                                       torch::kInt64)));

    CATCH_CHECK(result[0].positions_major == std::vector<int64_t>{0, 1, 1, 2});
    CATCH_CHECK(result[1].positions_major == std::vector<int64_t>{5, 6, 6, 7});

    CATCH_CHECK(result[0].read_ids_left == sample.read_ids_left);
    CATCH_CHECK(result[0].read_ids_right ==
                std::vector<std::string>{"__placeholder_0", "__placeholder_1"});
    CATCH_CHECK(result[1].read_ids_left ==
                std::vector<std::string>{"__placeholder_0", "__placeholder_1"});
    CATCH_CHECK(result[1].read_ids_right == sample.read_ids_right);
}

CATCH_TEST_CASE("split_sample_on_discontinuities: edge cases", TEST_GROUP) {
    CATCH_SECTION("Dense input is returned unsplit") {
        Sample sample;
        sample.seq_id = 16;
        sample.features = torch::arange(0, 30, torch::kInt64).view({6, 5});
        sample.positions_major = {20, 20, 21, 21, 22, 23};
        sample.positions_minor = {0, 1, 0, 1, 0, 0};
        sample.depth = torch::tensor({4, 4, 4, 4, 4, 4}, torch::kInt64);
        sample.read_ids_left = {"left_0", "left_1"};
        sample.read_ids_right = {"right_0", "right_1"};

        const std::vector<Sample> result = split_sample_on_discontinuities(sample);

        CATCH_REQUIRE(result.size() == 1);
        CATCH_CHECK(result[0].features.equal(sample.features));
        CATCH_CHECK(result[0].positions_major == sample.positions_major);
        CATCH_CHECK(result[0].positions_minor == sample.positions_minor);
        CATCH_CHECK(result[0].depth.equal(sample.depth));
        CATCH_CHECK(result[0].read_ids_left == sample.read_ids_left);
        CATCH_CHECK(result[0].read_ids_right == sample.read_ids_right);
    }

    CATCH_SECTION("Empty but valid sample returns no samples") {
        Sample sample;
        sample.seq_id = 17;
        sample.features = torch::empty({0, 5}, torch::kFloat32);
        sample.positions_major = {};
        sample.positions_minor = {};
        sample.depth = torch::empty({0}, torch::kInt64);

        const std::vector<Sample> result = split_sample_on_discontinuities(sample);

        CATCH_CHECK(result.empty());
    }
}

}  // namespace dorado::secondary::sample::tests
