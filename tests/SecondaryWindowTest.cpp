#include "secondary/common/region.h"
#include "secondary/common/window.h"
#include "secondary/consensus/window_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dorado::secondary::sample::tests {

#define TEST_GROUP "[SecondaryConsensus]"

CATCH_TEST_CASE("create_windows tests", TEST_GROUP) {
    struct TestCase {
        std::string name;
        int32_t seq_id = 0;
        int64_t seq_start = 0;
        int64_t seq_end = 0;
        int64_t seq_len = 0;
        int32_t window_len = 0;
        int32_t window_overlap = 0;
        int32_t source_region_id = 0;
        std::vector<Window> expected;
    };

    // clang-format off
    auto [test_case] = GENERATE(table<TestCase>({
        TestCase{
            "Coordinates and lengths all zero", 0, 0, 0, 0, 0, 0, 0,
            {},
        },
        TestCase{
            "Normal full-contig", 0, 0, 10000, 10000, 2000, 100, 1,
            {
                Window{0, 10000, 0, 2000, 0, 2000, 1},
                Window{0, 10000, 1900, 3900, 2000, 3900, 1},
                Window{0, 10000, 3800, 5800, 3900, 5800, 1},
                Window{0, 10000, 5700, 7700, 5800, 7700, 1},
                Window{0, 10000, 7600, 9600, 7700, 9600, 1},
                Window{0, 10000, 9500, 10000, 9600, 10000, 1},
            },
        },
        TestCase{
            "Normal short", 0, 0, 500, 10000, 2000, 100, 2,
            {
                Window{0, 10000, 0, 500, 0, 500, 2},
            },
        },
        TestCase{
            "Normal, internal region", 0, 100, 3000, 10000, 2000, 100, 3,
            {
                Window{0, 10000, 100, 2100, 100, 2100},
                Window{0, 10000, 2000, 3000, 2100, 3000},
            },
        },
        TestCase{
            "Zero-length input interval. Returns empty.", 0, 5, 5, 10000, 2000, 100, 4,
            {},
        },
        TestCase{
            "End coordinate over contig length. Returns empty.", 0, 0, 20000, 10000, 2000, 100, 5,
            {},
        },
        TestCase{
            "Start coordinate is < 0. Returns empty.", 0, -5, 10000, 10000, 2000, 100, 6,
            {},
        },
        TestCase{
            "Sequence length is < 0. Returns empty.", 0, 0, 10000, -10000, 2000, 100, 7,
            {},
        },
        TestCase{
            "Start coordinate is > end coordinate. Returns empty.", 0, 12000, 10000, 10000, 2000, 100, 8,
            {},
        },
        TestCase{
            "Window overlap is >= window_len / 2. Returns empty.", 0, 0, 10000, 10000, 2000, 1000, 9,
            {},
        },
        TestCase{
            "Window len == 0. Returns empty.", 0, 0, 10000, 10000, 0, 1000, 10,
            {},
        },
        TestCase{
            "Window len < 0. Returns empty.", 0, 0, 10000, 10000, -1, 1000, 11,
            {},
        },
        TestCase{
            "Window overlap < 0. Returns empty.", 0, 0, 10000, 10000, 2000, -1, 12,
            {},
        },
        TestCase{
            "Window overlap == 0. This is fine, should return non-overlapping windows.", 0, 0, 10000, 10000, 2000, 0, 13,
            {
                Window{0, 10000, 0, 2000, 0, 2000, 13},
                Window{0, 10000, 2000, 4000, 2000, 4000, 13},
                Window{0, 10000, 4000, 6000, 4000, 6000, 13},
                Window{0, 10000, 6000, 8000, 6000, 8000, 13},
                Window{0, 10000, 8000, 10000, 8000, 10000, 13},
            },
        },
    }));
    // clang-format on

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.name);
    const std::vector<Window> result = create_windows(
            test_case.seq_id, test_case.seq_start, test_case.seq_end, test_case.seq_len,
            test_case.window_len, test_case.window_overlap, test_case.source_region_id);
    CATCH_CHECK(result == test_case.expected);
}

CATCH_TEST_CASE("create_windows_from_regions expands and clamps regions", TEST_GROUP) {
    struct TestCase {
        std::string name;
        std::vector<Region> regions;
        std::unordered_map<std::string, std::pair<int64_t, int64_t>> draft_lookup;
        int32_t bam_chunk_len = 0;
        int32_t window_overlap = 0;
        std::vector<Window> expected;
    };

    // clang-format off
    auto [test_case] = GENERATE(table<TestCase>({
        TestCase{
            "Empty input returns no windows",
            {},
            {
                {"chr1", {0, 10000}},
            },
            2000, 100,
            {},
        },
        TestCase{
            "Splits multiple regions and preserves source_region_id",
            {
                Region{"chr1", 100, 3100},
                Region{"chr2", 0, 600},
            },
            {
                {"chr1", {0, 10000}},
                {"chr2", {1, 2000}},
            },
            2000, 100,
            {
                Window{0, 10000, 100, 2100, 100, 2100, 0},
                Window{0, 10000, 2000, 3100, 2100, 3100, 0},
                Window{1, 2000, 0, 600, 0, 600, 1},
            },
        },
        TestCase{
            "Clamps negative starts, open-ended ends, and overhanging ends to contig bounds",
            {
                Region{"chr1", -50, -1},
                Region{"chr1", 450, 600},
            },
            {
                {"chr1", {7, 500}},
            },
            200, 50,
            {
                Window{7, 500, 0, 200, 0, 200, 0},
                Window{7, 500, 150, 350, 200, 350, 0},
                Window{7, 500, 300, 500, 350, 500, 0},
                Window{7, 500, 450, 500, 450, 500, 1},
            },
        },
    }));
    // clang-format on

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.name);
    const std::vector<Window> result =
            create_windows_from_regions(test_case.regions, test_case.draft_lookup,
                                        test_case.bam_chunk_len, test_case.window_overlap);
    CATCH_CHECK(result == test_case.expected);
}

CATCH_TEST_CASE("create_windows_from_regions validates region lookup and coordinates", TEST_GROUP) {
    const std::unordered_map<std::string, std::pair<int64_t, int64_t>> draft_lookup = {
            {"chr1", {0, 500}},
    };

    CATCH_SECTION("Throws when the requested contig is not present in the draft lookup") {
        const std::vector<Region> regions = {
                Region{"missing_chr", 0, 100},
        };

        CATCH_CHECK_THROWS_WITH(create_windows_from_regions(regions, draft_lookup, 200, 50),
                                Catch::Matchers::ContainsSubstring(
                                        "Sequence specified by custom region not found in input"));
    }

    CATCH_SECTION("Throws when the clamped region coordinates become invalid") {
        const std::vector<Region> regions = {
                Region{"chr1", 600, 700},
        };

        CATCH_CHECK_THROWS_WITH(create_windows_from_regions(regions, draft_lookup, 200, 50),
                                Catch::Matchers::ContainsSubstring("Region coordinates not valid"));
    }
}

}  // namespace dorado::secondary::sample::tests
