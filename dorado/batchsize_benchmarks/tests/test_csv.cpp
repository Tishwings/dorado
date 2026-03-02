#include "SpeedEntry.h"
#include "csv_helpers.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <utility>

#define CUT_TAG "[batchsize_benchmarks]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)

using namespace dorado::batchsize_benchmarks;

namespace {

// SpeedEntry shouldn't need an equality check outside of the tests, so add that here.
bool entries_equal(std::span<const SpeedEntry> lhs, std::span<const SpeedEntry> rhs) {
    const auto compare = [](const SpeedEntry& a, const SpeedEntry& b) {
        return a.batch_size == b.batch_size && a.basecall_speed == b.basecall_speed &&
               a.memory_used == b.memory_used;
    };
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), compare);
}

DEFINE_TEST("Writing and reading") {
    const SpeedEntry gpu0_model0[]{
            {1, 2, 3},
    };
    const SpeedEntry gpu0_model1[]{
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12},
    };
    const SpeedEntry gpu1_model0[]{
            {1920, 123.456, 10'000'000'000},
    };
    const struct {
        std::string_view gpu_name;
        std::string_view model_name;
        std::span<const SpeedEntry> entries;
    } tests[] = {
            {"gpu0", "model0", gpu0_model0},
            {"gpu0", "model1", gpu0_model1},
            {"gpu1", "model0", gpu1_model0},
    };

    // Write entries to the stream.
    std::stringstream stream;
    for (const auto& test : tests) {
        csv_write_entries(stream, test.gpu_name, test.model_name, test.entries);
    }

    auto make_key = [](std::string_view gpu_name, std::string_view model_name) {
        return fmt::format("{}_{}", gpu_name, model_name);
    };

    // Read them back.
    std::unordered_map<std::string, std::vector<SpeedEntry>> read_entries;
    csv_read_lines(stream,
                   [&](std::string_view gpu_name, std::string_view model_name, SpeedEntry entry) {
                       read_entries[make_key(gpu_name, model_name)].push_back(entry);
                   });

    // Check that they match.
    CATCH_CHECK(read_entries.size() == std::size(tests));
    for (const auto& test : tests) {
        CATCH_CAPTURE(test.gpu_name, test.model_name);
        auto it = read_entries.find(make_key(test.gpu_name, test.model_name));
        CATCH_REQUIRE(it != read_entries.end());
        CATCH_CHECK(entries_equal(it->second, test.entries));
    }
}

DEFINE_TEST("Input with bad lines") {
    std::istringstream stream(
            "gpu,model,1,2\n"          // missing a column
            ",model,1,2,3\n"           // empty column
            "gpu,,1,2,3\n"             // empty column
            "gpu,model,,2,3\n"         // empty column
            "gpu,model,1,,3\n"         // empty column
            "gpu,model,1,2,\n"         // empty column
            "gpu,model,invalid,2,3\n"  // not a number
            "gpu,model,1,invalid,3\n"  // not a number
            "gpu,model,1,2,invalid\n"  // not a number
            "\n"                       // empty line
            // Valid lines come last to check that we continue parsing after the bad lines.
            "gpu,model,0,2,3\n"        // valid
            "gpu,model,1,2,3,extra\n"  // extra column should be ignored
    );

    std::size_t times_called = 0;
    csv_read_lines(stream,
                   [&](std::string_view gpu_name, std::string_view model_name, SpeedEntry entry) {
                       CATCH_CHECK(gpu_name == "gpu");
                       CATCH_CHECK(model_name == "model");
                       CATCH_CHECK(entry.batch_size == times_called);
                       CATCH_CHECK(entry.basecall_speed == 2);
                       CATCH_CHECK(entry.memory_used == 3);
                       times_called++;
                   });
    CATCH_CHECK(times_called == 2);
}

}  // namespace
