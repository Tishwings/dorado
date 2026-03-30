#include "utils/MoveOnlyFunction.h"

#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <type_traits>

#define CUT_TAG "[MoveOnlyFunction]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)

using namespace dorado::utils;

namespace {

DEFINE_TEST("Empty behaviour") {
    MoveOnlyFunction<void()> empty;
    CATCH_CHECK_FALSE(empty);
    CATCH_CHECK_THROWS_AS(empty(), std::bad_function_call);

    MoveOnlyFunction<void()> moved_ctor = std::move(empty);
    CATCH_CHECK_FALSE(empty);
    CATCH_CHECK_FALSE(moved_ctor);
    CATCH_CHECK_THROWS_AS(moved_ctor(), std::bad_function_call);

    MoveOnlyFunction<void()> moved_assigned;
    moved_assigned = std::move(moved_ctor);
    CATCH_CHECK_FALSE(moved_ctor);
    CATCH_CHECK_FALSE(moved_assigned);
    CATCH_CHECK_THROWS_AS(moved_assigned(), std::bad_function_call);
}

DEFINE_TEST("Callable is called once") {
    int counter = 0;

    MoveOnlyFunction<int(int)> add_to_counter = [&](int x) { return counter += x; };
    CATCH_CHECK(add_to_counter(2) == 2);
    CATCH_CHECK(counter == 2);

    MoveOnlyFunction<int(int)> moved_ctor = std::move(add_to_counter);
    CATCH_CHECK_FALSE(add_to_counter);
    CATCH_CHECK(moved_ctor(3) == 5);
    CATCH_CHECK(counter == 5);

    MoveOnlyFunction<int(int)> moved_assigned;
    moved_assigned = std::move(moved_ctor);
    CATCH_CHECK_FALSE(moved_ctor);
    CATCH_CHECK(moved_assigned(4) == 9);
    CATCH_CHECK(counter == 9);
}

DEFINE_TEST("Move only lambda") {
    auto counter_ptr = std::make_unique<int>();
    const int & counter = *counter_ptr;

    auto lambda = [ptr = std::move(counter_ptr)] { *ptr += 1; };
    using Lambda = std::remove_reference_t<decltype(lambda)>;
    static_assert(!std::is_copy_constructible_v<Lambda> && !std::is_copy_assignable_v<Lambda>);

    MoveOnlyFunction<void()> functor = std::move(lambda);
    CATCH_CHECK(counter == 0);
    functor();
    CATCH_CHECK(counter == 1);
    functor();
    CATCH_CHECK(counter == 2);
}

}  // namespace
