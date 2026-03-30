#pragma once

// Backport of std::move_only_function.

#include <functional>
#include <memory>
#include <type_traits>

namespace dorado::utils {

template <typename T>
class MoveOnlyFunction;

template <typename R, typename... Args>
class MoveOnlyFunction<R(Args...)> {
    struct ICallable {
        virtual ~ICallable() = default;
        virtual R invoke(Args&&...) = 0;
    };
    // TODO: small functor optimisation
    std::unique_ptr<ICallable> m_callable;

public:
    MoveOnlyFunction() noexcept = default;
    MoveOnlyFunction(MoveOnlyFunction&& o) noexcept = default;
    MoveOnlyFunction& operator=(MoveOnlyFunction&& o) noexcept = default;
    MoveOnlyFunction(const MoveOnlyFunction& o) noexcept = delete;
    MoveOnlyFunction& operator=(const MoveOnlyFunction& o) noexcept = delete;
    ~MoveOnlyFunction() = default;

    template <typename Func>
    MoveOnlyFunction(Func&& func) {
        using F = std::remove_const_t<Func>;
        struct Callable : ICallable {
            F m_func;
            explicit Callable(Func&& f) : m_func(std::forward<Func>(f)) {}
            R invoke(Args&&... args) override { return m_func(std::move(args)...); }
        };
        m_callable = std::make_unique<Callable>(std::forward<Func>(func));
    }

    template <typename... FuncArgs>
    R operator()(FuncArgs&&... args) {
        if (!*this) {
            throw std::bad_function_call();
        }
        return m_callable->invoke(std::forward<FuncArgs>(args)...);
    }

    explicit operator bool() const { return m_callable != nullptr; }
};

}  // namespace dorado::utils
