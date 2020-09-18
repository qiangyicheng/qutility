#pragma once

#include <type_traits>

namespace qutility {
	namespace c_array {
		template<class T, size_t N>
		struct c_array
		{
			T arr[N];

			constexpr T const& operator[](size_t p) const
			{
				return arr[p];
			}

			constexpr T const* begin() const
			{
				return arr + 0;
			}
			constexpr T const* end() const
			{
				return arr + N;
			}

			constexpr size_t size() const
			{
				return N;
			}
		};

		template<class T>
		struct c_array<T, 0> {};

		template<size_t... Is>
		struct seq {};

		template<size_t N, size_t... Is>
		struct gen_seq : gen_seq<N - 1, N - 1, Is...> {};

		template<size_t... Is>
		struct gen_seq<0, Is...> : seq<Is...> {};

		template<class T, size_t N, class U, size_t... Is>
		constexpr c_array<T, N + 1> append_impl(c_array<T, N> const& p, U const& e,
			seq<Is...>)
		{
			return { { p[Is]..., e } };
		}

		template<class T, size_t N, class U>
		constexpr c_array<T, N + 1> append(c_array<T, N> const& p, U const& e)
		{
			return append_impl(p, e, gen_seq<N>{});
		}

		template<class T, size_t N, size_t... Is>
		constexpr c_array<T, N - 1> pop_impl(c_array<T, N> const& p,
			seq<Is...>)
		{
			return { { p[Is]... } };
		}

		template<class T, size_t N>
		constexpr c_array<T, N - 1> pop(c_array<T, N> const& p)
		{
			return pop_impl(p, gen_seq<N - 1>{});
		}

		template<class T>
		constexpr c_array<T, 0> pop(c_array<T, 1> const& p)
		{
			return c_array<T, 0>();
		}

		template<class T, size_t N, size_t... Is>
		constexpr c_array<T, N> reverse_impl(c_array<T, N> const& p,
			seq<Is...>)
		{
			return { { p[N - 1 - Is]... } };
		}

		template<class T, size_t N>
		constexpr c_array<T, N> reverse(c_array<T, N> const& p)
		{
			return  reverse_impl(p, gen_seq<N>{});
		}

		template<class T, size_t M, size_t... Is >
		constexpr c_array<T, sizeof...(Is)> pick_last_impl(c_array<T, M> const& p,
			seq<Is...>)
		{
			return { {p[Is + (M - sizeof...(Is))]...} };
		}

		template<size_t N, class T, size_t M>
		constexpr c_array<T, N> pick_last(c_array<T, M> const& p)
		{
			static_assert((N <= M), "Lenth of sequence must be at least N");
			return pick_last_impl(p, gen_seq<N>{});
		}

		template <typename T, size_t M, size_t N, size_t ... Is>
		constexpr c_array<T, M* N> flattern_impl(c_array<c_array<T, M>, N> const& matrix, seq<Is...>) {
			return { { (matrix[Is / M][Is % M])... } };
		}

		template <typename T, size_t M, size_t N>
		constexpr c_array<T, M* N> flattern(c_array<c_array<T, M>, N> const& matrix) {
			return flattern_impl(matrix, gen_seq<M* N>{});
		}

		template <typename Test, typename U, U... e>
		struct append_case {};

		template <typename Test, typename U, U e, U... Else>
		struct append_case<Test, U, e, Else...> {
			template <typename T, size_t N>
			constexpr static auto apply(c_array<T, N> const& p) {
				return append_case<Test, U, Else...>::apply(
					append_case<std::bool_constant<Test::apply(e)>, U, e>::apply(p)
				);
			}
		};

		template <typename Test, typename U, U e>
		struct append_case<Test, U, e> {
			template <typename T, size_t N>
			constexpr static auto apply(c_array<T, N> const& p)
			{
				return append_case<std::bool_constant<Test::apply(e)>, U, e>::apply(p);
			}
		};

		template <typename U, U e>
		struct append_case<std::false_type, U, e> {
			template <typename T, size_t N>
			constexpr static c_array<T, N> apply(c_array<T, N> const& p) { return p; }
		};

		template <typename U, U e>
		struct append_case<std::true_type, U, e> {
			template <typename T, size_t N>
			constexpr static c_array<T, N + 1> apply(c_array<T, N> const& p) { return append(p, e); }
		};

	}
}