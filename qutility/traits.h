#pragma once

#include <type_traits>
#include <tuple>

namespace qutility {
	namespace traits {
		template <typename EleType, typename ... Args>
		struct is_list;

		template <typename EleType, typename First, typename ... Rest>
		struct is_list<EleType, First, Rest...> {
			static constexpr bool value = std::is_convertible<First, EleType>::value && is_list<EleType, Rest...>::value;
		};

		template <typename EleType, typename Last>
		struct is_list<EleType, Last> {
			static constexpr bool value = std::is_convertible<Last, EleType>::value;
		};

		template <size_t pos, typename ... Args>
		struct static_switch;

		template <size_t pos, typename First, typename ... Rest>
		struct static_switch<pos, First, Rest...> {
			static_assert(pos <= sizeof...(Rest), "The index must be smaller than the number of candidate types");
			using type = typename static_switch<pos - 1, Rest...>::type;
		};

		template <typename First, typename ... Rest>
		struct static_switch<0, First, Rest...> {
			using type = First;
		};

		template <typename T, typename... Args>
		struct prepend_tuple;

		template <typename T, typename... Args>
		struct prepend_tuple<T, std::tuple<Args...>> {
			using type = std::tuple<T, Args...>;
		};

		template <size_t n, typename ... Args>
		struct first_n;

		template <size_t n, typename First, typename ... Rest>
		struct first_n<n, First, Rest...> {
			static_assert(n <= sizeof...(Rest), "n must be smaller than the number of candidate types");
			using type = typename prepend_tuple<First, typename first_n<n - 1, Rest...>::type>::type;
		};

		template <typename ... Rest>
		struct first_n<0, Rest...> {
			using type = std::tuple<>;
		};

		template <size_t n, typename ... Args>
		struct rest_n;

		template<size_t n, typename First, typename... Rest>
		struct rest_n<n, First, Rest...> {
			static_assert(n <= sizeof...(Rest), "n must be smaller than the number of candidate types");
			using type = typename rest_n<n - 1, Rest...>::type;
		};

		template<typename ... Rest>
		struct rest_n<0, Rest...> {
			using type = std::tuple<Rest...>;
		};

		template<size_t pr, size_t pc, typename Requirements, typename Candidates>
		struct is_correct_tuple_impl {
			constexpr static bool value
				= std::tuple_element_t<pr, Requirements>::template apply<
				std::tuple_element_t<pc, Candidates>
				>::value
				&& is_correct_tuple_impl<
				pr == 0 ? (std::tuple_size<Requirements>::value - 1) : pr - 1,
				pc - 1,
				Requirements, Candidates
				>::value;
		};

		template<size_t pr, typename Requirements, typename Candidates>
		struct is_correct_tuple_impl<pr, 0, Requirements, Candidates> {
			constexpr static bool value = false;
		};

		template<typename Requirements, typename Candidates>
		struct is_correct_tuple_impl<0, 0, Requirements, Candidates> {
			constexpr static bool value = true;
		};

		template<typename Requirements, typename Candidates>
		struct is_correct_tuple {
			constexpr static bool value = is_correct_tuple_impl<
				std::tuple_size<Requirements>::value - 1,
				std::tuple_size<Candidates>::value - 1,
				Requirements, Candidates>::value;
		};

		template <size_t n, typename ... Args>
		struct is_correct_list_of_n {
			constexpr static size_t nr = n;
			constexpr static size_t nc = sizeof...(Args) - nr;
			static_assert(n < sizeof...(Args), "n must be smaller than the number of candidate types");
			using Requirements = typename first_n<n, Args...>::type;
			using Candidates = typename rest_n<n, Args...>::type;
			constexpr static bool value = (nc % nr == 0) && is_correct_tuple<Requirements, Candidates>::value;
		};

		template<typename T>
		struct is_tytpe_T {
			template<typename U>
			using apply = std::is_same<T, U>;
		};
	}
}