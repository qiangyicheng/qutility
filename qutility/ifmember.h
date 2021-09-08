#pragma once

#include <type_traits>
#include <functional>
#include <utility>
#include <iostream>
#include <string>
#include <algorithm>
#include <cassert>

namespace qutility {
	namespace ifmember {
		namespace detail {
			template < class... >
			using void_t = void;

			template<class X, class Y, class Op>
			struct op_valid_impl
			{
				template<class U, class L, class R>
				static auto test(int) -> decltype(std::declval<U>()(std::declval<L>(), std::declval<R>()),
					void(), std::true_type());

				template<class U, class L, class R>
				static auto test(...)->std::false_type;

				using type = decltype(test<Op, X, Y>(0));

			};

			template<class X, class Y, class Op> using op_valid = typename op_valid_impl<X, Y, Op>::type;

			namespace notstd {

				struct left_shift {

					template <class L, class R>
					constexpr auto operator()(L&& l, R&& r) const
						noexcept(noexcept(std::forward<L>(l) << std::forward<R>(r)))
						-> decltype(std::forward<L>(l) << std::forward<R>(r))
					{
						return std::forward<L>(l) << std::forward<R>(r);
					}
				};

				struct right_shift {

					template <class L, class R>
					constexpr auto operator()(L&& l, R&& r) const
						noexcept(noexcept(std::forward<L>(l) >> std::forward<R>(r)))
						-> decltype(std::forward<L>(l) >> std::forward<R>(r))
					{
						return std::forward<L>(l) >> std::forward<R>(r);
					}
				};

			}
		}

		template<class X, class Y> using has_operator_equality = detail::op_valid<X, Y, std::equal_to<>>;
		template<class X, class Y> using has_operator_inequality = detail::op_valid<X, Y, std::not_equal_to<>>;
		template<class X, class Y> using has_operator_less_than = detail::op_valid<X, Y, std::less<>>;
		template<class X, class Y> using has_operator_less_equal = detail::op_valid<X, Y, std::less_equal<>>;
		template<class X, class Y> using has_operator_greater_than = detail::op_valid<X, Y, std::greater<>>;
		template<class X, class Y> using has_operator_greater_equal = detail::op_valid<X, Y, std::greater_equal<>>;
		template<class X, class Y> using has_operator_bit_xor = detail::op_valid<X, Y, std::bit_xor<>>;
		template<class X, class Y> using has_operator_bit_or = detail::op_valid<X, Y, std::bit_or<>>;
		template<class X, class Y> using has_operator_left_shift = detail::op_valid<X, Y, detail::notstd::left_shift>;
		template<class X, class Y> using has_operator_right_shift = detail::op_valid<X, Y, detail::notstd::right_shift>;

	}
}

#define GENERATE_HAS_MEMBER_TYPE_ANY(Type)													\
namespace qutility::ifmember::detail {														\
	struct P { typedef int Type; };															\
	template <typename U>																	\
	struct has_member_type_##Type##_any : U, P												\
	{																						\
		template <typename T = has_member_type_##Type##_any, typename = typename T::Type>	\
		static std::false_type test(int);													\
		static std::true_type test(float);													\
	};																						\
}																							\
template <typename T>																		\
using has_member_type_##Type##_any =														\
std::integral_constant<																		\
	bool,																					\
	decltype(qutility::ifmember::detail::has_member_type_##Type##_any<T>::test(0)){}		\
>;																							\

#define GENERATE_HAS_MEMBER_TYPE_PUBLIC(Type)												\
namespace qutility::ifmember::detail {														\
	template < class, class = void_t<> >													\
	struct has_member_type_##Type##_public : std::integral_constant<bool,false> { };		\
	template < class T >																	\
	struct has_member_type_##Type##_public < T, void_t<typename T::Type>>					\
	: std::integral_constant<bool,true> { };												\
}																							\
template <typename T>																		\
using has_member_type_##Type##_public =														\
qutility::ifmember::detail::has_member_type_##Type##_public<T>;								\

#define GENERATE_HAS_MEMBER_TYPE_PROTECTED_OR_PUBLIC(Type)									\
namespace qutility::ifmember::detail{														\
	template <typename T>																	\
	struct private_derived_class_of_##Type : private T {									\
		friend struct has_member_type_##Type##_protected_or_public;							\
	};																						\
	struct has_member_type_##Type##_protected_or_public {									\
		template < class, class = void_t<> >												\
		struct apply																		\
			: std::integral_constant<bool,false> { };										\
		template < class T >																\
		struct apply < T, void_t<typename T::Type>>											\
		: std::integral_constant<bool,true> { };											\
	};																						\
}																							\
template <typename T>																		\
using has_member_type_##Type##_protected_or_public											\
	= qutility::ifmember::detail::has_member_type_##Type##_protected_or_public				\
	::apply<qutility::ifmember::detail::private_derived_class_of_##Type<T>>;				\

#define GENERATE_HAS_MEMBER_TYPE_PROTECTED(Type)											\
template <typename T>																		\
using has_member_type_##Type##_protected													\
	=std::integral_constant<bool,															\
		has_member_type_##Type##_protected_or_public<T>::value								\
		&&(!has_member_type_##Type##_public<T>::value)										\
	>;																						\

#define GENERATE_HAS_MEMBER_TYPE_PRIVATE_OR_NO_ACCESS(Type)									\
template <typename T>																		\
using has_member_type_##Type##_private_or_no_access											\
	=std::integral_constant<bool,															\
		(!has_member_type_##Type##_protected_or_public<T>::value)							\
		&&(has_member_type_##Type##_any<T>::value)											\
	>;																						\

#define GENERATE_HAS_MEMBER_TYPE(Type)														\
GENERATE_HAS_MEMBER_TYPE_ANY(Type)															\
GENERATE_HAS_MEMBER_TYPE_PUBLIC(Type)														\
GENERATE_HAS_MEMBER_TYPE_PROTECTED_OR_PUBLIC(Type)											\
GENERATE_HAS_MEMBER_TYPE_PROTECTED(Type)													\
GENERATE_HAS_MEMBER_TYPE_PRIVATE_OR_NO_ACCESS(Type)											\


