#pragma once

namespace qutility {
	namespace ifmember {
		namespace detail {
			template < class... >
			using void_t = void;
		}
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
