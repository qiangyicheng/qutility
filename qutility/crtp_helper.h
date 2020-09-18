#pragma once

#include <type_traits>

namespace qutility {
	namespace crtp_helper {
		template <typename ThisT, typename NextT, typename = void>
		struct NextFinal {
			using type = typename NextT::Final;
		};
		template <typename ThisT, typename NextT>
		struct NextFinal<ThisT, NextT, typename std::enable_if<std::is_same<ThisT, NextT>::value, void>::type>
		{
			using type = typename ThisT::NextT;
		};
	}
}

#define QUTILITY_CRTP_ACCESSOR_START(ThisType, DerivedType) \
struct crtp_accessor: DerivedT\
{\
	using ThisT = ThisType;\
	using NextT = DerivedType;\
	using FinalT = typename qutility::crtp_helper::NextFinal<crtp_accessor, typename NextT::crtp_accessor>::type;\
	struct crtp_final_accessor : FinalT \
	{ \
		using ThisT = crtp_accessor::ThisT; \
		using NextT = crtp_accessor::NextT; \
		using FinalT = crtp_accessor::FinalT; \
		static FinalT & cast_final(ThisT & instance){return static_cast<FinalT&>(instance);} \
		static NextT & cast_next(ThisT & instance){return static_cast<NextT&>(instance);} \

#define QUTILITY_CRTP_REGISTER_VOID_FUNCTION(funcname) \
	static void funcname(ThisT& instance) { \
		void (FinalT::* fn)() = &crtp_final_accessor::funcname##_impl; \
		return (static_cast<FinalT&>(instance).*fn)(); \
	} 

#define QUTILITY_CRTP_ACCESSOR_END \
	};\
};\

#define QUTILITY_CRTP_ACCESSOR crtp_accessor::crtp_final_accessor
