#pragma once

#include <vector>

namespace qutility {
	namespace array_wrapper {
		namespace detail {
			template<typename T, typename Alloc>
			std::vector<T, Alloc> duplicate(const std::vector<T, Alloc>& v, size_t size) {
				auto v_dup = v;
				v_dup.resize(size);
				return v_dup;
			}
		}
	}
}
