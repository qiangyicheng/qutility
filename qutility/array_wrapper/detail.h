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

			template<typename T1, typename Alloc1, typename T2, typename Alloc2>
			std::vector<T1, Alloc1> duplicate(const std::vector<T2, Alloc2>& v, size_t size) {
				auto v_dup = v;
				v_dup.resize(size);
				return std::vector<T1, Alloc1>(v_dup.cbegin(), v_dup.cend());
			}
		}
	}
}
