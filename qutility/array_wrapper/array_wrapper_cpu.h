#pragma once

#include <cstdio>
#include <vector>
#include "boost/align.hpp"
#include "detail.h"
#include "hbw_posix_allocator.h"

namespace qutility {
	namespace array_wrapper {

		class ArrayCPUBase {};

		template<class T, class Allocator, size_t A>
		class ArrayCPU : ArrayCPUBase {
		public:
			ArrayCPU() = delete;
			ArrayCPU(std::size_t S) : data_(S, T()), size_(S), pointer_(&(data_.at(0))) {	}
			ArrayCPU(const T& val, std::size_t S) : data_(S, val), size_(S), pointer_(&(data_.at(0))) {	}
			template<typename OtherT, typename OtherAlloc>
			ArrayCPU(const std::vector<OtherT, OtherAlloc>& v, std::size_t S) : data_(detail::duplicate(v, S)), size_(S), pointer_(&(data_.at(0))) {}
			ArrayCPU(const ArrayCPU& rhs) : data_(rhs.data_), size_(rhs.size_), pointer_(&(data_.at(0))) {	}
			ArrayCPU(ArrayCPU&& rhs) : data_(std::move(rhs.data_)), size_(rhs.size_), pointer_(&(data_.at(0))) {	}
			ArrayCPU& operator=(const ArrayCPU& rhs) {
				if (size_ < rhs.size_) throw std::logic_error("Assignment can not be done from a larger array to a smaller one");
				std::memcpy(pointer_, rhs.pointer(), sizeof(T) * rhs.size_);
				return *this;
			}
			ArrayCPU& operator=(ArrayCPU&& rhs) {
				if (size_ < rhs.size_) throw std::logic_error("Assignment can not be done from a larger array to a smaller one");
				std::memcpy(pointer_, rhs.pointer(), sizeof(T) * rhs.size_);
				return *this;
			}

			operator T* () { return pointer_; }
			operator const T* () const { return pointer_; }
			T* operator+(size_t shift) { return pointer_ + shift; }
			const T* operator+(size_t shift) const { return pointer_ + shift; }

			const std::size_t size_;
			constexpr static std::size_t Alignment = A;

			inline const T* pointer() const { return pointer_; }
			inline T* pointer() { return pointer_; }
			inline const T& operator[](size_t pos) const { return data_[pos]; }
			inline T& operator[](size_t pos) { return data_[pos]; }

		protected:
			std::vector<T, Allocator> data_;
			T* const pointer_;
		};

		template <class T, std::size_t A = 64>
		using DArrayDDR = ArrayCPU<T, boost::alignment::aligned_allocator<T, A>, A>;

		template <class T, std::size_t A = 64>
		using DArrayHBW = ArrayCPU<T, hbw::allocator<T, A>, A>;

		template <class T, std::size_t S, std::size_t A = 64>
		class ArrayDDR : public DArrayDDR<T, A> {
		public:
			ArrayDDR() : DArrayDDR<T, A>(S) {	}
			ArrayDDR(const T& val) : DArrayDDR<T, A>(val, S) {	}
			template<typename OtherT, typename OtherAlloc>
			ArrayDDR(const std::vector<OtherT, OtherAlloc>& v) : DArrayDDR<T, A>(v, S) {}
			ArrayDDR(const ArrayDDR&) = default;
			ArrayDDR(ArrayDDR&&) = default;
			ArrayDDR& operator=(const ArrayDDR&) = default;
			ArrayDDR& operator=(ArrayDDR&&) = default;

			constexpr static std::size_t Size = S;
		};

		template <class T, std::size_t S, std::size_t A = 64>
		class ArrayHBW : public DArrayHBW<T, A> {
		public:
			ArrayHBW() : DArrayHBW<T, A>(S) { }
			ArrayHBW(const T& val) : DArrayHBW<T, A>(val, S) { }
			template<typename OtherT, typename OtherAlloc>
			ArrayHBW(const std::vector<OtherT, OtherAlloc>& v) : DArrayHBW<T, A>(v, S) {}
			ArrayHBW(const ArrayHBW&) = default;
			ArrayHBW(ArrayHBW&&) = default;
			ArrayHBW& operator=(const ArrayHBW&) = default;
			ArrayHBW& operator=(ArrayHBW&&) = default;

			constexpr static std::size_t Size = S;
		};

	}
}
