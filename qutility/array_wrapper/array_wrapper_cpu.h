#pragma once

#include <cstdio>
#include <vector>
#include "boost/align.hpp"

#include "hbw_posix_allocator.h"

namespace qutility {
	namespace array_wrapper {

		template <class T, std::size_t S, std::size_t A = 64>
		class ArrayDDR {
		public:
			ArrayDDR() : data_(S, T()), pointer_(&(data_.at(0))) {	}
			ArrayDDR(const T& val) : data_(S, val), pointer_(&(data_.at(0))) {	}

			constexpr static std::size_t Size = S;
			constexpr static std::size_t Alignment = A;

			inline const T* pointer() const { return pointer_; }
			inline T* pointer() { return pointer_; }
			inline const T& operator[](size_t pos) const { return data_[pos]; }
			inline T& operator[](size_t pos) { return data_[pos]; }

		private:
			std::vector<T, boost::alignment::aligned_allocator<T, Alignment>> data_;
			T* const pointer_;
		};

		template <class T, std::size_t S, std::size_t A = 64>
		class ArrayHBW {
		public:
			ArrayHBW() : data_(S, T()), pointer_(&(data_.at(0))) { }
			ArrayHBW(const T& val) : data_(S, val), pointer_(&(data_.at(0))) { }

			constexpr static std::size_t Size = S;
			constexpr static std::size_t Alignment = A;

			inline const T* pointer() const { return pointer_; }
			inline T* pointer() { return pointer_; }
			inline const T& operator[](size_t pos) const { return data_[pos]; }
			inline T& operator[](size_t pos) { return data_[pos]; }

		private:
			std::vector<T, hbw::allocator<T, Alignment>> data_;
			T* const pointer_;
		};

		template <class T, std::size_t A = 64>
		class DArrayDDR {
		public:
			DArrayDDR() = delete;
			DArrayDDR(std::size_t S) : data_(S, T()), size_(S), pointer_(&(data_.at(0))) {	}
			DArrayDDR(const T& val, std::size_t S) : data_(S, val), size_(S), pointer_(&(data_.at(0))) {	}
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

		private:
			std::vector<T, boost::alignment::aligned_allocator<T, Alignment>> data_;
			T* const pointer_;
		};

		template <class T, std::size_t A = 64>
		class DArrayHBW {
		public:
			DArrayHBW() = delete;
			DArrayHBW(std::size_t S) : data_(S, T()), size_(S), pointer_(&(data_.at(0))) { }
			DArrayHBW(const T& val, std::size_t S) : data_(S, val), size_(S), pointer_(&(data_.at(0))) { }

			const std::size_t size_;
			constexpr static std::size_t Alignment = A;

			inline const T* pointer() const { return pointer_; }
			inline T* pointer() { return pointer_; }
			inline const T& operator[](size_t pos) const { return data_[pos]; }
			inline T& operator[](size_t pos) { return data_[pos]; }

		private:
			std::vector<T, hbw::allocator<T, Alignment>> data_;
			T* const pointer_;
		};
	}
}
