#pragma once

#include <cstdio>
#include <vector>
#include "cuda_runtime.h"
#include "thrust/device_vector.h"
#include "detail.h"

namespace qutility {
	namespace array_wrapper {
		class ArrayGPUBase {
		public:
			ArrayGPUBase() = delete;
			ArrayGPUBase(const ArrayGPUBase&) = delete;
			ArrayGPUBase(ArrayGPUBase&&) = delete;
			ArrayGPUBase& operator=(const ArrayGPUBase&) = delete;
			ArrayGPUBase(int device) :device_(device) { cudaSetDevice(device); }
			~ArrayGPUBase() { }
			const int device_;
		};

		//Allocate memory in GPU
		//Note that it is done through a thrust interface.
		template <class T, std::size_t S>
		class ArrayGPU :public ArrayGPUBase {
		public:
			ArrayGPU() = delete;
			ArrayGPU(int device) : ArrayGPUBase(device), data_(S, T()), pointer_(thrust::raw_pointer_cast(&(data_[0]))) {}
			ArrayGPU(const T& val, int device) : ArrayGPUBase(device), data_(S, val), pointer_(thrust::raw_pointer_cast(&(data_[0]))) {}
			template<typename OtherT, typename OtherAlloc>
			ArrayGPU(const std::vector<OtherT, OtherAlloc>& v, int device) : ArrayGPUBase(device), data_(detail::duplicate(v, S)), pointer_(thrust::raw_pointer_cast(&(data_[0]))) {}
			operator T* () { return pointer_; }
			operator const T* () const { return pointer_; }
			T* operator+(size_t shift) { return pointer_ + shift; }
			const T* operator+(size_t shift) const { return pointer_ + shift; }

			constexpr static std::size_t Size = S;

			const T* pointer() const { return pointer_; }
			T* pointer() { return pointer_; }
			inline const auto operator[](size_t pos) const { return data_[pos]; }
			inline auto operator[](size_t pos) { return data_[pos]; }

		private:
			thrust::device_vector<T> data_;
			T* const pointer_;
		};

		template <class T, std::size_t S, std::size_t A = 64>
		class ArrayDDRPinned {
		public:
			ArrayDDRPinned() : data_(S, T()), pointer_(&(data_.at(0))) {
				cudaHostRegister((void*)pointer_, Size * sizeof(T), cudaHostRegisterDefault);
			}
			ArrayDDRPinned(const T& val) : data_(S, val), pointer_(&(data_.at(0))) {
				cudaHostRegister((void*)pointer_, Size * sizeof(T), cudaHostRegisterDefault);
			}
			~ArrayDDRPinned() {
				cudaHostUnregister((void*)pointer_);
			}
			operator T* () { return pointer_; }
			operator const T* () const { return pointer_; }
			T* operator+(size_t shift) { return pointer_ + shift; }
			const T* operator+(size_t shift) const { return pointer_ + shift; }

			constexpr static std::size_t Size = S;
			constexpr static std::size_t Alignment = A;

			inline const T* pointer() const { return &(data_.at(0)); }
			inline T* pointer() { return &(data_.at(0)); }
			inline const T& operator[](size_t pos) const { return data_[pos]; }
			inline T& operator[](size_t pos) { return data_[pos]; }

		private:
			std::vector<T, boost::alignment::aligned_allocator<T, Alignment>> data_;
			T* const pointer_;
		};


		//Allocate memory in GPU
		//Note that it is done through a thrust interface.
		template <class T>
		class DArrayGPU :public ArrayGPUBase {
		public:
			DArrayGPU() = delete;
			DArrayGPU(std::size_t S, int device) : ArrayGPUBase(device), size_(S), data_(S, T()), pointer_(thrust::raw_pointer_cast(&(data_[0]))) {}
			DArrayGPU(const T& val, std::size_t S, int device) : ArrayGPUBase(device), size_(S), data_(S, val), pointer_(thrust::raw_pointer_cast(&(data_[0]))) {}
			template<typename OtherT, typename OtherAlloc>
			DArrayGPU(const std::vector<OtherT, OtherAlloc>& v, std::size_t S, int device) : ArrayGPUBase(device), size_(S), data_(detail::duplicate(v, S)), pointer_(thrust::raw_pointer_cast(&(data_[0]))) {}
			operator T* () { return pointer_; }
			operator const T* () const { return pointer_; }
			T* operator+(size_t shift) { return pointer_ + shift; }
			const T* operator+(size_t shift) const { return pointer_ + shift; }
			const std::size_t size_;
			const T* pointer() const { return pointer_; }
			T* pointer() { return pointer_; }
			inline const auto operator[](size_t pos) const { return data_[pos]; }
			inline auto operator[](size_t pos) { return data_[pos]; }

		private:
			thrust::device_vector<T> data_;
			T* const pointer_;
		};

		template <class T, std::size_t A = 64>
		class DArrayDDRPinned {
		public:
			DArrayDDRPinned() = delete;
			DArrayDDRPinned(std::size_t S) :size_(S), data_(S, T()), pointer_(&(data_.at(0))) {
				cudaHostRegister((void*)pointer_, size_ * sizeof(T), cudaHostRegisterDefault);
			}
			DArrayDDRPinned(const T& val, std::size_t S) :size_(S), data_(S, val), pointer_(&(data_.at(0))) {
				cudaHostRegister((void*)pointer_, size_ * sizeof(T), cudaHostRegisterDefault);
			}
			~DArrayDDRPinned() {
				cudaHostUnregister((void*)pointer_);
			}
			operator T* () { return pointer_; }
			operator const T* () const { return pointer_; }
			T* operator+(size_t shift) { return pointer_ + shift; }
			const T* operator+(size_t shift) const { return pointer_ + shift; }

			const std::size_t size_;
			constexpr static std::size_t Alignment = A;

			inline const T* pointer() const { return &(data_.at(0)); }
			inline T* pointer() { return &(data_.at(0)); }
			inline const T& operator[](size_t pos) const { return data_[pos]; }
			inline T& operator[](size_t pos) { return data_[pos]; }

		private:
			std::vector<T, boost::alignment::aligned_allocator<T, Alignment>> data_;
			T* const pointer_;
		};

	}
}

