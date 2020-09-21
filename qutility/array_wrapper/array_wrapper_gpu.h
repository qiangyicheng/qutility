#pragma once

#include <cstdio>
#include <vector>
#include "cuda_runtime.h"
#include "thrust/device_vector.h"
#include "detail.h"

namespace qutility {
	namespace array_wrapper {

		class ArrayGPUBase{};

		class ArrayGPUSelectDevice : ArrayGPUBase {
		public:
			ArrayGPUSelectDevice() = delete;
			ArrayGPUSelectDevice(const ArrayGPUSelectDevice&) = delete;
			ArrayGPUSelectDevice(ArrayGPUSelectDevice&&) = delete;
			ArrayGPUSelectDevice& operator=(const ArrayGPUSelectDevice&) = delete;
			ArrayGPUSelectDevice(int device) :device_(device) { cudaSetDevice(device); }
			~ArrayGPUSelectDevice() { }
			const int device_;
		};

		//Allocate memory in GPU
		//Note that it is done through a thrust interface.
		template <class T>
		class DArrayGPU :public ArrayGPUSelectDevice {
		public:
			DArrayGPU() = delete;
			DArrayGPU(std::size_t S, int device) : ArrayGPUSelectDevice(device), size_(S), data_(S, T()), pointer_(thrust::raw_pointer_cast(&(data_[0]))) {}
			DArrayGPU(const T& val, std::size_t S, int device) : ArrayGPUSelectDevice(device), size_(S), data_(S, val), pointer_(thrust::raw_pointer_cast(&(data_[0]))) {}
			template<typename OtherT, typename OtherAlloc>
			DArrayGPU(const std::vector<OtherT, OtherAlloc>& v, std::size_t S, int device) : ArrayGPUSelectDevice(device), size_(S), data_(detail::duplicate(v, S)), pointer_(thrust::raw_pointer_cast(&(data_[0]))) {}
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

		template <class T, std::size_t S>
		class ArrayGPU : public DArrayGPU<T> {
		public:
			ArrayGPU() = delete;
			ArrayGPU(int device) :DArrayGPU<T>(S, device) {}
			ArrayGPU(const T& val, int device) : DArrayGPU<T>(val, S, device) {}
			template<typename OtherT, typename OtherAlloc>
			ArrayGPU(const std::vector<OtherT, OtherAlloc>& v, int device) : DArrayGPU<T>(v, S, device) {}

			constexpr static std::size_t Size = S;
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

		template <class T, std::size_t S, std::size_t A = 64>
		class ArrayDDRPinned : public DArrayDDRPinned<T, A> {
		public:
			ArrayDDRPinned() : DArrayDDRPinned<T, A>(S) {}
			ArrayDDRPinned(const T& val) : DArrayDDRPinned<T, A>(val, S) {}
			~ArrayDDRPinned() {	}

			constexpr static std::size_t Size = S;
		};

	}
}

