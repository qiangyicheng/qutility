//! This file was modified from the hbw_allocator.h in memkind 1.10.0
//! The hbw_malloc was replaced by hbw_posix_memalign to keep the memory alignment
//! The infomation in original file is kept
//! Note that this actually will not compile successfully unless a memkind for Windows is present
//! My originally simple replacement for these functions are used for debug purpose if HBW_DEGUG_WIN is defined
//! Yicheng Qiang 20200602

// SPDX-License-Identifier: BSD-2-Clause
/* Copyright (C) 2014 - 2020 Intel Corporation. */

#pragma once

#include <boost/predef.h>

#if BOOST_OS_WINDOWS
#ifndef HBW_DEGUG_WIN
#define HBW_DEGUG_WIN
#endif // !HBW_DEGUG_WIN
#endif

#ifdef HBW_DEGUG_WIN
#include "hbw_debug_win.h"
#else
#include "hbwmalloc.h"
#endif // HBW_DEGUG_WIN

#include <stddef.h>
#include <new>
/*
 * Header file for the C++ allocator compatible with the C++ standard library allocator concepts.
 * More details in hbwallocator(3) man page.
 * Note: memory heap management is based on hbwmalloc, refer to the hbwmalloc man page for more information.
 *
 * Functionality defined in this header is considered as EXPERIMENTAL API.
 * API standards are described in memkind(3) man page.
 */
namespace hbw
{

	template <class T, size_t Alignment>
	class allocator
	{
	public:
		/*
		 *  Public member types required and defined by the standard library allocator concepts.
		 */
		typedef size_t size_type;
		typedef ptrdiff_t difference_type;
		typedef T* pointer;
		typedef const T* const_pointer;
		typedef T& reference;
		typedef const T& const_reference;
		typedef T value_type;

		template <class U>
		struct rebind {
			typedef hbw::allocator<U, Alignment> other;
		};

		/*
		 *  Public member functions required and defined by the standard library allocator concepts.
		 */
		allocator() throw() { }

		template <class U>
		allocator(const allocator<U, Alignment>&) throw() { }

		~allocator() throw() { }

		pointer address(reference x) const
		{
			return &x;
		}

		const_pointer address(const_reference x) const
		{
			return &x;
		}

		/*
		 *  Allocates n*sizeof(T) bytes of high bandwidth memory using hbw_posix_memalign() instead of hbw_malloc(). //Yicheng Qiang 20200602
		 *  Throws std::bad_alloc when cannot allocate memory.
		 */
		pointer allocate(size_type n, const void* = 0)
		{
			if (n > this->max_size()) {
				throw std::bad_alloc();
			}
			pointer result;
			hbw_posix_memalign((void**)(&result), Alignment, n * sizeof(T));
			if (!result) {
				throw std::bad_alloc();
			}
			return result;
		}

		/*
		 *  Deallocates memory associated with pointer returned by allocate() using hbw_free().
		 */
		void deallocate(pointer p, size_type n)
		{
			hbw_free(static_cast<void*>(p));
		}

		size_type max_size() const throw()
		{
			return size_t(-1) / sizeof(T);
		}

		template <class U, class... Args>
		void construct(U* p, Args&&... args) const
		{
			::new((void*)p) U(std::forward<Args>(args)...);
		}

		void destroy(pointer p)
		{
			p->~T();
		}
	};

	template <class T, class U, std::size_t Alignment>
	bool operator==(const allocator<T, Alignment>&, const allocator<U, Alignment>&)
	{
		return true;
	}

	template <class T, class U, std::size_t Alignment>
	bool operator!=(const allocator<T, Alignment>&, const allocator<U, Alignment>&)
	{
		return false;
	}

}
