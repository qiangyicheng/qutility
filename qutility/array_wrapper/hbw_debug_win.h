#pragma once

#include <immintrin.h>
#include <cerrno>

inline int hbw_check_available()
{
	return ENODEV;
}

inline int hbw_posix_memalign(void **memptr, size_t alignment, size_t size)
{
	*memptr = _mm_malloc(size, alignment);
	return 0;
}

inline void hbw_free(void * memptr) {
	_mm_free(memptr);
}
