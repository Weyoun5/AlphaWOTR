// For compilers other than MSVC (clang on macOS, gcc / clang on Linux).  The portable builder puts this folder on the
// include path, so the box's sources compile unchanged:
//   np_generator.inc: #include <intrin.h>, unsigned __int64, _umul128(a, b, &hi).
// Nothing else in the box is MSVC-specific (wotr_native.h guards __declspec with _WIN32).
// _umul128 is exact: the full 128-bit product of two 64-bit words (the low word returned, the high word in *hi), as the
// MSVC intrinsic gives it.
#pragma once
#if defined(_MSC_VER)
#error "portable/intrin.h is for compilers other than MSVC; MSVC has its own <intrin.h>"
#endif
#include <stdint.h>
#include <type_traits>
#ifndef __int64
// `unsigned __int64` must be the SAME type as uint64_t: np_generator.inc binds such fields to uint64_t& parameters.
// uint64_t is unsigned long on LP64 Linux and unsigned long long on macOS and on 64-bit Windows.
#if defined(__LP64__) && !defined(__APPLE__) && !defined(_WIN32)
#define __int64 long
#else
#define __int64 long long
#endif
#endif
static_assert(std::is_same<unsigned __int64, uint64_t>::value, "unsigned __int64 is not uint64_t on this platform");
static inline uint64_t _umul128(uint64_t a, uint64_t b, uint64_t* hi) {
    const unsigned __int128 p = static_cast<unsigned __int128>(a) * b;
    *hi = static_cast<uint64_t>(p >> 64);
    return static_cast<uint64_t>(p);
}
