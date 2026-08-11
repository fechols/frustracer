// MSVC built-in compatibility shim for the FidelityFX SDK 1.1.4 sources.
//
// The SDK is written against MSVC and only ever built on Windows upstream, so
// it reaches for a handful of Microsoft secure-CRT helpers (`_countof`,
// `wcscpy_s`, `strcpy_s`, `sprintf_s`, `swprintf_s`) that neither libstdc++ nor
// libc++ ships. The SDK gates `<windows.h>` itself; these are the gaps that
// remain. Force-included into every SDK translation unit by build.rs's
// `-include` flag rather than patched into the sources, because the sources are
// FETCHED (install-prerequisites.sh fsr3src) and must stay pristine — a local
// patch would silently un-apply on the next fetch or SDK bump.
//
// Ported from quinlight-player's cpp/ffx_msvc_compat.h, which established every
// workaround here on Linux/GCC; the same header serves macOS/clang unchanged,
// which is itself the useful datum — none of these gaps are compiler-specific,
// they are all "not MSVC".
//
// THE CONTEXT-SIZE OVERRIDE IS THE NON-OBVIOUS ONE. `FFX_SDK_DEFAULT_CONTEXT_SIZE`
// is a hard-coded 128 KB upstream, sized against a 2-byte `wchar_t`. Every other
// platform has a 4-byte `wchar_t`, which bloats the private FSR3 context struct
// past that bound. We pre-include `ffx_types.h` through its public path so its
// `#pragma once` is satisfied (making any later include of it a no-op), then
// redefine the constant to 1 MB so every subsequent SDK header sees the larger
// value. Order matters: define it after that include, not before.

#pragma once

#include <FidelityFX/host/ffx_types.h>

#undef FFX_SDK_DEFAULT_CONTEXT_SIZE
#define FFX_SDK_DEFAULT_CONTEXT_SIZE (1024 * 1024)

// `ffx_message.cpp` uses FFX_UNUSED in its non-Windows `#else` branch without
// including `ffx_util.h`, so that path does not compile as shipped. Mirrors the
// definition at ffx_util.h:55.
#ifndef FFX_UNUSED
  #define FFX_UNUSED(x) ((void)(x))
#endif

#if !defined(_MSC_VER)

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <cerrno>
// Not transitively available from the SDK's own includes on either standard
// library, so forced here for `std::wstring_convert`, `log2`, `floor`, etc.
#include <cmath>
#include <codecvt>
#include <locale>

#ifndef _countof
  #define _countof(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

// Two- and three-arg overloads of wcscpy_s: the two-arg form deduces the
// destination size from a fixed-size array reference, the three-arg form
// matches the explicit-count signature. Both null-terminate on every path,
// including the error ones — the secure-CRT contract the call sites assume.

template <std::size_t N>
inline int wcscpy_s(wchar_t (&dst)[N], const wchar_t* src) {
    if (!src) {
        if (N > 0) dst[0] = L'\0';
        return EINVAL;
    }
    std::size_t i = 0;
    while (i + 1 < N && src[i]) {
        dst[i] = src[i];
        ++i;
    }
    dst[i] = L'\0';
    return 0;
}

inline int wcscpy_s(wchar_t* dst, std::size_t count, const wchar_t* src) {
    if (!dst || count == 0) return EINVAL;
    if (!src) {
        dst[0] = L'\0';
        return EINVAL;
    }
    std::size_t i = 0;
    while (i + 1 < count && src[i]) {
        dst[i] = src[i];
        ++i;
    }
    dst[i] = L'\0';
    return 0;
}

// strcpy_s — the byte-copy twin, used by ffx_vk.cpp's UTF-8 conversion. Only
// the three-arg form: the SDK's single call site passes an explicit count.

inline int strcpy_s(char* dst, std::size_t count, const char* src) {
    if (!dst || count == 0) return EINVAL;
    if (!src) {
        dst[0] = '\0';
        return EINVAL;
    }
    std::size_t i = 0;
    while (i + 1 < count && src[i]) {
        dst[i] = src[i];
        ++i;
    }
    dst[i] = '\0';
    return 0;
}

// sprintf_s / swprintf_s: vsnprintf and vswprintf have the matching
// buffer-and-count truncation semantics, so both forms pass straight through.

template <std::size_t N>
inline int sprintf_s(char (&buf)[N], const char* fmt, ...) {
    std::va_list ap;
    va_start(ap, fmt);
    int r = std::vsnprintf(buf, N, fmt, ap);
    va_end(ap);
    return r;
}

inline int sprintf_s(char* buf, std::size_t count, const char* fmt, ...) {
    std::va_list ap;
    va_start(ap, fmt);
    int r = std::vsnprintf(buf, count, fmt, ap);
    va_end(ap);
    return r;
}

template <std::size_t N>
inline int swprintf_s(wchar_t (&buf)[N], const wchar_t* fmt, ...) {
    std::va_list ap;
    va_start(ap, fmt);
    int r = std::vswprintf(buf, N, fmt, ap);
    va_end(ap);
    return r;
}

inline int swprintf_s(wchar_t* buf, std::size_t count, const wchar_t* fmt, ...) {
    std::va_list ap;
    va_start(ap, fmt);
    int r = std::vswprintf(buf, count, fmt, ap);
    va_end(ap);
    return r;
}

#endif  // !_MSC_VER
