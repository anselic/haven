#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>

struct Slice {
    void* data;
    int length;
};

uint64_t rt_slice_len(struct Slice* slice) { return (uint64_t)slice->length; }

[[noreturn]] void rt_abort(const char* msg) {
    fprintf(stderr, "abort: %s\n", msg);
    exit(101);
}

void rt_puts(const char* s) { fputs(s, stdout); }

char* rt_f32_to_str(float f, int32_t precision) {
    static char buf[32];
    if (precision < 0)
        snprintf(buf, sizeof(buf), "%g", f);
    else
        snprintf(buf, sizeof(buf), "%.*g", precision, f);
    return buf;
}

char* rt_f64_to_str(double d, int32_t precision) {
    static char buf[32];
    if (precision < 0)
        snprintf(buf, sizeof(buf), "%g", d);
    else
        snprintf(buf, sizeof(buf), "%.*g", precision, d);
    return buf;
}

// Heap allocation, exposed to the language via `std/alloc` (see crt/std/alloc.hv).
// On failure these return NULL rather than aborting: `std/alloc` wraps the result
// in `Option<*T>`, so out-of-memory surfaces as `none` for the caller to handle.
void* rt_alloc(uint64_t size) {
    return malloc((size_t)size);
}
void* rt_realloc(void* ptr, uint64_t size) {
    return realloc(ptr, (size_t)size);
}
void rt_free(void* ptr) { free(ptr); }

// Denormal (subnormal) float handling, exposed via `std/dsp/denormal`.
//
// A denormal operand falls off the fast path on most cores, so a filter or a
// delay tail decaying towards silence - a signal made almost entirely of
// denormals - can take several times longer to process than the same block of
// ordinary audio. Real-time audio code therefore asks the FPU to flush
// denormals to zero instead of computing them, for the duration of a block.
//
//   x86: MXCSR bit 15 (FTZ, flush a denormal *result* to zero) and bit 6 (DAZ,
//        treat a denormal *input* as zero). SSE2 is baseline on x86-64.
//   aarch64: FPCR bit 24 (FZ) covers both.
//   elsewhere: a no-op, and correct - the block just runs at whatever denormals
//        cost on that target.
//
// `disable` returns the previous control word for `restore` to put back, because
// the setting is per-thread and shared with whoever called us: a plugin runs on
// the host's audio thread and must leave it as it found it.
#if defined(__x86_64__) || defined(_M_X64) || defined(__SSE2__) \
    || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <xmmintrin.h>
#define RT_DENORMAL_MXCSR 1
#endif

uint64_t rt_denormals_disable(void) {
#if defined(RT_DENORMAL_MXCSR)
    unsigned int csr = _mm_getcsr();
    _mm_setcsr(csr | 0x8040u); // (1 << 15) FTZ | (1 << 6) DAZ
    return (uint64_t)csr;
#elif defined(__aarch64__)
    uint64_t fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr | (1ull << 24)));
    return fpcr;
#else
    return 0;
#endif
}

void rt_denormals_restore(uint64_t saved) {
#if defined(RT_DENORMAL_MXCSR)
    _mm_setcsr((unsigned int)saved);
#elif defined(__aarch64__)
    __asm__ __volatile__("msr fpcr, %0" : : "r"(saved));
#else
    (void)saved;
#endif
}

// Bridges between haven's `str` (a NUL-terminated C string) and a `*u8` byte
// buffer, used by std/string. In haven's type system `str` is its own type: it
// is neither indexable nor `ptr_cast`-able, even though at the machine level it
// is just a bare pointer - the same representation as `*u8`. These are pure
// reinterpretations (no copy), the same way `strlen` already reads a `str`'s
// bytes. `rt_str_as_bytes` views a `str`'s bytes for reading; `rt_bytes_as_str`
// hands a byte buffer back as a `str` (the caller guarantees a trailing NUL).
uint8_t* rt_str_as_bytes(const char* s) { return (uint8_t*)s; }
const char* rt_bytes_as_str(uint8_t* p) { return (const char*)p; }