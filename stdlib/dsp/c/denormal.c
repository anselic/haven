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
#include <stdint.h>

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