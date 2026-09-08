// Atomic loads, stores and read-modify-write operations, exposed via `std/atomic`.
//
// haven has no atomic intrinsics and no way to spell a memory ordering in its
// own type system, so the operations are bound here as ordinary externs, one per
// (operation, width, ordering) triple. That is deliberately more symbols than a
// single entry point taking an ordering parameter would need: `memory_order` has
// to be a *compile-time constant* for the compiler to emit the right instruction
// (a relaxed load is a plain `mov` on x86, an acquire load on aarch64 is `ldar`,
// and a runtime ordering argument collapses both into a switch over the
// strongest case). Spelling the ordering into the symbol name keeps every one of
// these a single instruction.
//
// Only the orderings std actually needs are provided, rather than the full
// cross product: relaxed and acquire loads, relaxed and release stores,
// acq_rel read-modify-writes, and sequentially consistent versions of load and
// store for code that would rather not think about it. That is enough to build
// the SPSC queue in `std/spsc` and to hand a parameter value from a UI thread to
// an audio thread, which is what this exists for.
//
// The `_Atomic` cast: haven's externs deal in plain `uint32_t*`/`uint64_t*`, so
// each function casts to `_Atomic` before operating. This is well-defined in
// practice for exactly the reason it looks dubious - C requires `_Atomic T` and
// `T` to have the same size and alignment only for the lock-free types, and
// `uint32_t`/`uint64_t` are lock-free on every target haven targets. The
// `is_lock_free` probes below let a caller confirm that at runtime rather than
// take it on trust.

#include <stdatomic.h>
#include <stdint.h>

#define ATOMIC_OPS(W, T)                                                        \
    T rt_atomic_load_relaxed_##W(T* p) {                                        \
        return atomic_load_explicit((_Atomic T*)p, memory_order_relaxed);       \
    }                                                                           \
    T rt_atomic_load_acquire_##W(T* p) {                                        \
        return atomic_load_explicit((_Atomic T*)p, memory_order_acquire);       \
    }                                                                           \
    T rt_atomic_load_seqcst_##W(T* p) {                                         \
        return atomic_load_explicit((_Atomic T*)p, memory_order_seq_cst);       \
    }                                                                           \
    void rt_atomic_store_relaxed_##W(T* p, T v) {                               \
        atomic_store_explicit((_Atomic T*)p, v, memory_order_relaxed);          \
    }                                                                           \
    void rt_atomic_store_release_##W(T* p, T v) {                               \
        atomic_store_explicit((_Atomic T*)p, v, memory_order_release);          \
    }                                                                           \
    void rt_atomic_store_seqcst_##W(T* p, T v) {                                \
        atomic_store_explicit((_Atomic T*)p, v, memory_order_seq_cst);          \
    }                                                                           \
    T rt_atomic_fetch_add_relaxed_##W(T* p, T v) {                              \
        return atomic_fetch_add_explicit((_Atomic T*)p, v, memory_order_relaxed);\
    }                                                                           \
    T rt_atomic_fetch_add_acqrel_##W(T* p, T v) {                               \
        return atomic_fetch_add_explicit((_Atomic T*)p, v, memory_order_acq_rel);\
    }                                                                           \
    T rt_atomic_fetch_sub_acqrel_##W(T* p, T v) {                               \
        return atomic_fetch_sub_explicit((_Atomic T*)p, v, memory_order_acq_rel);\
    }                                                                           \
    T rt_atomic_fetch_and_acqrel_##W(T* p, T v) {                               \
        return atomic_fetch_and_explicit((_Atomic T*)p, v, memory_order_acq_rel);\
    }                                                                           \
    T rt_atomic_fetch_or_acqrel_##W(T* p, T v) {                                \
        return atomic_fetch_or_explicit((_Atomic T*)p, v, memory_order_acq_rel); \
    }                                                                           \
    T rt_atomic_exchange_acqrel_##W(T* p, T v) {                                \
        return atomic_exchange_explicit((_Atomic T*)p, v, memory_order_acq_rel);\
    }                                                                           \
    /* `expected` is in/out: on failure it is updated to the value actually  */ \
    /* found, which is what lets a CAS loop retry without a separate load.   */ \
    int32_t rt_atomic_cas_##W(T* p, T* expected, T desired) {                   \
        return atomic_compare_exchange_strong_explicit(                         \
            (_Atomic T*)p, expected, desired,                                   \
            memory_order_acq_rel, memory_order_acquire) ? 1 : 0;                \
    }                                                                           \
    int32_t rt_atomic_is_lock_free_##W(void) {                                  \
        _Atomic T probe;                                                        \
        return atomic_is_lock_free(&probe) ? 1 : 0;                             \
    }

ATOMIC_OPS(u32, uint32_t)
ATOMIC_OPS(u64, uint64_t)

#undef ATOMIC_OPS

// Standalone fences, for ordering ordinary (non-atomic) accesses around a
// relaxed atomic. The SPSC queue uses these rather than acquire/release loads
// and stores in one place: the payload it publishes is an ordinary buffer write,
// not an atomic one, so the ordering has to be established separately from the
// index update.
void rt_atomic_fence_acquire(void) { atomic_thread_fence(memory_order_acquire); }
void rt_atomic_fence_release(void) { atomic_thread_fence(memory_order_release); }
void rt_atomic_fence_seqcst(void)  { atomic_thread_fence(memory_order_seq_cst); }
